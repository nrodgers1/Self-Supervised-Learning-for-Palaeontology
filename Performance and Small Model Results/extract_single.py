#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import argparse
import resource
from pathlib import Path

# NOTE: These MUST be set before importing torch or transformers on this
# specific PyTorch + transformers + Apple Silicon combination to avoid
# MPS crashes and hangs during inference.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login


class PreprocessedImageDataset(torch.utils.data.Dataset):
    """
    Torch Dataset that loads images from an ImageFolder structure,
    validates them, and applies a Hugging Face image preprocessor.
    """
    def __init__(self, root_folder, preprocessor):
        self.dataset = datasets.ImageFolder(root_folder)
        self.preprocessor = preprocessor

        # Validate images
        valid_samples = []
        for path, label in tqdm(self.dataset.samples, desc="Validating images"):
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_samples.append((path, label))
            except (OSError, ValueError):
                print(f"Skipping {path}: not a valid image.")
        self.dataset.samples = valid_samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = Image.open(path).convert("RGB")
        inputs = self.preprocessor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs, label


def load_model_and_preprocessor(model_id, force_cpu=False):
    """Load model natively favoring Apple Silicon (MPS)."""
    if not force_cpu and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    try:
        model = AutoModel.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            dtype=dtype
        ).to(device)
        preprocessor = AutoImageProcessor.from_pretrained(model_id)
        print(f"Model and preprocessor loaded successfully on: {device} (dtype={dtype})")
    except Exception as e:
        raise ValueError(f"Could not load Hugging Face model/preprocessor '{model_id}': {e}")

    model.eval()
    return model, device, preprocessor


def extract_features_batch(image_folder_path, model, preprocessor, device, batch_size=32, save_loc=None):
    model.eval()
    dataset = PreprocessedImageDataset(image_folder_path, preprocessor)
    class_names = dataset.dataset.classes
    file_paths = [s[0] for s in dataset.dataset.samples]

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    all_features, all_labels = [], []
    device_str = str(device).lower()
    is_mps = (device_str == "mps")

    # --- BENCHMARK INITIALIZATION ---
    if is_mps:
        torch.mps.empty_cache()
        torch.mps.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch_inputs, labels in tqdm(dataloader, desc="Processing batches"):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            outputs = model(**batch_inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                raise ValueError("Model outputs do not contain expected features.")

            all_features.append(features.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy().astype(np.int64))

    #  Synchronize MPS before stopping timer
    if is_mps:
        torch.mps.synchronize()

    end_time = time.perf_counter()

    # --- CALCULATE AND PRINT BENCHMARK METRICS ---
    final_features = np.concatenate(all_features, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)

    total_seconds = end_time - start_time
    num_images = len(final_features)
    images_per_second = num_images / total_seconds

    # Memory reporting
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_gb = peak_rss_bytes / (1024 ** 3)

    if is_mps:
        mps_allocated_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
        mps_driver_gb = torch.mps.driver_allocated_memory() / (1024 ** 3)
        memory_section = (
            "Peak System RAM Usage:  {:.2f} GB\n"
            "MPS Tensors Allocated:  {:.2f} GB\n"
            "MPS Driver Allocated:   {:.2f} GB\n"
            "(Note: CPU and GPU share unified memory on Apple Silicon)\n"
        ).format(peak_rss_gb, mps_allocated_gb, mps_driver_gb)
    else:
        memory_section = (
            "Peak System RAM Usage:  {:.2f} GB\n"
            "(Note: CPU benchmark uses single-threaded execution due to\n"
            " stability requirements of the PyTorch/transformers/MPS stack)\n"
        ).format(peak_rss_gb)

    report = (
        "\n========================================\n"
        "  COMPUTATIONAL BENCHMARK PROFILE ({})\n"
        "========================================\n"
        "Processed Images:       {}\n"
        "Total Extraction Time:  {:.2f} seconds\n"
        "Throughput Speed:       {:.2f} images/second\n"
        "{}"
        "========================================\n"
    ).format(device_str.upper(), num_images, total_seconds, images_per_second, memory_section)

    print(report)

    if save_loc is not None:
        try:
            with open(save_loc, "w", encoding="utf-8") as f:
                f.write(report)
            print("Benchmark profile successfully saved to: {}".format(save_loc))
        except Exception as e:
            print("Warning: Could not save benchmark to file: {}".format(e))

    return final_features, final_labels, file_paths, class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from an image dataset and benchmark memory.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the root image folder")
    parser.add_argument("--save_loc", type=str, required=True, help="Path to save the text benchmark report")
    parser.add_argument("--model_id", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m", help="Hugging Face Model ID")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the DataLoader")
    parser.add_argument("--force_cpu", action="store_true", help="Force the script to run on CPU only")

    args = parser.parse_args()

    # Automatically logs into HuggingFace if HF_TOKEN is in the environment
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])

    # Load Model
    model, device, preprocessor = load_model_and_preprocessor(args.model_id, args.force_cpu)

    # Extract Features
    _ = extract_features_batch(
        image_folder_path=args.image_folder,
        model=model,
        preprocessor=preprocessor,
        device=device,
        batch_size=args.batch_size,
        save_loc=args.save_loc
    )