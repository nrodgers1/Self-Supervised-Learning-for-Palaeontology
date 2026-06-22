#!/usr/bin/env python
# coding: utf-8

import subprocess
import sys
import os
from tqdm import tqdm

# =================================================================
# CONFIGURATION
# =================================================================
DATASETS = {
    "pollen": '/Users/nrodger3/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Unsupervised Morphology Project/Paper Code and Data/Pollen',
    "radiolaria": '/Users/nrodger3/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Unsupervised Morphology Project/Paper Code and Data/Lille_Radiolarains_Image_Datasets/S',
    "tracks": '/Users/nrodger3/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Unsupervised Morphology Project/Paper Code and Data/Tracks',
    "diverse": '/Users/nrodger3/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Unsupervised Morphology Project/Paper Code and Data/Diverse Fossils',
    "foraminifera": '/Users/nrodger3/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Unsupervised Morphology Project/Paper Code and Data/Endless_Forams_training_set',
}

MODELS = {
    "DINO_Small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "DINO_Large": "facebook/dinov3-vitl16-pretrain-lvd1689m"
}

# These datasets will run on BOTH MPS and CPU
FORCE_CPU_DATASETS = ["tracks", "radiolaria"]

MASTER_OUTPUT_FILE = "all_benchmarks.txt"
# =================================================================

def run_benchmarks():
    tasks = []
    for model_key, model_id in MODELS.items():
        for dataset_name, dataset_path in DATASETS.items():
            # 1. Always run the default (MPS) version
            tasks.append({
                "model_key": model_key,
                "model_id": model_id,
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "force_cpu": False
            })
            
            # 2. If it's a CPU-specific dataset, also add a CPU version
            if dataset_name in FORCE_CPU_DATASETS:
                tasks.append({
                    "model_key": model_key,
                    "model_id": model_id,
                    "dataset_name": dataset_name,
                    "dataset_path": dataset_path,
                    "force_cpu": True
                })

    print(f"Total benchmarks to run: {len(tasks)}")
    
    # Initialize/Clear the master file
    with open(MASTER_OUTPUT_FILE, "w") as f:
        f.write("BENCHMARK REPORT CONSOLIDATION\n")
        f.write("==============================\n\n")

    for task in tqdm(tasks, desc="Overall Benchmarking Progress"):
        model_name = task["model_key"]
        ds_name = task["dataset_name"]
        device_label = "CPU" if task["force_cpu"] else "MPS"
        
        # Temp filename for this specific run
        temp_report = f"temp_{model_name}_{ds_name}_{device_label}.txt"

        # Build command
        cmd = [
            sys.executable, 
            "extract_single.py",
            "--image_folder", task["dataset_path"],
            "--save_loc", temp_report,
            "--model_id", task["model_id"],
            "--batch_size", "64" 
        ]

        if task["force_cpu"]:
            cmd.append("--force_cpu")

        # Execute
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"\n⚠️ Error running {model_name} on {ds_name} ({device_label}):")
            print(result.stderr)
        else:
            # Append content to master file
            try:
                with open(temp_report, "r") as f:
                    report_content = f.read()
                
                with open(MASTER_OUTPUT_FILE, "a") as master:
                    master.write(f"\n\n{'='*80}\n")
                    master.write(f"MODEL: {model_name} | DATASET: {ds_name} | DEVICE: {device_label}\n")
                    master.write(f"{'='*80}\n")
                    master.write(report_content)
                
                print(f"\n✅ Added: {model_name} on {ds_name} [{device_label}] to {MASTER_OUTPUT_FILE}")
            
            finally:
                # Cleanup temporary file
                if os.path.exists(temp_report):
                    os.remove(temp_report)

    print(f"\n🎉 All done! Final report saved to: {MASTER_OUTPUT_FILE}")

if __name__ == "__main__":
    if "HF_TOKEN" not in os.environ:
        print("Warning: HF_TOKEN not found. Ensure you have run 'huggingface-cli login'.")
    
    run_benchmarks()