This folder contains time and performance benchmarks comparing DINOv3 Vit-L and Vit-S in terms of performance. As well as giving the supervised learning results for Vit-S.

There is a small python script to run feature extraction and another python script which applies and this benchmarks it for each dataset. The benchmarking is setup to be system specfic and results may vary across systems. But the approximate trend that feature extraction is easy to run on consumer hardware is well and known and expected to generalise. The benchmark measures the timings of the performance on apple silicon so will need to be modified for other envirnoment types.

There is also a notebook which is the same as the main code but also gives the option to update the model code to select the small model which was used to generate the attached results.


The subfolders contain all the supervised learning results and the visulisation of the feature vectors for the small DINOv3 model Vit-S.
