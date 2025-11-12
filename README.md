# Self-Supervised-Learning-for-Palaeontology


This code is used to generate the results of the paper "Complexity Begets Simplicity: Self-Supervised Learning for
Palaeontological Images with Few or No Labels" written by Niall Rodgers at the UK Centre for Astrobiology, at the University of Edinburgh. He can be contacted at Niall.Rodgers@ed.ac.uk. 


This Github is considered a work in progess until the acceptance of the this work for publication when I will convert to a permenant repository with a fixed DOI. However, I wish to get the idea into the Paleontology community as fast as possible.


This code comprises of two jupyter notebooks. One notebook will take a given dataset and the generate all analysis and classifaction for that dataset as shown 
in the main text while the other notebook takes in an image and outputs a visualisation of the model output. 

The code is written at a relatively high-level and is commented for reproduciblity. For the purposes of the paper we do cross-validation and conduct the inference in batches which adds extra steps 
to the code complexity. The recommended way to test using a DINOv3 model would be to go directly the Hugging Face model card and follow the steps given there https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m .

The Hugging Face model card gives the steps you need to take to access the model, creating an account and shows the minimal code required to extract features from an image. 

To run this code you need to autheticate your Hugging Face account (this is free) due to the access rules of the model provider. We provide this code to support open science and highlight
how simple this pipeline is. In pratical use cases, it is probably easiest to follow the tutorials given by Hugging Face or online to implement your own version as this is a more widely known
appraoch outside of Paleontology.

Depending on your operating system, Hugging Face version and access to a GPU some functions may not run as expect and might need some minor edits. This code was written using a MacBook Pro and Apple Metal GPU running on the devlopment version of Hugging Face (although Dinov3 should now be included in the stable version). 



Results folders give CSV and .tex files of the raw results as well as all the figures which appear in the manuscript. 


Sources for the data used in this work are given in the manuscript in detail. We extract datasets from the repositories which were published alongside the works cited.


This work was "Built with DINOv3". And works using DINOv3 remain subject to the licensing agreement which was released with this model. The DINOv3 licensing agreement is also copied into this repo for reference. 
