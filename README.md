# MaskFreeVIS

Mask-Free Video Instance Segmentation [CVPR 2023].

This is the official pytorch implementation of [MaskFreeVIS](https://github.com/SysCV/MaskFreeVis/) built on the open-source detectron2. **We will finish the code organization/cleaning before the end of April. Stay tuned!**

Highlights
-----------------
- **High-performing** video instance segmentation **without using any** video masks or even image mask labels on all datasets. Using ResNet-50 and built on Mask2Former, MaskFreeVIS achieves 46.6 AP without using video masks, and 42.5 AP without any masks on YTVIS 2019.
- **Novelty:**: a new **parameter-free** Temporal KNN-patch Loss (TK-Loss), which leverages temporal masks consistency using unsupervised one-to-k patch correspondence.
- **Simple:** TK-Loss is flexible to intergrated with existing SOTA VIS models, has no trainable parameters.

Introduction
-----------------
The recent advancement in Video Instance Segmentation (VIS) has largely been driven by the use of deeper and increasingly data-hungry transformer-based models. However, video masks are tedious and expensive to annotate, limiting the scale and diversity of existing VIS datasets. In this work, we aim to remove the mask-annotation requirement. We propose MaskFreeVIS, achieving highly competitive VIS performance, while only using bounding box annotations for the object state. We leverage the rich temporal mask consistency constraints in videos by introducing the Temporal KNN-patch Loss (TK-Loss), providing strong mask supervision without any labels. Our TK-Loss finds one-to-many matches across frames, through an efficient patch-matching step followed by a K-nearest neighbor selection. A consistency loss is then enforced on the found matches. Our mask-free objective is simple to implement, has no trainable parameters, is computationally efficient, yet outperforms baselines employing, e.g., state-of-the-art optical flow to enforce temporal mask consistency. We validate MaskFreeVIS on the YouTube-VIS 2019/2021, OVIS and BDD100K MOTS benchmarks. The results clearly demonstrate the efficacy of our method by drastically narrowing the gap between fully and weakly-supervised VIS performance.


Methods
-----------------
<img width="1096" alt="image" src="https://user-images.githubusercontent.com/17427852/228353991-ff09784f-9afd-4ac2-bddf-c5b2763d25e6.png">
