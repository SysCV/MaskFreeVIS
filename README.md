# MaskFreeVIS

Mask-Free Video Instance Segmentation [CVPR 2023].

This is the official pytorch implementation of [MaskFreeVIS](https://github.com/SysCV/MaskFreeVis/) built on the open-source detectron2. **We will finish the code organization/cleaning before the end of April. Stay tuned!**

Highlights
-----------------
- **High-performing** video instance segmentation **without using any** video masks or even image mask labels on all datasets. Using ResNet-50 and built on Mask2Former, MaskFreeVIS achieves 46.6 AP without using video masks, and 42.5 AP without any masks on YTVIS 2019.
- **Novelty:**: a new **parameter-free** Temporal KNN-patch Loss (TK-Loss), which leverages temporal masks consistency using unsupervised one-to-k patch correspondence.
- **Simple:** TK-Loss is flexible to intergrated with existing SOTA VIS models, has no trainable parameters.


Methods
-----------------
<img width="1096" alt="image" src="https://user-images.githubusercontent.com/17427852/228353991-ff09784f-9afd-4ac2-bddf-c5b2763d25e6.png">
