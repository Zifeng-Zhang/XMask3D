# XMask3D: Cross-modal Mask Reasoning for Open Vocabulary 3D Semantic Segmentation

Created by [Ziyi Wang*](https://wangzy22.github.io/), [Yanbo Wang*](https://Yanbo-23.github.io/), [Xumin Yu](https://yuxumin.github.io/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=zh-CN).


This repository is a pyTorch implementation of our NeurIPS 2024 paper **XMask3D**.

XMask3D is a framework for open vocabulary 3D semantic segmentation that improves fine-grained boundary delineation by aligning 3D features with a 2D-text embedding space at the mask level. Using a mask generator based on a pre-trained diffusion model, it enables precise textual control over dense pixel representations, enhancing the versatility of generated masks. By integrating 3D global features into a 2D denoising UNet, XMask3D adds 3D geometry awareness to mask generation. The resulting 2D masks align 3D representations with vision-language features, yielding competitive segmentation performance across benchmarks.

[[arXiv]()][[Project Page]()]
![intro](fig/pipeline.jpg)


## Installation
Follow the [installation.md](installation.md) to install all required packages so you can do the training & evaluation afterwards.

## Data Preparation

### Scannet Dataset
- Download the 12 views image dataset of ShapeNet from [here]([http://maxwell.cs.umass.edu/mvcnn-data/shapenet55v1.tar](https://cloud.tsinghua.edu.cn/f/dd9ef45fb00e427eae23/?dl=1)). The images are rendered by [MVCNN](https://github.com/suhangpro/mvcnn).

