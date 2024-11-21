# XMask3D: Cross-modal Mask Reasoning for Open Vocabulary 3D Semantic Segmentation

Created by [Ziyi Wang*](https://wangzy22.github.io/), [Yanbo Wang*](https://Yanbo-23.github.io/), [Xumin Yu](https://yuxumin.github.io/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=zh-CN).


This repository is a pyTorch implementation of our NeurIPS 2024 paper **XMask3D**.

**XMask3D** is a framework for open vocabulary 3D semantic segmentation that improves fine-grained boundary delineation by aligning 3D features with a 2D-text embedding space at the mask level. Using a mask generator based on a pre-trained diffusion model, it enables precise textual control over dense pixel representations, enhancing the versatility of generated masks. By integrating 3D global features into a 2D denoising UNet, XMask3D adds 3D geometry awareness to mask generation. The resulting 2D masks align 3D representations with vision-language features, yielding competitive segmentation performance across benchmarks.

[[arXiv](https://arxiv.org/abs/2411.13243)]
![intro](fig/pipeline.jpg)


## Installation
- Follow the [installation.md](installation.md) to install all required packages so you can do the training & evaluation afterwards.

## Data Preparation

- For convenience, the download link for the processed dataset is provided here. You can download the dataset by executing the command below.
```bash
sh scripts/download_datasets.sh
```

## Pre-trained Model Preparation

- For this project, you will need the pre-trained CLIP model and the Stable Diffusion model. Due to the instability of official network links, we provide alternative download options below:
```bash
# CLIP ViT-Large Patch14
cd /path/to/your/workspace
wget -O openai.tar.gz https://cloud.tsinghua.edu.cn/f/3890f1df1c5248a7a6e8/?dl=1
tar -xzvf openai.tar.gz
# Stable Diffusion v1.3 Checkpoint
wget -O sd_model.tar.gz https://cloud.tsinghua.edu.cn/f/8dce9b137f574e6eb57c/?dl=1
tar -xzvf sd_model.tar.gz
```

## Usage

### Training

```
sh run/train.sh --exp_dir=<EXPERIMENT_DIRECTORY> --config=<CONFIG_FILE>
```

- For example, to train on the ScanNet B15N4 benchmark, run:

```
sh run/train.sh --exp_dir=out/exp_b15n4 --config=config/scannet/xmask3d_scannet_B15N4.yaml
```

### Resume

```
sh run/resume.sh --exp_dir=<EXPERIMENT_DIRECTORY> --config=<CONFIG_FILE>
```

- For example, to resume the last ckpt on the ScanNet B15N4 benchmark, run:

```
sh run/resume.sh --exp_dir=out/exp_b15n4 --config=config/scannet/xmask3d_scannet_B15N4.yaml
```
### Inference

```
sh run/infer.sh --exp_dir=<EXPERIMENT_DIRECTORY> --config=<CONFIG_FILE> --ckpt_name=<CKPT_NAME>
```

- For example, to run inference using the checkpoint ```b15n4.pth.tar``` on the ScanNet B15N4 benchmark, execute the following command:

```
sh run/infer.sh --exp_dir=out/exp_b15n4 --config=config/scannet/xmask3d_scannet_B15N4.yaml --ckpt_name=b15n4.pth.tar
```


## Checkpoint

| **Benchmark**         | **hIoU / mIoU<sub>b</sub> / mIoU<sub>n</sub>** | **Download Link**       |
|-----------------------|-----------------------------------------------|--------------------------|
| **Scannet B15N4**     | 70.0 / 69.8 / 70.2                            | [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/dc6459840fe542a288f8/?dl=1) [[Google]](https://drive.google.com/file/d/1A-QsKXwrvXLKedLQdWl6qoR-JuFdrRO-/view?usp=sharing)       |
| **Scannet B12N7**     | 61.7 / 70.2 / 55.1                            | [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/5d9671b1f0a9499d821e/?dl=1) [[Google]](https://drive.google.com/file/d/1ZSdoLcR8fr1MtXy5n3y-diFJJ5j1YLF7/view?usp=sharing)      |
| **Scannet B10N9**     | 55.7 / 76.5 / 43.8                            | [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/e7f41cd454a1469d865b/?dl=1) [[Google]](https://drive.google.com/file/d/1bsHBoFDXZIo-3UU1JE0zXRbAJ9q9_4Be/view?usp=sharing)      |
| **Scannet B170N30**   | 18.0 / 27.8 / 13.3                            | [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/b88c57c7093740f59d75/?dl=1) [[Google]](https://drive.google.com/file/d/1VgN6WukdOBBxL4C1t0mve6ZeLDVQKwoh/view?usp=sharing)      |
| **Scannet B150N50**   | 15.5 / 24.4 / 11.4                            | [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/a55ee7d62caa4e82bb77/?dl=1) [[Google]](https://drive.google.com/file/d/1tQjGznq2x8df7c_HnlnsOb_peMgq-_ZL/view?usp=sharing)       |

## Citation

If you find our work useful in your research, please consider citing:

```
@article{wang2024xmask3d,
  title={XMask3D: Cross-modal Mask Reasoning for Open Vocabulary 3D Semantic Segmentation},
  author={Wang, Ziyi and Wang, Yanbo and Yu, Xumin and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2411.13243},
  year={2024}
}
```
