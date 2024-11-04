import os
from setuptools import find_packages, setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 13], "Requires PyTorch >= 1.13.1"


setup(
    name="xmask3d",
    author="Yanbo Wang",
    url="https://github.com/wangzy22/XMask3D.git",
    description="Cross-modal Mask Reasoning for Open Vocabulary 3D Semantic Segmentation",
    python_requires=">=3.8",
    packages=find_packages(exclude=("config")),
    install_requires=[
        "timm==0.6.11",  # freeze timm version for stabliity
        "opencv-python==4.6.0.66",
        "diffdist==0.1",
        "nltk>=3.6.2",
        "einops>=0.3.0",
        "openai-clip",
        "wandb>=0.12.11",
        "Pillow == 9.4.0",
        "numpy == 1.26.4",
        "tensorboardX",
        "triton==2.0.0",
        "open3d", # optional, can be used for visualization
        "omegaconf==2.1.1",
        "open-clip-torch==2.0.2",
        "SharedArray",
        f"mask2former @ file://localhost/{os.getcwd()}/third_party/Mask2Former/",
        "stable-diffusion-sdkit==2.1.3",
    ],
    include_package_data=True,
)
