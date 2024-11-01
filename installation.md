# Installation
Start by cloning the repo:
```bash
git clone git@github.com:wangzy22/XMask3D.git
cd XMask3D
```

Then you can create an anaconda environment called `xmask3d` as below. 

```bash
conda create -n xmask3d python=3.9
conda activate xmask3d
```

Step 1: install PyTorch (we tested on 1.13.1, but the following versions should also work):

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev
```

Step 2 (Optional): install xformers:
```bash
conda install xformers -c xformers
```

If you encounter changes in the PyTorch version or other errors, refer to the [official installation page](https://github.com/facebookresearch/xformers).

Step 3: install MinkowskiNet:

```bash
pip install ninja
sudo apt install libopenblas-dev
```
If you do not have sudo right, try the following:
```
conda install openblas-devel -c anaconda
```
And now install MinkowskiNet:
```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
```
If it is still giving you error, please refer to their [official installation page](https://github.com/NVIDIA/MinkowskiEngine#installation).


Step 4: install all the remaining dependencies:
```bash

pip install -e .
```
