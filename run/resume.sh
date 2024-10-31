#!/bin/sh
set -x

exp_dir=out/B12N7
config=config/scannet/xmask3d_scannet_B12N7.yaml 
model_dir=${exp_dir}/model

export PYTHONPATH=.
python -u run/train.py \
  --config=${config} \
  save_path ${exp_dir} \
  resume ${model_dir}/model_last.pth.tar \
  2>&1 | tee ${exp_dir}/resume-$(date +"%Y%m%d_%H%M").log