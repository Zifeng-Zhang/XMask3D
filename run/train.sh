#!/bin/sh
set -x 

exp_dir=out/ttt

config=config/scannet/xmask3d_scannet_B10N9.yaml 

mkdir -p ${exp_dir} 

export PYTHONPATH=.
python -u run/train.py \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee -a ${exp_dir}/train-$(date +"%Y%m%d_%H%M").log
  