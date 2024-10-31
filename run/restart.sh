#!/bin/bash

while true; do


    if pgrep -f "python -u /home/admin/workspace/wangyanbo/openscene/run/our_train.py" > /dev/null
    then
        echo "start running."
    else
        echo "attempting running again"

        exp_dir=out/version_30_15B4N_PLA_view_3d_2d_fused_BS64_vit_L_14_B12N7_spconv

        config=config/scannet/xmask3d_scannet_B12N7.yaml
        model_dir=${exp_dir}/model
        export PYTHONPATH=.
        python -u run/train.py \
          --config=${config} \
          save_path ${exp_dir} \
          resume ${model_dir}/model_last.pth.tar \
          2>&1 | tee ${exp_dir}/train-$(date +"%Y%m%d_%H%M").log
    fi
    
    sleep 60
done
