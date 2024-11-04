#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: sh run/infer.sh --exp_dir=EXP_DIR --config=CONFIG --ckpt_dir=CKPT_DIR"
    exit 1
fi

for arg in "$@"; do
    case $arg in
        --exp_dir=*)
            exp_dir="${arg#*=}"
            shift
            ;;
        --config=*)
            config="${arg#*=}"
            shift
            ;;
        --ckpt_dir=*)
            ckpt_dir="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $arg"
            exit 1
            ;;
    esac
done

echo "Current ckpt: $ckpt_dir"

export PYTHONPATH=.
python -u run/infer.py \
    --config="${config}" \
    save_path "${exp_dir}" \
    resume "${exp_dir}/model/${ckpt_dir}" \
    2>&1 | tee "${exp_dir}/infer-${ckpt_dir}.log"
