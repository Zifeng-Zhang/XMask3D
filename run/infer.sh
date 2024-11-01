

echo "Please enter the ckpt name you want to infer: "
read -r input_list

array=$input_list
word_count=$(echo "$input_list" | wc -w)

echo $array
echo $word_count

for i in $(seq 1 $word_count)
do
    element=$(echo "$input_list" | cut -d" " -f$i)
    echo "Current ckpt: $element"
    set -x
    ckpt_dir=$element

    exp_dir=out/B12N7
    config=config/scannet/xmask3d_scannet_infer_B12N7.yaml

    

    model_dir=${exp_dir}/model

    export PYTHONPATH=.
    python -u run/infer.py \
    --config=${config} \
    save_path ${exp_dir} \
    resume ${model_dir}/${ckpt_dir} \
    2>&1 | tee ${exp_dir}/infer-$element.log
done
