#!/bin/bash
echo "This script downloads pre-processed datasets used in the xmask3d project."
echo "Choose from the following options:"
echo "0 - ScanNet 3D (point clouds with GT semantic labels)"
echo "1 - ScanNet 2D (RGB-D images with camera poses)"
echo "2 - ScanNet-200 3D (ScanNet-200 point clouds with GT semantic labels)"

read -p "Enter dataset ID you want to download: " ds_id

echo $ds_id
if [ "$ds_id" = "0" ]
then
    echo "You chose 0: ScanNet 3D"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget -O scannet_3d.tar.gz
    echo "Done! Start unzipping ..."
    tar -xzvf scannet_3d.tar.gz 
    echo "Done!"
elif [ "$ds_id" = "1" ]
then
    echo "You chose 1: ScanNet 2D"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget -O scannet_2d.tar.gz
    echo "Done! Start unzipping ..."
    tar -xzvf scannet_2d.tar.gz
    echo "Done!"
elif [ "$ds_id" = "2" ]
then
    echo "You chose 2: ScanNet-200 3D"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget -O scannet_3d_200.tar.gz https://cloud.tsinghua.edu.cn/f/16d62d0c7af246c49002/?dl=1
    echo "Done! Start unzipping ..."
    tar -xzvf scannet_3d_200.tar.gz
    echo "Done!"
else
    echo "You entered an invalid ID!"
fi
