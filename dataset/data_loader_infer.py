from glob import glob
from os.path import join
import imageio.v2 as imageio
import os
import torch
import numpy as np
import SharedArray as SA
from tqdm import tqdm
import cv2
import pandas as pd
import json
import numpy as np
from dataset.point_loader import Point3DLoader
from models.utils.mapping_util import getMapping


class ScannetLoaderFull(Point3DLoader):
    def __init__(
        self,
        datapath_prefix,
        datapath_prefix_2d,
        label_2d,
        category_split,
        scannet200=False,
        val_keep=10000000,
        caption_path="data/caption/caption_view_scannet_vit-gpt2-image-captioning_.json",
        voxel_size=0.05,
        split="train",
        aug=False,
        memcache_init=False,
        identifier=7791,
        loop=1,
        eval_all=False,
        input_color=False,
    ):
        super().__init__(
            datapath_prefix=datapath_prefix,
            voxel_size=voxel_size,
            split=split,
            aug=aug,
            memcache_init=memcache_init,
            identifier=identifier,
            loop=loop,
            eval_all=eval_all,
            input_color=input_color,
        )
        self.scannet200 = scannet200
        self.aug = aug
        self.input_color = input_color
        self.category_split = category_split

        self.datapath_2d = datapath_prefix_2d
        self.point2img_mapper = getMapping()
        self.val_keep = val_keep
        with open(caption_path, "r") as f:
            self.captions_view = json.load(f)
        self.data_ids_mapping = {}
        self.data_ids_mapping_all = {}
        label_2d_id = label_2d

        self.label_3d_id = label_2d
        if self.split in ["val", "test"]:
            self.label_2d_id = label_2d_id

        else:

            self.label_2d_id = [
                label_2d_id[category]
                for category in self.category_split["base_category"]
            ]
        for i, id in enumerate(self.label_2d_id):

            self.data_ids_mapping.update({id: i})
        for i, id in enumerate(self.label_3d_id):

            self.data_ids_mapping_all.update({id: i})

        if len(self.data_paths) == 0:
            raise Exception("0 file is loaded in the feature loader.")
        self.epoch = None

    def read_bytes(self, path):

        with open(path, "rb") as f:
            file_bytes = f.read()
        return file_bytes

    def __getitem__(self, index_long):

        index = index_long % len(self.data_paths)
        locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
        if self.scannet200:
            root_path = self.data_paths[index]

            scannet_200_label_path = root_path.replace(
                "/scannet_3d_/", "/scannet_3d_200/instance_gt/"
            )
            scannet_200_label_path = os.path.join(
                os.path.dirname(scannet_200_label_path),
                self.data_paths[index][:-15].split("/")[-1] + ".txt",
            )
            label_200 = np.loadtxt(scannet_200_label_path)

            vectorized_map_all = np.vectorize(
                lambda value: self.data_ids_mapping_all.get(value, value)
            )

            label_200[~np.isin(label_200, self.label_3d_id)] = -1
            label_200 = vectorized_map_all(label_200.astype(np.int64)).astype(
                np.float64
            )
            label_200[label_200 == -1] = self.category_split.ignore_category[-1]
            labels_in = label_200
        labels_in[labels_in == -100] = self.category_split.ignore_category[-1]
        labels_in[labels_in == 255] = self.category_split.ignore_category[-1]
        labels_in_clone = labels_in.copy()

        if self.split in ["val", "test"]:
            pass

        else:

            indices_to_replace = self.category_split["novel_category"] + [
                self.category_split.ignore_category[0]
            ]

            labels_in[
                np.isin(labels_in, indices_to_replace)
            ] = self.category_split.ignore_category[-1]
            for i, replace in enumerate(indices_to_replace):
                labels_in[labels_in > replace - i] -= 1

        if np.isscalar(feats_in) and feats_in == 0:

            feats_in = np.zeros_like(locs_in)
        else:
            feats_in = (feats_in + 1.0) * 127.5

        if "scannet_3d" in self.dataset_name:
            scene_name = self.data_paths[index][:-15].split("/")[-1]
        else:
            scene_name = self.data_paths[index][:-4].split("/")[-1]

        scene = join(self.datapath_2d, scene_name)
        img_dirs = sorted(
            glob(join(scene, "color/*")), key=lambda x: int(os.path.basename(x)[:-4])
        )
        ori_locals_3d = []
        labels_3d = []
        binary_label_3d = []
        coords_3d = []
        feats_3d = []
        labels_2d = []
        imgs = []
        x_labels = []
        y_labels = []
        mask_2ds = []
        captions_list = []
        inds_reconstructs = []

        for ii, img_dir in enumerate(img_dirs):

            img = imageio.imread(img_dir)
            h, w, c = img.shape

            posepath = img_dir.replace("color", "pose").replace(".jpg", ".txt")
            pose = np.loadtxt(posepath)

            depth = (
                imageio.imread(img_dir.replace("color", "depth").replace("jpg", "png"))
                / 1000
            )

            single_mapping = self.point2img_mapper.compute_mapping(pose, locs_in, depth)

            mask = single_mapping[:, 2]
            label_3d = labels_in[mask == 1].copy()
            feature_3d = feats_in[mask == 1].copy()
            locals_3d = locs_in[mask == 1].copy()
            label_3d_clone = labels_in_clone[mask == 1].copy()
            zero_rows = np.all(single_mapping != 0, axis=1)
            single_mapping = single_mapping[zero_rows]

            binary_label = label_3d_clone

            binary_label[
                np.isin(label_3d_clone, self.category_split["base_category"])
            ] = 1
            binary_label[
                np.isin(label_3d_clone, self.category_split["novel_category"])
            ] = 0

            binary_label = torch.from_numpy(binary_label).float()
            valid_point_num = np.sum(
                ~np.isin(binary_label, np.array(self.category_split.ignore_category))
            )

            if self.split == "train":

                if np.sum(mask) < 400 or valid_point_num < 10 or np.sum(mask) > 65000:
                    continue
            else:
                if (
                    np.sum(mask) < 400
                    or valid_point_num < 10
                    or np.sum(mask) > self.val_keep
                ):
                    continue

            img = cv2.resize(img, (512, 512))

            imgae_idx = os.path.basename(img_dir)[0:-4]

            captions = self.captions_view[scene_name][imgae_idx]

            if self.scannet200:
                label_2d = imageio.imread(
                    img_dir.replace("color", "label_200").replace(".jpg", ".png")
                ).astype(np.int32)
            else:
                label_2d = imageio.imread(
                    img_dir.replace("color", "label").replace(".jpg", ".png")
                ).astype(np.int32)

            vectorized_map = np.vectorize(
                lambda value: self.data_ids_mapping.get(value, value)
            )
            label_2d[~np.isin(label_2d, self.label_2d_id)] = 255
            label_2d = vectorized_map(label_2d)

            if self.split in ["val", "test"]:
                pass

            else:
                label_2d[label_2d == 255] = len(self.category_split["base_category"])

            label_2d = cv2.resize(label_2d, (512, 512), interpolation=cv2.INTER_NEAREST)
            img = torch.from_numpy(img).float()

            locals_3d = self.prevoxel_transforms(locals_3d) if self.aug else locals_3d

            locs, feats, _, inds_reconstruct = self.voxelizer.voxelize(
                locals_3d, feature_3d, label_3d
            )
            labels = label_3d

            if self.aug:
                locs, feats, labels = self.input_transforms(locs, feats, labels)
            coords = torch.from_numpy(locs).int()
            coords = torch.cat(
                (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1
            )
            if self.input_color:
                feats = torch.from_numpy(feats).float() / 127.5 - 1.0
            else:

                feats = torch.ones(coords.shape[0], 3)

            labels = torch.from_numpy(labels).long()

            label_2d = torch.from_numpy(label_2d).long()

            x_label = single_mapping[:, 0][single_mapping[:, 0] != 0]
            y_label = single_mapping[:, 1][single_mapping[:, 1] != 0]
            x_label = torch.from_numpy(x_label).long()
            y_label = torch.from_numpy(y_label).long()

            mask_2d = torch.from_numpy(mask).bool()

            inds_reconstruct = torch.from_numpy(inds_reconstruct).long()
            locals_3d = torch.from_numpy(locals_3d).float()
            locals_3d = torch.cat(
                (torch.ones(locals_3d.shape[0], 1, dtype=torch.float), locals_3d), dim=1
            )

            ori_locals_3d.append(locals_3d)
            coords_3d.append(coords)
            feats_3d.append(feats)
            labels_3d.append(labels)

            binary_label_3d.append(binary_label)
            labels_2d.append(label_2d)
            imgs.append(img)
            x_labels.append(x_label)
            y_labels.append(y_label)
            mask_2ds.append(mask_2d)
            inds_reconstructs.append(inds_reconstruct)
            captions_list.append(captions)

        locs_in = torch.from_numpy(locs_in).float()
        labels_in = torch.from_numpy(labels_in).long()
        return (
            locs_in,
            labels_in,
            ori_locals_3d,
            coords_3d,
            feats_3d,
            labels_3d,
            binary_label_3d,
            labels_2d,
            imgs,
            x_labels,
            y_labels,
            mask_2ds,
            inds_reconstructs,
            captions_list,
        )


def collation_fn_eval_all_full(batch):
    """
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    """
    (
        locs_in,
        labels_in,
        ori_locals_3d,
        coords_3d,
        feats_3d,
        labels_3d,
        binary_label_3d,
        labels_2d,
        imgs,
        x_labels,
        y_labels,
        mask_2ds,
        inds_reconstructs,
        captions_list,
    ) = list(zip(*batch))

    return (
        locs_in[0],
        labels_in[0],
        ori_locals_3d[0],
        coords_3d[0],
        feats_3d[0],
        labels_3d[0],
        binary_label_3d[0],
        labels_2d[0],
        imgs[0],
        x_labels[0],
        y_labels[0],
        mask_2ds[0],
        inds_reconstructs[0],
        captions_list[0],
    )
