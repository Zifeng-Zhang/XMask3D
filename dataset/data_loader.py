from glob import glob
from os.path import join
import imageio.v2 as imageio
import os
import torch
import numpy as np
import cv2
import json
import numpy as np
from dataset.point_loader import Point3DLoader
from models.utils.mapping_util import getMapping
from detectron2.data import detection_utils as utils


class ScannetLoader(Point3DLoader):
    def __init__(
        self,
        datapath_prefix,
        datapath_prefix_2d,
        category_split,
        label_2d,
        scannet200=False,
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

        vectorized_map_all = np.vectorize(
            lambda value: self.data_ids_mapping_all.get(value, value)
        )
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

            label_200[~np.isin(label_200, self.label_3d_id)] = -1
            label_200 = vectorized_map_all(label_200.astype(np.int64)).astype(
                np.float64
            )
            label_200[label_200 == -1] = self.category_split.ignore_category[-1]
            labels_in = label_200

        labels_in[labels_in == -100] = self.category_split.ignore_category[-1]
        labels_in[labels_in == 255] = self.category_split.ignore_category[-1]

        labels_in_clone = labels_in.copy()
        vectorized_map = np.vectorize(
            lambda value: self.data_ids_mapping.get(value, value)
        )

        if self.split == "train":

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

        if self.split in ["val", "test"]:
            img_idx = self.epoch % len(img_dirs)
        while True:

            if self.split in ["val", "test"]:
                img_idx = img_idx % len(img_dirs)
                img_dir = img_dirs[img_idx]

            else:
                img_dir = np.random.choice(img_dirs, 1, replace=False)[0]
                img_dir = np.random.choice(img_dirs, 1, replace=False)[0]

            img = utils.read_image(img_dir, format="RGB")

            posepath = img_dir.replace("color", "pose").replace(".jpg", ".txt")
            pose = np.loadtxt(posepath)

            depth = (
                imageio.imread(img_dir.replace("color", "depth").replace("jpg", "png"))
                / 1000
            )

            single_mapping = self.point2img_mapper.compute_mapping(pose, locs_in, depth)

            mask = single_mapping[:, 2]
            label_3d = labels_in[mask == 1]
            feature_3d = feats_in[mask == 1]
            locals_3d = locs_in[mask == 1]
            label_3d_clone = labels_in_clone[mask == 1]
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

                if np.sum(mask) > 400 and valid_point_num > 10 and np.sum(mask) < 65000:
                    break
            else:
                if np.sum(mask) > 400 and valid_point_num > 10 and np.sum(mask) < 65000:
                    break
            if self.split in ["val", "test"]:
                img_idx += 2

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

        binary_label_2d = label_2d.copy()
        binary_label_2d = cv2.resize(
            binary_label_2d, (128, 128), interpolation=cv2.INTER_NEAREST
        )
        binary_label_2d[~np.isin(binary_label_2d, self.label_3d_id)] = -1

        binary_label_2d = vectorized_map_all(binary_label_2d.astype(np.int64)).astype(
            np.float64
        )

        binary_label_2d[
            np.isin(binary_label_2d, self.category_split["base_category"])
        ] = 1
        binary_label_2d[
            np.isin(binary_label_2d, self.category_split["novel_category"])
        ] = 0
        binary_label_2d[binary_label_2d == -1] = 20
        binary_label_2d = torch.from_numpy(binary_label_2d).float()

        label_2d[~np.isin(label_2d, self.label_2d_id)] = -1

        label_2d = vectorized_map(label_2d)

        if self.split in ["val", "test"]:
            pass

        else:
            label_2d[label_2d == -1] = len(self.category_split["base_category"])

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

        inds_reconstruct = torch.from_numpy(inds_reconstruct).long()
        locals_3d = torch.from_numpy(locals_3d).float()
        locals_3d = torch.cat(
            (torch.ones(locals_3d.shape[0], 1, dtype=torch.float), locals_3d), dim=1
        )
        locs_in = torch.from_numpy(locs_in).float()
        if self.split == "train":
            return (
                locals_3d,
                coords,
                feats,
                labels,
                binary_label,
                binary_label_2d,
                label_2d,
                img,
                x_label,
                y_label,
                inds_reconstruct,
                captions,
            )
        else:
            return (
                locals_3d,
                coords,
                feats,
                labels,
                binary_label,
                binary_label_2d,
                label_2d,
                img,
                x_label,
                y_label,
                inds_reconstruct,
                captions,
            )


def collation_fn(batch):

    (
        locals_3d,
        coords,
        feats,
        labels,
        binary_label,
        binary_label_2d,
        label_2d,
        img,
        x_label,
        y_label,
        inds_reconstruct,
        captions,
    ) = list(zip(*batch))
    inds_recons = list(inds_reconstruct)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        locals_3d[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return (
        torch.cat(locals_3d),
        torch.cat(coords),
        torch.cat(feats),
        torch.cat(labels),
        torch.cat(binary_label),
        torch.stack(binary_label_2d),
        torch.stack(label_2d, dim=0),
        torch.stack(img, dim=0),
        torch.cat(x_label),
        torch.cat(y_label),
        torch.cat(inds_recons),
        captions,
    )
