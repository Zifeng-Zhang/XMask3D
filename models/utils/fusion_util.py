import os
import glob
import math
import numpy as np


def make_intrinsic(fx, fy, mx, my):
    """"""

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    """"""

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(
        math.floor(
            image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])
        )
    )
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])

    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


class PointCloudToImageMapper(object):
    def __init__(
        self, image_dim, visibility_threshold=0.25, cut_bound=0, intrinsics=None
    ):

        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):

        if self.intrinsics is not None:
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]

        pi = np.round(p).astype(int)

        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = (
                np.abs(
                    depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]
                )
                <= self.vis_thres * depth_cur
            )

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0
            inside_mask = front_mask * inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T


def obtain_intr_extr_matterport(scene):
    """"""

    img_dir = os.path.join(scene, "color")
    pose_dir = os.path.join(scene, "pose")
    intr_dir = os.path.join(scene, "intrinsic")
    img_names = sorted(glob.glob(img_dir + "/*.jpg"))

    intrinsics = []
    extrinsics = []
    for img_name in img_names:
        name = img_name.split("/")[-1][:-4]

        extrinsics.append(np.loadtxt(os.path.join(pose_dir, name + ".txt")))
        intrinsics.append(np.loadtxt(os.path.join(intr_dir, name + ".txt")))

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    img_names = np.asarray(img_names)

    return img_names, intrinsics, extrinsics


def get_matterport_camera_data(data_path, locs_in, args):
    """"""

    bbox_l = locs_in.min(axis=0)
    bbox_h = locs_in.max(axis=0)

    building_name = data_path.split("/")[-1].split("_")[0]
    scene_id = data_path.split("/")[-1].split(".")[0]

    scene = os.path.join(args.data_root_2d, building_name)
    img_names, intrinsics, extrinsics = obtain_intr_extr_matterport(scene)

    cam_loc = extrinsics[:, :3, -1]
    ind_in_scene = (
        (cam_loc[:, 0] > bbox_l[0])
        & (cam_loc[:, 0] < bbox_h[0])
        & (cam_loc[:, 1] > bbox_l[1])
        & (cam_loc[:, 1] < bbox_h[1])
        & (cam_loc[:, 2] > bbox_l[2])
        & (cam_loc[:, 2] < bbox_h[2])
    )

    img_names_in = img_names[ind_in_scene]
    intrinsics_in = intrinsics[ind_in_scene]
    extrinsics_in = extrinsics[ind_in_scene]
    num_img = len(img_names_in)

    if args.split == "test" and num_img == 0:
        print(
            "no views inside {}, take the nearest 100 images to fuse".format(scene_id)
        )

        centroid = (bbox_l + bbox_h) / 2
        dist_centroid = np.linalg.norm(cam_loc - centroid, axis=-1)
        ind_in_scene = np.argsort(dist_centroid)[:100]
        img_names_in = img_names[ind_in_scene]
        intrinsics_in = intrinsics[ind_in_scene]
        extrinsics_in = extrinsics[ind_in_scene]
        num_img = 100

    img_names_in = img_names_in.tolist()

    return intrinsics_in, extrinsics_in, img_names_in, scene_id, num_img
