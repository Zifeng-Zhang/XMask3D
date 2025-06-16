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
        """
        Modified compute_mapping function with proper handling of array dimensions

        Args:
            camera_to_world: Camera extrinsic matrix [4x4]
            coords: Point cloud coordinates [Nx3]
            depth: Depth image [HxW]
            intrinsic: Camera intrinsic matrix [3x3]

        Returns:
            mapping: Mapping between 3D points and 2D pixels [Nx3]
        """
        if self.intrinsics is not None:
            intrinsic = self.intrinsics

        # Initialize mapping array
        mapping = np.zeros((3, coords.shape[0]), dtype=int)

        # Convert to homogeneous coordinates
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        # Transform points to camera space
        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)

        # Convert 3D points to 2D pixel coordinates using the intrinsic matrix
        # Avoid division by zero by creating a safe divider
        safe_z = p[2].copy()
        safe_z[np.abs(safe_z) < 1e-8] = 1.0  # Avoid division by zero

        px = (p[0] * intrinsic[0][0]) / safe_z + intrinsic[0][2]
        py = (p[1] * intrinsic[1][1]) / safe_z + intrinsic[1][2]

        # Round to integer pixel coordinates
        pi_x = np.round(px).astype(int)
        pi_y = np.round(py).astype(int)

        # Create basic mask for points in front of camera with valid z
        front_mask = p[2] > 0

        # Check if pixels are within image bounds
        inside_mask = (
                front_mask &
                (pi_x >= self.cut_bound) &
                (pi_y >= self.cut_bound) &
                (pi_x < self.image_dim[0] - self.cut_bound) &
                (pi_y < self.image_dim[1] - self.cut_bound)
        )

        # Handle depth check if depth map is provided
        if depth is not None and np.any(inside_mask):
            # Get pixel coordinates for valid points
            valid_y = pi_y[inside_mask]
            valid_x = pi_x[inside_mask]
            valid_z = p[2][inside_mask]

            # Ensure pixel coordinates are within depth map bounds
            valid_coords = (
                    (valid_y >= 0) &
                    (valid_y < depth.shape[0]) &
                    (valid_x >= 0) &
                    (valid_x < depth.shape[1])
            )

            # Initialize occlusion mask with False (will be updated for valid points)
            occlusion_mask = np.zeros_like(inside_mask)

            if np.any(valid_coords):
                # Extract depth values only for valid coordinates
                filtered_y = valid_y[valid_coords]
                filtered_x = valid_x[valid_coords]
                filtered_z = valid_z[valid_coords]

                # Get depth values at those pixel locations
                depth_values = depth[filtered_y, filtered_x]

                # Check occlusion with depth threshold
                depth_check = np.abs(depth_values - filtered_z) <= self.vis_thres * depth_values

                # Create indices for the original inside_mask that correspond to valid points
                valid_indices = np.where(inside_mask)[0]
                valid_indices = valid_indices[valid_coords]

                # Update occlusion mask at the correct indices
                occlusion_mask[valid_indices[depth_check]] = True

                # Update inside_mask with occlusion check
                inside_mask = occlusion_mask

        # Set final mapping
        mapping[0, inside_mask] = pi_y[inside_mask]
        mapping[1, inside_mask] = pi_x[inside_mask]
        mapping[2, inside_mask] = 1

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
