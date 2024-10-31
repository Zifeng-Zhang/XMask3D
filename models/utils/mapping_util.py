import torch
import numpy as np
from models.utils.fusion_util import (
    PointCloudToImageMapper,
    adjust_intrinsic,
    make_intrinsic,
)


def getMapping():
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    img_dim = (320, 240)
    depth_scale = 1000.0
    fx = 577.870605
    fy = 577.870605
    mx = 319.5
    my = 239.5

    visibility_threshold = 0.25

    depth_scale = depth_scale
    cut_num_pixel_boundary = 10

    intrinsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
    intrinsic = adjust_intrinsic(
        intrinsic, intrinsic_image_dim=[640, 480], image_dim=img_dim
    )

    point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim,
        intrinsics=intrinsic,
        visibility_threshold=visibility_threshold,
        cut_bound=cut_num_pixel_boundary,
    )
    return point2img_mapper
