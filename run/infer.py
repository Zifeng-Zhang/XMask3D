import os
import time
import random
import numpy as np
import logging
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import imageio
from MinkowskiEngine import SparseTensor
from sklearn.neighbors import KDTree
from util import config
from util.util import (
    AverageMeter,
    intersectionAndUnionGPU,
)


from dataset.data_loader_infer import (
    ScannetLoaderFull,
    collation_fn_eval_all_full,
)

from models.xmask3d import XMASK3d as Model
from models.checkpoint import XMask3dCheckpointer
import MinkowskiEngine as ME


def worker_init_fn(worker_id):
    """"""
    random.seed(time.time() + worker_id)


def get_parser():
    """"""
    parser = argparse.ArgumentParser(description="xmask3d.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/scannet/xmask3d_scannet_infer_B12N7.yaml",
        help="config file",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, "model")
    result_dir = os.path.join(cfg.save_path, "result")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + "/last", exist_ok=True)
    os.makedirs(result_dir + "/best", exist_ok=True)
    return cfg


def get_logger():
    """"""

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0
    )


def main():
    """"""

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.infer_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:

        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    ""

    if not hasattr(args, "use_shm"):
        args.use_shm = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = len(args.infer_gpu)

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(
            main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args)
        )
    else:
        main_worker(args.infer_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        print("start")
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        print("over")

    model = get_model(args)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")

    def add_weight_decay(model, lr_1, lr_2):
        lr_3d = []
        lr_others = []
        for name, param in model.named_parameters():

            if "pc_decoder" in name or "pc_binary_head" in name:

                lr_3d.append(param)
            else:
                lr_others.append(param)
        return [{"params": lr_3d, "lr": lr_1}, {"params": lr_others, "lr": lr_2}]

    param_groups = add_weight_decay(model, args.lr_3d, args.lr_others)
    optimizer = torch.optim.AdamW(param_groups)

    args.index_split = 0

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.infer_batch_size_val = int(args.infer_batch_size_val / ngpus_per_node)
        args.workers = int(args.infer_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu], find_unused_parameters=True
        )
    else:
        model = model.cuda()

    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))

            ret = XMask3dCheckpointer(model, optimizer).load(args.resume, eval=True)
            args.start_epoch = ret["start_epoch"]

            if main_process():
                logger.info(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, ret["start_epoch"]
                    )
                )

        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if not hasattr(args, "input_color"):

        args.input_color = False

    OurLoaderFull = ScannetLoaderFull

    if args.evaluate:
        val_data = OurLoaderFull(
            datapath_prefix=args.data_root,
            datapath_prefix_2d=args.data_root_2d,
            category_split=args.category_split,
            label_2d=args.label_2d,
            caption_path=args.caption_path,
            scannet200=args.scannet200,
            val_keep=args.val_keep,
            voxel_size=args.voxel_size,
            split="val",
            aug=False,
            memcache_init=args.use_shm,
            eval_all=True,
            input_color=args.input_color,
        )
        val_sampler = (
            torch.utils.data.distributed.DistributedSampler(val_data)
            if args.distributed
            else None
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.infer_batch_size_val,
            shuffle=False,
            num_workers=args.infer_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collation_fn_eval_all_full,
            sampler=val_sampler,
        )

    val_sampler.set_epoch(args.start_epoch)
    try:
        assert args.infer_batch_size_val == 1
    except AssertionError:
        print(f"Error: Expected infer_batch_size_val to be 1, but got {args.infer_batch_size_val}.")

    (
        mIoU_Base,
        mIoU_Novel,
        mIoU_2d_Base,
        mIoU_2d_Novel,
        mIoU_3d_Base,
        mIoU_3d_Novel,
    ) = validate(val_loader, model)

    if main_process():
        writer.close()
        logger.info("==>Validation done!")


def get_model(cfg):
    """"""

    model = Model(cfg)
    return model


def validate(val_loader, model):
    """"""
    ""

    torch.backends.cudnn.enabled = False

    intersection_meter_Base = AverageMeter()
    intersection_meter_Novel = AverageMeter()
    union_meter_Base = AverageMeter()
    union_meter_Novel = AverageMeter()
    target_meter_Base = AverageMeter()
    target_meter_Novel = AverageMeter()

    intersection_meter_2d_Base = AverageMeter()
    intersection_meter_2d_Novel = AverageMeter()
    union_meter_2d_Base = AverageMeter()
    union_meter_2d_Novel = AverageMeter()
    target_meter_2d_Base = AverageMeter()
    target_meter_2d_Novel = AverageMeter()

    intersection_meter_3d_Base = AverageMeter()
    intersection_meter_3d_Novel = AverageMeter()
    union_meter_3d_Base = AverageMeter()
    union_meter_3d_Novel = AverageMeter()
    target_meter_3d_Base = AverageMeter()
    target_meter_3d_Novel = AverageMeter()

    model.eval()

    torch.rand(1)
    np.random.rand(1)
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            (
                scene_coords,
                scene_label,
                ori_coords_3ds,
                coords_3ds,
                feat_3ds,
                labels_3ds,
                binary_label_3ds,
                label_2ds,
                imgs,
                x_labels,
                y_labels,
                mask_2ds,
                inds_reconstructs,
                captions,
            ) = batch_data

            ii = 0
            scene_pred = torch.zeros(
                (
                    scene_coords.shape[0],
                    len(
                        args.category_split.base_category
                        + args.category_split.novel_category
                    ),
                ),
                dtype=scene_label.dtype,
            )

            scene_pred_2d = torch.zeros(
                (
                    scene_coords.shape[0],
                    len(
                        args.category_split.base_category
                        + args.category_split.novel_category
                    ),
                ),
                dtype=scene_label.dtype,
            )
            scene_pred_3d = torch.zeros(
                (
                    scene_coords.shape[0],
                    len(
                        args.category_split.base_category
                        + args.category_split.novel_category
                    ),
                ),
                dtype=scene_label.dtype,
            )

            counter = torch.zeros((scene_coords.shape[0]), dtype=scene_label.dtype)
            for (
                ori_coords_3d,
                coords_3d,
                feat_3d,
                labels_3d,
                binary_label_3d,
                label_2d,
                img,
                x_label,
                y_label,
                mask_2d,
                inds_reconstruct,
                caption,
            ) in zip(
                ori_coords_3ds,
                coords_3ds,
                feat_3ds,
                labels_3ds,
                binary_label_3ds,
                label_2ds,
                imgs,
                x_labels,
                y_labels,
                mask_2ds,
                inds_reconstructs,
                captions,
            ):
                ii += 1

                coords_3d[:, 0] *= 0
                ori_coords_3d[:, 0] *= 0
                label_2d = label_2d.unsqueeze(0)
                img = img.unsqueeze(0)

                sinput = SparseTensor(
                    feat_3d.cuda(non_blocking=True), coords_3d.cuda(non_blocking=True)
                )

                batch_input = {}
                img = img.permute(0, 3, 1, 2).contiguous()
                batch_input["coords"] = coords_3d
                batch_input["img"] = img
                batch_input["sinput"] = sinput

                batch_input["x_label"] = x_label
                batch_input["y_label"] = y_label
                batch_input["label_2d"] = label_2d
                batch_input["inds_reconstruct"] = inds_reconstruct
                batch_input["labels_3d"] = labels_3d
                batch_input["ori_coords"] = ori_coords_3d
                batch_input["captions"] = caption

                batch_input["use_pure_3d"] = False

                _, outputs = model(batch_input)

                fused_feature = outputs["fused_pred_feature"]
                text_features = outputs["text_embed"]

                logit_scale = outputs["logit_scale"]
                feature_2d = outputs["2d_pred_feature"]
                feature_3d = outputs["pure3d_pred_feature"]

                final_mask_3d = outputs["final_mask_3d"]
                final_pred_open_embedding = outputs["final_pred_open_embedding"]

                binary_pred = outputs["binary_pred"]

                binary_pred_copy = binary_pred.clone()

                binary_pred_copy[
                    torch.isin(
                        binary_label_3d.to(binary_pred),
                        torch.tensor(args.category_split.ignore_category).to(
                            binary_pred
                        ),
                    )
                ] = 2

                binary_label_3d[
                    torch.isin(
                        binary_label_3d.to(binary_pred),
                        torch.tensor(args.category_split.ignore_category).to(
                            binary_pred
                        ),
                    )
                ] = 2

                binary_pred_copy = binary_pred_copy.squeeze(1).cpu()
                binary_label_3d = binary_label_3d.long().cpu()

                del outputs

                mask = inds_reconstruct

                fused_feature = torch.cat(fused_feature)

                fused_feature = F.normalize(fused_feature, dim=-1)

                for feature_2d_single in feature_2d:
                    if (
                        len(feature_2d_single[torch.sum(feature_2d_single, dim=1) == 0])
                        == 0
                    ):
                        continue
                    coord_3d = ori_coords_3d[:, 1:].clone()

                    coord_3d = coord_3d.to(feature_2d_single.device)

                    scene_true = coord_3d[torch.sum(feature_2d_single, dim=1) != 0]
                    scene_false = coord_3d[torch.sum(feature_2d_single, dim=1) == 0]
                    flase_idx = torch.where(torch.sum(feature_2d_single, dim=1) == 0)[0]
                    true_idx = torch.where(torch.sum(feature_2d_single, dim=1) != 0)[0]

                    kdtree = KDTree(scene_true.cpu())

                    distances, indices = kdtree.query(scene_false.cpu(), k=1)

                    match = true_idx[indices.flatten()]

                    feature_2d_single[flase_idx] = feature_2d_single[match]

                feature_2d = torch.cat(feature_2d)

                feature_2d = F.normalize(feature_2d, dim=-1)
                feature_3d = torch.cat(feature_3d)

                feature_3d = F.normalize(feature_3d, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                logits_pred = logit_scale * (fused_feature @ text_features.t())

                final_pred_open_embedding = torch.cat(final_pred_open_embedding)
                final_pred_open_embedding = F.normalize(
                    final_pred_open_embedding, dim=-1
                )
                final_pred_open_logits = logit_scale * (
                    final_pred_open_embedding @ text_features.t()
                )
                final_mask_3d = torch.cat(final_mask_3d)
                logits_pred = logits_pred.softmax(dim=-1)
                final_pred_open_logits = final_pred_open_logits.softmax(dim=-1)

                category_overlapping_list = []

                train_labels = [l for l in args.category_split.base_category]
                test_labels = [l for l in args.category_split.all_category]

                for test_label in test_labels:
                    category_overlapping_list.append(int(test_label in train_labels))

                category_overlapping_mask = torch.tensor(
                    category_overlapping_list,
                    device=logits_pred.device,
                    dtype=torch.long,
                )

                for single_mask, final_pred_open_logit in zip(
                    final_mask_3d, final_pred_open_logits
                ):

                    pred_open_logits_base = (
                        logits_pred[single_mask] ** args.base_ratio
                        * final_pred_open_logit ** (1 - args.base_ratio)
                    ).log() * category_overlapping_mask

                    pred_open_logits_novel = (
                        logits_pred[single_mask] ** args.novel_ratio
                        * final_pred_open_logit ** (1 - args.novel_ratio)
                    ).log() * (1 - category_overlapping_mask)

                    logits_pred[single_mask] = (
                        pred_open_logits_base + pred_open_logits_novel
                    )

                logits_pred_novel = logits_pred.clone()
                logits_pred_base = logits_pred.clone()

                logits_pred_novel[:, args.category_split.base_category] = -1e10
                logits_pred_base[:, args.category_split.novel_category] = -1e10

                logits_pred = (
                    binary_pred * logits_pred_base
                    + (1 - binary_pred) * logits_pred_novel
                )

                logits_pred = torch.max(logits_pred, 1)[1].cpu()

                logits_pred_2d = logit_scale * (feature_2d @ text_features.t())

                logits_pred_novel = logits_pred_2d.clone()
                logits_pred_base = logits_pred_2d.clone()
                logits_pred_novel[:, args.category_split.base_category] = -1e10
                logits_pred_base[:, args.category_split.novel_category] = -1e10
                logits_pred_2d = (
                    binary_pred * logits_pred_base
                    + (1 - binary_pred) * logits_pred_novel
                )

                logits_pred_3d = logit_scale * (feature_3d @ text_features.t())

                logits_pred_novel = logits_pred_3d.clone()
                logits_pred_base = logits_pred_3d.clone()
                logits_pred_novel[:, args.category_split.base_category] = -1e10
                logits_pred_base[:, args.category_split.novel_category] = -1e10
                logits_pred_3d = (
                    binary_pred * logits_pred_base
                    + (1 - binary_pred) * logits_pred_novel
                )

                logits_pred_3d = torch.max(logits_pred_3d, 1)[1].cpu()

                logits_pred_2d = torch.max(logits_pred_2d, 1)[1].cpu()

                ""

                scene_pred[mask_2d, logits_pred] += 1

                scene_pred_2d[mask_2d, logits_pred_2d] += 1
                scene_pred_3d[mask_2d, logits_pred_3d] += 1

                counter[mask_2d] += 1

                torch.cuda.empty_cache()

            scene_coords = scene_coords.to(counter.device)
            scene_true = scene_coords.to(counter.device)[counter != 0]

            scene_false = scene_coords.to(counter.device)[counter == 0]
            flase_idx = torch.where(counter == 0)[0]
            true_idx = torch.where(counter != 0)[0]

            _, scene_pred = torch.max(scene_pred, dim=1)

            kdtree = KDTree(scene_true)

            distances, indices = kdtree.query(scene_false, k=1)

            match = true_idx[indices.flatten()]

            scene_pred[flase_idx] = scene_pred[match]

            _, scene_pred_2d = torch.max(scene_pred_2d, dim=1)
            scene_pred_2d[flase_idx] = scene_pred_2d[match]

            _, scene_pred_3d = torch.max(scene_pred_3d, dim=1)
            scene_pred_3d[flase_idx] = scene_pred_3d[match]

            intersection, union, target = intersectionAndUnionGPU(
                scene_pred.to(logits_pred.device),
                scene_label.detach(),
                args.test_classes,
                args.test_ignore_label,
            )

            intersection_2d, union_2d, target_2d = intersectionAndUnionGPU(
                scene_pred_2d.to(logits_pred.device),
                scene_label.detach(),
                args.test_classes,
                args.test_ignore_label,
            )

            intersection_3d, union_3d, target_3d = intersectionAndUnionGPU(
                scene_pred_3d.to(logits_pred.device),
                scene_label.detach(),
                args.test_classes,
                args.test_ignore_label,
            )

            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
                dist.all_reduce(intersection_2d), dist.all_reduce(
                    union_2d
                ), dist.all_reduce(target_2d)
                dist.all_reduce(intersection_3d), dist.all_reduce(
                    union_3d
                ), dist.all_reduce(target_3d)

            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_2d, union_2d, target_2d = (
                intersection_2d.cpu().numpy(),
                union_2d.cpu().numpy(),
                target_2d.cpu().numpy(),
            )
            intersection_3d, union_3d, target_3d = (
                intersection_3d.cpu().numpy(),
                union_3d.cpu().numpy(),
                target_3d.cpu().numpy(),
            )

            intersection_meter_Base.update(
                intersection[args.category_split.base_category]
            ), union_meter_Base.update(
                union[args.category_split.base_category]
            ), target_meter_Base.update(
                target[args.category_split.base_category]
            )

            intersection_meter_Novel.update(
                intersection[args.category_split.novel_category]
            ), union_meter_Novel.update(
                union[args.category_split.novel_category]
            ), target_meter_Novel.update(
                target[args.category_split.novel_category]
            )

            intersection_meter_2d_Base.update(
                intersection_2d[args.category_split.base_category]
            ), union_meter_2d_Base.update(
                union_2d[args.category_split.base_category]
            ), target_meter_2d_Base.update(
                target_2d[args.category_split.base_category]
            )

            intersection_meter_2d_Novel.update(
                intersection_2d[args.category_split.novel_category]
            ), union_meter_2d_Novel.update(
                union_2d[args.category_split.novel_category]
            ), target_meter_2d_Novel.update(
                target_2d[args.category_split.novel_category]
            )

            intersection_meter_3d_Base.update(
                intersection_3d[args.category_split.base_category]
            ), union_meter_3d_Base.update(
                union_3d[args.category_split.base_category]
            ), target_meter_3d_Base.update(
                target_3d[args.category_split.base_category]
            )

            intersection_meter_3d_Novel.update(
                intersection_3d[args.category_split.novel_category]
            ), union_meter_3d_Novel.update(
                union_3d[args.category_split.novel_category]
            ), target_meter_3d_Novel.update(
                target_3d[args.category_split.novel_category]
            )

            if main_process():
                logger.info(
                    "Checking: [{}/{}/{}]".format(
                        scene_pred.shape, scene_true.shape, scene_false.shape
                    )
                )
                logger.info("Process: [{}/{}]".format(i, len(val_loader)))
            if main_process():

                iou_class_Base = intersection_meter_Base.sum / (
                    union_meter_Base.sum + 1e-10
                )
                accuracy_class_Base = intersection_meter_Base.sum / (
                    target_meter_Base.sum + 1e-10
                )

                mIoU_Base = np.mean(iou_class_Base)
                mAcc_Base = np.mean(accuracy_class_Base)
                allAcc_Base = sum(intersection_meter_Base.sum) / (
                    sum(target_meter_Base.sum) + 1e-10
                )

                iou_class_Novel = intersection_meter_Novel.sum / (
                    union_meter_Novel.sum + 1e-10
                )
                accuracy_class_Novel = intersection_meter_Novel.sum / (
                    target_meter_Novel.sum + 1e-10
                )
                mIoU_Novel = np.mean(iou_class_Novel)
                mAcc_Novel = np.mean(accuracy_class_Novel)
                allAcc_Novel = sum(intersection_meter_Novel.sum) / (
                    sum(target_meter_Novel.sum) + 1e-10
                )

                iou_class_2d_Base = intersection_meter_2d_Base.sum / (
                    union_meter_2d_Base.sum + 1e-10
                )
                accuracy_class_2d_Base = intersection_meter_2d_Base.sum / (
                    target_meter_2d_Base.sum + 1e-10
                )

                mIoU_2d_Base = np.mean(iou_class_2d_Base)
                mAcc_2d_Base = np.mean(accuracy_class_2d_Base)
                allAcc_2d_Base = sum(intersection_meter_2d_Base.sum) / (
                    sum(target_meter_2d_Base.sum) + 1e-10
                )

                iou_class_2d_Novel = intersection_meter_2d_Novel.sum / (
                    union_meter_2d_Novel.sum + 1e-10
                )
                accuracy_class_2d_Novel = intersection_meter_2d_Novel.sum / (
                    target_meter_2d_Novel.sum + 1e-10
                )
                mIoU_2d_Novel = np.mean(iou_class_2d_Novel)
                mAcc_2d_Novel = np.mean(accuracy_class_2d_Novel)
                allAcc_2d_Novel = sum(intersection_meter_2d_Novel.sum) / (
                    sum(target_meter_2d_Novel.sum) + 1e-10
                )

                iou_class_3d_Base = intersection_meter_3d_Base.sum / (
                    union_meter_3d_Base.sum + 1e-10
                )
                accuracy_class_3d_Base = intersection_meter_3d_Base.sum / (
                    target_meter_3d_Base.sum + 1e-10
                )

                mIoU_3d_Base = np.mean(iou_class_3d_Base)
                mAcc_3d_Base = np.mean(accuracy_class_3d_Base)
                allAcc_3d_Base = sum(intersection_meter_3d_Base.sum) / (
                    sum(target_meter_3d_Base.sum) + 1e-10
                )

                iou_class_3d_Novel = intersection_meter_3d_Novel.sum / (
                    union_meter_3d_Novel.sum + 1e-10
                )
                accuracy_class_3d_Novel = intersection_meter_3d_Novel.sum / (
                    target_meter_3d_Novel.sum + 1e-10
                )
                mIoU_3d_Novel = np.mean(iou_class_3d_Novel)
                mAcc_3d_Novel = np.mean(accuracy_class_3d_Novel)
                allAcc_3d_Novel = sum(intersection_meter_3d_Novel.sum) / (
                    sum(target_meter_3d_Novel.sum) + 1e-10
                )

                logger.info(
                    "Val 2d result: mIoU_Base/mAcc_Base/allAcc_Base {:.4f}/{:.4f}/{:.4f}.".format(
                        mIoU_2d_Base, mAcc_2d_Base, allAcc_2d_Base
                    )
                )
                logger.info(
                    "Val 2d result: mIoU_Novel/mAcc_Novel/allAcc_Novel {:.4f}/{:.4f}/{:.4f}.".format(
                        mIoU_2d_Novel, mAcc_2d_Novel, allAcc_2d_Novel
                    )
                )
                logger.info(
                    "Val 3d result: mIoU_Base/mAcc_Base/allAcc_Base {:.4f}/{:.4f}/{:.4f}.".format(
                        mIoU_3d_Base, mAcc_3d_Base, allAcc_3d_Base
                    )
                )
                logger.info(
                    "Val 3d result: mIoU_Novel/mAcc_Novel/allAcc_Novel {:.4f}/{:.4f}/{:.4f}.".format(
                        mIoU_3d_Novel, mAcc_3d_Novel, allAcc_3d_Novel
                    )
                )

                logger.info(
                    "Val F result: mIoU_Base/mAcc_Base/allAcc_Base {:.4f}/{:.4f}/{:.4f}.".format(
                        mIoU_Base, mAcc_Base, allAcc_Base
                    )
                )
                logger.info(
                    "Val F result: mIoU_Novel/mAcc_Novel/allAcc_Novel {:.4f}/{:.4f}/{:.4f}.".format(
                        mIoU_Novel, mAcc_Novel, allAcc_Novel
                    )
                )

                logger.info("iou_class_Base '{}'".format(iou_class_Base))
                logger.info("iou_class_Novel '{}'".format(iou_class_Novel))

                mask = mask.cpu().numpy()


if __name__ == "__main__":
    main()
