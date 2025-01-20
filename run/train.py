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
from tqdm import tqdm
from tensorboardX import SummaryWriter
import imageio
from MinkowskiEngine import SparseTensor
from util import config
from util.util import (
    AverageMeter,
    intersectionAndUnionGPU,
    poly_learning_rate,
    cosine_learning_rate,
    save_checkpoint,
    export_pointcloud,
)

from dataset.data_loader import (
    ScannetLoader,
    collation_fn,
)
from models.xmask3d import XMASK3d as Model
from models.checkpoint import XMask3dCheckpointer
import MinkowskiEngine as ME


best_iou = 0.0


def worker_init_fn(worker_id):
    """"""
    random.seed(time.time() + worker_id)


def get_parser():
    """"""

    parser = argparse.ArgumentParser(description="xmask3d.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/scannet/xmask3d_scannet_B12N7.yaml",
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

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:

        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    if not hasattr(args, "use_shm"):
        args.use_shm = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = len(args.train_gpu)

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(
            main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args)
        )
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
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
        pa_1 = 0

        for name, param in model.named_parameters():
            if "pc_decoder" in name or "pc_binary_head" in name:
                lr_3d.append(param)
            elif "ldm_extractor.ldm.ldm" in name or "clip.clip" in name:
                pa_1 += param.numel()
                pass
            else:
                lr_others.append(param)

        return [{"params": lr_3d, "lr": lr_1}, {"params": lr_others, "lr": lr_2}]

    param_groups = add_weight_decay(model, args.lr_3d, args.lr_others)
    optimizer = torch.optim.AdamW(param_groups)

    args.index_split = 0

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu], find_unused_parameters=True
        )
        assert args.batch_size_val == 1, f"Expected batch size of 1 during validation, but got {args.batch_size_val}"
    else:
        model = model.cuda()
    
    if args.batch_size < 4:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))

            ret = XMask3dCheckpointer(model, optimizer).load(args.resume, eval=False)

            args.start_epoch = ret["start_epoch"]

            best_iou = ret["best_iou"]
            if main_process():
                logger.info(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, ret["start_epoch"]
                    )
                )

        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    OurLoader = ScannetLoader

    if not hasattr(args, "input_color"):

        args.input_color = False
    train_data = OurLoader(
        datapath_prefix=args.data_root,
        datapath_prefix_2d=args.data_root_2d,
        category_split=args.category_split,
        caption_path=args.caption_path,
        label_2d=args.label_2d,
        scannet200=args.scannet200,
        voxel_size=args.voxel_size,
        split="train",
        aug=args.aug,
        memcache_init=args.use_shm,
        loop=args.loop,
        input_color=args.input_color,
    )
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_data)
        if args.distributed
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collation_fn,
        worker_init_fn=worker_init_fn,
    )
    if args.evaluate:
        val_data = OurLoader(
            datapath_prefix=args.data_root,
            datapath_prefix_2d=args.data_root_2d,
            category_split=args.category_split,
            caption_path=args.caption_path,
            label_2d=args.label_2d,
            scannet200=args.scannet200,
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
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collation_fn,
            sampler=val_sampler,
        )

    miou_fused = []
    miou_2d = []
    miou_3d = []
    miou_2d_4N = []
    miou_fused_4N = []
    miou_3d_4N = []
    miou_binary_4N = []
    miou_binary_15B = []
    miou_binary = []

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)
        if args.train_s:
            if epoch < args.start_contra:
                if args.mask_contra_3d:
                    model.module.criterion.weight_dict["loss_3d_contra"] = 0
                    model.module.criterion.weight_dict["loss_3d_contra_v2"] = 0
                    model.module.criterion.mask_contra_3d = False

            elif epoch >= args.start_contra:
                if args.mask_contra_3d:
                    model.module.criterion.weight_dict[
                        "loss_3d_contra"
                    ] = args.loss_weight.loss_3d_contra
                    model.module.criterion.weight_dict[
                        "loss_3d_contra_v2"
                    ] = args.loss_weight.loss_3d_contra
                    model.module.criterion.mask_contra_3d = True

            loss_train = train_net(train_loader, model, optimizer, epoch)

        epoch_log = epoch + 1

        if args.train_s:
            if main_process():
                writer.add_scalar("loss_train", loss_train, epoch_log)

        is_best = False

        if args.evaluate:
            if epoch_log % args.eval_freq == 0:
                val_data.epoch = epoch - 1
                (
                    mIoU_Base,
                    mIoU_Novel,
                    mIoU_2d_Base,
                    mIoU_2d_Novel,
                    mIoU_3d_Base,
                    mIoU_3d_Novel,
                    mIou_binary,
                    mIou_binary_Novel,
                    mIou_binary_Base,
                ) = validate(val_loader, model)
                miou_fused.append(mIoU_Base)
                miou_2d.append(mIoU_2d_Base)
                miou_3d.append(mIoU_3d_Base)
                miou_2d_4N.append(mIoU_2d_Novel)
                miou_3d_4N.append(mIoU_3d_Novel)
                miou_fused_4N.append(mIoU_Novel)
                miou_binary_4N.append(mIou_binary_Novel)
                miou_binary_15B.append(mIou_binary_Base)
                miou_binary.append(mIou_binary)

                if main_process() and args.train_s:
                    writer.add_scalar("mIoU_Base", mIoU_Base, epoch_log)
                    writer.add_scalar("mIoU_Novel", mIoU_Novel, epoch_log)
                    writer.add_scalar("mIoU_2d_Base", mIoU_2d_Base, epoch_log)
                    writer.add_scalar("mIoU_2d_Novel", mIoU_2d_Novel, epoch_log)
                    writer.add_scalar("mIoU_3d_Base", mIoU_3d_Base, epoch_log)
                    writer.add_scalar("mIoU_3d_Novel", mIoU_3d_Novel, epoch_log)
                    writer.add_scalar("mIou_binary", mIou_binary, epoch_log)
                    writer.add_scalar("mIou_binary_Base", mIou_binary_Base, epoch_log)
                    writer.add_scalar("mIou_binary_Novel", mIou_binary_Novel, epoch_log)

        if args.train_s:
            if (epoch_log % args.save_freq == 0) and main_process():

                save_checkpoint(
                    {
                        "epoch": epoch_log,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_iou": best_iou,
                    },
                    is_best,
                    os.path.join(args.save_path, "model"),
                )
            if (epoch_log % 5 == 0) and main_process():
                save_checkpoint(
                    {
                        "epoch": epoch_log,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_iou": best_iou,
                    },
                    is_best,
                    os.path.join(args.save_path, "model"),
                    "model_" + str(epoch_log) + ".pth.tar",
                )
            if epoch_log >= 110 and main_process():
                save_checkpoint(
                    {
                        "epoch": epoch_log,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_iou": best_iou,
                    },
                    is_best,
                    os.path.join(args.save_path, "model"),
                    "model_" + str(epoch_log) + ".pth.tar",
                )
    if main_process():
        writer.close()
        logger.info("==>Training done!\nBest Iou: %.3f" % (best_iou))


def get_model(cfg):
    """"""

    model = Model(cfg)
    return model


def train_net(train_loader, model, optimizer, epoch):
    """"""

    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()

    mask_loss_meter = AverageMeter()

    ce_loss_meter = AverageMeter()

    dice_loss_meter = AverageMeter()

    threed_loss_meter = AverageMeter()

    pure_threed_loss_meter = AverageMeter()

    pure_2d_loss_meter = AverageMeter()

    contra_3d_loss_meter = AverageMeter()

    loss_explicit_contra_meter = AverageMeter()
    loss_explicit_contra_2d_pre_meter = AverageMeter()
    loss_explicit_contra_3d_meter = AverageMeter()

    loss_binary_meter = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    intersection_meter_2d = AverageMeter()
    union_meter_2d = AverageMeter()
    target_meter_2d = AverageMeter()

    intersection_meter_3d = AverageMeter()
    union_meter_3d = AverageMeter()
    target_meter_3d = AverageMeter()

    intersection_meter_binary = AverageMeter()
    union_meter_binary = AverageMeter()
    target_meter_binary = AverageMeter()

    intersection_meter_binary_Base = AverageMeter()
    intersection_meter_binary_Novel = AverageMeter()
    target_meter_binary_Base = AverageMeter()
    target_meter_binary_Novel = AverageMeter()
    union_meter_binary_Base = AverageMeter()
    union_meter_binary_Novel = AverageMeter()

    pure_2d_miou_meter = AverageMeter()
    pure_2d_macc_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    for i, batch_data in enumerate(train_loader):

        data_time.update(time.time() - end)

        (
            ori_coords_3d,
            coords_3d,
            feat_3d,
            labels_3d,
            binary_label_3d,
            binary_label_2d,
            label_2d,
            img,
            x_label,
            y_label,
            inds_reconstruct,
            captions,
        ) = batch_data

        coords_3d[:, 1:4] += (torch.rand(3) * 100).type_as(coords_3d)

        sinput = SparseTensor(
            feat_3d.cuda(non_blocking=True), coords_3d.cuda(non_blocking=True)
        )

        batch_input = {}

        img = img.permute(0, 3, 1, 2).contiguous()

        batch_input["coords"] = coords_3d
        batch_input["img"] = img
        batch_input["sinput"] = sinput
        batch_input["captions"] = captions
        batch_input["x_label"] = x_label
        batch_input["y_label"] = y_label
        batch_input["label_2d"] = label_2d
        batch_input["inds_reconstruct"] = inds_reconstruct
        batch_input["labels_3d"] = labels_3d
        batch_input["ori_coords"] = ori_coords_3d
        batch_input["binary_label_3d"] = binary_label_3d
        batch_input["binary_label_2d"] = binary_label_2d

        loss, outputs = model(batch_input)

        save_loss = loss.copy()

        loss = sum(loss.values())

        fused_feature = outputs["fused_pred_feature"]
        text_features = outputs["text_embed"]
        null_embed = outputs["null_embed"]
        logit_scale = outputs["logit_scale"]

        feature_2d = outputs["2d_pred_feature"]
        feature_3d = outputs["pure3d_pred_feature"]

        final_pred_mask = outputs["final_pred_mask"]
        binary_pred = outputs["binary_pred"]
        binary_pred[
            torch.isin(
                batch_input["binary_label_3d"].to(binary_pred),
                torch.tensor(args.category_split.ignore_category).to(binary_pred),
            )
        ] = 2

        binary_label_3d[
            torch.isin(
                batch_input["binary_label_3d"].to(binary_pred),
                torch.tensor(args.category_split.ignore_category).to(binary_pred),
            )
        ] = 2
        binary_pred = binary_pred.squeeze(1).cpu()
        binary_label_3d = binary_label_3d.long().cpu()
        del outputs

        loss.backward()

        if (i + 1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item(), args.batch_size)
        mask_loss_meter.update(save_loss["loss_ce"].item(), args.batch_size)
        ce_loss_meter.update(save_loss["loss_mask"].item(), args.batch_size)
        dice_loss_meter.update(save_loss["loss_dice"].item(), args.batch_size)

        threed_loss_meter.update(save_loss["loss_3d"].item(), args.batch_size)
        pure_threed_loss_meter.update(save_loss["loss_3d_pure"].item(), args.batch_size)

        if args.mask_contra_3d and model.module.criterion.mask_contra_3d:
            contra_3d_loss_meter.update(
                save_loss["loss_3d_contra"].item(), args.batch_size
            )

        if args.caption_contra:
            loss_explicit_contra_meter.update(
                save_loss["loss_explicit_contra"].item(), args.batch_size
            )

        if args.caption_contra_2d_pre:
            loss_explicit_contra_2d_pre_meter.update(
                save_loss["loss_explicit_contra_2d_pre"].item(), args.batch_size
            )
        if args.caption_contra_3d:
            loss_explicit_contra_3d_meter.update(
                save_loss["loss_explicit_contra_3d"].item(), args.batch_size
            )
        loss_binary_meter.update(save_loss["loss_binary"].item(), args.batch_size)

        batch_time.update(time.time() - end)

        current_iter = epoch * len(train_loader) + i + 1
        assert args.learning_rate_type in ["poly", "cosine"]
        if args.learning_rate_type == "poly":
            current_lr_1 = poly_learning_rate(
                args.lr_3d, current_iter, max_iter, power=args.power
            )
            current_lr_2 = poly_learning_rate(
                args.lr_others, current_iter, max_iter, power=args.power
            )
        elif args.learning_rate_type == "cosine":
            current_lr_1 = cosine_learning_rate(args.lr_3d, current_iter, max_iter)
            current_lr_2 = cosine_learning_rate(args.lr_others, current_iter, max_iter)
        optimizer.param_groups[0]["lr"] = current_lr_1
        optimizer.param_groups[1]["lr"] = current_lr_2

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))

        mask = inds_reconstruct

        fused_feature = torch.cat(fused_feature)

        fused_feature = F.normalize(fused_feature, dim=-1)

        feature_2d = torch.cat(feature_2d)

        feature_2d = F.normalize(feature_2d, dim=-1)

        feature_3d = torch.cat(feature_3d)

        feature_3d = F.normalize(feature_3d, dim=-1)

        text_features = F.normalize(text_features, dim=-1)
        null_embed = F.normalize(null_embed, dim=-1)
        text_features = torch.cat([text_features, null_embed])

        logits_pred = logit_scale * (fused_feature @ text_features.t())

        logits_pred = torch.max(logits_pred, 1)[1].cpu()

        logits_pred_2d = logit_scale * (feature_2d @ text_features.t())

        logits_pred_3d = logit_scale * (feature_3d @ text_features.t())

        logits_pred_2d = torch.max(logits_pred_2d, 1)[1].cpu()

        logits_pred_3d = torch.max(logits_pred_3d, 1)[1].cpu()

        intersection, union, target = intersectionAndUnionGPU(
            logits_pred, labels_3d.detach(), args.classes, [args.ignore_label]
        )

        intersection_2d, union_2d, target_2d = intersectionAndUnionGPU(
            logits_pred_2d, labels_3d.detach(), args.classes, [args.ignore_label]
        )

        intersection_3d, union_3d, target_3d = intersectionAndUnionGPU(
            logits_pred_3d, labels_3d.detach(), args.classes, [args.ignore_label]
        )

        intersection_binary, union_binary, target_binary = intersectionAndUnionGPU(
            binary_pred, binary_label_3d.detach(), 2, [2]
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
            dist.all_reduce(intersection_binary), dist.all_reduce(
                union_binary
            ), dist.all_reduce(target_binary)

        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter.update(intersection), union_meter.update(
            union
        ), target_meter.update(target)

        intersection_2d, union_2d, target_2d = (
            intersection_2d.cpu().numpy(),
            union_2d.cpu().numpy(),
            target_2d.cpu().numpy(),
        )
        intersection_meter_2d.update(intersection_2d), union_meter_2d.update(
            union_2d
        ), target_meter_2d.update(target_2d)

        intersection_3d, union_3d, target_3d = (
            intersection_3d.cpu().numpy(),
            union_3d.cpu().numpy(),
            target_3d.cpu().numpy(),
        )
        intersection_meter_3d.update(intersection_3d), union_meter_3d.update(
            union_3d
        ), target_meter_3d.update(target_3d)

        intersection_binary, union_binary, target_binary = (
            intersection_binary.cpu().numpy(),
            union_binary.cpu().numpy(),
            target_binary.cpu().numpy(),
        )
        intersection_meter_binary.update(
            intersection_binary
        ), union_meter_binary.update(union_binary), target_meter_binary.update(
            target_binary
        )
        intersection_meter_binary_Base.update(
            intersection_binary[[1]]
        ), union_meter_binary_Base.update(
            union_binary[[1]]
        ), target_meter_binary_Base.update(
            target_binary[[1]]
        )

        intersection_meter_binary_Novel.update(
            intersection_binary[[0]]
        ), union_meter_binary_Novel.update(
            union_binary[[0]]
        ), target_meter_binary_Novel.update(
            target_binary[[0]]
        )

        if (i + 1) % args.print_freq == 0 and main_process():

            logger.info(
                "Epoch: [{}/{}][{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Remain {remain_time} "
                "Loss {loss_meter.avg:.4f} "
                "ce_loss {ce_loss_meter.avg:.4f} "
                "mask_loss {mask_loss_meter.avg:.4f} "
                "dice_loss {dice_loss_meter.avg:.4f} "
                "fused_loss {threed_loss_meter.avg:.4f} "
                "2d_loss {pure_2d_loss_meter.avg:.4f} "
                "3d_loss {pure_threed_loss_meter.avg:.4f} "
                "contra_3d_loss {contra_3d_loss_meter.avg:.4f} "
                "explicit_contra_loss {loss_explicit_contra_meter.avg:.4f} "
                "explicit_contra_3d_loss {loss_explicit_contra_3d_meter.avg:.4f} "
                "binary_loss {loss_binary_meter.avg:.4f} ".format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    loss_meter=loss_meter,
                    ce_loss_meter=ce_loss_meter,
                    mask_loss_meter=mask_loss_meter,
                    dice_loss_meter=dice_loss_meter,
                    threed_loss_meter=threed_loss_meter,
                    pure_2d_loss_meter=pure_2d_loss_meter,
                    pure_threed_loss_meter=pure_threed_loss_meter,
                    contra_3d_loss_meter=contra_3d_loss_meter,
                    loss_explicit_contra_meter=loss_explicit_contra_meter,
                    loss_explicit_contra_3d_meter=loss_explicit_contra_3d_meter,
                    loss_binary_meter=loss_binary_meter,
                )
            )
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

            iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
            accuracy_class_2d = intersection_meter_2d.sum / (
                target_meter_2d.sum + 1e-10
            )
            mIoU_2d = np.mean(iou_class_2d)
            mAcc_2d = np.mean(accuracy_class_2d)
            allAcc_2d = sum(intersection_meter_2d.sum) / (
                sum(target_meter_2d.sum) + 1e-10
            )

            iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
            accuracy_class_3d = intersection_meter_3d.sum / (
                target_meter_3d.sum + 1e-10
            )
            mIoU_3d = np.mean(iou_class_3d)
            mAcc_3d = np.mean(accuracy_class_3d)
            allAcc_3d = sum(intersection_meter_3d.sum) / (
                sum(target_meter_3d.sum) + 1e-10
            )

            iou_class_binary = intersection_meter_binary.sum / (
                union_meter_binary.sum + 1e-10
            )
            accuracy_class_binary = intersection_meter_binary.sum / (
                target_meter_binary.sum + 1e-10
            )
            mIoU_binary = np.mean(iou_class_binary)
            mAcc_binary = np.mean(accuracy_class_binary)
            allAcc_binary = sum(intersection_meter_binary.sum) / (
                sum(target_meter_binary.sum) + 1e-10
            )

            iou_class_binary_Base = intersection_meter_binary_Base.sum / (
                union_meter_binary_Base.sum + 1e-10
            )
            accuracy_class_binary_Base = intersection_meter_binary_Base.sum / (
                target_meter_binary_Base.sum + 1e-10
            )
            mIoU_binary_Base = np.mean(iou_class_binary_Base)
            mAcc_binary_Base = np.mean(accuracy_class_binary_Base)
            allAcc_binary_Base = sum(intersection_meter_binary_Base.sum) / (
                sum(target_meter_binary_Base.sum) + 1e-10
            )

            iou_class_binary_Novel = intersection_meter_binary_Novel.sum / (
                union_meter_binary_Novel.sum + 1e-10
            )
            accuracy_class_binary_Novel = intersection_meter_binary_Novel.sum / (
                target_meter_binary_Novel.sum + 1e-10
            )
            mIoU_binary_Novel = np.mean(iou_class_binary_Novel)
            mAcc_binary_Novel = np.mean(accuracy_class_binary_Novel)
            allAcc_binary_Novel = sum(intersection_meter_binary_Novel.sum) / (
                sum(target_meter_binary_Novel.sum) + 1e-10
            )

            logger.info(
                "Traning result: 2d_mIoU/2d_mACC/2d_3d_mIoU/2d_3d_mAcc/2d_3d_allAcc/fused_mIoU/fused_mAcc/fused_allACC/3d_mIoU/3d_mAcc/3d_allAcc/binary_mIoU/binary_mAcc/binary_allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}\n/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(
                    pure_2d_miou_meter.avg,
                    pure_2d_macc_meter.avg,
                    mIoU_2d,
                    mAcc_2d,
                    allAcc_2d,
                    mIoU,
                    mAcc,
                    allAcc,
                    mIoU_3d,
                    mAcc_3d,
                    allAcc_3d,
                    mIoU_binary,
                    mAcc_binary,
                    allAcc_binary,
                )
            )

            logger.info(
                "Val Binary_Base result: mIoU_binary_Base/mAcc_binary_Base/allAcc_binary_Base {:.4f}/{:.4f}/{:.4f}.".format(
                    mIoU_binary_Base, mAcc_binary_Base, allAcc_binary_Base
                )
            )
            logger.info(
                "Val Binary_Novel result: mIoU_binary_Novel/mAcc_binary_Novel/allAcc_binary_Novel {:.4f}/{:.4f}/{:.4f}.".format(
                    mIoU_binary_Novel, mAcc_binary_Novel, allAcc_binary_Novel
                )
            )
        if main_process():
            writer.add_scalar("loss_train_batch", loss_meter.val, current_iter)
            writer.add_scalar("learning_rate_3d", current_lr_1, current_iter)
            writer.add_scalar("learning_rate_others", current_lr_2, current_iter)

        end = time.time()
        torch.cuda.empty_cache()

    mask = inds_reconstruct

    logits_pred = logits_pred.numpy()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
    accuracy_class_2d = intersection_meter_2d.sum / (target_meter_2d.sum + 1e-10)
    mIoU_2d = np.mean(iou_class_2d)
    mAcc_2d = np.mean(accuracy_class_2d)
    allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)

    if main_process():
        logger.info(
            "Val pure 2d result: mIoU/mAcc {:.4f}/{:.4f}.".format(
                pure_2d_miou_meter.avg, pure_2d_macc_meter.avg
            )
        )
        logger.info(
            "Val 2d result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU_2d, mAcc_2d, allAcc_2d
            )
        )
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

    torch.cuda.empty_cache()
    return loss_meter.avg


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

    intersection_meter_binary = AverageMeter()
    union_meter_binary = AverageMeter()
    target_meter_binary = AverageMeter()

    intersection_meter_binary_Base = AverageMeter()
    intersection_meter_binary_Novel = AverageMeter()
    target_meter_binary_Base = AverageMeter()
    target_meter_binary_Novel = AverageMeter()
    union_meter_binary_Base = AverageMeter()
    union_meter_binary_Novel = AverageMeter()

    pure_2d_miou_meter = AverageMeter()
    pure_2d_macc_meter = AverageMeter()

    mask_miou_meter = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_data in tqdm(val_loader):

            torch.cuda.empty_cache()

            (
                ori_coords_3d,
                coords_3d,
                feat_3d,
                labels_3d,
                binary_label_3d,
                binary_label_2d,
                label_2d,
                img,
                x_label,
                y_label,
                inds_reconstruct,
                captions,
            ) = batch_data

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
            batch_input["captions"] = captions
            batch_input["inds_reconstruct"] = inds_reconstruct
            batch_input["labels_3d"] = labels_3d
            batch_input["ori_coords"] = ori_coords_3d

            _, outputs = model(batch_input)

            final_pred_open_embedding = outputs["final_pred_open_embedding"]
            final_mask_3d = outputs["final_mask_3d"]

            fused_feature = outputs["fused_pred_feature"]
            text_features = outputs["text_embed"]
            logit_scale = outputs["logit_scale"]
            feature_2d = outputs["2d_pred_feature"]
            feature_3d = outputs["pure3d_pred_feature"]

            final_pred_mask = outputs["pred_masks"]
            binary_pred = outputs["binary_pred"]
            batch_input["binary_label_2d"] = binary_label_2d

            binary_pred_copy = binary_pred.clone()
            binary_pred_copy[
                torch.isin(
                    binary_label_3d.to(binary_pred),
                    torch.tensor(args.category_split.ignore_category).to(binary_pred),
                )
            ] = 2

            binary_label_3d[
                torch.isin(
                    binary_label_3d.to(binary_pred),
                    torch.tensor(args.category_split.ignore_category).to(binary_pred),
                )
            ] = 2
            binary_pred_copy = binary_pred_copy.squeeze(1).cpu()
            binary_label_3d = binary_label_3d.long().cpu()
            del outputs

            mask = inds_reconstruct

            fused_feature = torch.cat(fused_feature)

            fused_feature = F.normalize(fused_feature, dim=-1)

            feature_2d = torch.cat(feature_2d)

            feature_2d = F.normalize(feature_2d, dim=-1)
            feature_3d = torch.cat(feature_3d)

            feature_3d = F.normalize(feature_3d, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            logits_pred = logit_scale * (fused_feature @ text_features.t())

            final_pred_open_embedding = torch.cat(final_pred_open_embedding)
            final_pred_open_embedding = F.normalize(final_pred_open_embedding, dim=-1)
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
                category_overlapping_list, device=logits_pred.device, dtype=torch.long
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
                binary_pred * logits_pred_base + (1 - binary_pred) * logits_pred_novel
            )

            logits_pred = torch.max(logits_pred, 1)[1].cpu()

            logits_pred_2d = logit_scale * (feature_2d @ text_features.t())

            logits_pred_novel = logits_pred_2d.clone()
            logits_pred_base = logits_pred_2d.clone()
            logits_pred_novel[:, args.category_split.base_category] = -1e10
            logits_pred_base[:, args.category_split.novel_category] = -1e10
            logits_pred_2d = (
                binary_pred * logits_pred_base + (1 - binary_pred) * logits_pred_novel
            )

            logits_pred_3d = logit_scale * (feature_3d @ text_features.t())

            logits_pred_novel = logits_pred_3d.clone()
            logits_pred_base = logits_pred_3d.clone()
            logits_pred_novel[:, args.category_split.base_category] = -1e10
            logits_pred_base[:, args.category_split.novel_category] = -1e10
            logits_pred_3d = (
                binary_pred * logits_pred_base + (1 - binary_pred) * logits_pred_novel
            )

            logits_pred_3d = torch.max(logits_pred_3d, 1)[1].cpu()

            logits_pred_2d = torch.max(logits_pred_2d, 1)[1].cpu()
            torch.cuda.empty_cache()

            intersection, union, target = intersectionAndUnionGPU(
                logits_pred,
                labels_3d.detach(),
                args.test_classes,
                args.test_ignore_label,
            )

            intersection_2d, union_2d, target_2d = intersectionAndUnionGPU(
                logits_pred_2d,
                labels_3d.detach(),
                args.test_classes,
                args.test_ignore_label,
            )

            intersection_3d, union_3d, target_3d = intersectionAndUnionGPU(
                logits_pred_3d,
                labels_3d.detach(),
                args.test_classes,
                args.test_ignore_label,
            )
            intersection_binary, union_binary, target_binary = intersectionAndUnionGPU(
                binary_pred_copy, binary_label_3d.detach(), 2, [2]
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
                dist.all_reduce(intersection_binary), dist.all_reduce(
                    union_binary
                ), dist.all_reduce(target_binary)

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
            intersection_binary, union_binary, target_binary = (
                intersection_binary.cpu().numpy(),
                union_binary.cpu().numpy(),
                target_binary.cpu().numpy(),
            )
            intersection_meter_binary.update(
                intersection_binary
            ), union_meter_binary.update(union_binary), target_meter_binary.update(
                target_binary
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

            intersection_meter_binary_Base.update(
                intersection_binary[[1]]
            ), union_meter_binary_Base.update(
                union_binary[[1]]
            ), target_meter_binary_Base.update(
                target_binary[[1]]
            )

            intersection_meter_binary_Novel.update(
                intersection_binary[[0]]
            ), union_meter_binary_Novel.update(
                union_binary[[0]]
            ), target_meter_binary_Novel.update(
                target_binary[[0]]
            )

        iou_class_Base = intersection_meter_Base.sum / (union_meter_Base.sum + 1e-10)
        accuracy_class_Base = intersection_meter_Base.sum / (
            target_meter_Base.sum + 1e-10
        )

        mIoU_Base = np.mean(iou_class_Base)
        mAcc_Base = np.mean(accuracy_class_Base)
        allAcc_Base = sum(intersection_meter_Base.sum) / (
            sum(target_meter_Base.sum) + 1e-10
        )

        iou_class_Novel = intersection_meter_Novel.sum / (union_meter_Novel.sum + 1e-10)
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

        iou_class_binary = intersection_meter_binary.sum / (
            union_meter_binary.sum + 1e-10
        )
        accuracy_class_binary = intersection_meter_binary.sum / (
            target_meter_binary.sum + 1e-10
        )
        mIoU_binary = np.mean(iou_class_binary)
        mAcc_binary = np.mean(accuracy_class_binary)
        allAcc_binary = sum(intersection_meter_binary.sum) / (
            sum(target_meter_binary.sum) + 1e-10
        )

        iou_class_binary_Base = intersection_meter_binary_Base.sum / (
            union_meter_binary_Base.sum + 1e-10
        )
        accuracy_class_binary_Base = intersection_meter_binary_Base.sum / (
            target_meter_binary_Base.sum + 1e-10
        )
        mIoU_binary_Base = np.mean(iou_class_binary_Base)
        mAcc_binary_Base = np.mean(accuracy_class_binary_Base)
        allAcc_binary_Base = sum(intersection_meter_binary_Base.sum) / (
            sum(target_meter_binary_Base.sum) + 1e-10
        )

        iou_class_binary_Novel = intersection_meter_binary_Novel.sum / (
            union_meter_binary_Novel.sum + 1e-10
        )
        accuracy_class_binary_Novel = intersection_meter_binary_Novel.sum / (
            target_meter_binary_Novel.sum + 1e-10
        )
        mIoU_binary_Novel = np.mean(iou_class_binary_Novel)
        mAcc_binary_Novel = np.mean(accuracy_class_binary_Novel)
        allAcc_binary_Novel = sum(intersection_meter_binary_Novel.sum) / (
            sum(target_meter_binary_Novel.sum) + 1e-10
        )

        if main_process():
            logger.info(
                "Val pure 2d result: mIoU/mAcc {:.4f}/{:.4f}.".format(
                    pure_2d_miou_meter.avg, pure_2d_macc_meter.avg
                )
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
            logger.info(
                "Val Binary result: mIoU_binary/mAcc_binary/allAcc_binary {:.4f}/{:.4f}/{:.4f}.".format(
                    mIoU_binary, mAcc_binary, allAcc_binary
                )
            )
            logger.info(
                "Val Binary_Base result: mIoU_binary_Base/mAcc_binary_Base/allAcc_binary_Base {:.4f}/{:.4f}/{:.4f}.".format(
                    mIoU_binary_Base, mAcc_binary_Base, allAcc_binary_Base
                )
            )
            logger.info(
                "Val Binary_Novel result: mIoU_binary_Novel/mAcc_binary_Novel/allAcc_binary_Novel {:.4f}/{:.4f}/{:.4f}.".format(
                    mIoU_binary_Novel, mAcc_binary_Novel, allAcc_binary_Novel
                )
            )

    return (
        mIoU_Base,
        mIoU_Novel,
        mIoU_2d_Base,
        mIoU_2d_Novel,
        mIoU_3d_Base,
        mIoU_3d_Novel,
        mIoU_binary,
        mIoU_binary_Novel,
        mIoU_binary_Base,
    )


if __name__ == "__main__":
    main()
