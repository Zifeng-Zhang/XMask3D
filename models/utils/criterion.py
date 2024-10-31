import torch
import torch.nn as nn
import torch.nn.functional as F
from mask2former.modeling.criterion import SetCriterion
from detectron2.utils.comm import get_world_size

from models.modeling.meta_arch.clip import MaskCLIP
from .fuser import mask_mapper, is_dist_avail_and_initialized, FeatureMerger


class Criterion(SetCriterion):
    def __init__(self, *args, cfg, **kwargs):
        super(Criterion, self).__init__(*args, **kwargs)

        self.weight_dict.update({"loss_3d": cfg.loss_weight.loss_3d})
        self.weight_dict.update({"loss_3d_pure": cfg.loss_weight.loss_3d_pure})
        self.weight_dict.update(
            {"loss_explicit_contra": cfg.loss_weight.loss_explicit_contra}
        )
        self.weight_dict.update(
            {"loss_explicit_contra_3d": cfg.loss_weight.loss_explicit_contra_3d}
        )
        self.weight_dict.update(
            {"loss_explicit_contra_2d_pre": cfg.loss_weight.loss_explicit_contra_2d_pre}
        )
        self.weight_dict.update({"loss_binary": cfg.loss_weight.loss_binary})

        self.fuser = FeatureMerger(feature_dim=768)
        self.criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)
        self.ignore_label = cfg.ignore_label
        self.mask_contra_3d = cfg.mask_contra_3d

        self.fc1 = nn.Identity()
        self.fc2 = nn.Identity()
        self.contra_criterion = torch.nn.CosineSimilarity()
        self.cfg = cfg
        self.clip = MaskCLIP(name=cfg.clip_name)

    def loss_contra(
        self,
        x_list,
        y_list,
        masks,
        clip_mask_embeddings,
        binary_gts,
        outputs,
    ):
        masks = outputs["pred_masks"]
        mask_embeds = outputs["mask_embed"]
        clip_mask_embeddings = outputs["mask_embed_clip"]
        features_fused = outputs["fused_pred_feature"]
        features_3d = outputs["pure3d_pred_feature"]
        masks = F.interpolate(
            masks, size=tuple(self.cfg.mask_shape), mode="bilinear", align_corners=False
        )
        final_2d_mask = []
        embedding_3d = []
        embedding_fused = []
        embedding_gt = []
        contra_criterion = torch.nn.CosineSimilarity()
        bs_count = 0

        for (
            x_label,
            y_label,
            mask,
            feature_fused,
            mask_embed,
            feature_3d,
            clip_mask_embedding,
            binary_gt,
        ) in zip(
            x_list,
            y_list,
            masks,
            features_fused,
            mask_embeds,
            features_3d,
            clip_mask_embeddings,
            binary_gts,
        ):
            mask_copy = mask.clone()
            mask_3d = mask[:, x_label, y_label].clone()
            mask_3d = mask_3d.sigmoid()
            mask_3d = mask_3d >= 0.5

            if len(mask_3d[torch.sum(mask_3d, dim=1) >= 10]) == 0:
                mask_3d[0, :] = True

            keep = torch.sum(mask_3d, dim=1) >= 10
            mask_copys = mask_copy[keep]
            clip_mask_embedding = clip_mask_embedding[keep]
            mask_embed = mask_embed[keep]
            mask_3d = mask_3d[keep]

            base_list_idx = []
            novel_list_idx = []
            mask_count = 0

            for mask_3d_single, mask_copy in zip(mask_3d, mask_copys):
                mask_copy_ = mask_copy.clone()
                mask_copy_ = mask_copy_.sigmoid()
                binary_gt_ = binary_gt[mask_3d_single]
                novel_num = binary_gt_.eq(0).sum().item()
                base_num = len(binary_gt_) - novel_num
                base_num_ = binary_gt_.eq(1).sum().item()
                novel_num_ = len(binary_gt_) - base_num_

                if novel_num > 1.8 * base_num and novel_num > 10:
                    novel_list_idx.append(
                        (mask_count, torch.mean(mask_copy_[mask_copy_ > 0.5]))
                    )

                elif base_num_ > 20 * novel_num_ and base_num_ > 150:
                    base_list_idx.append(
                        (mask_count, torch.mean(mask_copy_[mask_copy_ > 0.5]))
                    )
                else:
                    mask_count += 1
                    continue

                mask_count += 1

            one_batch_embedding_fused = []
            one_batch_embedding_3d = []

            one_batch_embedding_gt = []
            mask_copy_list = []

            if len(novel_list_idx) != 0 or len(base_list_idx) != 0:
                if len(novel_list_idx) != 0:
                    novel_list_idx = sorted(
                        novel_list_idx, key=lambda x: x[1], reverse=True
                    )
                    novel_list_idx = [item[0] for item in novel_list_idx]
                    novel_list_idx = novel_list_idx[:4]

                if len(base_list_idx) != 0:
                    base_list_idx = sorted(
                        base_list_idx, key=lambda x: x[1], reverse=True
                    )

                    base_list_idx = [item[0] for item in base_list_idx]
                    base_list_idx = base_list_idx[:1]

                final_list_idx = novel_list_idx + base_list_idx

                for fidx in final_list_idx:

                    one_batch_embedding_gt.append(clip_mask_embedding[fidx])
                    mask_copy_list.append(mask_copys[fidx])
                    feature_fused_ = feature_fused[mask_3d[fidx]]
                    feature_fused_ = torch.mean(feature_fused_, dim=0)
                    one_batch_embedding_fused.append(feature_fused_)
                    feature_3d_ = feature_3d[mask_3d[fidx]]
                    feature_3d_ = torch.mean(feature_3d_, dim=0)
                    one_batch_embedding_3d.append(feature_3d_)

                embedding_fused.append(torch.stack(one_batch_embedding_fused))
                embedding_3d.append(torch.stack(one_batch_embedding_3d))
                embedding_gt.append(torch.stack(one_batch_embedding_gt))

                final_2d_mask.append((bs_count, torch.stack(mask_copy_list)))
            bs_count += 1

        if len(embedding_fused) != 0:
            embedding_fused = torch.cat(embedding_fused)
            embedding_3d = torch.cat(embedding_3d)

            embedding_gt = torch.cat(embedding_gt)
            embedding_gt = embedding_gt.detach()
            loss_3d_contra = (1 - contra_criterion(embedding_3d, embedding_gt)).mean()

        else:
            invalid_embed = torch.stack([mask_embed[0]])
            loss_3d_contra = (1 - contra_criterion(invalid_embed, invalid_embed)).mean()

        loss = {}

        loss.update({"loss_3d_contra": loss_3d_contra})

        return loss, final_2d_mask

    def loss_exact(self, outputs, gt):

        fused_features = outputs["fused_pred_feature"]
        feature_3d = outputs["pure3d_pred_feature"]
        text_embed = outputs["text_embed"]
        null_embed = outputs["null_embed"]
        feature_3d = torch.cat(feature_3d)
        fused_feature = torch.cat(fused_features)
        fused_feature = F.normalize(fused_feature, dim=-1)
        feature_3d = F.normalize(feature_3d, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        null_embed = F.normalize(null_embed, dim=-1)
        logit_scale = outputs["logit_scale"]
        text_embed = torch.cat([text_embed, null_embed])
        pred = logit_scale * (fused_feature @ text_embed.t())
        pred_3d = logit_scale * (feature_3d @ text_embed.t())

        if (gt == self.ignore_label).all():
            gt[0] = self.ignore_label - 1

        loss = self.criterion(pred, gt)
        loss_3d = self.criterion(pred_3d, gt)

        return {"loss_3d": loss, "loss_3d_pure": loss_3d}

    def forward(self, outputs, targets, batch_input):

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets)

        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        if self.training:
            losses = {}
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        mask_embed_list = outputs["mask_embed"]

        pred_masks_list = outputs["pred_masks"]

        clip_results = self.clip(
            outputs["images"],
            pred_masks_list,
        )
        outputs.update(clip_results)
        pred_open_embeddings = outputs["mask_embed_clip"]

        pred_masks_list = F.interpolate(
            pred_masks_list,
            size=tuple(self.cfg.mask_shape),
            mode="bilinear",
            align_corners=False,
        )

        gt = batch_input["labels_3d"]
        ori_coords = batch_input["ori_coords"]

        pred_logits_list = outputs["pred_logits"]
        binary_gts = batch_input["binary_label_3d"]

        ori_coords_list = []
        pred_3d_list = []
        x_list = []
        y_list = []
        pred_masks_final_list = []
        pred_embedding_final_list = []
        pred_open_embedding_final_list = []
        binary_gt_list = []
        for scene_idx in ori_coords[:, 0].unique():
            x_label = batch_input["x_label"][ori_coords[:, 0] == scene_idx]
            y_label = batch_input["y_label"][ori_coords[:, 0] == scene_idx]
            single_ori_coords = ori_coords[ori_coords[:, 0] == scene_idx]
            single_pred_3d = outputs["pred_3d"][ori_coords[:, 0] == scene_idx]
            binary_gt = binary_gts[ori_coords[:, 0] == scene_idx]
            binary_gt_list.append(binary_gt)
            mask_pred_result = pred_masks_list[int(scene_idx)]
            mask_cls = pred_logits_list[int(scene_idx)]
            pred_embedding = mask_embed_list[int(scene_idx)]
            pred_open_embedding = pred_open_embeddings[int(scene_idx)]

            num_classes = self.ignore_label
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred_result.sigmoid()
            labels[labels > num_classes - 1] = num_classes

            keep = scores > 0
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_embedding = pred_embedding[keep]
            cur_open_embedding = pred_open_embedding[keep]
            cur_masks = mask_pred[keep]
            if cur_masks.shape[0] == 0:
                pred_embedding = torch.zeros_like(pred_embedding)
                pred_open_embedding = torch.zeros_like(pred_open_embedding)

                mask_pred_result = torch.zeros_like(mask_pred)

                x_list.append(x_label)
                y_list.append(y_label)
                ori_coords_list.append(single_ori_coords[:, 1:4])
                pred_3d_list.append(single_pred_3d)
                pred_masks_final_list.append(mask_pred_result)
                pred_embedding_final_list.append(pred_embedding)
                pred_open_embedding_final_list.append(pred_open_embedding)
            else:
                cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
                cur_mask_ids = cur_prob_masks.argmax(0)
                final_keep = []
                final_mask = []
                for k in range(cur_classes.shape[0]):
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area <= 0:
                            continue
                        final_keep.append(k)
                        final_mask.append(mask)

                if len(final_keep) != 0:
                    pred_embedding = cur_embedding[final_keep]
                    pred_open_embedding = cur_open_embedding[final_keep]
                    mask_pred_result = torch.stack(final_mask).to(pred_embedding.device)
                else:

                    pred_embedding = torch.zeros_like(pred_embedding)
                    pred_open_embedding = torch.zeros_like(pred_open_embedding)
                    mask_pred_result = torch.zeros_like(mask_pred)

                x_list.append(x_label)
                y_list.append(y_label)
                ori_coords_list.append(single_ori_coords[:, 1:4])
                pred_3d_list.append(single_pred_3d)
                pred_masks_final_list.append(mask_pred_result)
                pred_embedding_final_list.append(pred_embedding)
                pred_open_embedding_final_list.append(pred_open_embedding)

        fused_feature, output_2d, output_3d, output_2d_pre = mask_mapper(
            x_list,
            y_list,
            pred_masks_final_list,
            pred_embedding_final_list,
            pred_3d_list,
            self.fuser,
            self.fc1,
            self.fc2,
            self.cfg,
        )
        outputs.update({"fused_pred_feature": fused_feature})
        outputs.update({"2d_pred_feature": output_2d})
        outputs.update({"pure3d_pred_feature": output_3d})

        outputs.update({"2d_pred_feature_pre": output_2d_pre})

        outputs.update({"final_pred_mask": pred_masks_final_list})

        if self.training:
            loss = self.loss_exact(outputs, gt)
            if self.mask_contra_3d:
                loss_contra, final_2d_mask = self.loss_contra(
                    x_list,
                    y_list,
                    pred_masks_final_list,
                    pred_open_embedding_final_list,
                    binary_gt_list,
                    outputs,
                )
                losses.update(loss_contra)
                outputs.update({"final_pred_mask": final_2d_mask})
            losses.update(loss)

        if not self.training:
            return outputs
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, outputs
