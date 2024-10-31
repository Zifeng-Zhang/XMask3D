from torch.nn import functional as F
import numpy as np
from torch import nn
import torch
from timm.models.layers import trunc_normal_
from models.modeling.meta_arch.pc_processor import (
    PC_Processor,
    PC_Binary_Processor,
)
from models.modeling.meta_arch.ldm import LdmImplicitCaptionerExtractor
from models.utils.criterion import Criterion

from models.modeling.meta_arch.odise import (
    ODISEMultiScaleMaskedTransformerDecoder,
    PooledMaskEmbed,
    CategoryEmbed,
    PseudoClassEmbed,
)

from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead
from mask2former.modeling.matcher import HungarianMatcher
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from models.modeling.meta_arch.helper import ensemble_logits_with_labels
from models.modeling.backbone.feature_extractor import FeatureExtractorBackbone
from detectron2.structures import ImageList


class XMASK3d(nn.Module):
    def __init__(self, cfg=None):
        super(XMASK3d, self).__init__()
        self.cfg = cfg
        num_classes = cfg.classes
        self.pixel_mean = cfg.pixel_mean
        self.pixel_std = cfg.pixel_std
        num_queries = cfg.num_queries
        self.seq_len = 77
        self.size_divisibility = 64

        self.pc_decoder = PC_Processor(arch_3d=cfg.arch_3d)

        self.pc_binary_head = PC_Binary_Processor(arch_3d=cfg.arch_binary_head)

        self.ignore_label = cfg.category_split.ignore_category

        self.binary_loss_func = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([cfg.data_ratio])
        )

        self.backbone = FeatureExtractorBackbone(
            feature_extractor=LdmImplicitCaptionerExtractor(
                encoder_block_indices=(5, 7),
                unet_block_indices=(2, 5, 8, 11),
                decoder_block_indices=(2, 5),
                steps=(0,),
                learnable_time_embed=True,
                num_timesteps=1,
                dim_latent=768,
                clip=None,
            ),
            out_features=["s2", "s3", "s4", "s5"],
            use_checkpoint=True,
            slide_training=False,
        )
        self.sem_seg_head = MaskFormerHead(
            ignore_value=255,
            num_classes=num_classes,
            pixel_decoder=MSDeformAttnPixelDecoder(
                conv_dim=256,
                mask_dim=256,
                norm="GN",
                transformer_dropout=0.0,
                transformer_nheads=8,
                transformer_dim_feedforward=1024,
                transformer_enc_layers=6,
                transformer_in_features=["s3", "s4", "s5"],
                common_stride=4,
                input_shape=self.backbone.output_shape(),
            ),
            loss_weight=1.0,
            transformer_in_feature="multi_scale_pixel_decoder",
            transformer_predictor=ODISEMultiScaleMaskedTransformerDecoder(
                class_embed=PseudoClassEmbed(num_classes=num_classes),
                hidden_dim=256,
                post_mask_embed=PooledMaskEmbed(
                    hidden_dim=256, mask_dim=256, projection_dim=768
                ),
                in_channels=256,
                mask_classification=True,
                num_classes=num_classes,
                num_queries=num_queries,
                nheads=8,
                dim_feedforward=2048,
                dec_layers=9,
                pre_norm=False,
                enforce_input_project=False,
                mask_dim=256,
            ),
            input_shape=self.backbone.output_shape(),
        )

        self.criterion = Criterion(
            num_layers=9,
            class_weight=2.0,
            mask_weight=5.0,
            dice_weight=5.0,
            num_classes=num_classes,
            matcher=HungarianMatcher(
                cost_class=2.0,
                cost_mask=5.0,
                cost_dice=5.0,
                num_points=12544,
            ),
            eos_coef=0.1,
            losses=["labels", "masks"],
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            cfg=cfg,
        )

        self.category_head = CategoryEmbed(
            clip_model_name=self.criterion.clip,
            labels=[[label] for label in cfg.label],
            test_labels=[[label] for label in cfg.all_label],
            projection_dim=-1,
        )
        self.clip_head = self.criterion.clip

    def cal_pred_logits(self, outputs):
        mask_embed = outputs["mask_embed"]
        text_embed = outputs["text_embed"]
        text_embed = outputs["text_embed"]
        null_embed = outputs["null_embed"]
        labels = outputs["labels"]
        mask_embed = F.normalize(mask_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        logit_scale = outputs["logit_scale"]
        pred = logit_scale * (mask_embed @ text_embed.t())
        pred = ensemble_logits_with_labels(pred, labels, ensemble_method="max")
        null_embed = F.normalize(null_embed, dim=-1)
        null_pred = logit_scale * (mask_embed @ null_embed.t())
        pred = torch.cat([pred, null_pred], dim=-1)
        return pred

    def forward(self, batch_input):

        label_2d = batch_input["label_2d"].detach().cpu()
        imp_condition, pred_3d, _idx_ = self.pc_decoder(batch_input["sinput"])
        caption_embed = self.category_head.clip.embed_text(
            batch_input["captions"]
        ).text_embed
        pred_3d = pred_3d[batch_input["inds_reconstruct"], :]
        imp_condition_input = []
        for scene_idx in torch.unique(_idx_):
            single = imp_condition[_idx_ == scene_idx]
            sparse_x = torch.max(single, dim=0, keepdim=False)[0]
            imp_condition_input.append(sparse_x)

        imp_condition_input = torch.stack(imp_condition_input)
        images = [
            (x - torch.tensor(self.pixel_mean).view(3, 1, 1).to(x.device))
            / torch.tensor(self.pixel_std).view(3, 1, 1).to(x.device)
            for x in batch_input["img"]
        ]
        images = ImageList.from_tensors(images, self.size_divisibility)
        denormalized_images = ImageList.from_tensors(
            [x.to(x.device) / 255.0 for x in batch_input["img"]]
        )
        feature = self.backbone(images.tensor, imp_condition_input)

        binary_scores = self.pc_binary_head(batch_input["sinput"])
        binary_scores = binary_scores[batch_input["inds_reconstruct"], :]

        outputs = self.sem_seg_head(feature)

        outputs.update({"pred_3d": pred_3d})

        outputs.update({"images": denormalized_images.tensor})

        targets = []

        if self.training:
            binary_pred = (torch.sigmoid(binary_scores) > 0.5).long()

            caption_embed = self.category_head.text_proj(caption_embed)

            h_pad, w_pad = batch_input["img"].shape[-2:]

            for idx in range(label_2d.shape[0]):
                unique_values = np.unique(label_2d[idx])

                masks = np.stack([label_2d[idx] == value for value in unique_values])
                labels = unique_values.astype(np.int64)

                masks = torch.from_numpy(masks).to(batch_input["img"].device)

                labels = torch.from_numpy(labels).to(batch_input["img"].device)

                targets.append(
                    {
                        "labels": labels,
                        "masks": masks,
                    }
                )
            new_targets = []
            for target in targets:

                gt_masks = target["masks"]

                padded_masks = torch.zeros(
                    (gt_masks.shape[0], h_pad, w_pad),
                    dtype=gt_masks.dtype,
                    device=gt_masks.device,
                )

                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": target["labels"],
                        "masks": padded_masks.float(),
                    }
                )

            targets = new_targets

            if self.category_head is not None:
                category_head_outputs = self.category_head(outputs, targets)
                outputs.update(category_head_outputs)

                outputs["pred_logits"] = self.cal_pred_logits(outputs)
                if "aux_outputs" in outputs:
                    for aux_outputs in outputs["aux_outputs"]:
                        aux_outputs.update(category_head_outputs)

                        aux_outputs["pred_logits"] = self.cal_pred_logits(aux_outputs)

            losses, outputs = self.criterion(outputs, targets, batch_input)

            fused_features = outputs["fused_pred_feature"]

            fused_features_ = []
            for fused_feature in fused_features:
                fused_features_.append(fused_feature.mean(0, keepdim=False))
            fused_features_ = torch.stack(fused_features_)

            loss_explicit_contra = (
                1 - torch.nn.CosineSimilarity()(fused_features_, caption_embed)
            ).mean()

            features_3d = outputs["pure3d_pred_feature"]

            features_3d_ = []
            for feature_3d in features_3d:
                features_3d_.append(feature_3d.mean(0, keepdim=False))
            features_3d_ = torch.stack(features_3d_)
            loss_explicit_contra_3d = (
                1 - torch.nn.CosineSimilarity()(features_3d_, caption_embed)
            ).mean()

            if self.cfg.caption_contra_2d_pre:

                features_2d_pre = outputs["2d_pred_feature_pre"]

                features_2d_pre_ = []
                for feature_2d_pre in features_2d_pre:
                    features_2d_pre_.append(feature_2d_pre.mean(0, keepdim=False))
                features_2d_pre_ = torch.stack(features_2d_pre_)

                loss_explicit_contra_2d_pre = (
                    1 - torch.nn.CosineSimilarity()(features_2d_pre_, caption_embed)
                ).mean()

            binary_labels = batch_input["binary_label_3d"]

            mask_binary = ~torch.isin(
                batch_input["binary_label_3d"],
                torch.tensor(self.ignore_label).to(binary_labels),
            )

            binary_labels = binary_labels[mask_binary]
            binary_scores = binary_scores[mask_binary]

            loss_binary = self.binary_loss_func(
                binary_scores, binary_labels.reshape(-1, 1)
            )

            if self.cfg.caption_contra:
                losses.update({"loss_explicit_contra": loss_explicit_contra})

            if self.cfg.caption_contra_2d_pre:
                losses.update(
                    {"loss_explicit_contra_2d_pre": loss_explicit_contra_2d_pre}
                )
            if self.cfg.caption_contra_3d:
                losses.update({"loss_explicit_contra_3d": loss_explicit_contra_3d})

            losses.update({"loss_binary": loss_binary})
            outputs.update({"binary_pred": binary_pred})
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:

                    losses.pop(k)
            return losses, outputs
        else:

            binary_pred = (torch.sigmoid(binary_scores) > 0.5).long()
            outputs.update(self.category_head(outputs))

            outputs["pred_logits"] = self.cal_pred_logits(outputs)

            mask_pred_results = outputs["pred_masks"]
            mask_cls_results = outputs["pred_logits"]

            pred_embeddings = outputs["mask_embed"]
            if self.clip_head is not None:

                clip_results = self.clip_head(
                    outputs["images"],
                    mask_pred_results,
                )
                outputs.update(clip_results)

                pred_open_embeddings = outputs["mask_embed_clip"]
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=tuple(self.cfg.mask_shape),
                mode="bilinear",
                align_corners=False,
            )

            ori_coords = batch_input["ori_coords"]
            pred_logits = outputs["pred_logits"]
            output = []
            output_2d = []
            output_3d = []
            mask_3d_list = []
            pred_open_embedding_list = []

            for scene_idx in range(torch.max(ori_coords[:, 0]).int() + 1):

                x_label = batch_input["x_label"][ori_coords[:, 0] == scene_idx]
                y_label = batch_input["y_label"][ori_coords[:, 0] == scene_idx]

                single_pred_3d = outputs["pred_3d"][ori_coords[:, 0] == scene_idx]

                pred_embedding = pred_embeddings[scene_idx]
                pred_open_embedding = pred_open_embeddings[scene_idx]
                mask_pred_result = mask_pred_results[scene_idx]

                test_ignore_label = self.cfg.test_ignore_label
                num_classes = test_ignore_label[0]
                mask_cls = pred_logits[scene_idx]

                mask_3d_full = mask_pred_result[:, x_label, y_label].sigmoid()
                mask_3d_full = mask_3d_full > 0.5
                keep_full = torch.sum(mask_3d_full, dim=1) > 0

                mask_3d_full = mask_3d_full[keep_full]

                binary_scores = torch.sigmoid(binary_scores).view(1, -1)

                binary_scores_full = binary_scores * mask_3d_full
                binary_pred_full = torch.sum(binary_scores_full, dim=1) / (
                    torch.sum(mask_3d_full, dim=1) + 1e-10
                )
                binary_pred_full_base = (
                    binary_pred_full > self.cfg.binary_2d_thresh
                ).view(-1, 1)
                binary_pred_full_novel = (
                    binary_pred_full <= self.cfg.binary_2d_thresh
                ).view(-1, 1)

                mask_cls_full = mask_cls[keep_full]
                mask_pred_result = mask_pred_result[keep_full]
                pred_embedding = pred_embedding[keep_full]
                pred_open_embedding = pred_open_embedding[keep_full]
                logits_pred_novel = mask_cls_full.clone()
                logits_pred_base = mask_cls_full.clone()
                logits_pred_novel[
                    :, self.cfg.category_split.base_category + [num_classes]
                ] = -1e10
                logits_pred_base[:, self.cfg.category_split.novel_category] = -1e10

                modified_logits = (
                    binary_pred_full_base * logits_pred_base
                    + binary_pred_full_novel * logits_pred_novel
                )

                scores, labels = F.softmax(modified_logits, dim=-1).max(-1)
                mask_pred = mask_pred_result.sigmoid()

                labels[labels > num_classes - 1] = num_classes

                keep = scores > self.cfg.scores_keep_thresh
                cur_scores = scores[keep]
                cur_classes = labels[keep]
                cur_embedding = pred_embedding[keep]
                cur_open_embedding = pred_open_embedding[keep]
                cur_masks = mask_pred[keep]

                cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

                final_fused_feature = torch.zeros_like(
                    single_pred_3d, device=single_pred_3d.device
                )
                counter = torch.zeros(
                    (single_pred_3d.shape[0], 1), device=single_pred_3d.device
                )
                if cur_masks.shape[0] == 0:
                    single_2d_feature = torch.zeros_like(
                        single_pred_3d, device=single_pred_3d.device
                    )

                else:

                    cur_mask_ids = cur_prob_masks.argmax(0)
                    final_keep = []
                    final_mask = []
                    for k in range(cur_classes.shape[0]):

                        mask_area = (cur_mask_ids == k).sum().item()
                        original_area = (cur_masks[k] >= 0.5).sum().item()
                        mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                        if (
                            mask_area > 0
                            and original_area > 0
                            and mask.sum().item() > 0
                        ):
                            if mask_area / original_area <= 0:
                                continue
                            final_keep.append(k)
                            final_mask.append(mask)

                    pred_embedding = cur_embedding[final_keep]
                    pred_open_embedding = cur_open_embedding[final_keep]
                    mask_pred_result = torch.stack(final_mask).to(pred_embedding.device)

                    mask_3d = mask_pred_result[:, x_label, y_label]
                    mask_3d = mask_3d >= 0.5

                    mask_3d_feature = torch.zeros_like(
                        single_pred_3d, device=single_pred_3d.device
                    )

                    for single_mask, mask_emb in zip(mask_3d, pred_embedding):

                        mask_3d_feature[single_mask] += mask_emb
                        counter[single_mask] += 1

                    counter[counter == 0] = 1e-5

                    single_2d_feature = mask_3d_feature / counter

                single_2d_feature_need_fused = single_2d_feature[
                    torch.sum(counter, dim=1) >= 1
                ]
                single_pred_3d_need_fused = single_pred_3d[
                    torch.sum(counter, dim=1) >= 1
                ]
                single_pred_3d_no_need_fused = single_pred_3d[
                    torch.sum(counter, dim=1) < 1
                ]

                fused_feature = self.criterion.fuser(
                    single_2d_feature_need_fused, single_pred_3d_need_fused
                )
                final_fused_feature[torch.sum(counter, dim=1) >= 1] = fused_feature
                final_fused_feature[
                    torch.sum(counter, dim=1) < 1
                ] = single_pred_3d_no_need_fused

                output.append(final_fused_feature)
                output_2d.append(single_2d_feature)
                output_3d.append(self.criterion.fc1(single_pred_3d))
                mask_3d_list.append(mask_3d)
                pred_open_embedding_list.append(pred_open_embedding)

            outputs.update({"fused_pred_feature": output})
            outputs.update({"2d_pred_feature": output_2d})
            outputs.update({"mask_cls_results": mask_cls_results})
            outputs.update({"pure3d_pred_feature": output_3d})
            outputs.update({"binary_pred": binary_pred})
            outputs.update({"final_mask_3d": mask_3d_list})
            outputs.update({"final_pred_open_embedding": pred_open_embedding_list})

            return None, outputs


class FeatureMerger(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureMerger, self).__init__()
        self.linear = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, X, Y):
        combined_features = torch.cat((X, Y), dim=1)
        output = self.linear(combined_features)
        return output


class UNetFPN(nn.Module):
    def __init__(self, out_dim=256, ldm_prior=512):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(2 * ldm_prior, ldm_prior, kernel_size=1),
            nn.GroupNorm(16, ldm_prior),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(2 * ldm_prior, ldm_prior, kernel_size=1),
            nn.GroupNorm(16, ldm_prior),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(2 * ldm_prior, ldm_prior, kernel_size=1),
            nn.GroupNorm(16, ldm_prior),
            nn.ReLU(),
            nn.Conv2d(ldm_prior, out_dim, kernel_size=1),
        )

        self.fc = nn.Conv2d(out_dim, 1, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, outs):

        x1 = self.layer1(
            torch.cat([F.interpolate(outs[3], scale_factor=2), outs[2]], dim=1)
        )

        x2 = self.layer2(torch.cat([F.interpolate(x1, scale_factor=2), outs[1]], dim=1))

        x3 = self.layer3(torch.cat([F.interpolate(x2, scale_factor=2), outs[0]], dim=1))

        x = self.fc(x3)

        return x
