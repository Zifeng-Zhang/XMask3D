import logging
import numpy as np
import operator
from collections import OrderedDict
from typing import Any, Mapping
import diffdist.functional as diff_dist
import torch
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.memory import retry_if_cuda_oom
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MLP,
    MultiScaleMaskedTransformerDecoder,
)
from torch import nn
from torch.nn import functional as F


from .clip import ClipAdapter, MaskCLIP, build_clip_text_embed
from .helper import ensemble_logits_with_labels

logger = logging.getLogger(__name__)


def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


@torch.no_grad()
def _concat_all_gather(tensor):

    if comm.get_world_size() == 1:
        return tensor
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_world_batch_sizes(batch_size: int, device):
    batch_size = torch.as_tensor([batch_size], dtype=torch.long, device=device)
    global_batch_sizes = _concat_all_gather(batch_size)
    return global_batch_sizes


@torch.no_grad()
def concat_all_gather(tensor):

    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones(
            (max_batch_size, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for _ in range(comm.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, padded_tensor, async_op=False)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


def dist_collect(tensor):

    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones(
            (max_batch_size, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for _ in range(comm.get_world_size())
    ]
    tensors_gather = diff_dist.all_gather(tensors_gather, padded_tensor)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


class ODISE(MaskFormer):
    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return destination

    def _open_state_dict(self):
        return {
            "sem_seg_head.num_classes": self.sem_seg_head.num_classes,
            "metadata": self.metadata,
            "test_topk_per_image": self.test_topk_per_image,
            "semantic_on": self.semantic_on,
            "panoptic_on": self.panoptic_on,
            "instance_on": self.instance_on,
        }

    def _save_open_state_dict(self, destination, prefix):
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "open_state_dict"):
                module.open_state_dict(destination, prefix + name + ".")
        return destination

    def load_open_state_dict(self, state_dict: Mapping[str, Any]):
        for k, v in state_dict.items():

            if len(k.rsplit(".", 1)) == 2:
                prefix, suffix = k.rsplit(".", 1)
                operator.attrgetter(prefix)(self).__setattr__(suffix, v)
            else:
                self.__setattr__(k, v)
            assert operator.attrgetter(k)(self) == v, f"{k} is not loaded correctly"


class CategoryODISE(ODISE):
    def __init__(
        self,
        *,
        category_head=None,
        clip_head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.category_head = category_head
        self.clip_head = clip_head

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

    def forward(self, batched_inputs):

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        denormalized_images = ImageList.from_tensors(
            [x["image"].to(self.device) / 255.0 for x in batched_inputs]
        )

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        outputs["images"] = denormalized_images.tensor

        if self.training:

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            if self.category_head is not None:
                category_head_outputs = self.category_head(outputs, targets)
                outputs.update(category_head_outputs)

                outputs["pred_logits"] = self.cal_pred_logits(outputs)
                if "aux_outputs" in outputs:
                    for aux_outputs in outputs["aux_outputs"]:
                        aux_outputs.update(category_head_outputs)

                        aux_outputs["pred_logits"] = self.cal_pred_logits(aux_outputs)

            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:

                    losses.pop(k)

            return losses
        else:

            outputs.update(self.category_head(outputs))

            outputs["pred_logits"] = self.cal_pred_logits(outputs)

            mask_pred_results = outputs["pred_masks"]
            mask_cls_results = outputs["pred_logits"]

            if self.clip_head is not None:
                if self.clip_head.with_bg:

                    outputs["pred_open_logits"] = outputs["pred_logits"]
                    outputs.update(self.clip_head(outputs))
                    mask_cls_results = outputs["pred_open_logits"]
                else:

                    outputs["pred_open_logits"] = outputs["pred_logits"][..., :-1]
                    outputs.update(self.clip_head(outputs))

                    open_logits = outputs["pred_open_logits"]

                    binary_probs = torch.zeros(
                        (mask_cls_results.shape[0], mask_cls_results.shape[1], 2),
                        device=mask_cls_results.device,
                        dtype=mask_cls_results.dtype,
                    )
                    binary_probs[..., -1] = F.softmax(mask_cls_results, dim=-1)[..., -1]
                    binary_probs[..., 0] = 1 - binary_probs[..., -1]

                    masks_class_probs = F.softmax(open_logits, dim=-1)

                    mask_cls_results = torch.cat(
                        [
                            masks_class_probs * binary_probs[..., 0:1],
                            binary_probs[..., 1:2],
                        ],
                        dim=-1,
                    )

                    mask_cls_results = torch.log(mask_cls_results + 1e-8)

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(
                            r, image_size, height, width
                        )
                    processed_results[-1]["sem_seg"] = r

                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["instances"] = instance_r

            return processed_results


class ODISEMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    def __init__(
        self,
        *,
        class_embed=None,
        mask_embed=None,
        post_mask_embed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.mask_classification

        if class_embed is not None:
            self.class_embed = class_embed
        if mask_embed is not None:
            self.mask_embed = mask_embed
        if post_mask_embed is not None:
            assert mask_embed is None
        self.post_mask_embed = post_mask_embed

    def forward(self, x, mask_features, mask=None, *, inputs_dict=None):

        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(
                self.input_proj[i](x[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_extra_results = []

        (
            outputs_class,
            outputs_mask,
            attn_mask,
            extra_results,
        ) = self.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0],
            inputs_dict=inputs_dict,
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_extra_results.append(extra_results)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            output = self.transformer_ffn_layers[i](output)

            (
                outputs_class,
                outputs_mask,
                attn_mask,
                extra_results,
            ) = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                inputs_dict=inputs_dict,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_extra_results.append(extra_results)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
            ),
        }

        for k in predictions_extra_results[-1].keys():
            out[k] = predictions_extra_results[-1][k]
            for i in range(len(predictions_extra_results) - 1):
                out["aux_outputs"][i][k] = predictions_extra_results[i][k]

        return out

    def forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size, *, inputs_dict=None
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)

        extra_results = dict()

        mask_embed_results = self.mask_embed(decoder_output)
        if isinstance(mask_embed_results, dict):
            mask_embed = mask_embed_results.pop("mask_embed")
            extra_results.update(mask_embed_results)

        else:
            mask_embed = mask_embed_results

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if self.post_mask_embed is not None:
            post_mask_embed_results = self.post_mask_embed(
                decoder_output, mask_embed, mask_features, outputs_class, outputs_mask
            )

            if "outputs_mask" in post_mask_embed_results:
                outputs_mask = post_mask_embed_results.pop("outputs_mask")

            extra_results.update(post_mask_embed_results)

        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )

        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, extra_results


class PseudoClassEmbed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):

        fg_logits = torch.ones(
            (*x.shape[:-1], self.num_classes), dtype=x.dtype, device=x.device
        )
        bg_logits = torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
        logits = torch.cat([fg_logits, bg_logits], dim=-1)
        return logits


class MaskPooling(nn.Module):
    def __init__(
        self,
        hard_pooling=True,
        mask_threshold=0.5,
    ):
        super().__init__()

        self.hard_pooling = hard_pooling
        self.mask_threshold = mask_threshold

    def extra_repr(self) -> str:
        return (
            f"hard_pooling={self.hard_pooling}\n"
            f"mask_threshold={self.mask_threshold}\n"
        )

    def forward(self, x, mask):

        assert x.shape[-2:] == mask.shape[-2:]

        mask = mask.detach()

        mask = mask.sigmoid()

        if self.hard_pooling:
            mask = (mask > self.mask_threshold).to(mask.dtype)

        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )

        output = {"mask_pooled_features": mask_pooled_x}

        return output


class PooledMaskEmbed(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mask_dim,
        projection_dim,
        temperature=0.07,
    ):
        super().__init__()
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )
        self.mask_embed = nn.Sequential(
            nn.LayerNorm(mask_dim), MLP(mask_dim, hidden_dim, projection_dim, 3)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        self.mask_pooling = MaskPooling()

    def forward(
        self, decoder_output, input_mask_embed, mask_features, pred_logits, pred_masks
    ):

        mask_pooled_x = self.mask_pooling(mask_features, pred_masks)

        mask_pooled_results = self.mask_pooling(mask_features, pred_masks)
        mask_pooled_x = mask_pooled_results["mask_pooled_features"]
        outputs_mask = mask_pooled_results.get("outputs_mask", None)

        mask_pooled_x = self.pool_proj(mask_pooled_x)

        mask_pooled_x += decoder_output

        mask_embed = self.mask_embed(mask_pooled_x)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)

        output = {
            "mask_embed": mask_embed,
            "mask_pooled_features": mask_pooled_x,
            "logit_scale": logit_scale,
        }

        if outputs_mask is not None:
            output["outputs_mask"] = outputs_mask

        return output


class CategoryEmbed(nn.Module):
    def __init__(
        self,
        labels,
        test_labels,
        projection_dim,
        clip_model_name="ViT-L-14",
        prompt=None,
    ):
        super().__init__()
        self.labels = labels

        if isinstance(clip_model_name, str):
            self.clip_model_name = clip_model_name
            self.clip = ClipAdapter(name=self.clip_model_name, normalize=False)
        else:
            self.clip_model_name = "<FROM OTHER PLACE> "
            self.clip = clip_model_name

        if projection_dim < 0:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(self.clip.dim_latent, projection_dim)
        self.register_buffer(
            "text_embed", self.build_text_embed(labels, verbose=True), False
        )
        self.null_embed = nn.Parameter(self.build_text_embed(""))

        self.prompt = prompt

        self.test_labels = test_labels
        self._test_text_embed_dict = dict()

    def extra_repr(self) -> str:
        return f"clip_model_name={self.clip_model_name},\n"

    @property
    def device(self):
        return self.clip.device

    def _open_state_dict(self):
        return {"test_labels": self.test_labels}

    def _save_open_state_dict(self, destination, prefix):
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "open_state_dict"):
                module.open_state_dict(destination, prefix + name + ".")
        return destination

    @torch.no_grad()
    def build_text_embed(self, labels, verbose=False):
        return build_clip_text_embed(
            clip_model_name=self.clip.clip,
            labels=labels,
            verbose=verbose,
        )

    def get_and_cache_test_text_embed(self, labels):
        labels = to_tuple(labels)
        if labels not in self._test_text_embed_dict:
            text_embed = self.build_text_embed(labels, verbose=True)
            self._test_text_embed_dict[labels] = text_embed.cpu()
        else:
            text_embed = self._test_text_embed_dict[labels].to(self.device)
        return text_embed

    def forward(self, outputs, targets=None):
        if self.training:

            text_embed = self.text_proj(self.text_embed)
            null_embed = self.text_proj(self.null_embed)

            return {
                "text_embed": text_embed,
                "null_embed": null_embed,
                "labels": self.labels,
            }

        else:

            labels = self.test_labels
            text_embed = self.get_and_cache_test_text_embed(labels)

            text_embed = self.text_proj(text_embed)

            null_embed = self.text_proj(self.null_embed)

            return {
                "text_embed": text_embed,
                "null_embed": null_embed,
                "labels": labels,
            }
