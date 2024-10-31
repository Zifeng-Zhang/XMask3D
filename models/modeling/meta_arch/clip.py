import logging
from collections import OrderedDict, namedtuple
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from detectron2.utils import comm
from einops import rearrange

from .helper import ensemble_logits_with_labels

logger = logging.getLogger(__name__)

EmbeddedText = namedtuple(
    "EmbedTextReturn", ["text_embed", "text_encodings", "text_mask"]
)
EmbeddedImage = namedtuple("EmbedImageReturn", ["image_embed", "image_encodings"])


def build_clip_text_embed(clip_model_name, labels, device="cuda", verbose=True):
    if isinstance(clip_model_name, str):
        clip, _, _ = open_clip.create_model_and_transforms(
            model_name=clip_model_name,
            pretrained="openai",
            device=device if torch.cuda.is_available() else "cpu",
        )
        if verbose:
            logger.info(f"Loading CLIP model {clip_model_name}")
    else:
        clip = clip_model_name
        if verbose:
            logger.info("Using provided CLIP model")
    clip_device = next(clip.parameters()).device
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(labels[0], str):
        labels = [[t] for t in labels]

    labels = tuple(tuple(t) for t in labels)

    assert isinstance(
        labels[0], (list, tuple)
    ), f"labels should be a list of list of str, but got {type(labels[0])}"

    flatten_text = [t for sublist in labels for t in sublist]

    text_embed_list = []

    local_batch_size = 256

    for i in range(0, len(flatten_text), local_batch_size):
        cur_text = flatten_text[i : i + local_batch_size]
        text_embed = clip.encode_text(open_clip.tokenize(cur_text).to(clip_device))
        text_embed_list.extend(list(text_embed))

    out_text_embed = torch.stack(text_embed_list)
    if verbose:
        logger.info(
            f"Built text_embed of shape {out_text_embed.shape} for {len(labels)} labels: {labels}"
        )

    return out_text_embed


class ClipAdapter(nn.Module):
    def __init__(self, name="ViT-B-32", normalize=True):

        open_clip.create_model_and_transforms(name, pretrained="openai")

        openai_clip, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained="openai"
        )
        super().__init__()
        self.clip = openai_clip

        self.clip_preprocess = T.Compose(
            [*preprocess.transforms[:2], preprocess.transforms[-1]]
        )
        self._freeze()
        self.name = name
        self.normalize = normalize

    def extra_repr(self) -> str:
        return f"name={self.name}, normalize={self.normalize}"

    def _freeze(self):
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return super().state_dict(destination=destination, prefix=prefix)

    @property
    def device(self):
        return next(self.parameters()).device

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return OrderedDict()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze()
        return self

    @property
    def dim_latent(self):
        return self.clip.text_projection.shape[-1]

    @property
    def image_size(self):
        if isinstance(self.clip.visual.image_size, tuple):
            return self.clip.visual.image_size
        else:
            return (self.clip.visual.image_size, self.clip.visual.image_size)

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length

    def _encode_text(self, text):
        x = self.clip.token_embedding(text)
        x = x + self.clip.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x)
        text_encodings = x

        text_embed = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection
        )

        return text_embed, text_encodings

    @torch.no_grad()
    def embed_text(self, captions):
        text = open_clip.tokenize(captions).to(next(self.parameters()).device)
        text = text[..., : self.max_text_len]
        text_mask = (text != 0).long()

        text_embed, text_encodings = self._encode_text(text)
        if self.normalize:
            return EmbeddedText(
                F.normalize(text_embed.float(), dim=-1),
                text_encodings.float(),
                text_mask,
            )
        else:
            return EmbeddedText(text_embed.float(), text_encodings.float(), text_mask)

    def _encode_image(self, image):
        if hasattr(self.clip.visual, "positional_embedding"):

            x = self.clip.visual.conv1(image)

            x = x.reshape(x.shape[0], x.shape[1], -1)

            x = x.permute(0, 2, 1)

            x = torch.cat(
                [
                    self.clip.visual.class_embedding.to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    x,
                ],
                dim=1,
            )
            x = x + self.clip.visual.positional_embedding.to(x.dtype)

            x = self.clip.visual.ln_pre(x)

            x = x.permute(1, 0, 2)

            x = self.clip.visual.transformer(x)
            x = x.permute(1, 0, 2)

            x = self.clip.visual.ln_post(x)

            batch_size, num_tokens, _ = x.shape

            if self.clip.visual.proj is not None:
                x = rearrange(x, "b n c -> (b n) c", b=batch_size, n=num_tokens)
                x = x @ self.clip.visual.proj
                x = rearrange(x, "(b n) c -> b n c", b=batch_size, n=num_tokens)

            image_embed = x[:, 0, :]
            image_encodings = x[:, 1:, :]

            width = height = int(image_encodings.shape[1] ** 0.5)

            image_encodings = rearrange(
                image_encodings, "b (h w) c -> b c h w", h=height, w=width
            )

            return image_embed, image_encodings
        else:

            image_embed = self.clip.encode_image(image)
            return image_embed, None

    @torch.no_grad()
    def embed_image(self, image):
        image_embed, image_encodings = self._encode_image(self.clip_preprocess(image))
        if self.normalize:
            return EmbeddedImage(
                F.normalize(image_embed.float(), dim=-1), image_encodings
            )
        else:
            return EmbeddedImage(image_embed.float(), image_encodings)

    @torch.no_grad()
    def build_text_embed(self, labels):
        return build_clip_text_embed(self.clip, labels)


class MaskCLIP(ClipAdapter):
    def __init__(self, name="ViT-L-14-336"):
        super().__init__(name=name, normalize=False)

    @property
    def logit_scale(self):
        logit_scale = torch.clamp(self.clip.logit_scale.exp(), max=100)
        return logit_scale

    def _mask_clip_forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor, num_mask_tokens: int
    ):
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.clip.visual.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        cls_embed = x[0:1]
        cls_embed = cls_embed.expand(num_mask_tokens, -1, -1)
        x = torch.cat([cls_embed, x], dim=0)
        x = self.clip.visual.transformer(x, attn_mask)
        x = x.permute(1, 0, 2)

        x = self.clip.visual.ln_post(x[:, :num_mask_tokens, :])

        if self.clip.visual.proj is not None:
            x = torch.einsum("nld,dc->nlc", x, self.clip.visual.proj)

        return x

    def encode_image_with_mask(self, image, mask):
        assert hasattr(self.clip.visual, "positional_embedding")
        image = self.clip_preprocess(image)
        batch_size = image.shape[0]
        assert batch_size == mask.shape[0]
        num_queries = mask.shape[1]

        mask = mask.sigmoid()

        patch_mask = F.max_pool2d(
            mask,
            kernel_size=self.clip.visual.conv1.kernel_size,
            stride=self.clip.visual.conv1.stride,
        )

        mask_token_attn_mask = patch_mask < 0.5

        mask_token_attn_mask = mask_token_attn_mask.reshape(batch_size, num_queries, -1)

        num_mask_token = num_queries
        num_image_cls_token = self.clip.visual.positional_embedding.shape[0]
        num_image_token = num_image_cls_token - 1
        num_all_token = num_mask_token + num_image_cls_token

        attn_mask = torch.zeros(
            (num_all_token, num_all_token), dtype=torch.bool, device=image.device
        )

        attn_mask[:, :num_mask_token] = True

        attn_mask = attn_mask.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        attn_mask[:, :num_mask_token, -num_image_token:] = mask_token_attn_mask
        num_heads = self.clip.visual.conv1.out_channels // 64
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn_mask = attn_mask.reshape(
            batch_size * num_heads, num_all_token, num_all_token
        )

        return self._mask_clip_forward(image, attn_mask, num_mask_token)

    def get_mask_embed(self, image, mask):

        image = F.interpolate(
            image,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

        mask = F.interpolate(
            mask, size=image.shape[-2:], mode="bilinear", align_corners=False
        )

        mask_embed = self.encode_image_with_mask(image, mask)

        return mask_embed

    def pred_logits(self, mask_embed, text_embed, labels):
        logit_per_mask = (
            torch.einsum(
                "bqc,nc->bqn",
                F.normalize(mask_embed, dim=-1),
                F.normalize(text_embed, dim=-1),
            )
            * self.logit_scale
        )

        logit_per_mask = ensemble_logits_with_labels(logit_per_mask, labels)

        return logit_per_mask

    def forward(self, image, mask):

        mask_embed = self.get_mask_embed(image, mask)
        output = {"mask_embed_clip": mask_embed}

        return output
