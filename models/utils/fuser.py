import torch
import torch.nn as nn
import torch.distributed as dist


def mask_mapper(x_list, y_list, masks, mask_embeds, pred_3ds, fuser, fc1, fc2, cfg):

    output = []
    output_2d = []
    output_3d = []
    output_2d_pre = []
    for x_label, y_label, mask, mask_embed, pred_3d in zip(
        x_list, y_list, masks, mask_embeds, pred_3ds
    ):

        mask_3d = mask[:, x_label, y_label].clone()
        mask_3d = mask_3d >= 0.5

        if len(mask_3d[torch.sum(mask_3d, dim=1) != 0]) == 0:
            mask_3d[0][0] = True

        counter = torch.zeros((pred_3d.shape[0], 1), device=pred_3d.device)
        final_fused_feature = torch.zeros_like(pred_3d, device=pred_3d.device)
        mask_3d_feature = torch.zeros_like(pred_3d, device=pred_3d.device)
        ii = 0
        for single_mask, mask_emb in zip(mask_3d, mask_embed):
            if torch.sum(single_mask) == 0:
                continue
            ii += 1
            mask_3d_feature[single_mask] += mask_emb
            counter[single_mask] += 1

        counter[counter == 0] = 1e-5
        mask_3d_feature = mask_3d_feature / counter

        single_2d_feature_need_fused = mask_3d_feature[torch.sum(counter, dim=1) >= 1]
        single_pred_3d_need_fused = pred_3d[torch.sum(counter, dim=1) >= 1]
        single_pred_3d_no_need_fused = pred_3d[torch.sum(counter, dim=1) < 1]
        fused_feature = fuser(single_2d_feature_need_fused, single_pred_3d_need_fused)
        final_fused_feature[torch.sum(counter, dim=1) >= 1] = fused_feature
        final_fused_feature[
            torch.sum(counter, dim=1) < 1
        ] = single_pred_3d_no_need_fused

        pure3d_feature = fc1(pred_3d)
        mask_3d_feature = fc2(mask_3d_feature)
        output.append(final_fused_feature)
        output_2d.append(mask_3d_feature)
        output_3d.append(pure3d_feature)
        if cfg.caption_contra_2d_pre:
            output_2d_pre.append(single_2d_feature_need_fused)

    return output, output_2d, output_3d, output_2d_pre


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class FeatureMerger(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureMerger, self).__init__()
        self.linear = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, X, Y):
        combined_features = torch.cat((X, Y), dim=1)
        output = self.linear(combined_features)
        return output
