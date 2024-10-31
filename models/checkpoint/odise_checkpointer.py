import os.path as osp
from collections import defaultdict
from typing import List
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts
from fvcore.common.checkpoint import Checkpointer

import torch
from typing import Any, Dict, List
from models.utils.file_io import PathManager


def _longest_common_prefix(names: List[str]) -> str:
    """
    ["abc.zfg", "abc.zef"] -> "abc."
    """
    names = [n.split(".") for n in names]
    m1, m2 = min(names), max(names)
    ret = []
    for a, b in zip(m1, m2):
        if a == b:
            ret.append(a)
        else:

            break
    ret = ".".join(ret) + "." if len(ret) else ""
    return ret


def group_by_prefix(names):
    grouped_names = defaultdict(list)

    for name in names:
        grouped_names[name.split(".")[0]].append(name)

    return grouped_names


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


class XMask3dCheckpointer(Checkpointer):
    def __init__(
        self, model, optimizer, save_dir="", *, save_to_disk=None, **checkpointables
    ):
        super().__init__(
            model=model, save_dir=save_dir, save_to_disk=save_to_disk, **checkpointables
        )
        self.path_manager = PathManager
        self.optimizer = optimizer

    def _load_model(self, checkpoint):

        if hasattr(self.model, "preprocess_state_dict"):
            self.logger.info("Preprocessing model state_dict")
            checkpoint["model"] = self.model.preprocess_state_dict(checkpoint["model"])
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])

            checkpoint["model"] = align_and_update_state_dicts(
                self.model.state_dict(),
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )

        incompatible = super(DetectionCheckpointer, self)._load_model(checkpoint)

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:

            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        for k in incompatible.unexpected_keys[:]:

            if "anchor_generator.cell_anchors" in k:
                incompatible.unexpected_keys.remove(k)

        removed_keys = []
        for k in incompatible.missing_keys[:]:
            if hasattr(self.model, "ignored_state_dict"):
                ignored_keys = set(self.model.ignored_state_dict().keys())
            else:
                ignored_keys = set()

            if k in ignored_keys:
                incompatible.missing_keys.remove(k)
                removed_keys.append(k)
        if len(removed_keys) > 0:
            prefix_list = [
                _longest_common_prefix(grouped_names)
                for grouped_names in group_by_prefix(removed_keys).values()
            ]
            self.logger.warn(
                "Keys with prefix are removed from state_dict:\n"
                + ",".join(prefix_list)
            )

            self.logger.warn(
                f"Removed {len(removed_keys)} ignored_state_dict keys from missing_keys"
            )

        return incompatible

    def load(self, path, checkpointables=None, eval=False):
        ret = {}

        checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda())

        ret["start_epoch"] = checkpoint["epoch"]

        checkpoints = checkpoint["state_dict"]
        ret["checkpoint"] = checkpoints
        if not eval:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        ret["best_iou"] = checkpoint["best_iou"]

        checkpoints = {"model": checkpoints}
        _strip_prefix_if_present(self.model.state_dict(), "module.")

        incompatible = super()._load_model(checkpoints)

        if incompatible is not None:
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoints:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoints.pop(key))

        return ret

    @staticmethod
    def has_checkpoint_in_dir(save_dir) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = osp.join(save_dir, "last_checkpoint")
        return osp.exists(save_file)


class LdmCheckpointer(Checkpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(
            model=model, save_dir=save_dir, save_to_disk=save_to_disk, **checkpointables
        )
        self.path_manager = PathManager

    def _load_model(self, checkpoint):

        checkpoint["model"] = checkpoint.pop("state_dict")
        return super()._load_model(checkpoint)
