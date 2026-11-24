"""Qwen3 DSpARK global fields for the offline rank-local FT writer."""

from typing import Any, List

import torch

from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_v3 import QWenV3Weight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose


def dspark_offset_d2t_to_absolute(ts: List[torch.Tensor]) -> torch.Tensor:
    """Normalize Speculators' offset d2t table to absolute target token ids."""
    offsets = identity(ts).to(torch.int64)
    return offsets + torch.arange(
        offsets.numel(), dtype=offsets.dtype, device=offsets.device
    )


class Qwen3DSparkWeight(QWenV3Weight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._d2t_path = None
        self._hidden_norm_path = "model.hidden_norm.weight"

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        super()._process_meta(meta_dicts, weight_keys)
        if "hidden_norm.weight" in weight_keys:
            self._hidden_norm_path = "hidden_norm.weight"
        if "d2t" in weight_keys:
            self._d2t_path = "d2t"
        elif "model.d2t" in weight_keys:
            self._d2t_path = "model.d2t"

    def _get_weight_info(self):
        info = super()._get_weight_info()
        extras = (
            (W.dspark_fc_w, "fc.weight", transpose),
            (W.dspark_hidden_norm_gamma, self._hidden_norm_path, identity),
            (W.dspark_markov_w1, "markov_head.markov_w1.weight", identity),
            (W.dspark_markov_w2, "markov_head.markov_w2.weight", identity),
        )
        info.weights.extend(
            AtomicWeight(name, [CkptWeightInfo(path, identity)], transform)
            for name, path, transform in extras
        )
        if self._d2t_path is not None:
            transform = (
                dspark_offset_d2t_to_absolute if self._d2t_path == "d2t" else identity
            )
            info.weights.append(
                AtomicWeight(
                    W.multi_tokens_predict_d2t_map,
                    [CkptWeightInfo(self._d2t_path, identity)],
                    transform,
                    data_type=torch.int64,
                )
            )
        return info
