"""Resident Main/Indexer Q-B fusion, after Mega Attention's Q permutation."""

import logging
from copy import copy

import torch

from rtp_llm.model_loader.weight_memory_saver import feature_weights_region
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear


class MergedQB:
    @classmethod
    def create(cls, attn, layer_weights=None):
        # Only the verified V4.1 Flash TP1 geometry uses this decode path.
        if not attn.is_index_source:
            return None
        main, index = attn.wq_b, attn.index_wq
        if not (
            type(main) is type(index) is V41MXFP8Linear
            and attn.n_heads == 64
            and main.weight.shape == (32768, 1280)
            and index.weight.shape == (4096, 1280)
            and main.weight.dtype == index.weight.dtype == torch.float8_e4m3fn
            and main.weight_scales.dtype == index.weight_scales.dtype == torch.int32
            and main.weight.device == index.weight.device
        ):
            return None
        result = cls(main, index)
        if layer_weights is not None:
            from rtp_llm.utils.model_weight import W

            # Checkpoint reload must target the shared Indexer slice, too.
            layer_weights[W.v4_indexer_wq_b_w] = index.weight
        logging.info("[dsv41] merged Q-B enabled: layer=%s", attn.layer_id)
        return result

    def __init__(self, main, index):
        self.main_width = main.N
        self.linear = copy(main)
        self.linear.N = main.N + index.N
        # Main Q is already permuted; its original checkpoint stays separate.
        with feature_weights_region():
            self.linear.weight = torch.cat(
                (main.weight.view(torch.uint8), index.weight.view(torch.uint8))
            ).view(main.weight.dtype)
            # Preserve the packed scales' column-major layout for TMA.
            self.linear.weight_scales = (
                torch.cat((main.weight_scales, index.weight_scales)).T.contiguous().T
            )
        main.weight, index.weight = self.linear.weight.split(
            (self.main_width, index.N), dim=0
        )
        # Indexer needs no permutation. Its checkpoint destination can be a view.
        index._sleep_raw_weight_source = index.weight

    def reload(self, main, index):
        # Weight views are already updated in place by checkpoint/Mega reload.
        # Separate GEMMs need their own compact scale strides, so only SF copies.
        self.linear.weight_scales[: self.main_width].copy_(main.weight_scales)
        self.linear.weight_scales[self.main_width :].copy_(index.weight_scales)

    def forward_quantized(self, *pair):
        output = self.linear.forward_quantized(*pair)
        # Both outputs are views: consumers must honor their token stride.
        return output.split((self.main_width, self.linear.N - self.main_width), dim=-1)
