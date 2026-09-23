"""Optional W4A16 FFN execution using independently loaded weights."""

import logging

import torch
from torch import nn

from rtp_llm.model_loader.w4a16_weight import w4a16_key
from rtp_llm.models_py.kernels.cuda.w4a16_sm120 import gemm, rotate, support
from rtp_llm.utils.model_weight import W


class RotatedW4A16Linear(nn.Module):
    def __init__(self, packed, scales, signs, bias):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("signs", signs)
        self.register_buffer("bias", bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = gemm(rotate(inputs, self.signs), self.packed, self.scales)
        return output if self.bias is None else output + self.bias


class W4A16DenseMLP(nn.Module):
    def __init__(self, up_proj, down_proj):
        super().__init__()
        self.up_proj = up_proj
        self.down_proj = down_proj

    @classmethod
    def create(cls, weights, up_bias, down_bias, is_gated):
        up_key = W.ffn_w13 if is_gated else W.ffn_w3
        components = ("packed", "scales", "signs")
        if is_gated and w4a16_key(up_key, "packed") not in weights:
            if all(
                w4a16_key(key, part) in weights
                for key in (W.ffn_w1, W.ffn_w3)
                for part in components
            ):
                for part in components:
                    gate = weights.pop(w4a16_key(W.ffn_w1, part))
                    up = weights.pop(w4a16_key(W.ffn_w3, part))
                    if part == "signs" and not torch.equal(gate, up):
                        raise ValueError(
                            "W4A16 w1/w3 sign vectors must match to merge into w13"
                        )
                    weights[w4a16_key(up_key, part)] = (
                        gate if part == "signs" else torch.cat((gate, up), dim=1)
                    )
        keys = (up_key, W.ffn_w2)
        if not all(
            w4a16_key(key, part) in weights for key in keys for part in components
        ):
            for key in (W.ffn_w1, W.ffn_w3, W.ffn_w13, W.ffn_w2):
                for part in components:
                    weights.pop(w4a16_key(key, part), None)
            return None
        for key in keys:
            packed = weights[w4a16_key(key, "packed")]
            n, k = packed.shape[1] * 2, packed.shape[0] * 16
            if not support(n, k):
                logging.warning("W4A16 FFN uses the default path for shape %s", (n, k))
                for name in keys:
                    for part in components:
                        weights.pop(w4a16_key(name, part))
                return None
        projections = [
            RotatedW4A16Linear(
                *(weights[w4a16_key(key, part)] for part in components), bias
            )
            for key, bias in zip(keys, (up_bias, down_bias))
        ]
        logging.info("W4A16 FFN initialized: block=128, 0<M<64")
        return cls(*projections)

    def forward(self, inputs: torch.Tensor, activation: nn.Module) -> torch.Tensor:
        return self.down_proj(activation(self.up_proj(inputs)))
