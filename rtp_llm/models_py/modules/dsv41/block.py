"""Single-pass mHC block with explicit state passed across sublayers."""

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.math import hc_mixes, hc_post, hc_pre, rms_norm


class V41Block(nn.Module):
    def __init__(
        self, attention: nn.Module, moe: nn.Module, weights: dict[str, torch.Tensor]
    ):
        super().__init__()
        self.attention = attention
        self.moe = moe
        self.weights = nn.ParameterDict(
            {
                name.replace(".", "_"): nn.Parameter(value, requires_grad=False)
                for name, value in weights.items()
            }
        )

    def _mixes(self, hidden: torch.Tensor, sublayer: str):
        return hc_mixes(
            hidden,
            self.weights[f"hc_{sublayer}_fn"],
            self.weights[f"hc_{sublayer}_scale"],
            self.weights[f"hc_{sublayer}_base"],
        )

    def forward(
        self, hidden: torch.Tensor, pre_mix: torch.Tensor, context, image_mask=None
    ):
        compact = getattr(context, "model_row_indices", None) is not None
        if compact:
            hidden = context.pack_model_rows(hidden)
            pre_mix = context.pack_model_rows(pre_mix)
            if image_mask is not None:
                image_mask = context.pack_model_rows(image_mask)
            if hidden.shape[0] == 0:
                empty = hidden.new_empty((0, hidden.shape[-1]))
                self.attention(context.unpack_model_rows(empty), context)
                # Empty owners still participate in every EP layer collective.
                self.moe(empty, image_mask)
                return (
                    context.unpack_model_rows(hidden),
                    context.unpack_model_rows(pre_mix),
                )
        attn_pre, attn_post, attn_comb = self._mixes(hidden, "attn")
        attn_input = rms_norm(hc_pre(hidden, pre_mix), self.weights["attn_norm_weight"])
        if compact:
            attn_output = context.pack_model_rows(
                self.attention(context.unpack_model_rows(attn_input), context)
            )
        else:
            attn_output = self.attention(attn_input, context)
        hidden = hc_post(attn_output, hidden, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = self._mixes(hidden, "ffn")
        ffn_input = rms_norm(hc_pre(hidden, attn_pre), self.weights["ffn_norm_weight"])
        hidden = hc_post(self.moe(ffn_input, image_mask), hidden, ffn_post, ffn_comb)
        if compact:
            return (
                context.unpack_model_rows(hidden),
                context.unpack_model_rows(ffn_pre),
            )
        return hidden, ffn_pre
