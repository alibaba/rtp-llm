"""One resident GEMM for Mega Attention Q and Indexer Q, sharing FP8 storage."""

import torch

from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear
from rtp_llm.utils.model_weight import W


class MergedQB(torch.nn.Module):
    @classmethod
    def create(cls, attn, mega, layer_weights):
        # Preserve TP/fallback layouts. Only the qualified Flash TP1 geometry
        # combines the *permuted* Mega Q with the ordinary Indexer Q.
        if not attn.is_index_source or layer_weights is None:
            return None
        main, index = attn.wq_b, attn.index_wq
        if not (
            type(main) is type(index) is V41MXFP8Linear
            and attn.n_heads == 64
            and mega.query_weight.shape == (32768, 1280)
            and index.weight.shape == (4096, 1280)
            and mega.query_weight.dtype == index.weight.dtype == torch.float8_e4m3fn
            and mega.query_scale.dtype == index.weight_scales.dtype == torch.int32
            and mega.query_weight.device == index.weight.device
        ):
            return None
        result = cls(mega, index, layer_weights[W.v4_indexer_wq_b_s])
        # Reload writes into this view, not a redundant checkpoint allocation.
        layer_weights[W.v4_indexer_wq_b_w] = index.weight
        return result

    def __init__(self, mega, index, index_scale_source):
        super().__init__()
        self.main_width = mega.query_weight.shape[0]
        self.index_width = index.N
        self.K = index.K
        self.index_scale_source = index_scale_source
        weight = torch.cat(
            (mega.query_weight.view(torch.uint8), index.weight.view(torch.uint8))
        ).view(mega.query_weight.dtype)
        scales = torch.cat((mega.query_scale, index.weight_scales)).T.contiguous().T
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("weight_scales", scales, persistent=False)
        mega.query_weight, index.weight = weight.split(
            (self.main_width, self.index_width), dim=0
        )
        # Original, unpermuted Main Q stays intact for non-Mega execution.
        # Separate linears retain their own TMA-compatible scale strides.

    @torch.no_grad()
    def reload(self, mega, index):
        # Mega reload has already copied the permuted weights into our view;
        # Indexer checkpoint writes also target its shared view. Refresh SF only.
        index.weight_scales.copy_(index.pack_scales(self.index_scale_source))
        self.weight_scales[: self.main_width].copy_(mega.query_scale)
        self.weight_scales[self.main_width :].copy_(index.weight_scales)

    def forward_quantized(self, data, scales):
        import deep_gemm

        width = self.main_width + self.index_width
        output = torch.empty(
            (*data.shape[:-1], width), dtype=torch.bfloat16, device=data.device
        )
        deep_gemm.fp8_fp4_gemm_nt(
            (data.reshape(-1, self.K), scales),
            (self.weight, self.weight_scales),
            output.view(-1, width),
            recipe=(1, 1, 32),
        )
        return output.split((self.main_width, self.index_width), dim=-1)
