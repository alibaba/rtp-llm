"""One resident GEMM for Mega Attention Q and Indexer Q, sharing FP8 storage."""

import os

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
        self._postprocess_op = None
        props = torch.cuda.get_device_properties(weight.device)
        if (props.major, props.minor, props.multi_processor_count) == (10, 3, 148):
            import deep_gemm

            op = getattr(deep_gemm, "dsv41_qb_indexer", None)
            configure = getattr(deep_gemm, "dsv41_qb_indexer_config", None)
            if callable(op) and callable(configure):
                with torch.cuda.device(weight.device):
                    configure()
                self._postprocess_op = op

    def try_forward_prepared(self, data, scales, weights, frequencies, positions):
        """Decode-only; retain resident weights/SF and the original fallback.

        Returns Main Q and (FP4 payload, packed SF, scaled BF16 head weights).
        No uninitialized BF16 Indexer view escapes this interface.
        """
        rows = data.numel() // self.K
        if (
            os.environ.get("DSV41_FUSED_QB_POSTPROCESS", "1") != "1"
            or self._postprocess_op is None
            or rows not in (256, 384, 512)
            or (self.main_width, self.index_width, self.K) != (32768, 4096, 1280)
            or weights is None
            or weights.dtype != torch.bfloat16
            or weights.numel() != rows * 32
            or not weights.is_contiguous()
            or not data.is_contiguous()
            or data.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.int32
            or scales.shape != (rows, 10)
            or scales.stride() != (1, rows)
            or frequencies.dtype != torch.complex64
            or frequencies.ndim != 2
            or frequencies.shape[1] != 32
            or not frequencies.is_contiguous()
            or positions.dtype not in (torch.int32, torch.int64)
            or positions.numel() != rows
            or not positions.is_contiguous()
            or any(
                t.device != data.device
                for t in (scales, weights, frequencies, positions)
            )
        ):
            return None
        q, payload, sf, scaled = self._postprocess_op(
            data.view(rows, self.K),
            scales,
            self.weight,
            self.weight_scales,
            weights.view(rows, 32),
            frequencies,
            positions.view(rows),
        )
        shape = data.shape[:-1]
        return q.view(*shape, self.main_width), (
            payload.view(*shape, 32, 64),
            sf.view(*shape, 32),
            scaled.view(*shape, 32),
        )

    @torch.no_grad()
    def reload(self, mega, index):
        # Mega reload has already copied the permuted weights into our view;
        # Indexer checkpoint writes also target its shared view. Refresh SF only.
        index.weight_scales.copy_(index.pack_scales(self.index_scale_source))
        self.weight_scales[: self.main_width].copy_(mega.query_scale)
        self.weight_scales[self.main_width :].copy_(index.weight_scales)

    def forward_quantized(self, data, scales, *, wait_pdl=False):
        import deep_gemm

        from ._v41_qb_kv import try_project

        result = try_project(self, data, scales, wait_pdl=wait_pdl)
        if result is not None:
            return result

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
