"""Batched GLM53 routing with compensated BF16 weight decomposition."""

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.distributed.sequence_parallel import TokenShardLayout
from rtp_llm.models_py.modules.dsv4.forward_metadata import (
    metadata_cache,
    tokenwise_router_enabled,
)

_PREFILL_PROJECTION_ROWS = 256


class Glm53FP32Router(nn.Module):
    def __init__(self, weight: torch.Tensor, chunk_rows: int = 8192):
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.float32:
            raise ValueError(
                "GLM-5.3 router requires a loaded FP32 [hidden, experts] weight"
            )
        if chunk_rows <= 0:
            raise ValueError("router chunk_rows must be positive")
        self.register_buffer("weight", weight.T.contiguous())
        self.chunk_rows = chunk_rows
        self.split_parts = int(os.environ.get("GLM53_ROUTER_BF16_PARTS", "2"))
        if self.split_parts not in (0, 2, 3):
            raise ValueError("GLM53_ROUTER_BF16_PARTS must be 0, 2, or 3")
        self.prefill_split_parts = int(
            os.environ.get("GLM53_PREFILL_ROUTER_BF16_PARTS", str(self.split_parts))
        )
        if self.prefill_split_parts not in (0, 2, 3):
            raise ValueError("GLM53_PREFILL_ROUTER_BF16_PARTS must be 0, 2, or 3")
        if (self.split_parts or self.prefill_split_parts) and self.weight.shape in (
            (256, 4096),
            (288, 4096),
        ):
            logging.info(
                "[GLM53 router opt] BF16 parts decode=%d prefill=%d weight_shape=%s",
                self.split_parts,
                self.prefill_split_parts,
                tuple(self.weight.shape),
            )
            from rtp_llm.models_py.triton_kernels.glm53_router_split import (
                split_weights,
            )

            for index, component in enumerate(split_weights(self.weight)):
                self.register_buffer(f"split_weight_{index}", component)

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self.split_parts
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and hidden_states.is_contiguous()
            and metadata_cache() is None
            and 0 < hidden_states.shape[0] <= 384
            and self.weight.shape[0] in (256, 288)
            and self.weight.shape[1] == 4096
            and torch.cuda.get_device_capability(hidden_states.device)
            in ((10, 0), (10, 3))
        ):
            if (
                os.environ.get("GLM53_ROUTER_SPLIT_K", "1") == "1"
                and self.split_parts == 2
                and hidden_states.shape[0] in (48, 192, 240, 288)
                and self.weight.shape[0] == 288
            ):
                from rtp_llm.models_py.triton_kernels.glm53_router_splitk import router

                return router(
                    hidden_states,
                    (self.split_weight_0, self.split_weight_1, self.split_weight_2),
                )
            from rtp_llm.models_py.triton_kernels.glm53_router_split import router

            return router(
                hidden_states,
                (self.split_weight_0, self.split_weight_1, self.split_weight_2),
                parts=self.split_parts,
            )
        return F.linear(hidden_states.float(), self.weight)

    def forward_routes(
        self, hidden_states, weights, ids, bias, groups, selected, k, norm, scale
    ):
        """Optional GEMM epilogue path; unsupported configurations use forward."""
        if (
            os.environ.get("GLM53_ROUTER_REDUCE_TOPK", "1") != "1"
            or metadata_cache() is not None
            or self.split_parts != 2
            or os.environ.get("GLM53_ROUTER_SPLIT_K", "1") != "1"
            or torch.backends.cuda.matmul.allow_tf32
            or not hidden_states.is_cuda
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 2
            or hidden_states.shape[0] not in (48, 192, 240, 288)
            or hidden_states.shape[1] != 4096
            or not hidden_states.is_contiguous()
            or torch.cuda.get_device_capability(hidden_states.device)
            not in ((10, 0), (10, 3))
            or self.weight.shape != (288, 4096)
            or groups != 1
            or selected != 1
            or k != 8
        ):
            return False
        if (
            weights.shape != (hidden_states.shape[0], 8)
            or weights.dtype != torch.float32
            or not weights.is_contiguous()
            or ids.shape != weights.shape
            or ids.dtype not in (torch.int32, torch.int64)
            or not ids.is_contiguous()
            or bias.shape != (288,)
            or bias.dtype != torch.float32
            or any(t.device != hidden_states.device for t in (weights, ids, bias))
        ):
            return False
        from rtp_llm.models_py.triton_kernels.glm53_router_splitk import gemm
        from rtp_llm.models_py.modules.glm53_router_topk import fused_splitk_router_topk
        import triton

        m = hidden_states.shape[0]
        partial = torch.empty(
            (8, m, 288), device=hidden_states.device, dtype=torch.float32
        )
        bm = 16 if m == 48 else 32
        gemm[(triton.cdiv(m, bm), triton.cdiv(288, 64), 8)](
            hidden_states,
            self.split_weight_0,
            self.split_weight_1,
            partial,
            m,
            288,
            4096,
            bm,
            64,
            64,
            8,
            num_warps=4,
            num_stages=3,
        )
        fused_splitk_router_topk(partial, bias, weights, ids, norm, scale)
        return True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("GLM-5.3 FP32 router requires matmul.allow_tf32=False")
        if metadata_cache() is not None and not tokenwise_router_enabled():
            return self._project_prefill(hidden_states)
        # Keep Decode/MTP rows together, including the explicit FP32 fallback.
        # The optional BF16 decomposition approximates the FP32 weight; it is
        # independent of the Prefill fixed-M precision policy.
        if hidden_states.shape[0] <= self.chunk_rows:
            return self._project(hidden_states)
        return torch.cat(
            [self._project(x) for x in hidden_states.split(self.chunk_rows)],
            dim=0,
        )

    def _project_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not (
            self.prefill_split_parts
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and hidden_states.is_contiguous()
            and hidden_states.shape[0] > 0
            and hidden_states.shape[1] == 4096
            and self.weight.shape in ((256, 4096), (288, 4096))
            and torch.cuda.get_device_capability(hidden_states.device) in ((10, 0), (10, 3))
        ):
            return self._fixed_projection(hidden_states, _PREFILL_PROJECTION_ROWS)

        # Use the same BF16 components as Decode. Large M benefits from
        # DeepGEMM's BF16 GEMM and in-place FP32 residual accumulation.
        try:
            from deep_gemm import bf16_gemm_nt
        except ImportError:
            from rtp_llm.models_py.triton_kernels.glm53_router_split import router

            return router(
                hidden_states,
                (self.split_weight_0, self.split_weight_1, self.split_weight_2),
                config=(128, 64, 64, 4, 3),
                parts=self.prefill_split_parts,
            )

        logits = hidden_states.new_empty(
            (hidden_states.shape[0], self.weight.shape[0]), dtype=torch.float32
        )
        bf16_gemm_nt(hidden_states, self.split_weight_0, logits)
        bf16_gemm_nt(hidden_states, self.split_weight_1, logits, c=logits)
        if self.prefill_split_parts == 3:
            bf16_gemm_nt(hidden_states, self.split_weight_2, logits, c=logits)
        return logits

    def _fixed_projection(self, hidden_states: torch.Tensor, rows: int) -> torch.Tensor:
        # The final cuBLAS chunk otherwise changes M with the request batch.
        # Fixed 256-row GEMMs preserve the FP32 reduction at every token,
        # including shard boundaries, and avoid the less accurate large-M path.
        from rtp_llm.models_py.modules.glm53_fixed_projection_graph import (
            fixed_projection_graph,
        )

        projected = fixed_projection_graph(hidden_states, self.weight, rows)
        if projected is not None:
            return projected
        tokens = hidden_states.shape[0]
        output = hidden_states.new_empty(
            (tokens, self.weight.shape[0]), dtype=torch.float32
        )
        for begin in range(0, tokens, rows):
            count = min(rows, tokens - begin)
            chunk = hidden_states.narrow(0, begin, count)
            if count < rows:
                padded = hidden_states.new_zeros((rows, hidden_states.shape[1]))
                padded[:count].copy_(chunk)
                chunk = padded
            output[begin : begin + count].copy_(
                F.linear(chunk.float(), self.weight)[:count]
            )
        return output

    def forward_fixed_rows(self, hidden_states: torch.Tensor, rows: int):
        if rows > self.chunk_rows or metadata_cache() is None:
            return None
        # Preserve explicit fixed-M overrides independently of the native
        # Prefill policy, including when projection graph replay is disabled.
        return self._fixed_projection(hidden_states, rows)

    def forward_shard(
        self, hidden_states: torch.Tensor, layout: TokenShardLayout
    ) -> torch.Tensor:
        """Project local rows with the unsharded router's GEMM shapes.

        Only a boundary chunk needs zero rows in place of another rank's
        tokens. Keeping its original M and row offsets avoids changing the
        FP32 reduction algorithm at a top-k near tie. No remote token data is
        needed. Aligned large shards take the ordinary bounded fast path.
        """
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("GLM-5.3 FP32 router requires matmul.allow_tf32=False")
        if hidden_states.ndim != 2 or hidden_states.shape[0] != layout.local_tokens:
            raise ValueError("router input must match the local token layout")
        begin = layout.local_start
        valid = layout.local_valid_tokens
        end = begin + valid
        if (
            begin < 0
            or valid < 0
            or valid > layout.local_tokens
            or (valid and end > layout.logical_tokens)
        ):
            raise ValueError("invalid router token layout")
        if metadata_cache() is not None and not tokenwise_router_enabled():
            # Token shards contain all input features. Prefill projects local
            # rows directly and clears padding before routing.
            output = self._project_prefill(hidden_states)
            output[valid:].zero_()
            return output
        if (
            valid == layout.local_tokens
            and begin % self.chunk_rows == 0
            and (end % self.chunk_rows == 0 or end == layout.logical_tokens)
        ):
            return self(hidden_states)
        output = hidden_states.new_empty(
            (layout.local_tokens, self.weight.shape[0]), dtype=torch.float32
        )
        output[valid:].zero_()
        for chunk_begin in range(
            begin // self.chunk_rows * self.chunk_rows, end, self.chunk_rows
        ):
            chunk_end = min(chunk_begin + self.chunk_rows, layout.logical_tokens)
            first, last = max(begin, chunk_begin), min(end, chunk_end)
            if first >= last:
                continue
            local_first, chunk_first = first - begin, first - chunk_begin
            count = last - first
            local = hidden_states.narrow(0, local_first, count)
            if count == chunk_end - chunk_begin:
                chunk = local
            else:
                chunk = hidden_states.new_zeros(
                    (chunk_end - chunk_begin, hidden_states.shape[1])
                )
                chunk.narrow(0, chunk_first, count).copy_(local)
            logits = self(chunk)
            output.narrow(0, local_first, count).copy_(
                logits.narrow(0, chunk_first, count)
            )
        return output
