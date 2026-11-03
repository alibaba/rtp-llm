"""TokenSpeed's ragged MLA kernel behind RTP's cache planning."""

import logging
from functools import cache
from importlib.metadata import version

import torch

from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path


@cache
def _load_backend():
    setup_cutlass_import_path()
    try:
        from tokenspeed_mla.mla_prefill import tokenspeed_mla_prefill
    except ImportError as error:
        raise ImportError(
            "TokenSpeed MLA requires the pinned tokenspeed-mla runtime"
        ) from error
    return tokenspeed_mla_prefill, version("tokenspeed-mla")


class TokenspeedMlaPrefill:
    """FP8 kernel adapter; request planning belongs to RTP's MLA params."""

    operand_dtype = torch.float8_e4m3fn

    def __init__(self, num_heads, scale):
        self.heads = num_heads
        self.scale = scale
        self._run, backend_version = _load_backend()
        logging.info(
            "K3 MLA prefill backend=tokenspeed_mla version=%s Q/K/V=E4M3 output=BF16",
            backend_version,
        )

    def run(self, q, k, v, params):
        shapes = (
            (params.q_tokens, self.heads, 192),
            (params.kv_tokens, self.heads, 192),
            (params.kv_tokens, self.heads, 128),
        )
        for name, tensor, shape in zip(("q", "k", "v"), (q,k,v), shapes):
            if tensor.dtype != self.operand_dtype or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"TokenSpeed MLA prefill {name} does not match its plan: "
                    f"actual_shape={tuple(tensor.shape)} actual_dtype={tensor.dtype} "
                    f"expected_shape={shape} expected_dtype={self.operand_dtype}; "
                    f"batch={len(params.q_lens_host)} "
                    f"q_tokens={params.q_tokens} k_tokens={params.kv_tokens}"
                )
            if tensor.device != params.qo_indptr_d.device:
                raise ValueError("TokenSpeed MLA prefill tensor and metadata devices must match")
        if params.q_tokens == 0:
            return torch.empty((0, self.heads, 128), dtype=torch.bfloat16, device=q.device)
        # TokenSpeed 0.1.8 assumes a contiguous V allocation, even though the
        # up-projection produces a strided split view. vLLM applies this copy too.
        return self._run(
            query=q.contiguous(),
            key=k.contiguous(),
            value=v.contiguous(),
            seq_lens=params.kv_lens_d,
            cum_seq_lens=params.prefill_ragged_kv_len_indptr_d,
            max_seq_len=params.max_k,
            batch_size=len(params.q_lens_host),
            softmax_scale=self.scale,
            is_causal=True,
            cum_seq_lens_q=params.qo_indptr_d,
            max_seq_len_q=params.max_q,
            enable_pdl=False,
        )

    def run_partial(
        self, q, k, v, *, qo_indptr, kv_indptr, max_q, max_k, causal
    ):
        """Return output and natural-log LSE for one bounded KV segment."""
        if (q.dtype != self.operand_dtype
                or k.dtype != self.operand_dtype or v.dtype != self.operand_dtype
                or q.ndim != 3 or k.ndim != 3 or v.ndim != 3
                or q.shape[1:] != (self.heads, 192)
                or k.shape[1:] != (self.heads, 192)
                or v.shape[1:] != (self.heads, 128)
                or k.shape[0] != v.shape[0]
                or qo_indptr.ndim != 1 or kv_indptr.shape != qo_indptr.shape
                or max_q < 0 or max_k < 0):
            raise ValueError("K3 FP8 MLA partial attention operand mismatch")
        if any(x.device != q.device for x in (k, v, qo_indptr, kv_indptr)):
            raise ValueError("K3 FP8 MLA partial attention device mismatch")
        return self._run(
            query=q.contiguous(),
            key=k.contiguous(),
            value=v.contiguous(),
            seq_lens=kv_indptr.diff(),
            cum_seq_lens=kv_indptr,
            max_seq_len=max_k,
            batch_size=qo_indptr.numel() - 1,
            softmax_scale=self.scale,
            is_causal=causal,
            return_lse=True,
            cum_seq_lens_q=qo_indptr,
            max_seq_len_q=max_q,
            enable_pdl=False,
        )
