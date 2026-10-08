"""TokenSpeed's native ragged MLA kernel behind RTP's cache planning."""

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
            "K3 native MLA requires deps/requirements_kimi_k3_native.txt"
        ) from error
    return tokenspeed_mla_prefill, version("tokenspeed-mla")


class KimiK3TokenspeedPrefill:
    """Eager prefill plan; paged decode/verification own their graph metadata."""

    def __init__(self, fp8_compute: bool = False):
        self.fp8_compute = fp8_compute
        self.operand_dtype = torch.float8_e4m3fn if fp8_compute else torch.bfloat16
        self._run, backend_version = _load_backend()
        logging.info(
            "K3 MLA prefill backend=tokenspeed_mla version=%s Q/K/V=%s output=BF16",
            backend_version,
            "E4M3" if fp8_compute else "BF16",
        )

    def plan(
        self,
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        *,
        sm_scale,
        causal,
        q_data_type,
        kv_data_type,
        qo_indptr_host,
        kv_indptr_host,
    ):
        if q_data_type != self.operand_dtype or kv_data_type != self.operand_dtype:
            raise ValueError(f"K3 prefill requires {self.operand_dtype} Q, K and V")
        if not causal or num_qo_heads <= 0 or num_qo_heads != num_kv_heads:
            raise ValueError("K3 expanded MLA requires causal attention and equal Q/K heads")
        if (head_dim_qk, head_dim_vo) != (192, 128):
            raise ValueError("K3 TokenSpeed prefill requires QK/V dimensions 192/128")
        if not qo_indptr.is_cuda or qo_indptr.device != kv_indptr.device:
            raise ValueError("K3 prefill requires Q/K indptrs on the same CUDA device")
        if qo_indptr.ndim != 1 or qo_indptr.shape != kv_indptr.shape:
            raise ValueError("K3 prefill requires matching one-dimensional Q/K indptrs")
        if qo_indptr.numel() < 2:
            raise ValueError("K3 prefill requires at least one request")
        if qo_indptr.dtype not in (torch.int32,torch.int64) or kv_indptr.dtype not in (torch.int32,torch.int64):
            raise ValueError("K3 prefill indptrs require integer storage")
        if (
            qo_indptr_host.device.type != "cpu"
            or kv_indptr_host.device.type != "cpu"
            or qo_indptr_host.shape != qo_indptr.shape
            or kv_indptr_host.shape != kv_indptr.shape
            or qo_indptr_host.dtype not in (torch.int32, torch.int64)
            or kv_indptr_host.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("K3 prefill requires matching CPU indptr mirrors")
        # RTP fills these host mirrors and the device arrays together. Reading
        # the device arrays back here would synchronize all earlier GPU work.
        q_host = qo_indptr_host.tolist()
        k_host = kv_indptr_host.tolist()
        q_lens = [b-a for a,b in zip(q_host,q_host[1:])]
        k_lens = [b-a for a,b in zip(k_host,k_host[1:])]
        if q_host[0] != 0 or k_host[0] != 0 or any(q < 0 or k < q for q,k in zip(q_lens,k_lens)):
            raise ValueError("K3 prefill Q/K lengths must be nonnegative with Q <= KV")
        if max(q_host[-1],k_host[-1]) > torch.iinfo(torch.int32).max:
            raise ValueError("K3 prefill token offsets exceed int32 storage")
        self.qo_indptr = qo_indptr.to(torch.int32).contiguous()
        self.kv_indptr = kv_indptr.to(torch.int32).contiguous()
        self.kv_lens = self.kv_indptr.diff()
        self.max_q, self.max_k = max(q_lens), max(k_lens)
        self.q_tokens, self.k_tokens = q_host[-1], k_host[-1]
        self.batch = len(q_lens)
        self.heads = num_qo_heads
        self.scale = sm_scale

    def run(self, q, k, v):
        shapes = ((self.q_tokens,self.heads,192), (self.k_tokens,self.heads,192), (self.k_tokens,self.heads,128))
        for name, tensor, shape in zip(("q", "k", "v"), (q,k,v), shapes):
            if tensor.dtype != self.operand_dtype or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"K3 native prefill {name} does not match its plan: "
                    f"actual_shape={tuple(tensor.shape)} actual_dtype={tensor.dtype} "
                    f"expected_shape={shape} expected_dtype={self.operand_dtype}; "
                    f"batch={self.batch} q_tokens={self.q_tokens} k_tokens={self.k_tokens}"
                )
            if tensor.device != self.qo_indptr.device:
                raise ValueError("K3 native prefill tensor and metadata devices must match")
        if self.q_tokens == 0:
            return torch.empty((0,self.heads,128), dtype=torch.bfloat16, device=q.device)
        # TokenSpeed 0.1.8 assumes a contiguous V allocation, even though the
        # up-projection produces a strided split view. vLLM applies this copy too.
        return self._run(
            query=q.contiguous(),
            key=k.contiguous(),
            value=v.contiguous(),
            seq_lens=self.kv_lens,
            cum_seq_lens=self.kv_indptr,
            max_seq_len=self.max_k,
            batch_size=self.batch,
            softmax_scale=self.scale,
            is_causal=True,
            cum_seq_lens_q=self.qo_indptr,
            max_seq_len_q=self.max_q,
            enable_pdl=False,
        )

    def run_partial(
        self, q, k, v, *, qo_indptr, kv_indptr, max_q, max_k, causal
    ):
        """Return output and natural-log LSE for one bounded KV segment."""
        if (not self.fp8_compute or q.dtype != self.operand_dtype
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
