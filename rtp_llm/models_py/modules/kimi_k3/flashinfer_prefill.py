"""Unit-scale FP8 K3 prefill through FlashInfer's ragged TRTLLM kernel."""

import logging
import math
from functools import cache
from importlib.metadata import PackageNotFoundError, version

import torch
import triton
import triton.language as tl

# All layers on a device execute serially and reuse this scratch. Keep earlier
# generations alive because an existing CUDA graph can retain their addresses.
# Geometric growth bounds retained scratch to less than twice the latest size.
_WORKSPACES = {}
_WORKSPACE_BASE_BYTES = 128 * 1024**2
_WORKSPACE_ALIGNMENT = 256


@triton.jit(do_not_specialize=["ROWS", "BATCH"])
def _normalize_prefill_output(
    output,
    lse,
    qo_indptr,
    kv_lengths,
    ROWS,
    BATCH,
    HEADS: tl.constexpr,
    HAS_LSE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    live = row < ROWS
    token = row // HEADS
    # upper_bound on Q offsets skips empty Q sequences without host reads.
    lo = tl.full((BLOCK_ROWS,), 0, tl.int32)
    hi = tl.where(live, BATCH, 0)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        searching = lo < hi
        mid = (lo + hi) // 2
        end = tl.load(qo_indptr + mid + 1, searching, other=0)
        before = token < end
        hi = tl.where(searching & before, mid, hi)
        lo = tl.where(searching & ~before, mid + 1, lo)
    empty = tl.load(kv_lengths + lo, live, other=0) == 0
    if HAS_LSE:
        # Native zero-KV sequences leave output/statistics unspecified. Do not
        # read that storage; publish RTP's identity and natural-log contract.
        value = tl.load(lse + row, live & ~empty, other=0.0)
        tl.store(
            lse + row, tl.where(empty, -float("inf"), value * 0.6931471805599453), live
        )
    if tl.sum((live & empty).to(tl.int32), 0) > 0:
        column = tl.arange(0, 128)
        tl.store(
            output + row.to(tl.int64)[:, None] * 128 + column[None, :],
            0.0,
            live[:, None] & empty[:, None],
        )


@cache
def _load_backend():
    try:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek
    except ImportError as error:
        raise ImportError(
            "K3 FP8 prefill requires the pinned FlashInfer ragged TRTLLM API"
        ) from error
    try:
        backend_version = version("flashinfer-python")
    except PackageNotFoundError:
        backend_version = "unknown"
    return trtllm_ragged_attention_deepseek, backend_version


def _get_workspace(device, query_tokens, heads):
    # The native launcher reserves 8 MiB for counters and Q*H float2 softmax
    # statistics. The base leaves the remaining 120 MiB for its scratch.
    required = _WORKSPACE_BASE_BYTES + query_tokens * heads * 8
    required = (
        (required + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    with torch.cuda.device(device):
        key = torch.cuda.current_device()
        generations = _WORKSPACES.setdefault(key, [])
        if generations and generations[-1].numel() >= required:
            return generations[-1]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3 FP8 prefill cannot allocate or grow workspace during CUDA "
                "graph capture; warm up the largest query shape eagerly first"
            )
        size = max(required, generations[-1].numel() * 2 if generations else 0)
        workspace = torch.empty(size, dtype=torch.uint8, device=device)
        generations.append(workspace)
        return workspace


class FlashinferMlaPrefill:
    """Preserve RTP's ragged planning and natural-log attention-state contract."""

    operand_dtype = torch.float8_e4m3fn

    def __init__(self, num_heads, scale):
        if num_heads <= 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("K3 FP8 prefill requires positive heads and scale")
        self.heads = num_heads
        self.scale = float(scale)
        self._run, backend_version = _load_backend()
        logging.info(
            "K3 MLA prefill backend=flashinfer_trtllm version=%s "
            "Q/K/V=E4M3 output=BF16",
            backend_version,
        )

    def _validate(self, q, k, v, qo_indptr, kv_indptr, max_q, max_k):
        if (
            q.ndim != 3
            or k.ndim != 3
            or v.ndim != 3
            or q.shape[1:] != (self.heads, 192)
            or k.shape[1:] != (self.heads, 192)
            or v.shape[1:] != (self.heads, 128)
            or k.shape[0] != v.shape[0]
            or any(x.dtype != self.operand_dtype for x in (q, k, v))
        ):
            raise ValueError("K3 FP8 prefill requires E4M3 Q/K192 and V128")
        if (
            qo_indptr.ndim != 1
            or kv_indptr.shape != qo_indptr.shape
            or qo_indptr.numel() < 2
            or qo_indptr.dtype != torch.int32
            or kv_indptr.dtype != torch.int32
            or max_q < 0
            or max_k < 0
            or (q.shape[0] > 0 and max_q == 0)
            or (k.shape[0] > 0 and max_k == 0)
        ):
            raise ValueError("K3 FP8 prefill ragged metadata does not match its plan")
        if not q.is_cuda or any(
            x.device != q.device for x in (k, v, qo_indptr, kv_indptr)
        ):
            raise ValueError("K3 FP8 prefill operands and metadata must share one GPU")

    def _execute(
        self,
        q,
        k,
        v,
        *,
        qo_indptr,
        kv_indptr,
        seq_lens,
        max_q,
        max_k,
        causal,
        return_lse,
    ):
        self._validate(q, k, v, qo_indptr, kv_indptr, max_q, max_k)
        if (
            seq_lens.shape != (qo_indptr.numel() - 1,)
            or seq_lens.dtype != torch.int32
            or seq_lens.device != q.device
        ):
            raise ValueError("K3 FP8 prefill KV lengths do not match its ragged batch")
        if q.shape[0] == 0 or k.shape[0] == 0:
            output = torch.zeros(
                (q.shape[0], self.heads, 128), dtype=torch.bfloat16, device=q.device
            )
            if return_lse:
                lse = torch.full(
                    q.shape[:2], -math.inf, dtype=torch.float32, device=q.device
                )
                return output, lse
            return output
        workspace = _get_workspace(q.device, q.shape[0], self.heads)
        try:
            result = self._run(
                query=q.contiguous(),
                key=k.contiguous(),
                value=v.contiguous(),
                workspace_buffer=workspace,
                seq_lens=seq_lens,
                max_q_len=int(max_q),
                max_kv_len=int(max_k),
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=qo_indptr.numel() - 1,
                window_left=-1,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_indptr,
                enable_pdl=False,
                is_causal=causal,
                return_lse=return_lse,
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                "FlashInfer TRTLLM K3 FP8 prefill failed for "
                f"Q={tuple(q.shape)}, K={tuple(k.shape)}, V={tuple(v.shape)}, "
                f"max_q={max_q}, max_k={max_k}, causal={causal}: {error}"
            ) from error
        output, lse = result if return_lse else (result, None)
        # Also normalize mixed ragged zero-KV requests: the native kernel does
        # not initialize those O rows. Combine it with base-2 -> natural LSE
        # conversion without device-to-host metadata synchronization.
        _normalize_prefill_output[(triton.cdiv(q.shape[0] * self.heads, 128),)](
            output,
            lse if lse is not None else output,
            qo_indptr,
            seq_lens,
            q.shape[0] * self.heads,
            qo_indptr.numel() - 1,
            HEADS=self.heads,
            HAS_LSE=return_lse,
            BLOCK_ROWS=128,
            num_warps=4,
        )
        return (output, lse) if return_lse else output

    def run(self, q, k, v, params):
        if q.shape[0] != params.q_tokens or k.shape[0] != params.kv_tokens:
            raise ValueError("K3 FP8 prefill token counts do not match its plan")
        if params.qo_indptr_d.numel() != len(params.q_lens_host) + 1:
            raise ValueError("K3 FP8 prefill batch does not match its plan")
        return self._execute(
            q,
            k,
            v,
            qo_indptr=params.qo_indptr_d,
            kv_indptr=params.prefill_ragged_kv_len_indptr_d,
            seq_lens=params.kv_lens_d,
            max_q=params.max_q,
            max_k=params.max_k,
            causal=True,
            return_lse=False,
        )

    def run_partial(self, q, k, v, *, qo_indptr, kv_indptr, max_q, max_k, causal):
        """Return BF16 output and natural-log FP32 LSE for a KV segment."""
        return self._execute(
            q,
            k,
            v,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            seq_lens=kv_indptr.diff(),
            max_q=max_q,
            max_k=max_k,
            causal=causal,
            return_lse=True,
        )
