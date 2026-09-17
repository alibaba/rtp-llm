"""Optional CUDA GDN prefill adapter with sparse, V-first SSM checkpoints.

FlashInfer consumes multiplicative forget gates; RTP consumes their logarithm.
Q/K normalization is explicit because the SM100 wrapper does not apply its
public use_qk_l2norm_in_kernel argument. Serving selects this adapter explicitly.
"""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from rtp_llm.models_py.triton_kernels.fla.exact_qk_norm import (
    fused_l2norm_qk_exact,
    supports_exact_qk_norm,
)
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


@lru_cache(maxsize=1)
def _flashinfer_blackwell_kernel():
    # Bazel does not execute wheel .pth files. Resolve the wheel's own DSL
    # package directory, never an unrelated system installation.
    if importlib.util.find_spec("cutlass") is None:
        spec = importlib.util.find_spec("nvidia_cutlass_dsl")
        if spec is not None:
            for package in spec.submodule_search_locations or ():
                for subdir in ("dsl_packages", "python_packages"):
                    directory = Path(package) / subdir
                    if (directory / "cutlass").is_dir():
                        sys.path.insert(0, str(directory))
                        break
    from flashinfer.gdn_kernels.blackwell.gdn_prefill import (
        chunk_gated_delta_rule_sm100,
    )

    return chunk_gated_delta_rule_sm100


def flashinfer_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    checkpoint_interval: int = 2048,
):
    """Return (output[1,T,H,V], final[N,H,V,K], checkpoints, checkpoint_starts).

    Checkpoint capacity is a host-known upper bound T//interval; the valid
    prefix is described by checkpoint_starts. This avoids a device-to-host
    synchronization for ragged sequence lengths.
    """
    kernel = _flashinfer_blackwell_kernel()

    if q.ndim != 4 or q.shape[0] != 1 or q.shape != k.shape:
        raise ValueError("Expected packed Q/K [1,T,H,K] with matching shapes")
    if q.shape[-1] != 128 or v.shape[-1] != 128 or v.shape[:2] != q.shape[:2]:
        raise ValueError("FlashInfer Blackwell requires K=V=128")
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] != 10:
        raise ValueError("This adapter requires Blackwell CUDA")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("The validated prefill contract uses BF16 Q/K/V")
    if checkpoint_interval < 128 or checkpoint_interval % 64:
        raise ValueError(
            "Checkpoint interval must be a multiple of 64 and at least 128 tokens"
        )
    if v.shape[2] % q.shape[2]:
        raise ValueError("V heads must be divisible by Q/K heads")
    if g.shape != beta.shape or g.shape != v.shape[:3]:
        raise ValueError("Expected gate/beta shape [1,T,Hv]")
    if initial_state is not None and initial_state.shape != (
        cu_seqlens.numel() - 1,
        v.shape[2],
        128,
        128,
    ):
        raise ValueError("Expected V-first initial state [N,Hv,V,K]")
    tensors = (k, v, g, beta, cu_seqlens)
    if any(t.device != q.device for t in tensors) or (
        initial_state is not None and initial_state.device != q.device
    ):
        raise ValueError("All prefill tensors must share one CUDA device")
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 2
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected nonempty packed sequence offsets")
    if q.shape[1] == 0 or q.shape[2] == 0:
        raise ValueError("Expected positive token and head counts")
    if supports_exact_qk_norm(q, k):
        qn, kn = fused_l2norm_qk_exact(q, k)
    else:
        qn, kn = l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous())
    cu = cu_seqlens.to(dtype=torch.int64).contiguous()
    counts = (cu[1:] - cu[:-1]) // checkpoint_interval
    starts = torch.cat(
        (torch.zeros(1, device=cu.device, dtype=torch.int64), counts.cumsum(0))
    )
    checkpoints = torch.empty(
        (q.shape[1] // checkpoint_interval, v.shape[2], 128, 128),
        device=q.device,
        dtype=torch.float32,
    )
    output = torch.empty_like(v[0], memory_format=torch.contiguous_format)
    state = torch.empty(
        (cu.numel() - 1, v.shape[2], 128, 128), device=q.device, dtype=torch.float32
    )
    enabled = bool(checkpoints.shape[0])
    kernel(
        qn[0],
        kn[0],
        v[0].contiguous(),
        g[0].float().exp(),
        beta[0].float().contiguous(),
        output,
        cu.to(torch.int32),
        None if initial_state is None else initial_state.float().contiguous(),
        state,
        128**-0.5 if scale is None else scale,
        checkpoint_every_n_tokens=checkpoint_interval if enabled else 0,
        cu_checkpoints=starts.to(torch.int32) if enabled else None,
        output_checkpoints=checkpoints if enabled else None,
    )
    return output.unsqueeze(0), state, checkpoints, starts


# Sparse checkpoints avoid materializing one H*V*K state for every 64 tokens.
import triton
import triton.language as tl


@triton.jit
def _store_flashinfer_checkpoints(
    CP,
    STARTS,
    FINAL,
    PREFIX,
    CU,
    MAP,
    CACHE,
    MAP_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    INTERVAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    block = tl.program_id(1)
    x = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(CU + seq + 1) - tl.load(CU + seq)
    prefix = tl.load(PREFIX + seq)
    if block * INTERVAL < length:
        end = tl.minimum((block + 1) * INTERVAL, length)
        dest = (prefix + end - 1) // INTERVAL
        index = tl.load(MAP + seq * MAP_STRIDE + dest).to(tl.int64)
        # The final state owns a shared destination for a non-aligned prefix.
        final_dest = (prefix + length - 1) // INTERVAL
        if index > 0 and (end == length or dest != final_dest):
            if end == length:
                value = tl.load(
                    FINAL + seq.to(tl.int64) * STATE_SIZE + x, x < STATE_SIZE, 0
                )
            else:
                source = tl.load(STARTS + seq).to(tl.int64) + block
                value = tl.load(CP + source * STATE_SIZE + x, x < STATE_SIZE, 0)
            tl.store(CACHE + index * CACHE_STRIDE + x, value, x < STATE_SIZE)


def store_flashinfer_ssm_state(
    checkpoints,
    checkpoint_starts,
    final_state,
    prefix_lengths,
    cu_seqlens,
    block_map,
    ssm_states,
    seq_size_per_block,
    total_tokens,
):
    """Match RTP's block-map writeback, including partial final blocks and padding."""
    if not final_state.is_contiguous() or not checkpoints.is_contiguous():
        raise ValueError("Expected contiguous V-first state buffers")
    state_size = final_state[0].numel()
    if (
        ssm_states.shape[1:] != final_state.shape[1:]
        or ssm_states.stride(-2) != final_state.shape[-1]
        or ssm_states.stride(-1) != 1
        or ssm_states.stride(1) != final_state.shape[2] * final_state.shape[3]
    ):
        raise ValueError("Expected cache with contiguous per-block V-first states")
    _store_flashinfer_checkpoints[
        (
            final_state.shape[0],
            triton.cdiv(total_tokens, seq_size_per_block),
            triton.cdiv(state_size, 256),
        )
    ](
        checkpoints,
        checkpoint_starts,
        final_state,
        prefix_lengths,
        cu_seqlens,
        block_map,
        ssm_states,
        block_map.stride(0),
        ssm_states.stride(0),
        state_size,
        seq_size_per_block,
        256,
        num_warps=4,
    )
