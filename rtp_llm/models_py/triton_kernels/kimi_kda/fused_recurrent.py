# Fused recurrent KDA kernel for rtp-llm decode.
# Combines fla's gate logic (USE_GATE_IN_KERNEL, softplus, A_log, dt_bias)
# with rtp-llm's block_map-based continuous batching for ssm_state indexing.

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.op import exp, softplus

logger = logging.getLogger(__name__)

_FLA37_SM103_CUBIN_SHA256 = (
    "6a486a94bae75c683e1ef961e2bfa9cdb18270f1c6980a8d3097f766164f3374"
)


@triton.jit
def cal_block_idx(x, seq_size_per_block):
    return (x - 1) // seq_size_per_block


@triton.jit(do_not_specialize=["N", "T"])
def _fla37_recurrent_launcher_template(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    lower_bound,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Build only the Triton 3.6 launch ABI for a validated 3.7 cubin.

    This body is never launched: ``fused_recurrent_kda_fla37_precompiled``
    replaces its compiled image before the first invocation.  Keeping the
    source signature identical to FLA 0.5.1 makes Triton's generated argument
    launcher safe to reuse without loading a second Python/Torch environment.
    """

    pid = tl.program_id(0)
    offset = pid * 0
    value = tl.load(q + offset).to(tl.float32)
    value += tl.load(k + offset).to(tl.float32)
    value += tl.load(v + offset).to(tl.float32)
    value += tl.load(g + offset).to(tl.float32)
    value += tl.load(beta + offset).to(tl.float32)
    value += tl.load(A_log + offset).to(tl.float32)
    value += tl.load(dt_bias + offset).to(tl.float32)
    value += tl.load(h0 + offset).to(tl.float32)
    value += tl.load(cu_seqlens + offset).to(tl.float32)
    value += lower_bound + N * 0.0 + T * 0.0
    tl.store(o + offset, value.to(o.dtype.element_ty))
    tl.store(ht + offset, value)


def _install_fla37_precompiled_image(
    compiled,
    precompiled_dir: Path,
) -> None:
    if getattr(compiled, "_kimi_fla37_precompiled", False):
        return
    cubin_path = precompiled_dir / "fused_recurrent_kda_fwd_kernel.cubin"
    metadata_path = precompiled_dir / "fused_recurrent_kda_fwd_kernel.json"
    cubin = cubin_path.read_bytes()
    digest = hashlib.sha256(cubin).hexdigest()
    if digest != _FLA37_SM103_CUBIN_SHA256:
        raise RuntimeError(
            "K3 FLA37 recurrent cubin SHA256 mismatch: "
            f"{digest} != {_FLA37_SM103_CUBIN_SHA256}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target = metadata.get("target", {})
    if (
        target.get("backend") != "cuda"
        or int(target.get("arch", -1)) != 103
        or metadata.get("triton_version") != "3.7.0"
        or int(metadata.get("num_warps", -1)) != 4
        or int(metadata.get("num_stages", -1)) != 2
    ):
        raise RuntimeError(f"unsupported K3 FLA37 cubin metadata: {metadata}")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise RuntimeError(
            "K3 FLA37 recurrent cubin requires SM103, got "
            f"compute capability {capability}"
        )

    updates = {}
    for name in (
        "name",
        "shared",
        "num_warps",
        "num_ctas",
        "num_stages",
        "tmem_size",
    ):
        if name in compiled.metadata._fields and name in metadata:
            updates[name] = metadata[name]
    compiled.metadata = compiled.metadata._replace(**updates)
    compiled.name = metadata["name"]
    from triton.compiler.compiler import make_backend

    compiled.packed_metadata = make_backend(compiled.metadata.target).pack_metadata(
        compiled.metadata
    )
    compiled.asm["cubin"] = cubin
    compiled.kernel = cubin
    compiled.module = None
    compiled.function = None
    compiled._kimi_fla37_precompiled = True
    logger.warning(
        "K3 decode accuracy backend loaded FLA 0.5.1/Triton 3.7 SM103 "
        "recurrent cubin sha256=%s",
        digest,
    )


def fused_recurrent_kda_fla37_precompiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the bitwise-Dummy recurrent KDA image in a Triton 3.6 process.

    This is a guarded accuracy backend, not the default production path.  It
    exists because Triton 3.6 and 3.7 lower the same FLA reduction differently,
    and near-tied K3 routers can amplify the resulting fp32 state ULPs.  The
    caller must explicitly provide ``KIMI_K3_KDA_FLA37_PRECOMPILED_DIR``.
    """

    precompiled = os.environ.get("KIMI_K3_KDA_FLA37_PRECOMPILED_DIR")
    if not precompiled:
        raise RuntimeError(
            "KIMI_K3_KDA_FLA37_PRECOMPILED_DIR is required for "
            "KIMI_K3_KDA_BACKEND=fla37_precompiled"
        )
    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError("K3 precompiled recurrent q/k/g shapes must match")
    batch, length, heads, key_dim = q.shape
    if (
        length != 1
        or heads != 96
        or key_dim != 128
        or tuple(v.shape) != (batch, length, 96, 128)
        or tuple(beta.shape) != (batch, length, 96)
        or initial_state.shape[-3:] != (96, 128, 128)
    ):
        raise ValueError(
            "K3 FLA37 recurrent cubin supports decode "
            "[B,1,96,128] with [N,96,128,128] V-first state only"
        )
    sequence_count = len(cu_seqlens) - 1
    output = torch.zeros_like(v)
    final_state = q.new_empty(
        sequence_count,
        96,
        128,
        128,
        dtype=torch.float32,
    )
    grid = (4 * sequence_count * 96,)
    arguments = {
        "q": q,
        "k": k,
        "v": v,
        "g": g.contiguous(),
        "beta": beta.contiguous(),
        "A_log": A_log,
        "dt_bias": dt_bias,
        "o": output,
        "h0": initial_state,
        "ht": final_state,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": None,
        "num_accepted_tokens": None,
        "lower_bound": lower_bound,
        "scale": key_dim**-0.5,
        "N": sequence_count,
        "T": length,
        "H": heads,
        "HV": 96,
        "K": key_dim,
        "V": 128,
        "BK": 128,
        "BV": 32,
        "stride_init_state_token": initial_state.stride(0),
        "stride_final_state_token": final_state.stride(0),
        "stride_indices_seq": 1,
        "stride_indices_tok": 1,
        "USE_INITIAL_STATE": True,
        "INPLACE_FINAL_STATE": False,
        "IS_BETA_HEADWISE": False,
        "USE_QK_L2NORM_IN_KERNEL": True,
        "IS_VARLEN": True,
        "IS_CONTINUOUS_BATCHING": False,
        "IS_SPEC_DECODING": False,
        "STORE_FINAL_STATE": True,
        "HAS_DT_BIAS": True,
        "USE_GATE_IN_KERNEL": True,
        "USE_LOWER_BOUND": True,
        "APPLY_BETA_SIGMOID": True,
        "ALLOW_NEG_EIGVAL": False,
        "STATE_V_FIRST": True,
        "num_stages": 2,
        "num_warps": 4,
    }
    compiled = _fla37_recurrent_launcher_template.warmup(grid=grid, **arguments)
    _install_fla37_precompiled_image(compiled, Path(precompiled))
    _fla37_recurrent_launcher_template[grid](**arguments)
    return output, final_state


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["block_map"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_kda_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    block_map,
    sequence_lengths,
    max_block_size: tl.int32,
    lower_bound,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    NK = tl.cdiv(K, BK)
    i_k = pid % NK
    pid_rest = pid // NK
    i_v = pid_rest % NV
    i_nh = pid_rest // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    if IS_CONTINUOUS_BATCHING:
        sequence_length = tl.load(sequence_lengths + i_n).to(tl.int64)
    else:
        sequence_length = 0

    if T <= 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    # KDA: g is per-dim [B, T, HV, K]
    p_g = g + (bos * HV + i_hv) * K + o_k
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # Match FLA's transpose_state_layout=True arithmetic: keep the register
    # tile [V,K], while translating addresses to RTP's canonical [K,V] cache.
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            load_block_offset = cal_block_idx(sequence_length - 1, SEQ_SIZE_PER_BLOCK)
            read_block_id = tl.load(
                block_map + i_n * max_block_size + load_block_offset
            ).to(tl.int64)
            if read_block_id <= 0:
                return
            p_h0 = h0 + read_block_id * stride_init_state_token + i_hv * K * V
        else:
            p_h0 = h0 + (i_n * HV + i_hv) * K * V
        if STATE_V_FIRST:
            p_h0 = p_h0 + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = p_h0 + o_k[None, :] * V + o_v[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in tl.range(0, T, num_stages=num_stages):
        b_q = tl.load(p_q, mask=mask_k, other=0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_k = tl.load(p_k, mask=mask_k, other=0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_v = tl.load(p_v, mask=mask_v, other=0, eviction_policy="evict_first").to(
            tl.float32
        )

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        # Load raw gate and apply gate activation if needed.
        b_g = tl.load(p_g, mask=mask_k, other=0, eviction_policy="evict_last").to(
            tl.float32
        )
        if USE_GATE_IN_KERNEL:
            b_A = tl.load(A_log + i_hv).to(tl.float32)
            if HAS_DT_BIAS:
                b_bias = tl.load(
                    dt_bias + i_hv * K + o_k,
                    mask=mask_k,
                    other=0,
                ).to(tl.float32)
                b_g = b_g + b_bias
            if USE_LOWER_BOUND:
                b_gk = lower_bound * tl.sigmoid(exp(b_A) * b_g)
            else:
                b_gk = -exp(b_A) * softplus(b_g)
        else:
            b_gk = b_g

        b_h *= exp(b_gk[None, :])

        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(
                p_beta,
                mask=mask_v,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
            if ALLOW_NEG_EIGVAL:
                b_beta = b_beta * 2
        b_v *= b_beta

        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(
            p_o,
            b_o.to(p_o.dtype.element_ty),
            mask=mask_v,
            eviction_policy="evict_first",
        )

        if IS_CONTINUOUS_BATCHING:
            if INPLACE_FINAL_STATE:
                write_block_offset = (
                    cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK) + i_t
                )
                write_block_id = tl.load(
                    block_map + i_n * max_block_size + write_block_offset
                ).to(tl.int64)
                p_ht = ht + write_block_id * stride_final_state_token
            else:
                p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * K * V
            if STATE_V_FIRST:
                p_ht = p_ht + o_v[:, None] * K + o_k[None, :]
            else:
                p_ht = p_ht + o_k[None, :] * V + o_v[:, None]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    if not IS_CONTINUOUS_BATCHING:
        p_ht = ht + (i_n * HV + i_hv) * K * V
        if STATE_V_FIRST:
            p_ht = p_ht + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = p_ht + o_k[None, :] * V + o_v[:, None]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), 32
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    o = torch.zeros_like(v)
    if inplace_final_state:
        assert initial_state is not None
        final_state = initial_state
    else:
        state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
        final_state = q.new_empty(*state_shape, dtype=torch.float32)

    stride_init_state_token = (
        initial_state.stride(0) if initial_state is not None else 1
    )
    stride_final_state_token = final_state.stride(0) if final_state is not None else 1

    max_block_size = 0
    if block_map is not None:
        assert block_map.ndim == 2, "block_map must be a 2D tensor"
        max_block_size = block_map.shape[1]

    grid = (NK * NV * N * HV,)
    fused_recurrent_kda_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        block_map=block_map,
        sequence_lengths=sequence_lengths,
        max_block_size=max_block_size,
        lower_bound=lower_bound,
        scale=scale,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        ALLOW_NEG_EIGVAL=allow_neg_eigval,
        STATE_V_FIRST=state_v_first,
        num_warps=4,
        num_stages=2,
    )
    return o, final_state


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        A_log=A_log,
        dt_bias=dt_bias,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        block_map=block_map,
        seq_size_per_block=seq_size_per_block,
        sequence_lengths=sequence_lengths,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        lower_bound=lower_bound,
        state_v_first=state_v_first,
    )
    return o, final_state
