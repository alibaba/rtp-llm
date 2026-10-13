"""Bit-exact Triton elementwise/gather kernels for the MiniCPM-V-4.7 vision lane.

Every kernel replicates the exact floating-point semantics of the aten CUDA
kernels it replaces (fp32 opmath, identical constant values, left-associated
multiplies, round-to-nearest-even bf16 stores), so the ported vision tower
stays bit-exact while removing native-Torch elementwise launches from the
forward (zero-native-Torch fusion-gate census):

- ``vit_gelu``: ``F.gelu`` (tanh approximation and exact erf formulation),
  replicating aten ``ActivationGeluKernel.cu`` op-for-op in fp32 opmath.
- ``pos_embed_gather_add``: NaViT position-embedding gather + broadcast add
  over the channel-major Conv2d output (``aten::embedding`` + ``aten::add``
  -> 1 launch).
- ``window_gather_add``: in-tower window-merger residual row gather + add
  (``aten::index`` + ``aten::add`` -> 1 launch).
- ``merger_row_gather``: final 2x2 merger source-row gather into the per-crop
  ``[tokens, merge_h * merge_w * D]`` input layout (per-crop
  view/permute/reshape materializations -> 1 launch).

Contracts: contiguous CUDA bf16 tensors (row-major); anything else falls back
to the equivalent aten expression with identical results.  ``vit_gelu`` is
additionally gated by a one-time exhaustive bitwise self-check against aten
per (variant, device), because the libdevice bitcode linked by triton is not
guaranteed to match the one nvcc used to build the aten kernels.  No
autotune, no atomics, no reductions: fixed launch configs and deterministic
output.
"""
import torch
import triton
import triton.language as tl

try:
    import triton.language.extra.libdevice as tld
except ImportError:  # older triton layouts
    import triton.language.extra.cuda.libdevice as tld

# aten ActivationGeluKernel.cu constants, pre-rounded to their exact float32
# values: kBeta = M_SQRT2 * M_2_SQRTPI * 0.5 (= sqrt(2/pi)), kKappa = 0.044715
# (tanh approximation), kAlpha = M_SQRT1_2 (exact erf formulation).
_M_SQRT2 = 1.4142135623730951
_M_2_SQRTPI = 1.1283791670955126
_GELU_TANH_BETA = torch.tensor(_M_SQRT2 * _M_2_SQRTPI * 0.5, dtype=torch.float32).item()
_GELU_TANH_KAPPA = torch.tensor(0.044715, dtype=torch.float32).item()
_GELU_ERF_ALPHA = torch.tensor(0.7071067811865476, dtype=torch.float32).item()

_GELU_BLOCK = 1024
_GATHER_BLOCK = 1024

# Per-(approximate, device) result of the one-time exhaustive bit-exactness
# self-check of the Triton gelu against aten (all 65536 bf16 patterns).
# The libdevice bitcode linked by triton can differ from the one nvcc used
# for the aten kernels (erf/tanh last-ulp differences amplified by the
# 1 + erf(...) cancellation in the gelu tail), so the kernel is only used
# on a device where the check has passed bit-for-bit; otherwise the wrapper
# permanently falls back to ``F.gelu`` (identical values by definition).
_GELU_TRITON_OK: dict = {}


def _all_bf16_values(device) -> torch.Tensor:
    return (
        torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(device)
    )


def _gelu_triton_available(approximate: str, device: torch.device) -> bool:
    key = (approximate, device)
    if key in _GELU_TRITON_OK:
        return _GELU_TRITON_OK[key]
    ok = False
    try:
        x = _all_bf16_values(device)
        n = x.numel()
        y = torch.empty_like(x)
        _gelu_kernel[(triton.cdiv(n, _GELU_BLOCK),)](
            x,
            y,
            n,
            USE_TANH=approximate == "tanh",
            FMA_INNER=True,
            BETA=_GELU_TANH_BETA,
            KAPPA=_GELU_TANH_KAPPA,
            ALPHA=_GELU_ERF_ALPHA,
            BLOCK=_GELU_BLOCK,
            num_warps=4,
        )
        ref = torch.nn.functional.gelu(x, approximate=approximate)
        both_nan = torch.isnan(y.float()) & torch.isnan(ref.float())
        ok = not bool(((y != ref) & ~both_nan).any())
    except Exception:
        ok = False
    _GELU_TRITON_OK[key] = ok
    return ok


@triton.jit
def _gelu_kernel(
    X,
    Y,
    N,
    USE_TANH: tl.constexpr,
    FMA_INNER: tl.constexpr,
    BETA: tl.constexpr,
    KAPPA: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0).to(tl.float32)
    if USE_TANH:
        # aten: x_cube = x * x * x; inner = kBeta * (x + kKappa * x_cube);
        #       tanh_inner = tanh(inner); y = x * 0.5 * (1 + tanh_inner).
        # FMA_INNER selects the fused multiply-add form of (x + kKappa*x_cube)
        # to match nvcc's default contraction of the aten expression.
        x_cube = x * x * x
        if FMA_INNER:
            inner = BETA * tld.fma(KAPPA, x_cube, x)
        else:
            inner = BETA * (x + KAPPA * x_cube)
        tanh_inner = tld.tanh(inner)
        y = x * 0.5 * (1.0 + tanh_inner)
    else:
        # aten: y = x * 0.5 * (1 + erf(x * kAlpha)).
        y = x * 0.5 * (1.0 + tld.erf(x * ALPHA))
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


def vit_gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Bit-exact Triton ``F.gelu`` for contiguous CUDA bf16 tensors.

    The Triton path is enabled per (approximate, device) only after a
    one-time exhaustive bitwise self-check against ``F.gelu`` over all 65536
    bf16 patterns (see ``_gelu_triton_available``); anything else -- other
    device/dtype/layout, or a failed check -- falls back to the aten kernel
    with identical values.
    """
    if approximate not in ("none", "tanh"):
        return torch.nn.functional.gelu(x, approximate=approximate)
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()):
        return torch.nn.functional.gelu(x, approximate=approximate)
    if not _gelu_triton_available(approximate, x.device):
        return torch.nn.functional.gelu(x, approximate=approximate)
    y = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return y
    _gelu_kernel[(triton.cdiv(n, _GELU_BLOCK),)](
        x,
        y,
        n,
        USE_TANH=approximate == "tanh",
        FMA_INNER=True,
        BETA=_GELU_TANH_BETA,
        KAPPA=_GELU_TANH_KAPPA,
        ALPHA=_GELU_ERF_ALPHA,
        BLOCK=_GELU_BLOCK,
        num_warps=4,
    )
    return y


@triton.jit
def _pos_embed_gather_add_kernel(
    X,
    TABLE,
    POS,
    OUT,
    N,
    P,
    E,
    BLOCK: tl.constexpr,
):
    # X: [E, P] contiguous (channel-major Conv2d output); TABLE: [N, E];
    # POS: [P] int64; OUT: [P, E].  Out[s, d] = X[d, s] + Table[pos[s], d].
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    s = offs // E
    d = offs % E
    x = tl.load(X + d * P + s, mask=mask, other=0)
    pos = tl.load(POS + s, mask=mask, other=0)
    t = tl.load(TABLE + pos * E + d, mask=mask, other=0)
    out = x.to(tl.float32) + t.to(tl.float32)
    tl.store(OUT + offs, out.to(OUT.dtype.element_ty), mask=mask)


def pos_embed_gather_add(
    conv_out: torch.Tensor,
    pos_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Fused NaViT position-embedding gather + add (bit-exact vs
    ``embedding(pos_ids) + conv_out.transpose(1, 2)``).

    conv_out: [*, E, P] contiguous bf16 CUDA (Conv2d output, channel-major);
    pos_ids: [P] int64 CUDA; weight: [N, E] contiguous embedding table.
    Returns [1, P, E] contiguous.
    """
    if not (
        conv_out.is_cuda
        and conv_out.dtype == torch.bfloat16
        and conv_out.is_contiguous()
        and weight.is_contiguous()
        and pos_ids.is_cuda
        and pos_ids.dtype == torch.int64
    ):
        embeddings = conv_out.transpose(-2, -1)
        position_embeddings = torch.nn.functional.embedding(pos_ids, weight).unsqueeze(0)
        return embeddings + position_embeddings
    e, p = conv_out.shape[-2], conv_out.shape[-1]
    out = torch.empty((1, p, e), dtype=conv_out.dtype, device=conv_out.device)
    n = p * e
    _pos_embed_gather_add_kernel[(triton.cdiv(n, _GATHER_BLOCK),)](
        conv_out,
        weight,
        pos_ids,
        out,
        n,
        p,
        e,
        BLOCK=_GATHER_BLOCK,
        num_warps=4,
    )
    return out


@triton.jit
def _window_gather_add_kernel(
    R,
    INDEX,
    A,
    OUT,
    N,
    D,
    BLOCK: tl.constexpr,
):
    # R: [S, D] contiguous residual; INDEX: [Sw] int64; A/OUT: [Sw, D].
    # Out[w, d] = R[index[w], d] + A[w, d].
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    w = offs // D
    d = offs % D
    idx = tl.load(INDEX + w, mask=mask, other=0)
    r = tl.load(R + idx * D + d, mask=mask, other=0)
    a = tl.load(A + offs, mask=mask, other=0)
    out = r.to(tl.float32) + a.to(tl.float32)
    tl.store(OUT + offs, out.to(OUT.dtype.element_ty), mask=mask)


def window_gather_add(
    residual: torch.Tensor,
    window_index: torch.Tensor,
    attn_out: torch.Tensor,
) -> torch.Tensor:
    """Fused window-merger residual gather + add (bit-exact vs
    ``residual[:, window_index, :] + attn_out``).

    residual: [1, S, D] contiguous bf16 CUDA; window_index: [Sw] int64 CUDA;
    attn_out: [1, Sw, D] contiguous.  Returns [1, Sw, D] contiguous.
    """
    if not (
        residual.is_cuda
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and attn_out.is_contiguous()
        and window_index.is_cuda
        and window_index.dtype == torch.int64
    ):
        return residual[:, window_index, :] + attn_out
    d = residual.shape[-1]
    out = torch.empty_like(attn_out)
    n = attn_out.numel()
    if n == 0:
        return out
    _window_gather_add_kernel[(triton.cdiv(n, _GATHER_BLOCK),)](
        residual,
        window_index,
        attn_out,
        out,
        n,
        d,
        BLOCK=_GATHER_BLOCK,
        num_warps=4,
    )
    return out


@triton.jit
def _merger_row_gather_kernel(
    H,
    ROWMAP,
    OUT,
    N,
    D,
    K,
    BLOCK: tl.constexpr,
):
    # H: [S, D] contiguous hidden states; ROWMAP: [T*K] int64;
    # OUT: [T, K*D].  Out[t, k*D + d] = H[rowmap[t*K + k], d].
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    kd = K * D
    t = offs // kd
    rem = offs % kd
    k = rem // D
    d = rem % D
    src = tl.load(ROWMAP + t * K + k, mask=mask, other=0)
    h = tl.load(H + src * D + d, mask=mask, other=0)
    tl.store(OUT + offs, h, mask=mask)


def merger_row_gather(
    hidden_states: torch.Tensor,
    rowmap: torch.Tensor,
    src_dim: int,
    block_cols: int,
) -> torch.Tensor:
    """Gather the final 2x2 merger source rows (bit-exact copy, replaces the
    per-crop view/permute/reshape materializations).

    hidden_states: [1, S, D] contiguous bf16 CUDA; rowmap: [T*K] int64 CUDA
    (source row per output token, k-major within each token); src_dim: D;
    block_cols: K*D.  Returns [T, K*D] contiguous: the per-crop merger MLP
    input rows, identical values to ``view(mh, mh2, mw, mw2,
    D).permute(0, 2, 1, 3, 4).reshape(-1, K*D)``.
    """
    k = block_cols // src_dim
    total_tokens = rowmap.numel() // k
    out = torch.empty((total_tokens, block_cols), dtype=hidden_states.dtype, device=hidden_states.device)
    n = out.numel()
    if n == 0:
        return out
    _merger_row_gather_kernel[(triton.cdiv(n, _GATHER_BLOCK),)](
        hidden_states,
        rowmap,
        out,
        n,
        src_dim,
        k,
        BLOCK=_GATHER_BLOCK,
        num_warps=4,
    )
    return out
