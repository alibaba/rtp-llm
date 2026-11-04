"""DeepSeek V4.1 PPU mHC (shifted / delayed hyper-connections) op.

Implements the ``mhc`` op of the V4.1 backend contract
(``rtp_llm.models_py.modules.dsv41.backend.V41BackendOps.mhc``) for the PPU
provider: a pure-torch FP32 path (the numerical authority, portable to any
device) plus a Triton fast path (``ppu/_mhc_triton.py``) selected at call time
via ``is_ppu()`` + triton availability. The torch path stays authoritative;
remote PPU validation can force either path with ``MHCConfig.use_triton``.

Numerics
--------
The authority is the rtp-llm non-CUDA path:
* ``hc/delayed.py`` ``DelayedHCUnit`` (CPU path): FP32 projection
  ``m = (flat @ fn.T) * rsqrt(mean(flat^2) + rms_eps)`` with the RMS taken
  over the FULL ``hc*dim`` group, FP32 coefficients, FP32 readout
  accumulation, first-sublayer readout = lane 0.
* ``hc/fallback_impl.py:38-51`` ``_hc_split_sinkhorn``: the authoritative
  split with exact eps placement:
  ``pre = sigmoid(m[:hc]*scale[0] + base[:hc]) + eps``,
  ``post = 2*sigmoid(m[hc:2hc]*scale[1] + base[hc:2hc])``,
  ``comb = sinkhorn(m[2hc:]*scale[2] + base[2hc:])`` where sinkhorn is
  ``softmax(-1)+eps -> /(sum(-2)+eps) -> (iters-1) x {/(sum(-1)+eps);
  /(sum(-2)+eps)}``.

RMSNorm is deliberately NOT applied here. The caller norms the pre readout
AFTER the mHC op (block.py:662 ``x_pre = self.attn_norm(x_pre)`` for the
attention sublayer, block.py:777 ``x_pre = self.ffn_norm(x_pre)`` for the FFN
sublayer). sglang-for-sail folds that norm inside its ``combine``
(deepseek_v4_mhc.py:470); this module follows rtp semantics and returns the
un-normalized readout. The final model RMSNorm also belongs to the caller.

Dtype boundaries (matching hc/delayed.py exactly):
* residual / branch arrive bf16 (any float dtype works);
* all coefficient math runs FP32 (fn/base/scale are cast to FP32;
  delayed.py:58 requires FP32 weights already);
* ``pre_mix`` ``[..., hc]``, ``post_mix`` ``[..., hc, 1]`` and
  ``residual_mix`` ``[..., hc, hc]`` are stored FP32;
* ``hidden`` is returned in the residual's dtype (FP32 accumulate, cast once,
  like ``collapse_delayed`` and ``DelayedHCUnit._post_impl``).

State threading (PPUStepMix)
----------------------------
``PPUStepMix`` extends the frozen contract dataclass
``V41MHCResult`` (backend.py:56) with the shifted-chain field ``pre_mix``.
``V41Backend.validate()`` only checks op callables, so the extra field is
contract-compatible. The model threads the ``PPUStepMix`` returned by each
``pre`` call:

(a) into the SAME sublayer's ``post`` via the ``state`` argument, which
    consumes ``post_mix`` and ``residual_mix`` (the comb term);
(b) into the NEXT sublayer's ``pre`` as ``state``, which consumes ``pre_mix``
    for the delayed readout (``state=None`` or a state without ``pre_mix``
    selects the lane-0 one-hot entry readout);
(c) into the final ``head`` (parameterless), which consumes the LAST FFN
    sublayer's ``pre_mix``.

There is no cross-request hidden state anywhere in this module: every tensor
in ``PPUStepMix`` lives for one forward step of one request batch.

The call order mirrors block.py:645-808 / transformer.py:229-235:
``attn.pre -> norm -> attn -> attn.post -> ffn.pre -> norm -> ffn ->
ffn.post -> ... -> head``; the ``pre`` result of sublayer N is the ``state``
of sublayer N+1's ``pre`` (ffn_hc -> next layer's attn_hc included).

# The state/threading design and the Triton fast path are adapted from
# sglang-for-sail python/sglang/srt/models/deepseek_v4_mhc.py and
# python/sglang/kernels/ops/layernorm/mhc.py (512b03d), Apache-2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.backend import V41MHCResult

__all__ = [
    "MHCConfig",
    "PPUStepMix",
    "PPUMHC",
    "mhc_pre",
    "mhc_post",
    "mhc_head",
]


@dataclass(frozen=True)
class MHCConfig:
    """Per-call mHC knobs; defaults mirror the V4.1 model config.

    Nothing is hardcoded at the call sites: every phase reads these values.
    """

    hc_mult: int = 4
    sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_eps: float = 1e-6
    post_mult: float = 2.0
    # Bit-match hc/fallback_impl.py's bf16 fn GEMM (rsqrt cast to the input
    # dtype, fn cast to bf16). Only affects the 'pre' mixes projection; the
    # readout/post keep the delayed.py FP32 semantics. Off by default: the
    # PPU reference does NOT replicate NVIDIA bf16 GEMM rounding.
    strict_bf16_fn: bool = False
    # None = auto (is_ppu() + triton available + accelerator tensors);
    # True/False force the fast/torch path at call time (remote validation).
    use_triton: Optional[bool] = None


@dataclass(frozen=True)
class PPUStepMix(V41MHCResult):
    """The V41MHCResult plus the shifted-chain pre-mix.

    ``pre_mix`` ``[..., hc_mult]`` (FP32) carries this sublayer's pre
    coefficients to the NEXT sublayer's ``pre`` readout and to the final
    ``head``; ``post_mix`` ``[..., hc_mult, 1]`` and ``residual_mix``
    ``[..., hc_mult, hc_mult]`` carry this sublayer's own ``post``. No
    cross-request state.
    """

    pre_mix: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _get_config(config: Optional[MHCConfig]) -> MHCConfig:
    return config if config is not None else MHCConfig()


def _mix_width(hc_mult: int) -> int:
    return hc_mult * (hc_mult + 2)


def _require_parameters(
    parameters: Optional[Mapping[str, torch.Tensor]], keys, phase: str
) -> None:
    if parameters is None:
        raise ValueError(
            f"mhc phase {phase!r} requires a parameters mapping with keys "
            f"{list(keys)}; got None"
        )
    missing = [key for key in keys if key not in parameters]
    if missing:
        raise KeyError(
            f"mhc phase {phase!r} is missing required parameters {missing}; "
            f"got {sorted(parameters.keys())}"
        )


def _scale_1d(scale: torch.Tensor) -> torch.Tensor:
    """Accept [3] or [3, 1] (the loader's trailing-singleton promotion)."""
    if scale.dim() == 2 and scale.shape[-1] == 1:
        scale = scale.squeeze(-1)
    if scale.dim() != 1 or scale.numel() != 3:
        raise ValueError(
            f"mhc 'scale' must have 3 elements (optionally shaped [3, 1]); "
            f"got shape {tuple(scale.shape)}"
        )
    return scale.reshape(3)


def _is_ppu() -> bool:
    from rtp_llm.device.device_type import is_ppu

    return is_ppu()


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def _prefer_triton(config: MHCConfig, residual: torch.Tensor) -> bool:
    """Select the Triton fast path at CALL time (torch path authoritative)."""
    want = config.use_triton if config.use_triton is not None else _is_ppu()
    if not want:
        return False
    # Kernels launch on the accelerator only; CPU tensors always take torch.
    if not residual.is_cuda:
        return False
    return _triton_available()


def _triton_kernels():
    from rtp_llm.models_py.modules.dsv41.ppu import _mhc_triton

    return _mhc_triton


# ---------------------------------------------------------------------------
# The authoritative split (copied from hc/fallback_impl.py:22-52)
# ---------------------------------------------------------------------------


def _hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    post_mult: float = 2.0,
):
    """PyTorch reference split. Copied verbatim from hc/fallback_impl.py
    (rtp-llm) with the post multiplier parametrised; eps placement is exact."""
    hc = hc_mult
    *batch, mix_hc = mixes.size()

    pre_raw = mixes[..., :hc] * hc_scale[0] + hc_base[:hc]
    pre = pre_raw.sigmoid() + eps

    post_raw = mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    post = post_mult * post_raw.sigmoid()

    comb_raw = mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb_raw.view(*batch, hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def _pre_mixes(flat: torch.Tensor, fn: torch.Tensor, config: MHCConfig) -> torch.Tensor:
    """m = (flat @ fn.T) * rsqrt(mean(flat^2) + rms_eps), FP32 authority path.

    Matches hc/delayed.py:94-97 expression-for-expression. With
    ``strict_bf16_fn`` and a bf16 input, replicates hc/fallback_impl.py's
    bf16 fn GEMM bit-for-bit (deviation from the FP32 authority, opt-in for
    oracle comparisons only).
    """
    if config.strict_bf16_fn and flat.dtype == torch.bfloat16:
        # hc/fallback_impl.py:61-66, 90-94: fn cast to bf16, rsqrt cast to the
        # input dtype, GEMM in bf16, mixes stored fp32.
        rsqrt = torch.rsqrt(
            flat.float().square().mean(dim=-1, keepdim=True) + config.rms_eps
        ).to(flat.dtype)
        return (F.linear(flat, fn.to(torch.bfloat16)) * rsqrt).float()
    flat_f = flat.float()
    fn_f = fn.float()
    return F.linear(flat_f, fn_f) * torch.rsqrt(
        flat_f.square().mean(dim=-1, keepdim=True) + config.rms_eps
    )


def _readout(residual: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """hidden = sum_h pre_mix[..., h] * residual[..., h, :]; FP32 accumulate.

    Matches hc/delayed.py:44 (collapse_delayed) exactly.
    """
    return (
        (residual.float() * pre_mix.float().unsqueeze(-1))
        .sum(dim=-2)
        .to(residual.dtype)
    )


def _state_pre_mix(state) -> Optional[torch.Tensor]:
    if state is None:
        return None
    # A plain V41MHCResult carries no pre-mix; treat it as the entry state.
    return getattr(state, "pre_mix", None)


def _check_pre_mix_shape(pre_mix: torch.Tensor, residual: torch.Tensor, hc: int):
    expected = tuple(residual.shape[:-2]) + (hc,)
    if tuple(pre_mix.shape) != expected:
        raise ValueError(
            f"mhc state pre_mix shape {tuple(pre_mix.shape)} does not match the "
            f"residual token layout {expected}"
        )


# ---------------------------------------------------------------------------
# Phase implementations (torch authority + Triton fast path)
# ---------------------------------------------------------------------------


def _pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    config: MHCConfig,
):
    mixes = _pre_mixes(residual.flatten(-2), fn, config)
    return _hc_split_sinkhorn_torch(
        mixes,
        scale,
        base,
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.sinkhorn_iters,
        eps=config.hc_eps,
        post_mult=config.post_mult,
    )


def _readout_triton(residual: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    kernels = _triton_kernels()
    hc = residual.shape[-2]
    dim = residual.shape[-1]
    flat = residual.reshape(-1, hc * dim)
    if flat.stride(1) != 1:
        flat = flat.contiguous()
    pre = pre_mix.reshape(-1, hc).float().contiguous()
    y = kernels.mhc_combine(flat, pre, hc, residual.dtype)
    return y.reshape(tuple(residual.shape[:-2]) + (dim,))


def _pre_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    config: MHCConfig,
):
    kernels = _triton_kernels()
    hc = config.hc_mult
    dim = residual.shape[-1]
    flat = residual.reshape(-1, hc * dim)
    if flat.stride(1) != 1:
        flat = flat.contiguous()
    mixes = kernels.hc_mix_stats(flat, fn.float().contiguous(), config.rms_eps)
    return kernels.hc_split_sinkhorn(
        mixes,
        scale,
        base,
        hc_mult=hc,
        sinkhorn_iters=config.sinkhorn_iters,
        eps=config.hc_eps,
    )


def mhc_pre(
    residual: torch.Tensor,
    parameters: Mapping[str, torch.Tensor],
    *,
    state: Optional[V41MHCResult] = None,
    config: Optional[MHCConfig] = None,
) -> PPUStepMix:
    """Predict the mix triplet and read the (un-normalized) sublayer input.

    Args:
        residual: ``[..., hc_mult, dim]`` (typically ``[T, 4, 5120]`` bf16).
        parameters: ``"fn"`` ``[mix_width, hc*dim]`` (fp32), ``"base"``
            ``[mix_width]``, ``"scale"`` ``[3]`` or ``[3, 1]``.
        state: the PREVIOUS sublayer's ``pre`` result (uses ``pre_mix``);
            ``None`` selects the lane-0 one-hot entry readout.
        config: :class:`MHCConfig` knobs.

    Returns:
        ``PPUStepMix(hidden=readout, post_mix=[..., hc, 1],
        residual_mix=[..., hc, hc] comb, pre_mix=[..., hc])``. The caller
        applies RMSNorm to ``hidden`` afterwards (block.py:662/777).
    """
    cfg = _get_config(config)
    _require_parameters(parameters, ("fn", "base", "scale"), "pre")
    fn = parameters["fn"]
    base = parameters["base"]
    scale = _scale_1d(parameters["scale"])
    hc = cfg.hc_mult

    if residual.dim() < 3:
        raise ValueError(
            f"mhc 'pre' residual must be [..., hc_mult={hc}, dim]; got shape "
            f"{tuple(residual.shape)}"
        )
    if residual.shape[-2] != hc:
        raise ValueError(
            f"mhc 'pre' residual lane count {residual.shape[-2]} != hc_mult {hc}"
        )
    dim = residual.shape[-1]
    mix_width = _mix_width(hc)
    if fn.dim() != 2 or tuple(fn.shape) != (mix_width, hc * dim):
        raise ValueError(
            f"mhc 'fn' shape {tuple(fn.shape)} != {(mix_width, hc * dim)} "
            f"(mix_width, hc_mult * dim)"
        )
    if base.numel() != mix_width:
        raise ValueError(
            f"mhc 'base' must have {mix_width} elements; got {base.numel()}"
        )

    pre_state = _state_pre_mix(state)
    if pre_state is not None:
        _check_pre_mix_shape(pre_state, residual, hc)

    if _prefer_triton(cfg, residual):
        pre, post, comb = _pre_triton(residual, fn, scale, base, cfg)
    else:
        pre, post, comb = _pre_torch(residual, fn, scale, base, cfg)

    if pre_state is None:
        # Identity readout at entry selects lane zero, not the new mix
        # (hc/delayed.py:109-111).
        hidden = residual[..., 0, :].contiguous()
    elif _prefer_triton(cfg, residual):
        hidden = _readout_triton(residual, pre_state)
    else:
        hidden = _readout(residual, pre_state)

    return PPUStepMix(
        hidden=hidden,
        post_mix=post.unsqueeze(-1),
        residual_mix=comb,
        pre_mix=pre,
    )


def mhc_post(
    residual: torch.Tensor,
    branch: torch.Tensor,
    *,
    state: Optional[V41MHCResult] = None,
    parameters: Optional[Mapping[str, torch.Tensor]] = None,
    config: Optional[MHCConfig] = None,
) -> PPUStepMix:
    """Rebuild the residual streams from the sublayer output.

    ``r'[t, i] = state.post_mix[t, i] * branch[t] + sum_j
    state.residual_mix[t, j, i] * r[t, j]`` -- the comb orientation matches
    hc/fallback_impl.py:141-149 (``matmul(comb.transpose(-1, -2), residual)``)
    and hc/delayed.py:120-123. ``parameters`` is accepted for the contract
    signature but unused.

    Returns ``PPUStepMix(hidden=r')``; post_mix/residual_mix are consumed and
    returned as ``None``.
    """
    cfg = _get_config(config)
    if state is None:
        raise ValueError(
            "mhc 'post' requires the PPUStepMix returned by the SAME sublayer's "
            "'pre' call (pass it via state=...)"
        )
    post = state.post_mix
    comb = state.residual_mix
    if post is None or comb is None:
        raise ValueError(
            "mhc 'post' state is missing post_mix/residual_mix; it must be the "
            "'pre' result of the same sublayer, not of a 'post' or 'head' call"
        )
    hc = residual.shape[-2] if residual.dim() >= 2 else -1
    if residual.dim() < 3 or hc != cfg.hc_mult:
        raise ValueError(
            f"mhc 'post' residual must be [..., hc_mult={cfg.hc_mult}, dim]; got "
            f"shape {tuple(residual.shape)}"
        )
    if branch.dim() != residual.dim() - 1 or branch.shape[-1] != residual.shape[-1]:
        raise ValueError(
            f"mhc 'post' branch shape {tuple(branch.shape)} must be the residual's "
            f"token layout with shape {tuple(residual.shape[:-2]) + (residual.shape[-1],)}"
        )
    if comb.dim() != residual.dim() or tuple(comb.shape[-2:]) != (hc, hc):
        raise ValueError(
            f"mhc 'post' state residual_mix must be [..., {hc}, {hc}]; got "
            f"{tuple(comb.shape)}"
        )
    if post.dim() == residual.dim() and post.shape[-1] == 1:
        post_b = post.squeeze(-1)
    elif post.dim() == residual.dim() - 1:
        post_b = post
    else:
        raise ValueError(
            f"mhc 'post' state post_mix must be [..., {hc}, 1] or [..., {hc}]; got "
            f"{tuple(post.shape)}"
        )
    if tuple(post_b.shape) != tuple(residual.shape[:-2]) + (hc,):
        raise ValueError(
            f"mhc 'post' state post_mix shape {tuple(post_b.shape)} does not match "
            f"the residual token layout {tuple(residual.shape[:-2]) + (hc,)}"
        )

    if _prefer_triton(cfg, residual) and residual.dim() == 3:
        kernels = _triton_kernels()
        r_prime = kernels.mhc_post(
            branch.reshape(-1, branch.shape[-1]).contiguous(),
            residual.reshape(-1, hc, residual.shape[-1]).contiguous(),
            post_b.reshape(-1, hc).float().contiguous(),
            comb.reshape(-1, hc, hc).float().contiguous(),
        ).reshape(residual.shape)
    else:
        # hc/delayed.py:117-123 CPU path, expression-for-expression: the post
        # coefficients keep their trailing singleton so [..., hc, 1] broadcasts
        # against branch's [..., 1, dim].
        post_hc1 = post if post.dim() == residual.dim() else post_b.unsqueeze(-1)
        r_prime = (
            post_hc1.float() * branch.float().unsqueeze(-2)
            + torch.matmul(comb.float().transpose(-1, -2), residual.float())
        ).to(residual.dtype)
    return PPUStepMix(hidden=r_prime, post_mix=None, residual_mix=None)


def mhc_head(
    residual: torch.Tensor,
    *,
    state: Optional[V41MHCResult] = None,
    parameters: Optional[Mapping[str, torch.Tensor]] = None,
    config: Optional[MHCConfig] = None,
) -> PPUStepMix:
    """Parameterless final readout using the LAST FFN sublayer's pre_mix.

    hidden = sum_h state.pre_mix[..., h] * residual[..., h, :] (FP32
    accumulate; hc/delayed.py:126-133 DelayedHCHead). The final model
    RMSNorm belongs to the caller. Only ``pre_mix`` is read; post_mix and
    residual_mix are ignored.
    """
    cfg = _get_config(config)
    if state is None:
        raise ValueError(
            "mhc 'head' requires the PPUStepMix of the last FFN sublayer's "
            "'pre' call (pass it via state=...)"
        )
    pre = _state_pre_mix(state)
    if pre is None:
        raise ValueError(
            "mhc 'head' state has no pre_mix; pass the last FFN 'pre' result, "
            "not a 'post' result"
        )
    hc = cfg.hc_mult
    if residual.dim() < 3 or residual.shape[-2] != hc:
        raise ValueError(
            f"mhc 'head' residual must be [..., hc_mult={hc}, dim]; got shape "
            f"{tuple(residual.shape)}"
        )
    _check_pre_mix_shape(pre, residual, hc)

    if _prefer_triton(cfg, residual):
        hidden = _readout_triton(residual, pre)
    else:
        hidden = _readout(residual, pre)
    return PPUStepMix(hidden=hidden, post_mix=None, residual_mix=None)


# ---------------------------------------------------------------------------
# Contract-shaped callable
# ---------------------------------------------------------------------------


class PPUMHC:
    """``V41BackendOps.mhc`` implementation for the PPU provider.

    ``__call__`` matches the frozen contract signature
    ``mhc(residual, branch, *, phase, parameters, context, state)`` and
    dispatches to :func:`mhc_pre` / :func:`mhc_post` / :func:`mhc_head`.
    ``context`` (``V41StepContext``) is currently unused -- mHC is
    context-free by construction -- but kept in the signature per contract.
    """

    def __init__(self, config: Optional[MHCConfig] = None) -> None:
        self.config = config

    def __call__(
        self,
        residual: torch.Tensor,
        branch: Optional[torch.Tensor],
        *,
        phase: str,
        parameters: Optional[Mapping[str, torch.Tensor]] = None,
        context: object = None,
        state: Optional[V41MHCResult] = None,
        config: Optional[MHCConfig] = None,
    ) -> PPUStepMix:
        cfg = config if config is not None else self.config
        if phase == "pre":
            return mhc_pre(residual, parameters, state=state, config=cfg)
        if phase == "post":
            return mhc_post(
                residual, branch, state=state, parameters=parameters, config=cfg
            )
        if phase == "head":
            return mhc_head(residual, state=state, parameters=parameters, config=cfg)
        raise ValueError(
            f"unsupported mhc phase {phase!r}; expected one of 'pre', 'post', 'head'"
        )

    # Explicit per-phase helpers for direct call sites.
    def pre(self, residual, parameters, *, state=None, config=None) -> PPUStepMix:
        return mhc_pre(residual, parameters, state=state, config=config or self.config)

    def post(self, residual, branch, *, state=None, config=None) -> PPUStepMix:
        return mhc_post(residual, branch, state=state, config=config or self.config)

    def head(self, residual, *, state=None, config=None) -> PPUStepMix:
        return mhc_head(residual, state=state, config=config or self.config)
