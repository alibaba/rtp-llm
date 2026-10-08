"""V4.1 block numerics with an explicit attention publication/exchange boundary.

The stage applies Engram before prepare(). Between prepare() and finish(), the
caller publishes KV rows, exchanges CP data, selects keys and executes sparse
attention. No request, cache or predecessor state is stored on the block.
"""

from dataclasses import dataclass

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv4.hc.delayed import (
    DelayedHCMixes,
    DelayedHCUnit,
    collapse_delayed,
    identity_pre_mix,
)
from rtp_llm.models_py.modules.dsv4.v41_attention_projection import (
    V41AttentionProjection,
    V41QueryFrame,
)
from rtp_llm.models_py.modules.dsv4.v41_feed_forward import V41FeedForward
from rtp_llm.models_py.modules.dsv4.v41_moe import _assert_values
from rtp_llm.models_py.modules.dsv4.v41_projection import (
    V41KVFrame,
    V41KVProjection,
    _rms_norm,
    _tensor,
)


@dataclass(frozen=True)
class V41ResidualState:
    residual: torch.Tensor
    previous_pre_mix: torch.Tensor
    next_layer_id: int


@dataclass(frozen=True)
class V41BlockFrame:
    owner: object
    residual: torch.Tensor
    mixes: DelayedHCMixes
    query: V41QueryFrame
    kv: V41KVFrame


def _clean_state(state, valid_rows, *, dim, device, max_rows, next_layer_id):
    if (
        not isinstance(state, V41ResidualState)
        or type(state.next_layer_id) is not int
        or state.next_layer_id != next_layer_id
    ):
        raise ValueError("V4.1 residual state belongs to a different next global layer")
    residual, previous = state.residual, state.previous_pre_mix
    rows = residual.shape[0] if residual.ndim == 3 else -1
    if (
        not 0 <= rows <= max_rows
        or residual.shape != (rows, 4, dim)
        or residual.dtype != torch.bfloat16
        or residual.device != device
        or not residual.is_contiguous()
        or residual.requires_grad
        or previous.shape != (rows, 4)
        or previous.dtype != torch.float32
        or previous.device != device
        or not previous.is_contiguous()
        or previous.requires_grad
    ):
        raise ValueError(
            "V4.1 block requires bounded BF16 residual and FP32 predecessor coefficients"
        )
    _tensor(valid_rows, (rows,), torch.bool, device, "block mask")
    clean = torch.where(valid_rows[:, None, None], residual, 0)
    previous = torch.where(valid_rows[:, None], previous, 0)
    _assert_values(
        torch.isfinite(clean).all() & torch.isfinite(previous).all(),
        "V4.1 live block state must be finite",
    )
    return clean, previous


def start_v41_residual(embeddings, valid_rows):
    """Expand stage-zero token embeddings; the initial readout selects lane zero."""
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] <= 0
        or embeddings.dtype != torch.bfloat16
        or not embeddings.is_contiguous()
        or embeddings.requires_grad
    ):
        raise ValueError("V4.1 embeddings require contiguous inference BF16 [R,D]")
    _tensor(
        valid_rows,
        (embeddings.shape[0],),
        torch.bool,
        embeddings.device,
        "embedding mask",
    )
    clean = torch.where(valid_rows[:, None], embeddings, 0)
    _assert_values(torch.isfinite(clean).all(), "V4.1 live embeddings must be finite")
    residual = clean[:, None].repeat(1, 4, 1)
    previous = torch.where(valid_rows[:, None], identity_pre_mix(residual), 0)
    return V41ResidualState(residual, previous, 0)


def readout_v41_residual(state, norm_weight, valid_rows):
    """Final FFN coefficients collapse the stream before the checkpoint RMS norm."""
    if norm_weight.ndim != 1 or norm_weight.numel() <= 0:
        raise ValueError("V4.1 final norm requires a nonempty weight vector")
    device, dim = norm_weight.device, norm_weight.numel()
    _tensor(norm_weight, (dim,), torch.bfloat16, device, "final norm")
    residual, previous = _clean_state(
        state, valid_rows, dim=dim, device=device, max_rows=128, next_layer_id=40
    )
    return _rms_norm(collapse_delayed(residual, previous), norm_weight)


class V41Block(nn.Module):
    """Checkpoint delayed-mHC, attention projections and complete FFN.

    Frames own residual/mask snapshots and must return to their creating block.
    The caller owns frame scheduling and same-stream cache readiness. Engram is
    a stage operation before this block, matching the checkpoint Transformer.
    This numerical component is not registered as a serving model.
    """

    def __init__(
        self, attention, kv, attn_hc, ffn_hc, attn_norm, ffn_norm, feed_forward
    ):
        super().__init__()
        if (
            not isinstance(attention, V41AttentionProjection)
            or not isinstance(kv, V41KVProjection)
            or not isinstance(feed_forward, V41FeedForward)
        ):
            raise ValueError("V4.1 block requires V4.1 projections and complete FFN")
        dim, device = attention.wq_a.K, attention.wq_a.weight.device
        layer_id = attention.sources.layer_id
        if (
            kv.sources != attention.sources
            or kv.wkv.K != dim
            or kv.wkv.backend != attention.wq_a.backend
            or kv.wkv.weight.device != device
            or kv.frequencies is not attention.frequencies
            or feed_forward.layer_id != layer_id
            or feed_forward.dim != dim
            or feed_forward.device != device
            or (feed_forward.shared.backend == "eager")
            != (attention.wq_a.backend == "eager")
        ):
            raise ValueError(
                "V4.1 block component layer/geometry/device/backend mismatch"
            )
        for unit in (attn_hc, ffn_hc):
            if (
                not isinstance(unit, DelayedHCUnit)
                or unit.dim != dim
                or unit.hc_mult != 4
                or unit.fn.device != device
                or unit.norm_eps != 1e-20
                or unit.hc_sinkhorn_iters != 20
                or unit.hc_eps != 1e-6
                or any(t.requires_grad for t in (unit.fn, unit.base, unit.scale))
            ):
                raise ValueError(
                    "V4.1 block requires checkpoint delayed-mHC geometry and epsilons"
                )
        _tensor(attn_norm, (dim,), torch.bfloat16, device, "attention norm")
        _tensor(ffn_norm, (dim,), torch.bfloat16, device, "FFN norm")
        self.attention, self.kv = attention, kv
        self.attn_hc, self.ffn_hc = attn_hc, ffn_hc
        self.attn_norm, self.ffn_norm, self.feed_forward = (
            attn_norm,
            ffn_norm,
            feed_forward,
        )
        self.dim, self.device, self.layer_id = dim, device, layer_id
        self.max_rows = min(
            attention.max_rows, kv.max_rows, feed_forward.router.max_rows
        )
        self._owner = object()

    def _check_precision(self):
        if self.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("V4.1 FP32 delayed-mHC requires TF32 disabled")

    def prepare(
        self, state, positions, valid_rows, *, pair_state=None, pair_metadata=None
    ):
        """Project a forward without writing caches; stage has already applied Engram."""
        self._check_precision()
        residual, previous = _clean_state(
            state,
            valid_rows,
            dim=self.dim,
            device=self.device,
            max_rows=self.max_rows,
            next_layer_id=self.layer_id,
        )
        collapsed, mixes = self.attn_hc.pre(residual, previous)
        hidden = _rms_norm(collapsed, self.attn_norm)
        query = self.attention.prepare(hidden, positions, valid_rows)
        kv = self.kv.prepare(
            hidden,
            positions,
            valid_rows,
            pair_state=pair_state,
            pair_metadata=pair_metadata,
        )
        return V41BlockFrame(self._owner, residual, mixes, query, kv)

    def finish(self, frame, attended):
        """Complete one block after sparse attention using this frame's queries."""
        self._check_precision()
        if not isinstance(frame, V41BlockFrame) or frame.owner is not self._owner:
            raise ValueError("V4.1 attention frame belongs to a different block")
        output = self.attention.output(attended, frame.query)
        residual = self.attn_hc.post(output, frame.residual, frame.mixes)
        collapsed, mixes = self.ffn_hc.pre(residual, frame.mixes.next_pre_mix)
        output = self.feed_forward(
            _rms_norm(collapsed, self.ffn_norm), frame.query.valid_rows
        )
        residual = self.ffn_hc.post(output, residual, mixes)
        valid = frame.query.valid_rows
        return V41ResidualState(
            torch.where(valid[:, None, None], residual, 0),
            torch.where(valid[:, None], mixes.next_pre_mix, 0),
            self.layer_id + 1,
        )


def build_v41_block(layer_weights, attention, kv, feed_forward):
    """Bind checkpoint norms and FP32 HC tensors without consuming loader state."""
    from rtp_llm.utils.model_weight import W

    def unit(kind):
        return DelayedHCUnit(
            layer_weights[getattr(W, "v4_hc_" + kind + "_fn")],
            layer_weights[getattr(W, "v4_hc_" + kind + "_base")],
            layer_weights[getattr(W, "v4_hc_" + kind + "_scale")],
            dim=attention.wq_a.K,
            norm_eps=1e-20,
        )

    return V41Block(
        attention,
        kv,
        unit("attn"),
        unit("ffn"),
        layer_weights[W.v4_attn_norm],
        layer_weights[W.v4_ffn_norm],
        feed_forward,
    )
