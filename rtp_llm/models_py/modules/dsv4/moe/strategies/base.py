"""Routed-expert strategy interface + registry.

A *strategy* owns the per-rank routed-expert compute. The MoE layer drives
``Gate`` (token → expert routing) and the *shared* expert; the strategy takes
``(x, weights, indices)`` and returns the per-token routed sum in fp32.

The framework is intentionally NOT involved here — see
``.claude/plans/optimized-riding-mist.md`` for why we keep this dsv4-internal
rather than going through ``rtp_llm.models_py.modules.factory.fused_moe``.

Strategies (priority high→low for ``forced=None``):

    ep_size  env / kernel                 → strategy
    --------------------------------------------------------
    >1       mega available + dist-init    MegaMoEStrategy
    >1       mega unavailable/disabled     RuntimeError
    1        grouped FP4 kernel available  GroupedFP4Strategy
    1        grouped unavailable           LocalLoopStrategy

A model can override the auto-pick via:
  - ``MoE(strategy="mega"|"grouped_fp4"|"local_loop"|"deepep")`` ctor kwarg
  - ``DSV4_MOE_STRATEGY`` env var (overrides ctor kwarg)
  - the legacy ``DSV4_USE_MEGA_MOE=0`` / ``DSV4_USE_GROUPED_FP4=1`` toggles
    (translated to forced=... internally; conflicting toggles → RuntimeError)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Type

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MoeCfg:
    """Per-layer MoE configuration shared across all strategies.

    Frozen because strategies cache stuff keyed off it; mutating after
    construction would silently invalidate those caches.
    """

    layer_id: int
    dim: int
    moe_inter_dim: int
    n_routed_experts: int
    n_activated_experts: int  # topk
    swiglu_limit: float
    ep_size: int
    ep_rank: int
    n_local_experts: int
    local_expert_start: int
    local_expert_end: int
    max_tokens_per_rank: int


class RoutedExpertsStrategy(nn.Module):
    """Single-card or multi-card routed-expert compute.

    Inherits ``nn.Module`` so that strategies (notably ``LocalLoopStrategy``)
    can hold ``nn.ModuleList`` of ``Expert`` children whose Parameters propagate
    correctly through ``MoE.to(device)`` / state_dict traversal.

    The MoE layer is responsible for:
      - ``Gate`` (routing scores + topk)
      - the shared expert (one ``Expert`` instance)
      - dispatching to the chosen strategy

    A strategy is responsible for:
      - holding its own slice of routed-expert weights (loaded in ``setup_weights``)
      - producing ``[N, D] fp32`` per-token routed-sum from
        ``(x: [N, D] BF16, weights: [N, topk] FP32, indices: [N, topk] int64)``

    A strategy MUST handle cuda-graph capture state internally (e.g.
    ``LocalLoopStrategy.forward`` checks ``torch.cuda.is_current_stream_capturing()``
    and dispatches to a graph-safe variant). The MoE layer does NOT switch
    strategies based on capture state.

    Subclasses MUST call ``super().__init__()`` first so nn.Module bookkeeping
    is initialised. They override ``forward`` directly (it doubles as both
    nn.Module's forward hook and the strategy interface contract) and must
    define ``setup_weights`` + ``can_handle``.
    """

    name: ClassVar[str]  # "mega" / "grouped_fp4" / "local_loop" / "deepep"

    def __init__(self, cfg: MoeCfg):
        super().__init__()
        self.cfg = cfg

    def setup_weights(self, layer_weights: Dict) -> None:
        """Pop the strategy's own routed-expert stacks from ``layer_weights``
        (the framework's per-layer ``ModelWeights.weights[layer_id]`` dict
        keyed by ``W.v4_*`` enum). The stacks are already EP-sliced by the
        loader: each ``W.v4_routed_w{1,2,3}_{w,s}`` has shape ``[E_local, ...]``.

        Each strategy's docstring lists the exact W keys it pops, so a
        post-init audit can detect leftover keys (= bug).
        """
        raise NotImplementedError

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,        # [N, D] BF16
        weights: torch.Tensor,  # [N, topk] FP32
        indices: torch.Tensor,  # [N, topk] int64 GLOBAL expert id
    ) -> torch.Tensor:          # [N, D] FP32
        """Route + compute. Returns per-token routed-expert sum in fp32."""
        raise NotImplementedError

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        """Whether this strategy is applicable for ``cfg`` in the current
        runtime (env vars, kernel availability, dist init, SM arch, ...).

        Does NOT check cuda-graph capture state — that is forward's concern.
        """
        raise NotImplementedError


# --- selection -------------------------------------------------------------

# All known strategies, in priority order. Populated by ``register_strategy``
# from each strategy module's import side-effect (see strategies/__init__.py
# — importing a strategy class registers it).
_STRATEGY_PRIORITY: list[Type[RoutedExpertsStrategy]] = []


def register_strategy(cls: Type[RoutedExpertsStrategy]) -> Type[RoutedExpertsStrategy]:
    """Decorator: append ``cls`` to ``_STRATEGY_PRIORITY``.

    Order of import = order of priority. Convention: strategies/__init__.py
    imports them in priority order high→low.
    """
    if cls not in _STRATEGY_PRIORITY:
        _STRATEGY_PRIORITY.append(cls)
    return cls


def _resolve_forced(strategy_arg: Optional[str]) -> tuple[Optional[str], bool]:
    """Apply env-var overrides on top of constructor kwarg.

    Returns ``(forced_name, strict)``:
      - ``forced_name``: the strategy name to try, or ``None`` (auto-pick)
      - ``strict``: ``True`` → fail loudly if can_handle is False (explicit
        opt-in via ``DSV4_MOE_STRATEGY=...`` env or ctor kwarg);
        ``False`` → silently fall through to auto-pick if can_handle is False
        (legacy ``DSV4_USE_MEGA_MOE=1`` / ``DSV4_USE_GROUPED_FP4=1`` toggles —
        historically a "use if applicable" hint, NOT a hard force; e.g.
        ``DSV4_USE_MEGA_MOE=1`` was commonly left on for ep_size=1 smokes
        where Mega is fundamentally incompatible).

    Precedence (highest first):
      1. ``DSV4_MOE_STRATEGY`` env var (if not "auto") — strict
      2. legacy toggles (translated; non-strict)
      3. ``strategy_arg`` ctor kwarg — strict

    Raises ``RuntimeError`` on conflicting toggles.
    """
    env = os.environ.get("DSV4_MOE_STRATEGY", "").strip()
    if env and env != "auto":
        return env, True

    use_mega = os.environ.get("DSV4_USE_MEGA_MOE")
    use_grouped = os.environ.get("DSV4_USE_GROUPED_FP4")
    legacy_pos: list[str] = []
    if use_mega == "1":
        legacy_pos.append("mega")
    if use_grouped == "1":
        legacy_pos.append("grouped_fp4")

    if len(legacy_pos) > 1:
        raise RuntimeError(
            f"Conflicting legacy MoE toggles (multiple positive): {legacy_pos}. "
            "Set at most one of DSV4_USE_MEGA_MOE / DSV4_USE_GROUPED_FP4 to 1, "
            "or use DSV4_MOE_STRATEGY=<name>."
        )
    if legacy_pos and strategy_arg and strategy_arg not in legacy_pos:
        raise RuntimeError(
            f"Conflicting MoE strategy: ctor strategy={strategy_arg!r} but legacy "
            f"toggle forces {legacy_pos[0]!r}. Pick one source of truth."
        )
    if legacy_pos:
        return legacy_pos[0], False
    return strategy_arg, strategy_arg is not None  # ctor kwarg → strict


def select_strategy(
    cfg: MoeCfg,
    forced: Optional[str] = None,
    strict: bool = True,
) -> Type[RoutedExpertsStrategy]:
    """Pick a strategy class for ``cfg``.

    ``forced``: strategy name to try (after ``_resolve_forced`` env merge).
    ``strict``: when True (explicit opt-in: ctor kwarg or
    ``DSV4_MOE_STRATEGY``), fail loudly if ``forced`` can't handle cfg.
    When False (legacy env toggle), fall through silently to auto-pick.
    """
    if forced is not None:
        for cls in _STRATEGY_PRIORITY:
            if cls.name == forced:
                if cls.can_handle(cfg):
                    if cfg.ep_size > 1 and cls.name not in ("mega", "deepep"):
                        raise RuntimeError(
                            "DSV4 EP MoE requires MegaMoEStrategy by default. "
                            f"Requested strategy {forced!r} would bypass Mega "
                            f"(layer_id={cfg.layer_id}, ep_size={cfg.ep_size})."
                        )
                    return cls
                if strict:
                    raise RuntimeError(
                        f"Forced MoE strategy {forced!r} cannot handle cfg "
                        f"(layer_id={cfg.layer_id}, ep_size={cfg.ep_size}). "
                        "Check env / kernel availability."
                    )
                # Non-strict (legacy toggle) → fall through to auto-pick.
                break
        else:
            names = [c.name for c in _STRATEGY_PRIORITY]
            raise RuntimeError(
                f"Unknown MoE strategy {forced!r}. Available: {names}"
            )

    if cfg.ep_size > 1:
        mega_cls = next((c for c in _STRATEGY_PRIORITY if c.name == "mega"), None)
        if mega_cls is None:
            raise RuntimeError(
                "DSV4 EP MoE requires MegaMoEStrategy, but it is not registered."
            )
        if mega_cls.can_handle(cfg):
            return mega_cls
        from rtp_llm.models_py.modules.dsv4.moe.mega_buf import (
            _mega_moe_disabled_or_unavailable_reason,
        )

        raise RuntimeError(
            "DSV4 EP MoE requires MegaMoEStrategy by default; fallback to "
            "DeepEP/LocalLoop is disabled. "
            f"layer_id={cfg.layer_id}, ep_size={cfg.ep_size}. "
            f"Reason: {_mega_moe_disabled_or_unavailable_reason()}."
        )

    for cls in _STRATEGY_PRIORITY:
        if cls.can_handle(cfg):
            return cls
    raise RuntimeError(
        f"No MoE strategy can handle cfg (layer_id={cfg.layer_id}, "
        f"ep_size={cfg.ep_size})"
    )
