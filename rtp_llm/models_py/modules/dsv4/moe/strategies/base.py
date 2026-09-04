"""Routed-expert strategy interface + registry.

A *strategy* owns the per-rank routed-expert compute. The MoE layer drives
``Gate`` (token → expert routing) and, normally, the *shared* expert; a fused
strategy may own the shared expert too and return ``routed + shared``.

The framework is intentionally NOT involved here — see
``.claude/plans/optimized-riding-mist.md`` for why we keep this dsv4-internal
rather than going through ``rtp_llm.models_py.modules.factory.fused_moe``.

Strategies (priority high→low for ``forced=None``):

    ep_size  env / capability             → strategy
    --------------------------------------------------------------
    >1       SE enabled + SE capable       MegaMoEStrategySE (default)
    >1       SE auto + SE incapable        MegaMoEStrategy (warn once/process)
    >1       DSV4_USE_MEGA_MOE_SE=0        MegaMoEStrategy / legacy fused
    >1       base Mega unavailable         RuntimeError
    1        grouped FP4 kernel available  GroupedFP4Strategy
    1        grouped unavailable           LocalLoopStrategy

``DSV4_MOE_STRATEGY`` is the highest-precedence environment override.  Without
it, ``DSV4_USE_MEGA_MOE_SE=1`` is a strict request, while an unset variable is
an automatic preference and ``=0`` is an explicit opt-out.  Incompatible
legacy requests fail instead of silently selecting another backend.
"""

from __future__ import annotations

import logging
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

    The MoE layer is normally responsible for:
      - ``Gate`` (routing scores + topk)
      - the shared expert (one ``Expert`` instance)
      - dispatching to the chosen strategy

    A strategy with ``routed_includes_shared=True`` instead owns the shared
    expert weights and returns the combined result.

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

    # Registered names currently include mega, mega_se, mega_fused,
    # grouped_fp4, local_loop, and deepep.
    name: ClassVar[str]

    # True when ``forward`` already returns ``routed + shared`` (the strategy
    # fuses the shared expert internally). The ``MoE`` layer then skips its own
    # shared-expert executor and the ``combine_routed_and_shared`` add. Only
    # Mega variants that fuse the shared expert set this True.
    routed_includes_shared: ClassVar[bool] = False

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
        x: torch.Tensor,  # [N, D] BF16
        weights: torch.Tensor,  # [N, topk] FP32
        indices: torch.Tensor,  # [N, topk] int64 GLOBAL expert id
    ) -> torch.Tensor:  # [N, D] FP32
        """Route + compute. Returns per-token routed-expert sum in fp32."""
        raise NotImplementedError

    def can_use_gate_pack_static(self, gate) -> bool:
        """Whether this strategy can use the MegaMoE gate-pack fast path.

        The default strategy contract is "not supported"; Mega strategies
        override it after checking env/static model properties.
        """
        return False

    def forward_with_gate_pack(
        self,
        x: torch.Tensor,
        gate,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Optional fast path that fuses router gate + MegaMoE input packing."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support MegaMoE gate-pack"
        )

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
_MEGA_SE_AUTO_FALLBACK_WARNED = False


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
        opt-in via ``DSV4_MOE_STRATEGY=...``, ``DSV4_USE_MEGA_MOE_SE=1``, or
        ctor kwarg);
        ``False`` → silently fall through to auto-pick if can_handle is False
        (legacy ``DSV4_USE_MEGA_MOE=1`` / ``DSV4_USE_GROUPED_FP4=1`` toggles —
        historically a "use if applicable" hint, NOT a hard force; e.g.
        ``DSV4_USE_MEGA_MOE=1`` was commonly left on for ep_size=1 smokes
        where Mega is fundamentally incompatible).

    Precedence (highest first):
      1. ``DSV4_MOE_STRATEGY`` env var (if not "auto") — strict
      2. compatibility toggles (Mega-SE strict; historical toggles non-strict)
      3. ``strategy_arg`` ctor kwarg — strict

    Raises ``RuntimeError`` on conflicting toggles.
    """
    env = os.environ.get("DSV4_MOE_STRATEGY", "").strip()
    if env and env != "auto":
        return env, True

    use_mega = os.environ.get("DSV4_USE_MEGA_MOE")
    use_mega_se = os.environ.get("DSV4_USE_MEGA_MOE_SE")
    use_grouped = os.environ.get("DSV4_USE_GROUPED_FP4")
    if use_mega == "0" and use_mega_se == "1":
        raise RuntimeError(
            "DSV4_USE_MEGA_MOE=0 disables the Mega MoE family and conflicts "
            "with DSV4_USE_MEGA_MOE_SE=1."
        )

    legacy_pos: list[str] = []
    # Explicit Mega-SE=1 is strict. A generic Mega=1 hint is compatible with
    # it and is subsumed by the more specific selection.
    if use_mega_se == "1":
        legacy_pos.append("mega_se")
    elif use_mega == "1":
        legacy_pos.append("mega")
    if use_grouped == "1":
        legacy_pos.append("grouped_fp4")

    if len(legacy_pos) > 1:
        raise RuntimeError(
            f"Conflicting legacy MoE toggles (multiple positive): {legacy_pos}. "
            "Set at most one of DSV4_USE_MEGA_MOE_SE / DSV4_USE_MEGA_MOE / "
            "DSV4_USE_GROUPED_FP4 to 1, "
            "or use DSV4_MOE_STRATEGY=<name>."
        )
    compatible_ctor = legacy_pos == ["mega_se"] and strategy_arg in ("mega", "mega_se")
    if (
        legacy_pos
        and strategy_arg
        and strategy_arg not in legacy_pos
        and not compatible_ctor
    ):
        raise RuntimeError(
            f"Conflicting MoE strategy: ctor strategy={strategy_arg!r} but legacy "
            f"toggle forces {legacy_pos[0]!r}. Pick one source of truth."
        )
    if legacy_pos:
        # Unlike historical best-effort toggles, the SE switch exists to prove
        # that the fused path is active and therefore must fail if unavailable.
        return legacy_pos[0], legacy_pos[0] == "mega_se"
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
    global _MEGA_SE_AUTO_FALLBACK_WARNED

    explicit_env_value = os.environ.get("DSV4_MOE_STRATEGY", "").strip()
    explicit_env = bool(explicit_env_value and explicit_env_value != "auto")
    mega_disabled = os.environ.get("DSV4_USE_MEGA_MOE", "1") == "0"

    # A named DSV4_MOE_STRATEGY is authoritative. Otherwise resolve the
    # compatibility switches before probing kernels so contradictory requests
    # fail deterministically on every rank.
    if not explicit_env:
        from rtp_llm.models_py.modules.dsv4.moe.mega_fused_buf import (
            mega_moe_fused_requested,
        )
        from rtp_llm.models_py.modules.dsv4.moe.mega_se_buf import (
            _mega_moe_se_unavailable_reason,
            mega_moe_se_requested,
        )

        se_value = os.environ.get("DSV4_USE_MEGA_MOE_SE")
        se_is_explicit = se_value == "1"
        se_requested = mega_moe_se_requested()
        fused_requested = mega_moe_fused_requested()
        grouped_requested = os.environ.get("DSV4_USE_GROUPED_FP4") == "1"

        if mega_disabled and (se_is_explicit or fused_requested):
            requested = (
                "DSV4_USE_MEGA_MOE_SE=1"
                if se_is_explicit
                else "DSV4_USE_MEGA_MOE_FUSED=1"
            )
            raise RuntimeError(
                "DSV4_USE_MEGA_MOE=0 disables the Mega MoE family and "
                f"conflicts with {requested}."
            )

        if cfg.ep_size == 1:
            if fused_requested:
                raise RuntimeError(
                    "DSV4_USE_MEGA_MOE_FUSED=1 requires ep_size > 1; "
                    "single-rank MoE uses grouped FP4 or local-loop execution."
                )
        else:
            if grouped_requested:
                raise RuntimeError(
                    "DSV4_USE_GROUPED_FP4=1 is incompatible with ep_size > 1; "
                    "distributed MoE requires a Mega strategy."
                )
            if mega_disabled:
                # In particular, do not let the new default SE preference
                # override the existing whole-family disable switch.
                se_requested = False
            if se_requested and not se_is_explicit and forced is not None and strict:
                se_requested = False
            if se_requested and fused_requested:
                raise RuntimeError(
                    "DSV4_USE_MEGA_MOE_SE (enabled by default) conflicts with "
                    "DSV4_USE_MEGA_MOE_FUSED=1; set "
                    "DSV4_USE_MEGA_MOE_SE=0 to select the legacy fused path."
                )
            if se_requested:
                if forced not in (None, "mega", "mega_se"):
                    raise RuntimeError(
                        "MegaMoE-SE is incompatible with requested MoE strategy "
                        f"{forced!r}. Set DSV4_USE_MEGA_MOE_SE=0 or choose "
                        "DSV4_MOE_STRATEGY explicitly."
                    )
                if se_is_explicit or (forced == "mega_se" and strict):
                    # Explicit =1 or a strict named SE request must run SE.
                    forced, strict = "mega_se", True
                else:
                    # Unset means prefer SE on compatible runtimes, but retain
                    # the routed-only Mega path with a visible warning when the
                    # additional shared-expert API is unavailable.
                    se_cls = next(
                        (c for c in _STRATEGY_PRIORITY if c.name == "mega_se"),
                        None,
                    )
                    if se_cls is not None and se_cls.can_handle(cfg):
                        return se_cls
                    reason = (
                        _mega_moe_se_unavailable_reason()
                        if se_cls is not None
                        else "MegaMoEStrategySE is not registered"
                    )
                    if not _MEGA_SE_AUTO_FALLBACK_WARNED:
                        logging.warning(
                            "DSV4 MegaMoE-SE automatic default is unavailable "
                            "(%s); falling back to routed-only MegaMoE.",
                            reason or "unknown capability failure",
                        )
                        _MEGA_SE_AUTO_FALLBACK_WARNED = True

            # The legacy fused implementation is strict and only becomes
            # reachable after an explicit SE opt-out.
            if fused_requested:
                if forced not in (None, "mega", "mega_fused"):
                    raise RuntimeError(
                        "DSV4_USE_MEGA_MOE_FUSED=1 conflicts with requested MoE "
                        f"strategy {forced!r}."
                    )
                forced, strict = "mega_fused", True

    if forced in ("mega", "mega_se", "mega_fused") and mega_disabled:
        raise RuntimeError(
            "DSV4_USE_MEGA_MOE=0 disables the Mega MoE family and conflicts "
            f"with requested strategy {forced!r}."
        )

    if forced is None and cfg.ep_size > 1 and mega_disabled:
        raise RuntimeError(
            "DSV4_USE_MEGA_MOE=0 disables the Mega MoE family, but "
            "distributed MoE requires a Mega strategy."
        )

    if (
        forced in ("mega", "mega_se", "mega_fused")
        and cfg.ep_size == 1
        and strict
    ):
        raise RuntimeError(
            f"Forced MoE strategy {forced!r} requires ep_size > 1; "
            "single-rank MoE uses grouped FP4 or local-loop execution."
        )

    if forced is not None:
        for cls in _STRATEGY_PRIORITY:
            if cls.name == forced:
                if cls.can_handle(cfg):
                    if cfg.ep_size > 1 and cls.name not in (
                        "mega",
                        "mega_fused",
                        "mega_se",
                    ):
                        raise RuntimeError(
                            "DSV4 EP MoE requires MegaMoEStrategy. "
                            f"Requested strategy {forced!r} would bypass Mega "
                            f"(layer_id={cfg.layer_id}, ep_size={cfg.ep_size})."
                        )
                    return cls
                if strict:
                    reason = None
                    if forced == "mega_se":
                        from rtp_llm.models_py.modules.dsv4.moe.mega_se_buf import (
                            _mega_moe_se_disabled_or_unavailable_reason,
                        )

                        reason = _mega_moe_se_disabled_or_unavailable_reason()
                    elif forced == "mega":
                        from rtp_llm.models_py.modules.dsv4.moe.mega_buf import (
                            _mega_moe_disabled_or_unavailable_reason,
                        )

                        reason = _mega_moe_disabled_or_unavailable_reason()
                    elif forced == "mega_fused":
                        from rtp_llm.models_py.modules.dsv4.moe.mega_fused_buf import (
                            _mega_moe_fused_disabled_or_unavailable_reason,
                        )

                        reason = _mega_moe_fused_disabled_or_unavailable_reason()
                    detail = f" Reason: {reason}." if reason else ""
                    raise RuntimeError(
                        f"Forced MoE strategy {forced!r} cannot handle cfg "
                        f"(layer_id={cfg.layer_id}, ep_size={cfg.ep_size})."
                        f"{detail}"
                    )
                # Non-strict (legacy toggle) → fall through to auto-pick.
                break
        else:
            names = [c.name for c in _STRATEGY_PRIORITY]
            raise RuntimeError(f"Unknown MoE strategy {forced!r}. Available: {names}")

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
        if cls.name in ("mega", "mega_se", "mega_fused"):
            continue
        if cls.can_handle(cfg):
            return cls
    raise RuntimeError(
        f"No MoE strategy can handle cfg (layer_id={cfg.layer_id}, "
        f"ep_size={cfg.ep_size})"
    )
