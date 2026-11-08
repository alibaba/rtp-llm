"""Startup-only PPU prefill/cache semantics, without accelerator imports."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from rtp_llm.models_py.modules.dsv41.ppu.shared_dspark_modes import SharedDSparkMode

_CONFIG_KEY = "_ppu_prefill_policy"


def _flag(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"invalid PPU replay boolean: {value!r}")


@dataclass(frozen=True)
class PrefillPolicy:
    ced: bool = False
    bounded_replay: bool = False
    encoder_replay: bool = False

    def validate(self):
        if self.bounded_replay and not self.ced:
            raise ValueError("DSV41_SWA_BOUNDED_REPLAY requires DSV41_CED=1")
        if self.encoder_replay and not self.bounded_replay:
            raise ValueError("encoder SWA recovery requires decoder bounded replay")
        return self

    @classmethod
    def from_config(cls, config=None, env=None):
        values = (config or {}).get(_CONFIG_KEY)
        if values is not None:
            return cls(
                **{key: _flag(value) for key, value in values.items()}
            ).validate()
        env = os.environ if env is None else env
        return cls(
            ced=_flag(env.get("DSV41_CED", "0")),
            bounded_replay=_flag(env.get("DSV41_SWA_BOUNDED_REPLAY", "0")),
            encoder_replay=_flag(env.get("DSV41_ENCODER_SWA_REPLAY", "0")),
        ).validate()

    def freeze(self, config):
        return dict(config, **{_CONFIG_KEY: asdict(self)})

    def decoder_start(self, geometry):
        return max(geometry.kv_source_layer_ids, default=-1) + 1

    def framework_tag(self, owner, tag, geometry):
        if (
            tag == "swa_kv"
            and self.bounded_replay
            and int(owner) >= self.decoder_start(geometry)
        ):
            return "decoder_swa_kv"
        return {
            "global_kv_1": "hca_kv",
            "global_kv_2": "csa_kv",
        }.get(tag, tag)


@dataclass(frozen=True)
class SharedPrefillPolicy:
    """Explicit shared runtime identity; legacy PrefillPolicy stays unchanged."""

    mode: str = "full"
    keep_tokens: int = 140
    window: int = 128
    max_rollback_tokens: int = 0
    encoder_replay: bool = False

    def validate(self):
        if self.mode not in ("full", "exact", "compact"):
            raise ValueError("shared Prefill mode must be full, exact or compact")
        if self.keep_tokens != 140 or self.window != 128:
            raise ValueError("shared PPU profile requires keep_tokens140/window128")
        if not 0 <= self.max_rollback_tokens <= self.keep_tokens - self.window:
            raise ValueError("keep_tokens must cover window + maximum rollback")
        if self.encoder_replay and self.mode != "compact":
            raise ValueError("encoder recovery requires compact replay")
        return self

    @classmethod
    def from_config(cls, config=None, env=None):
        frozen = (config or {}).get("_ppu_shared_prefill_policy")
        if frozen is not None:
            policy = cls(**frozen).validate()
            mode = SharedDSparkMode.from_config(config, env)
            if mode is not None and policy.max_rollback_tokens != mode.max_verified_rows:
                raise ValueError("frozen profile disagrees with actual maximum verified rows")
            return policy
        env = os.environ if env is None else env
        base = PrefillPolicy.from_config(config, env)
        # Dense target verification consumes gamma+1 INPUT rows; cancellation
        # may accept zero (MtpExecutor::previewSpeculativeCommitRows). B covers
        # this whole interval, independent of the proposal sampling budget.
        mode = SharedDSparkMode.from_config(config, env)
        rollback = mode.max_verified_rows if mode is not None else 0
        configured = int(env.get("RTP_V41_PPU_MAX_ROLLBACK_TOKENS", str(rollback)))
        if configured != rollback:
            raise ValueError(f"shared runtime consumes actual maximum rollback B={rollback}, got {configured}")
        return cls(
            mode="compact" if base.bounded_replay else "exact" if base.ced else "full",
            keep_tokens=int(env.get("RTP_V41_PPU_YOCO_KEEP_TOKENS", "140")),
            max_rollback_tokens=configured,
            encoder_replay=base.encoder_replay,
        ).validate()

    def freeze(self, config):
        mode = SharedDSparkMode.from_config(config)
        config = mode.freeze(config) if mode is not None else config
        return dict(config, _ppu_shared_prefill_policy=asdict(self))
