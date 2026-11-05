"""Startup-only PPU prefill/cache semantics, without accelerator imports."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

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
