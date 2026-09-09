"""GLM53 Prefill-only sequence parallel and dedicated MLA CP selection."""

import os
from typing import Any

from rtp_llm.ops import RoleType


def _enabled(name: str) -> bool:
    value = os.environ.get(name, "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def mla_cp_enabled(model_type: str, role_type: Any) -> bool:
    return bool(
        model_type == "glm5_3_flash"
        and role_type == RoleType.PREFILL
        and _enabled("GLM53_PREFILL_MLA_CP")
    )


def shared_expert_local_enabled(model_type: str, role_type: Any) -> bool:
    """One selection shared by checkpoint loading and MoE execution."""
    return bool(
        model_type == "glm5_3_flash"
        and role_type == RoleType.PREFILL
        and _enabled("GLM53_PREFILL_SHARED_EXPERT_LOCAL")
    )


def sequence_parallel_enabled(model_type: str, parallelism: Any) -> bool:
    return bool(
        model_type == "glm5_3_flash"
        and parallelism.role_type == RoleType.PREFILL
        and parallelism.tp_size > 1
        and (
            _enabled("GLM53_PREFILL_SEQUENCE_PARALLEL")
            or mla_cp_enabled(model_type, parallelism.role_type)
        )
    )


class _SequenceCPView:
    def __init__(self, config):
        self._config = config
        if not config.kv_cache_sharded:
            raise ValueError("GLM53 MLA CP requires PREFILL_CP_KV_CACHE_SHARDED=true")

    def is_enabled(self):
        return True

    def __getattr__(self, name):
        return getattr(self._config, name)


class MlaCPParallelismView:
    """Use full MLA heads while retaining the real TP group and page-RR geometry."""

    def __init__(self, parallelism):
        if parallelism.tp_size <= 1:
            raise ValueError("GLM53 MLA CP requires TP size > 1")
        if parallelism.prefill_cp_config.is_enabled():
            raise ValueError(
                "GLM53 dedicated MLA CP requires cp_rotate_method=DISABLED for the surrounding model"
            )
        self._parallelism = parallelism
        self.prefill_cp_config = _SequenceCPView(parallelism.prefill_cp_config)

    def get_attn_tp_size(self):
        return 1

    def get_attn_tp_rank(self):
        return 0

    def __getattr__(self, name):
        return getattr(self._parallelism, name)
