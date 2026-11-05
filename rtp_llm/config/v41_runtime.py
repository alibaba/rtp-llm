"""V4.1 PPU release constraints, independent of torch and compiled bindings."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Mapping

PPU_STRATEGY = "bf16_dense_w4a16_moe_sparse_mla_host_enp_v1"
CONFIG_SHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"


def enum_name(value) -> str:
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].upper()


def env_bool(env: Mapping[str, str], name: str, default=False) -> bool:
    value = env.get(name, str(int(default))).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def select_v41_backend(device=None, env=None) -> str:
    env = os.environ if env is None else env
    if device is None:
        from rtp_llm.device.device_type import get_device_type

        device = get_device_type()
    device = enum_name(device)
    requested = env.get("RTP_V41_BACKEND", "auto").lower()
    if requested not in ("auto", "ppu", "cuda", "reference"):
        raise ValueError(f"unsupported RTP_V41_BACKEND={requested!r}")
    if requested == "reference":
        return requested
    actual = {"PPU": "ppu", "CUDA": "cuda"}.get(device)
    if actual is None or requested not in ("auto", actual):
        raise RuntimeError(
            f"V4.1 backend {requested!r} cannot run on {device}; "
            "select reference explicitly for diagnostic execution"
        )
    return actual


def v41_backend_for_config():
    """Permit checkpoint metadata inspection on a CPU development machine."""
    from rtp_llm.device.device_type import get_device_type

    device = get_device_type()
    if (
        enum_name(device) == "CPU"
        and os.environ.get("RTP_V41_BACKEND", "auto") == "auto"
    ):
        return "cuda"
    return select_v41_backend(device)


@dataclass(frozen=True)
class V41RuntimeConfig:
    role: str
    world_size: int
    physical_tp: int
    attention_dp: int
    ep: int
    enp: int
    cp_method: str
    source_cp: int
    max_seq_len: int
    kv_sharded: bool = False
    cuda_graph: bool = False
    speculative: str = "NONE"
    eplb: str = "NONE"
    low_latency: bool = False
    ced: bool = False
    bounded_replay: bool = False
    encoder_replay: bool = False
    redundant_experts: int = 0
    normal_ep: bool = True
    all_gather_moe: bool = False
    strategy: str = PPU_STRATEGY
    moe_backend: str = "grouped"
    moe_precision: str = "w4a8"

    @property
    def attention_cp(self):
        return self.physical_tp if self.role == "PREFILL" else 1

    @property
    def attention_tp(self):
        return 1

    def validate(self):
        expected = {
            "PREFILL": ((8, 1, "ALL_GATHER", 8), (1, 8, "DISABLED", 1)),
            "DECODE": ((1, 8, "PREFILL_CP", 8), (1, 8, "DISABLED", 1)),
        }
        errors = []
        if self.role not in expected:
            errors.append("role must be PREFILL or DECODE")
        elif (
            self.physical_tp,
            self.attention_dp,
            self.cp_method,
            self.source_cp,
        ) not in expected[self.role]:
            errors.append(
                f"{self.role} requires (tp, dp, cp_method, source_cp) in "
                f"{expected[self.role]}"
            )
        if (self.world_size, self.ep, self.enp) != (8, 8, 8):
            errors.append("world_size=EP=ENP=8 (separate P/D worlds)")
        if self.max_seq_len <= 0:
            errors.append("max_seq_len must be positive and cover input + output")
        if self.kv_sharded or self.low_latency:
            errors.append("KV sharding and low-latency EP must be disabled")
        if self.bounded_replay and not self.ced:
            errors.append("DSV41_SWA_BOUNDED_REPLAY requires DSV41_CED=1")
        if self.encoder_replay and not self.bounded_replay:
            errors.append("encoder SWA recovery requires decoder bounded replay")
        if self.cuda_graph and (
            self.role != "DECODE"
            or self.moe_backend != "megamoe"
            or self.moe_precision != "w4a4"
        ):
            errors.append("PPU Graph requires DP8 DECODE with MegaMoE/W4A4")
        if self.speculative != "NONE" or self.eplb != "NONE":
            errors.append("speculative execution and dynamic EPLB must be NONE")
        if self.redundant_experts != 0 or not self.normal_ep or self.all_gather_moe:
            errors.append(
                "normal DeepEP without redundant experts or all-gather MoE is required"
            )
        if self.strategy != PPU_STRATEGY:
            errors.append(f"release strategy must be {PPU_STRATEGY}")
        if errors:
            raise ValueError("V4.1 PPU configuration: " + "; ".join(errors))
        return self

    def summary(self):
        return dict(
            asdict(self),
            attention_tp=1,
            attention_cp=self.attention_cp,
            local_experts=48,
            model_top_k=6,
            communication_slots=8,
        )

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        from rtp_llm.models_py.modules.dsv41.ppu.moe_precision import (
            moe_backend,
            moe_precision,
        )

        return cls(
            role=env.get("ROLE_TYPE", "").upper(),
            world_size=int(env.get("WORLD_SIZE", "1")),
            physical_tp=int(env.get("TP_SIZE", "1")),
            attention_dp=int(env.get("DP_SIZE", "1")),
            ep=int(env.get("EP_SIZE", "0")),
            enp=int(env.get("RTP_V41_ENP_SIZE", "8")),
            cp_method=env.get("CP_ROTATE_METHOD", "DISABLED").upper(),
            source_cp=int(env.get("PREFILL_CP_SIZE", "0")),
            max_seq_len=int(env.get("MAX_SEQ_LEN", "32768")),
            kv_sharded=env_bool(env, "PREFILL_CP_KV_CACHE_SHARDED"),
            cuda_graph=env_bool(env, "ENABLE_CUDA_GRAPH"),
            speculative=(env.get("SP_TYPE") or "NONE").upper(),
            eplb=env.get("EPLB_MODE", "NONE").upper(),
            low_latency=env_bool(env, "USE_DEEPEP_LOW_LATENCY"),
            ced=env_bool(env, "DSV41_CED"),
            bounded_replay=env_bool(env, "DSV41_SWA_BOUNDED_REPLAY"),
            encoder_replay=env_bool(env, "DSV41_ENCODER_SWA_REPLAY"),
            redundant_experts=int(env.get("REDUNDANT_EXPERT", "0")),
            normal_ep=env_bool(env, "USE_DEEPEP_MOE", default=True),
            all_gather_moe=env_bool(env, "USE_ALL_GATHER"),
            strategy=env.get("RTP_V41_PPU_STRATEGY", PPU_STRATEGY),
            moe_backend=moe_backend(env),
            moe_precision=moe_precision(env),
        ).validate()

    @classmethod
    def from_server(cls, config):
        from rtp_llm.models_py.modules.dsv41.ppu.moe_precision import (
            moe_backend,
            moe_precision,
        )

        p = config.parallelism_config
        cp = p.prefill_cp_config
        env = os.environ
        if config.model_args.max_seq_len is None:
            raise ValueError("V4.1 PPU requires explicit positive --max_seq_len")
        if p.local_world_size != 8:
            raise ValueError("V4.1 PPU requires local_world_size=8 for each role")
        return cls(
            role=enum_name(config.role_config.role_type),
            world_size=p.world_size,
            physical_tp=p.tp_size,
            attention_dp=p.dp_size,
            ep=p.ep_size,
            enp=int(env.get("RTP_V41_ENP_SIZE", "8")),
            cp_method=enum_name(cp.method),
            source_cp=cp.prefill_cp_size,
            max_seq_len=int(config.model_args.max_seq_len),
            kv_sharded=cp.kv_cache_sharded,
            cuda_graph=config.py_hw_kernel_config.enable_cuda_graph,
            speculative=enum_name(config.sp_config.type),
            eplb=enum_name(config.eplb_config.eplb_mode),
            low_latency=config.moe_config.use_deepep_low_latency,
            ced=env_bool(env, "DSV41_CED"),
            bounded_replay=env_bool(env, "DSV41_SWA_BOUNDED_REPLAY"),
            encoder_replay=env_bool(env, "DSV41_ENCODER_SWA_REPLAY"),
            redundant_experts=config.eplb_config.redundant_expert,
            normal_ep=config.moe_config.use_deepep_moe,
            all_gather_moe=config.moe_config.use_all_gather,
            strategy=env.get("RTP_V41_PPU_STRATEGY", PPU_STRATEGY),
            moe_backend=moe_backend(env),
            moe_precision=moe_precision(env),
        ).validate()


def validate_v41_server_config(config):
    model_type = config.model_args.model_type
    if not model_type or not model_type.startswith("deepseek_v41"):
        return
    if v41_backend_for_config() != "ppu":
        return
    if model_type != "deepseek_v41":
        raise ValueError(
            "V4.1 PPU release supports the text target model only; DSpark is disabled"
        )
    runtime = V41RuntimeConfig.from_server(config)
    logging.info("V4.1 actual topology and strategies: %s", runtime.summary())
    from rtp_llm.models_py.modules.dsv41.backend import get_v41_backend

    get_v41_backend("ppu").check_runtime(runtime)
