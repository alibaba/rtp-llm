import gc
import os
import platform
from enum import IntEnum, auto
from typing import Optional, Tuple

import torch
from deep_ep import Buffer as DeepEPBuffer
from deep_ep import Config as DeepEPConfig
from torch.distributed import ProcessGroup

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops.compute_ops import DeviceType, get_device

__all__ = [
    "DeepEPBuffer",
    "DeepEPConfig",
    "init_deepep_wrapper",
    "get_deepep_wrapper",
    "destroy_deepep_wrapper",
]


def use_accl_ep():
    device_type = get_device().get_device_type()
    return not device_type == DeviceType.ROCm


def allow_mnnvl():
    is_sm_100 = torch.cuda.get_device_capability()[0] in [10]
    return "aarch64" in platform.machine() and is_sm_100


class DeepEPMode(IntEnum):
    """
    The mode of deep_ep.
    """

    NORMAL = auto()
    LOW_LATENCY = auto()
    LOW_LATENCY_M2N = auto()


class DeepEPWrapper:
    """
    A wrapper for deep_ep.
    """

    _buffer: Optional[DeepEPBuffer]
    _ep_rank: int = 0
    _ep_size: int = 0
    _hidden_size: int = 0
    _num_experts: int = 0
    _num_topk: int = 0
    _ll_num_max_token_per_rank: int = 0
    _config_ll_num_max_token_per_rank: int = 0
    _num_sms: int = 24
    _use_accl_ep: bool = True
    _mode: DeepEPMode = DeepEPMode.NORMAL

    def __init__(
        self,
        group: ProcessGroup,
        config_adapter: MoEConfigAdapter,
    ) -> None:
        """Initialize DeepEPWrapper with ProcessGroup and MoEConfigAdapter.

        Args:
            group: ProcessGroup for distributed communication
            config_adapter: MoEConfigAdapter containing all necessary configuration
        """
        # Extract configurations from MoEConfigAdapter
        model_config = config_adapter.model_config
        parallelism_config = config_adapter.parallelism_config
        moe_config = config_adapter.moe_config

        self._ep_rank = parallelism_config.ep_rank
        self._ep_size = parallelism_config.ep_size
        self._hidden_size = model_config.hidden_size
        self._num_experts = model_config.expert_num
        self._num_topk = model_config.moe_k
        self._num_sms = moe_config.deep_ep_num_sm
        self._use_accl_ep = use_accl_ep()
        self._model_config = model_config
        self._parallelism_config = parallelism_config
        self._moe_config = moe_config
        self._ffn_disaggregate_config = parallelism_config.ffn_disaggregate_config
        # Use provided ll_num_max_token_per_rank if available, otherwise calculate it
        self._config_ll_num_max_token_per_rank = (
            config_adapter.ll_num_max_token_per_rank
            if config_adapter.ll_num_max_token_per_rank > 0
            else config_adapter.max_generate_batch_size
        )
        self._mode, self._buffer = self._init_deepep_buffer(group)

    @property
    def buffer(self) -> DeepEPBuffer:
        assert self._buffer is not None, "deep_ep buffer is not initialized"
        return self._buffer

    @property
    def mode(self) -> DeepEPMode:
        return self._mode

    @property
    def ep_rank(self) -> int:
        return self._ep_rank

    @property
    def ep_size(self) -> int:
        return self._ep_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def num_topk(self) -> int:
        return self._num_topk

    @property
    def ll_num_max_token_per_rank(self) -> int:
        return self._ll_num_max_token_per_rank

    @property
    def num_sms(self) -> int:
        return self._num_sms

    @property
    def use_accl_ep(self) -> bool:
        return self._use_accl_ep

    def _init_deepep_buffer(
        self, group: ProcessGroup
    ) -> Tuple[DeepEPMode, DeepEPBuffer]:
        # init deep_ep buffer
        ep_rank = self._ep_rank
        use_deepep_low_latency: bool = self._moe_config.use_deepep_low_latency
        enable_ffn_disaggregate: bool = (
            self._ffn_disaggregate_config.enable_ffn_disaggregate
            if self._ffn_disaggregate_config
            else False
        )
        if use_deepep_low_latency and enable_ffn_disaggregate:
            if self._use_accl_ep:
                return DeepEPMode.LOW_LATENCY_M2N, self._init_low_latency_m2n_buffer(
                    group
                )
            else:
                raise RuntimeError(
                    f"[rank: {ep_rank}] init deep_ep buffer failed, current deep_ep provider "
                    f"does not support use_deepep_low_latency: {use_deepep_low_latency} "
                    f"and enable_ffn_disaggregate: {enable_ffn_disaggregate}"
                )
        elif use_deepep_low_latency and not enable_ffn_disaggregate:
            return DeepEPMode.LOW_LATENCY, self._init_low_latency_buffer(group)
        elif not use_deepep_low_latency and not enable_ffn_disaggregate:
            return DeepEPMode.NORMAL, self._init_normal_buffer(group)
        else:
            raise RuntimeError(
                f"[rank: {ep_rank}] init deep_ep buffer failed, unsupported "
                f"use_deepep_low_latency: {use_deepep_low_latency} and "
                f"enable_ffn_disaggregate: {enable_ffn_disaggregate}"
            )

    def _calc_low_latency_max_token_per_rank(
        self, ll_num_max_token_per_rank: int, tp_size: int
    ) -> int:
        ll_num_max_token_per_rank = (ll_num_max_token_per_rank + tp_size - 1) // tp_size

        matched_tokens = [
            16,
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
        ]
        if ll_num_max_token_per_rank > 128:
            ll_num_max_token_per_rank = ((ll_num_max_token_per_rank + 127) // 128) * 128
            return ll_num_max_token_per_rank
        for t in matched_tokens:
            if ll_num_max_token_per_rank <= t:
                ll_num_max_token_per_rank = t
                return ll_num_max_token_per_rank
        return 128

    def _init_normal_buffer(self, group: ProcessGroup) -> DeepEPBuffer:
        num_nvl_bytes = 0
        num_rdma_bytes = 0
        num_qps_per_rank = 1
        use_deepep_internode: bool = self._moe_config.use_deepep_internode
        num_experts: int = self._num_experts
        ep_size: int = self._ep_size
        assert num_experts > 0 and ep_size > 0, "num_experts and ep_size must be set"
        # normal-kernel internode
        if use_deepep_internode:
            num_nvl_bytes = int(2e9)
            num_rdma_bytes = int(1e9)
            # normal ibgda
            if os.environ.get("ACCL_NORMAL_MODE", "IBRC") == "IBGDA":
                os.environ["ACCL_NORMAL_MODE"] = "IBGDA"
                num_qps_per_rank = max(self._num_sms // 2, (int)(num_experts / ep_size))
            # normal ibrc
            else:
                os.environ["ACCL_NORMAL_MODE"] = "IBRC"
                num_qps_per_rank = self._num_sms // 2
        # normal-kernel intranode
        else:
            num_nvl_bytes = int(2e9)
            num_qps_per_rank = 1
        init_kwargs = {
            "group": group,
            "num_nvl_bytes": num_nvl_bytes,
            "num_rdma_bytes": num_rdma_bytes,
            "low_latency_mode": False,
            "num_qps_per_rank": num_qps_per_rank,
        }
        if self._use_accl_ep:
            init_kwargs["allow_nvlink_for_low_latency_mode"] = True
            if allow_mnnvl():
                init_kwargs["allow_mnnvl"] = True
                init_kwargs["use_fabric"] = True
            else:
                init_kwargs["allow_mnnvl"] = False
        return DeepEPBuffer(**init_kwargs)  # type: ignore

    def _init_low_latency_buffer(self, group: ProcessGroup) -> DeepEPBuffer:
        ll_num_max_token_per_rank: int = self._config_ll_num_max_token_per_rank
        tp_size: int = self._parallelism_config.tp_size
        assert (
            ll_num_max_token_per_rank > 0 and tp_size > 0
        ), "ll_num_max_token_per_rank and tp_size must be set"
        ll_num_max_token_per_rank = self._calc_low_latency_max_token_per_rank(
            ll_num_max_token_per_rank, tp_size
        )
        self._ll_num_max_token_per_rank = ll_num_max_token_per_rank

        num_nvl_bytes = 0
        num_rdma_bytes = 0
        num_qps_per_rank = 1
        hidden_size: int = self._hidden_size
        ep_size: int = self._ep_size
        num_experts: int = self._num_experts
        assert (
            hidden_size > 0 and ep_size > 0 and num_experts > 0
        ), "hidden_size, ep_size and num_experts must be set"
        num_rdma_bytes = DeepEPBuffer.get_low_latency_rdma_size_hint(
            ll_num_max_token_per_rank,
            hidden_size,
            ep_size,
            num_experts,
        )
        local_rank: int = self._parallelism_config.local_rank
        if local_rank == 0:
            print(
                f"Allocating buffer size: {num_rdma_bytes / 1e6} MB, "
                f"ll_num_max_token_per_rank: {ll_num_max_token_per_rank}, "
                f"hidden_size: {hidden_size}, "
                f"ep_size: {ep_size}, "
                f"num_experts: {num_experts}",
                flush=True,
            )
        num_qps_per_rank = num_experts / ep_size

        init_kwargs = {
            "group": group,
            "num_nvl_bytes": num_nvl_bytes,
            "num_rdma_bytes": num_rdma_bytes,
            "low_latency_mode": True,
            "num_qps_per_rank": num_qps_per_rank,
            "allow_mnnvl": True,
        }
        if self._use_accl_ep:
            os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
            init_kwargs["allow_nvlink_for_low_latency_mode"] = True
            if allow_mnnvl():
                init_kwargs["allow_mnnvl"] = True
            else:
                init_kwargs["allow_mnnvl"] = False
        return DeepEPBuffer(**init_kwargs)  # type: ignore

    def _init_low_latency_m2n_buffer(self, group: ProcessGroup) -> DeepEPBuffer:
        if self._ffn_disaggregate_config is None:
            raise RuntimeError(
                "ffn_disaggregate_config is required for low-latency m2n mode"
            )
        ll_num_max_token_per_rank: int = self._config_ll_num_max_token_per_rank
        attention_tp_size: int = self._ffn_disaggregate_config.attention_tp_size
        assert (
            ll_num_max_token_per_rank > 0 and tp_size > 0
        ), "ll_num_max_token_per_rank and tp_size must be set"
        ll_num_max_token_per_rank = self._calc_low_latency_max_token_per_rank(
            ll_num_max_token_per_rank, attention_tp_size
        )

        attention_dp_size: int = self._ffn_disaggregate_config.attention_dp_size
        ffn_dp_size: int = self._ffn_disaggregate_config.ffn_dp_size
        ffn_tp_size: int = self._ffn_disaggregate_config.ffn_tp_size
        assert (
            attention_dp_size > 0 and ffn_dp_size > 0 and ffn_tp_size > 0
        ), "attention_dp_size, ffn_dp_size and ffn_tp_size must be set"
        num_m = attention_dp_size * attention_tp_size
        num_n = ffn_dp_size * ffn_tp_size

        num_nvl_bytes = 0
        num_rdma_bytes = 0
        num_qps_per_rank = 1
        if not hasattr(DeepEPBuffer, "get_low_latency_rdma_size_hint_m2n"):
            raise RuntimeError(
                "current deep_ep provider does not support low-latency m2n"
            )
        hidden_size: int = self._hidden_size
        num_experts: int = self._num_experts
        assert (
            hidden_size > 0 and num_experts > 0
        ), "hidden_size and num_experts must be set"
        num_rdma_bytes = DeepEPBuffer.get_low_latency_rdma_size_hint_m2n(
            ll_num_max_token_per_rank,
            hidden_size,
            num_m + num_n,
            num_experts,
            num_m,
        )
        local_rank: int = self._parallelism_config.local_rank
        if local_rank == 0:
            print(
                f"Allocating buffer size: {num_rdma_bytes / 1e6} MB, "
                f"ll_num_max_token_per_rank: {ll_num_max_token_per_rank}, "
                f"hidden_size: {hidden_size}, "
                f"expert_num: {num_experts}, "
                f"num_m: {num_m}, "
                f"num_n: {num_n}",
                flush=True,
            )
        num_qps_per_rank = num_experts / num_n

        init_kwargs = {
            "group": group,
            "num_nvl_bytes": num_nvl_bytes,
            "num_rdma_bytes": num_rdma_bytes,
            "low_latency_mode": True,
            "num_qps_per_rank": num_qps_per_rank,
        }
        if self._use_accl_ep:
            init_kwargs["allow_nvlink_for_low_latency_mode"] = True
            init_kwargs["allow_mnnvl"] = False
        return DeepEPBuffer(**init_kwargs)  # type: ignore

    def destroy_deepep_buffer(self) -> None:
        if self._buffer is not None:
            del self._buffer
            self._buffer = None
        gc.collect()


_DEEP_EP: Optional[DeepEPWrapper] = None


def get_deepep_wrapper() -> DeepEPWrapper:
    assert _DEEP_EP is not None, "deep_ep wrapper is not initialized"
    return _DEEP_EP


def init_deepep_wrapper(
    group: ProcessGroup,
    config_adapter: MoEConfigAdapter,
) -> None:
    """Initialize DeepEP wrapper with ProcessGroup and MoEConfigAdapter.

    Args:
        group: ProcessGroup for distributed communication
        config_adapter: MoEConfigAdapter containing all necessary configuration
    """
    global _DEEP_EP
    _DEEP_EP = DeepEPWrapper(
        group, config_adapter
    )  # pyright: ignore[reportConstantRedefinition]


def destroy_deepep_wrapper() -> None:
    global _DEEP_EP
    if _DEEP_EP:
        _DEEP_EP.destroy_deepep_buffer()
    _DEEP_EP = None  # pyright: ignore[reportConstantRedefinition]
    gc.collect()
