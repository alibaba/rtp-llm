import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)

__all__ = [
    "MegaMoeWrapperConfig",
    "MegaMoeWrapper",
    "init_megamoe_wrapper",
    "init_megamoe_wrapper_from_config",
]


def _next_pow2(n: int) -> int:
    """Round n up to the next power of two (FlyDSL requires pow2 max_tok_per_rank)."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


@dataclass(eq=True)
class MegaMoeWrapperConfig:
    """Config for the FlyDSL 2-stage fused MegaMoE op (FusedMoEZeroCopyFp8)."""

    rank: int
    world_size: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    max_tok_per_rank: int
    stage1_tile_m: int = 64
    stage2_tile_m: int = 32
    tile_n: int = 128
    tile_k: int = 256
    stage2_tile_n: int = 256
    stage2_tile_k: int = 128

    @classmethod
    def from_config_adapter(
        cls, config_adapter: MoEConfigAdapter
    ) -> "MegaMoeWrapperConfig":
        """Create MegaMoeWrapperConfig from MoEConfigAdapter."""
        model_config = config_adapter.model_config
        parallelism_config = config_adapter.parallelism_config
        moe_config = config_adapter.moe_config

        assert parallelism_config.ep_size == parallelism_config.world_size, (
            f"MegaMoE currently requires ep_size == world_size, "
            f"got ep_size={parallelism_config.ep_size}, "
            f"world_size={parallelism_config.world_size}"
        )

        experts = model_config.expert_num
        world_size = parallelism_config.ep_size
        assert experts % world_size == 0, (
            f"MegaMoE requires experts % world_size == 0, "
            f"got experts={experts}, world_size={world_size}"
        )
        topk = model_config.moe_k
        assert topk <= world_size, (
            f"MegaMoE requires topk <= world_size, "
            f"got topk={topk}, world_size={world_size}"
        )

        # max tokens a single rank can see; FlyDSL requires a power-of-two cap.
        # FlyDSL JIT compile time scales strongly with this value, so we cap it
        # (ll_num_max_token=4096 makes stage1 compilation impractically slow).
        # Override with MEGAMOE_MAX_TOK; default 512 (matches fast smoke-test shape).
        max_tok_cap = _next_pow2(int(os.environ.get("MEGAMOE_MAX_TOK", "512")))
        max_tokens = min(_next_pow2(int(moe_config.ll_num_max_token)), max_tok_cap)

        return cls(
            rank=parallelism_config.ep_rank,
            world_size=world_size,
            model_dim=model_config.hidden_size,
            inter_dim=model_config.moe_inter_size,
            experts=experts,
            topk=topk,
            max_tok_per_rank=max_tokens,
        )

    def to_op_kwargs(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "model_dim": self.model_dim,
            "inter_dim": self.inter_dim,
            "experts": self.experts,
            "topk": self.topk,
            "max_tok_per_rank": self.max_tok_per_rank,
            "stage1_tile_m": self.stage1_tile_m,
            "stage2_tile_m": self.stage2_tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "stage2_tile_n": self.stage2_tile_n,
            "stage2_tile_k": self.stage2_tile_k,
        }


class MegaMoeWrapper:
    """Singleton wrapper around the FlyDSL FusedMoEZeroCopyFp8 op."""

    _instance: Optional["MegaMoeWrapper"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __init__(self, config: MegaMoeWrapperConfig) -> None:
        self._config = config
        self._op = None
        self._init_op()

    @classmethod
    def supported(cls) -> bool:
        try:
            from flydsl.kernels.moe_fused_chained_fp8 import (  # noqa: F401
                FusedMoEZeroCopyFp8,
            )

            return True
        except ImportError:
            try:
                # Fallback: FlyDSL checked out as a sibling repo (kernels/ top-level).
                from kernels.moe_fused_chained_fp8 import (  # noqa: F401
                    FusedMoEZeroCopyFp8,
                )

                return True
            except ImportError:
                return False

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._initialized

    @classmethod
    def get_instance(cls) -> Optional["MegaMoeWrapper"]:
        """Return the initialized singleton, or None if not yet initialized."""
        with cls._lock:
            return cls._instance if cls._initialized else None

    @classmethod
    def _create(cls, config: MegaMoeWrapperConfig) -> "MegaMoeWrapper":
        """Internal creation, only called by init_megamoe_wrapper."""
        with cls._lock:
            if cls._initialized:
                if cls._instance is None:
                    raise RuntimeError("MegaMoE state is inconsistent")
                if cls._instance._config != config:
                    raise RuntimeError(
                        f"MegaMoE already initialized with different config, "
                        f"origin: {cls._instance._config}, new: {config}"
                    )
                return cls._instance

            if not cls.supported():
                raise RuntimeError(
                    "FlyDSL MegaMoE is not supported in this environment"
                )
            if not torch.distributed.is_initialized():
                raise RuntimeError("Distributed environment is not initialized")

            cls._instance = cls(config)
            cls._initialized = True
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance._op = None
                cls._instance = None
            cls._initialized = False

    @property
    def config(self) -> MegaMoeWrapperConfig:
        return self._config

    @property
    def op(self) -> Any:
        if self._op is None:
            raise RuntimeError("FlyDSL FusedMoEZeroCopyFp8 is not initialized")
        return self._op

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.op.forward(*args, **kwargs)

    def _init_op(self) -> None:
        try:
            from flydsl.kernels.moe_fused_chained_fp8 import FusedMoEZeroCopyFp8
        except ImportError:
            from kernels.moe_fused_chained_fp8 import FusedMoEZeroCopyFp8

        self._op = FusedMoEZeroCopyFp8(**self._config.to_op_kwargs())


def _mori_flydsl_link_fixup() -> None:
    """Link mori device functions into FlyDSL (required before any MoE kernel).

    FlyDSL's fused MoE kernels call mori shmem device functions; those symbols
    must be wrapped via link_extern before the first kernel launch. This mirrors
    the ``_mori_flydsl_link_fixup`` helper duplicated across FlyDSL's tests.
    Idempotent: guarded by ``mori.ir.flydsl._link_fixup_applied``.
    """
    import mori.ir.flydsl as mif

    if getattr(mif, "_link_fixup_applied", False):
        return

    import mori.shmem as ms
    from flydsl.compiler.extern_link import link_extern
    from flydsl.expr.extern import ffi
    from mori.ir.flydsl.runtime import get_bitcode_path
    from mori.ir.ops import MORI_DEVICE_FUNCTIONS

    bc = get_bitcode_path()

    def _init(hip_module: int) -> None:
        ms.shmem_module_init(hip_module)

    for name, meta in MORI_DEVICE_FUNCTIONS.items():
        wrapped = link_extern(
            ffi(
                meta["symbol"],
                meta["args"],
                meta["ret"],
                is_pure=meta.get("pure", False),
            ),
            bitcode_path=bc,
            module_init_fn=_init,
        )
        setattr(mif, name, wrapped)
    mif._link_fixup_applied = True
    logging.info("MegaMoE: mori<->FlyDSL device function link fixup applied")


def _init_shmem(shmem_group_name: str) -> None:
    """Register the WORLD process group + init mori shmem (FlyDSL uses mori.shmem)."""
    dist_world_size = torch.distributed.get_world_size()
    world_group = torch.distributed.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group(shmem_group_name, world_group)

    import mori.shmem

    mori.shmem.shmem_torch_process_group_init(shmem_group_name)
    # FlyDSL MoE kernels require the mori device-function link fixup up-front.
    _mori_flydsl_link_fixup()
    logging.info(
        "MegaMoE shmem initialized on group '%s' (world_size=%d)",
        shmem_group_name,
        dist_world_size,
    )


def init_megamoe_wrapper_from_config(
    megamoe_config: MegaMoeWrapperConfig,
    shmem_group_name: str = "default",
) -> None:
    """Initialize MegaMoe wrapper from a MegaMoeWrapperConfig.

    Must be called after torch.distributed.init_process_group().
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed environment is not initialized. "
            "Call torch.distributed.init_process_group() first."
        )

    dist_world_size = torch.distributed.get_world_size()
    assert megamoe_config.world_size == dist_world_size, (
        f"MegaMoE config world_size ({megamoe_config.world_size}) must equal "
        f"distributed world_size ({dist_world_size}) when using WORLD group for shmem."
    )

    if not MegaMoeWrapper.is_initialized():
        _init_shmem(shmem_group_name)

    logging.info("Start initialize MegaMoE wrapper (from_config)")
    MegaMoeWrapper._create(megamoe_config)
    logging.info("Finish initialize MegaMoE wrapper (from_config)")


def init_megamoe_wrapper(
    engine_config: EngineConfig,
    model_config: ModelConfig,
    shmem_group_name: str = "default",
) -> None:
    """Register shmem process group + create MegaMoeWrapper singleton.

    Must be called after torch.distributed.init_process_group().
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed environment is not initialized. "
            "Call torch.distributed.init_process_group() first."
        )

    ep_size = engine_config.parallelism_config.ep_size
    world_size = engine_config.parallelism_config.world_size
    assert ep_size == world_size, (
        f"MegaMoE currently requires ep_size == world_size, "
        f"got ep_size={ep_size}, world_size={world_size}. "
        f"Using WORLD group for shmem is invalid when EP is a subset of ranks."
    )

    if not MegaMoeWrapper.is_initialized():
        _init_shmem(shmem_group_name)

    enable_cuda_graph = (
        engine_config.hw_kernel_config.enable_cuda_graph
        if engine_config.hw_kernel_config is not None
        else False
    )

    config_adapter = MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=engine_config.parallelism_config,
        moe_config=engine_config.moe_config,
        quant_config=model_config.quant_config,
        enable_cuda_graph=enable_cuda_graph,
    )
    megamoe_config = MegaMoeWrapperConfig.from_config_adapter(config_adapter)
    logging.info("Start initialize MegaMoE wrapper")
    MegaMoeWrapper._create(megamoe_config)
    logging.info("Finish initialize MegaMoE wrapper")
