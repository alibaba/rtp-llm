import threading
from dataclasses import dataclass
from typing import Any, Optional

import mori
import torch

__all__ = [
    "MoriEPWrapperConfig",
    "MoriEPWrapper",
    "init_moriep_wrapper",
]


@dataclass(eq=True)
class MoriEPWrapperConfig:
    """Config for MORI EP dispatch/combine wrapper."""

    data_type: Any
    rank: int
    world_size: int
    hidden_dim: int
    scale_dim: int
    scale_type_size: int
    max_token_type_size: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    warp_num_per_block: int = 16
    block_num: int = 80
    use_external_inp_buf: bool = True
    kernel_type: mori.ops.EpDispatchCombineKernelType = (
        mori.ops.EpDispatchCombineKernelType.IntraNode
    )
    gpu_per_node: Optional[int] = None
    rdma_block_num: int = 0
    num_qp_per_pe: int = 1
    quant_type: str = "none"

    def to_mori_kwargs(self) -> dict[str, Any]:
        return {
            "data_type": self.data_type,
            "rank": self.rank,
            "world_size": self.world_size,
            "hidden_dim": self.hidden_dim,
            "scale_dim": self.scale_dim,
            "scale_type_size": self.scale_type_size,
            "max_token_type_size": self.max_token_type_size,
            "max_num_inp_token_per_rank": self.max_num_inp_token_per_rank,
            "num_experts_per_rank": self.num_experts_per_rank,
            "num_experts_per_token": self.num_experts_per_token,
            "warp_num_per_block": self.warp_num_per_block,
            "block_num": self.block_num,
            "use_external_inp_buf": self.use_external_inp_buf,
            "kernel_type": self.kernel_type,
            "gpu_per_node": (
                self.world_size if self.gpu_per_node is None else self.gpu_per_node
            ),
            "rdma_block_num": self.rdma_block_num,
            "num_qp_per_pe": self.num_qp_per_pe,
            "quant_type": self.quant_type,
        }


class MoriEPWrapper:
    """Thread-safe singleton wrapper around MORI EpDispatchCombineOp."""

    _instance: Optional["MoriEPWrapper"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __init__(self, config: MoriEPWrapperConfig, shmem_group_name: str) -> None:
        self._config = config
        self._shmem_group_name = shmem_group_name
        self._op = None
        self._init_op()

    @classmethod
    def supported(cls) -> bool:
        try:
            import mori

            return True
        except ImportError:
            return False

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._initialized

    @classmethod
    def get_instance(cls) -> Optional["MoriEPWrapper"]:
        """返回已初始化的单例，未初始化则返回 None。"""
        with cls._lock:
            return cls._instance if cls._initialized else None

    @classmethod
    def _create(
        cls,
        config: MoriEPWrapperConfig,
        shmem_group_name: str,
    ) -> "MoriEPWrapper":
        """内部创建方法，仅由 init_moriep_wrapper 调用。"""
        with cls._lock:
            if cls._initialized:
                if cls._instance is None:
                    raise RuntimeError("MoriEP state is inconsistent")
                if cls._instance._config != config:
                    raise RuntimeError(
                        f"MoriEP already initialized with different config, "
                        f"origin: {cls._instance._config}, new: {config}"
                    )
                if cls._instance._shmem_group_name != shmem_group_name:
                    raise RuntimeError(
                        f"MoriEP already initialized with different shmem group, "
                        f"origin: {cls._instance._shmem_group_name}, new: {shmem_group_name}"
                    )
                return cls._instance

            if not cls.supported():
                raise RuntimeError("MORI is not supported in current environment")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Distributed environment is not initialized")

            cls._instance = cls(config, shmem_group_name)
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
    def config(self) -> MoriEPWrapperConfig:
        return self._config

    @property
    def op(self) -> Any:
        if self._op is None:
            raise RuntimeError("MORI EpDispatchCombineOp is not initialized")
        return self._op

    def dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return self.op.dispatch(*args, **kwargs)

    def combine(self, *args: Any, **kwargs: Any) -> Any:
        return self.op.combine(*args, **kwargs)

    def reset_op(self) -> None:
        if self._op is not None and hasattr(self._op, "reset"):
            self._op.reset()

    def _init_op(self) -> None:
        mori_config = mori.ops.EpDispatchCombineConfig(**self._config.to_mori_kwargs())
        self._op = mori.ops.EpDispatchCombineOp(mori_config)


def init_moriep_wrapper(
    config: MoriEPWrapperConfig,
    shmem_group_name: str = "default",
) -> None:
    """注册 shmem process group + 创建 MoriEPWrapper 单例。

    必须在 torch.distributed.init_process_group() 之后调用。
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed environment is not initialized. "
            "Call torch.distributed.init_process_group() first."
        )

    # 注册 WORLD group 并初始化 mori shmem
    world_group = torch.distributed.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group(shmem_group_name, world_group)
    mori.shmem.shmem_torch_process_group_init(shmem_group_name)

    MoriEPWrapper._create(config, shmem_group_name=shmem_group_name)
