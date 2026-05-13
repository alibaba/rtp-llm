"""Multi-GPU tests for Mori EP router 行为（直接实例化生产 MoriEpIntranodeRouter）。

与 distributed/test/moriep_test.py 相同策略：
- 用 importlib 按文件路径加载 moriep_wrapper.py，无需整包安装/编译 rtp_llm；
- 在加载前注入最小 rtp_llm 桩模块，满足 moriep_wrapper.py 顶层 import；
- 分布式使用 torch.distributed 直接 init（同 moriep_test），不依赖 collective_torch / PortManager；
- 通过 importlib 加载生产 MoriEpIntranodeRouter 及其依赖的 fused_moe defs 模块，
  直接调用 router.prepare() / router.finalize()，确保测试覆盖生产代码路径。

依赖：mori、多卡 CUDA/ROCm。无 mori 时 __main__ 退出 0。

spawn 多进程时父进程勿用 torch.cuda（见 _gpu_count_without_parent_cuda_init）；
可设环境变量 MORIEP_TEST_MAX_GPUS=8 指定卡数，避免走兜底 torch.cuda.device_count()。

调试：默认 MORIEP_TEST_DEBUG=1 打印各阶段（含 rank/pid/相对时间）；
关闭详细日志：MORIEP_TEST_DEBUG=0
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import pathlib
import random
import re
import socket
import subprocess
import sys
import time
import types
import unittest
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# 路径：models_py/distributed/moriep_wrapper.py（与 moriep_test 中 _DIST_DIR 相对关系一致）
# 本文件: .../models_py/modules/factory/fused_moe/impl/rocm/test/moriep_intranode_router_test.py
# parents[6] -> models_py
# ---------------------------------------------------------------------------
_MODELS_PY = pathlib.Path(__file__).resolve().parents[6]
_DIST_DIR = _MODELS_PY / "distributed"
_MORIEP_WRAPPER_PATH = _DIST_DIR / "moriep_wrapper.py"

# 调试：MORIEP_TEST_DEBUG=0 可关闭；MORIEP_TEST_DEBUG=1（默认）打印各阶段
_T0 = time.monotonic()


def _moriep_debug_enabled() -> bool:
    return os.environ.get("MORIEP_TEST_DEBUG", "1").strip() != "0"


def _moriep_log(rank: Optional[int], msg: str) -> None:
    if not _moriep_debug_enabled():
        return
    elapsed = time.monotonic() - _T0
    pid = os.getpid()
    r = "?" if rank is None else str(rank)
    print(
        f"[moriep_intranode_router_test +{elapsed:8.3f}s pid={pid} rank={r}] {msg}",
        flush=True,
    )


def _load_module(name: str, path: pathlib.Path):
    """Load a single Python file as *name* in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_packages_up_to(full_name: str) -> None:
    """注册 package 链，如 rtp_llm.models_py.modules..."""
    parts = full_name.split(".")
    for i in range(1, len(parts) + 1):
        p = ".".join(parts[:i])
        if p in sys.modules:
            continue
        m = types.ModuleType(p)
        if i < len(parts):
            m.__path__ = []
        sys.modules[p] = m


def _register_stub_module(full_name: str, **attrs: Any) -> None:
    """注册叶子模块（如 rtp_llm.config.engine_config）并挂属性。"""
    parent = full_name.rsplit(".", 1)[0]
    _ensure_packages_up_to(parent)
    mod = types.ModuleType(full_name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[full_name] = mod


def _to_torch_dtype_stub(maybe_str_dtype):
    """Minimal stub for rtp_llm.utils.util.to_torch_dtype."""
    import torch as _torch

    if isinstance(maybe_str_dtype, _torch.dtype):
        return maybe_str_dtype
    if hasattr(maybe_str_dtype, "name"):
        maybe_str_dtype = maybe_str_dtype.name
    elif not isinstance(maybe_str_dtype, str):
        maybe_str_dtype = str(maybe_str_dtype)
    if maybe_str_dtype.startswith("TYPE_"):
        maybe_str_dtype = maybe_str_dtype[5:]
    _map = {
        "bf16": _torch.bfloat16,
        "fp16": _torch.float16,
        "fp32": _torch.float32,
        "bfloat16": _torch.bfloat16,
        "float16": _torch.float16,
        "float32": _torch.float32,
        "int8": _torch.int8,
        "int32": _torch.int32,
    }
    key = maybe_str_dtype.lower()
    if key in _map:
        return _map[key]
    raise ValueError(f"Unknown dtype: {maybe_str_dtype}")


def _install_rtp_llm_stubs_for_moriep_wrapper() -> None:
    """满足 moriep_wrapper.py 顶层 `from rtp_llm... import ...`，无需安装 rtp_llm。"""
    _register_stub_module(
        "rtp_llm.config.engine_config",
        EngineConfig=type("EngineConfig", (), {}),
    )
    _register_stub_module(
        "rtp_llm.config.model_config",
        ModelConfig=type("ModelConfig", (), {}),
    )
    _register_stub_module(
        "rtp_llm.utils.util",
        to_torch_dtype=_to_torch_dtype_stub,
    )


# ---------------------------------------------------------------------------
# Production router 加载（加载 fused_moe defs + mori_ep_intranode_router）
# ---------------------------------------------------------------------------

_FUSED_MOE_DEFS_DIR = _MODELS_PY / "modules" / "factory" / "fused_moe" / "defs"
_ROUTER_FILE = (
    _MODELS_PY
    / "modules"
    / "factory"
    / "fused_moe"
    / "impl"
    / "rocm"
    / "routers"
    / "mori_ep_intranode_router.py"
)


def _load_production_router_modules() -> None:
    """Load production fused_moe def modules so MoriEpIntranodeRouter can be imported."""
    # Ensure package hierarchy exists in sys.modules
    _ensure_packages_up_to("rtp_llm.models_py.modules.factory.fused_moe.defs")
    # 1. type.py (RouterType, ExecutorType)
    type_mod = _load_module(
        "rtp_llm.models_py.modules.factory.fused_moe.defs.type",
        _FUSED_MOE_DEFS_DIR / "type.py",
    )
    # 2. quant_config.py (FusedMoEQuantConfig)
    quant_mod = _load_module(
        "rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config",
        _FUSED_MOE_DEFS_DIR / "quant_config.py",
    )
    # 3. config_adapter.py (MoEConfigAdapter) — stub with minimal class
    _register_stub_module(
        "rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter",
        MoEConfigAdapter=type("MoEConfigAdapter", (), {}),
    )
    # 4. fused_moe.py (FusedMoeDataRouter, dataclasses)
    _load_module(
        "rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe",
        _FUSED_MOE_DEFS_DIR / "fused_moe.py",
    )


def _load_production_router():
    """Load the production MoriEpIntranodeRouter class."""
    _load_production_router_modules()
    _ensure_packages_up_to(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers"
    )
    mod = _load_module(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.mori_ep_intranode_router",
        _ROUTER_FILE,
    )
    return mod.MoriEpIntranodeRouter


def _load_moriep_wrapper():
    """importlib 加载 moriep_wrapper.py（需先桩 rtp_llm）。"""
    _install_rtp_llm_stubs_for_moriep_wrapper()
    mod = _load_module("moriep_wrapper_under_test", _MORIEP_WRAPPER_PATH)
    # Also register under canonical name so production router's import resolves
    _ensure_packages_up_to("rtp_llm.models_py.distributed")
    sys.modules["rtp_llm.models_py.distributed.moriep_wrapper"] = mod
    return mod.MoriEPWrapper, mod.MoriEPWrapperConfig


# 延迟到 worker / main 中赋值，避免 import 时即依赖 mori
MoriEPWrapper: Any = None
MoriEPWrapperConfig: Any = None
MoriEpIntranodeRouter: Any = None


def _ensure_moriep_symbols_loaded() -> None:
    global MoriEPWrapper, MoriEPWrapperConfig, MoriEpIntranodeRouter
    if MoriEPWrapper is None:
        MoriEPWrapper, MoriEPWrapperConfig = _load_moriep_wrapper()
        MoriEpIntranodeRouter = _load_production_router()


# ---------------------------------------------------------------------------
# 最小 MoEConfigAdapter mock，仅提供 MoriEpIntranodeRouter.__init__ 所需字段
# ---------------------------------------------------------------------------


class _MockMoEConfig:
    """Minimal mock satisfying MoriEpIntranodeRouter's config requirements."""

    def __init__(self, ep_size: int, ep_rank: int, expert_num: int):
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.expert_num = expert_num


# ---------------------------------------------------------------------------
# 分布式 & Mori shmem（对齐 moriep_test）
# ---------------------------------------------------------------------------


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gpu_count_without_parent_cuda_init() -> Tuple[int, str]:
    """spawn 子进程前不要在父进程调用 torch.cuda.*，否则易与子进程争用 GPU 导致死锁。

    优先：环境变量 MORIEP_TEST_MAX_GPUS、CUDA_VISIBLE_DEVICES；
    其次：nvidia-smi / rocm-smi；最后才用 torch.cuda.device_count()（会 init CUDA）。

    Returns:
        (gpu_count, source): source 为可观测字符串，标明本次实际走到的分支。
    """
    env_max = os.environ.get("MORIEP_TEST_MAX_GPUS")
    if env_max is not None and env_max.strip().isdigit():
        n = int(env_max.strip())
        return n, f"MORIEP_TEST_MAX_GPUS={n}"
    cv = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cv is not None and cv.strip():
        n = len([x for x in cv.split(",") if x.strip()])
        return n, f"CUDA_VISIBLE_DEVICES ({n} id(s): {cv!r})"
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if r.returncode == 0 and r.stdout:
            n = len([ln for ln in r.stdout.splitlines() if ln.strip()])
            if n > 0:
                return n, "nvidia-smi -L (non-empty line count)"
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass
    try:
        r = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if r.returncode == 0 and r.stdout:
            gpus = re.findall(r"GPU\[(\d+)\]", r.stdout)
            if gpus:
                n = len(set(gpus))
                return n, "rocm-smi --showid (regex GPU[n])"
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass
    # 最后手段：会初始化父进程 CUDA，仅作兜底
    n = int(torch.cuda.device_count())
    return n, "torch.cuda.device_count() [fallback, may init CUDA in parent]"


def _init_dist(rank: int, world_size: int, port: int) -> None:
    _moriep_log(rank, f"set MASTER_ADDR/PORT port={port}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    _moriep_log(rank, "before torch.cuda.set_device")
    torch.cuda.set_device(rank)
    _moriep_log(rank, "before dist.init_process_group (Gloo/NCCL)")
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
    )
    _moriep_log(rank, "after init_process_group, before barrier #1 (sync all ranks)")
    # 所有 rank 都进入 PG 后再做 Mori shmem，避免一侧先进 shmem 另一侧还在 init
    dist.barrier()
    _moriep_log(rank, "after barrier #1")


def _cleanup_dist() -> None:
    if dist.is_initialized():
        r = dist.get_rank()
        _moriep_log(r, "before barrier in _cleanup_dist")
        dist.barrier()
        _moriep_log(r, "before destroy_process_group")
        dist.destroy_process_group()
        _moriep_log(r, "after destroy_process_group")


def _init_moriep_shmem_and_wrapper(
    mori_cfg: Any, mori_mod: Any, shmem_group_name: str = "default"
) -> None:
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_group = dist.group.WORLD
    assert world_group is not None
    _moriep_log(rank, f"before _register_process_group({shmem_group_name!r})")
    torch._C._distributed_c10d._register_process_group(shmem_group_name, world_group)
    _moriep_log(
        rank,
        "after _register_process_group, before mori.shmem.shmem_torch_process_group_init",
    )
    mori_mod.shmem.shmem_torch_process_group_init(shmem_group_name)
    _moriep_log(
        rank,
        "after shmem_torch_process_group_init, before barrier #2 (sync before EpDispatchCombineOp)",
    )
    dist.barrier()
    _moriep_log(
        rank, "after barrier #2, before MoriEPWrapper._create / EpDispatchCombineOp"
    )
    MoriEPWrapper._create(mori_cfg)
    _moriep_log(rank, "after MoriEPWrapper._create")


def _build_mori_ep_config(
    *,
    rank: int,
    world_size: int,
    expert_num: int,
    hidden_size: int,
    max_tokens_on_rank: int,
    num_experts_per_token: int,
    data_dtype: torch.dtype,
    mori_mod: Any,
) -> Any:
    num_experts_per_rank = expert_num // world_size
    return MoriEPWrapperConfig(
        data_type=data_dtype,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_size,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=torch.tensor([], dtype=data_dtype).element_size(),
        max_num_inp_token_per_rank=max_tokens_on_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        use_external_inp_buf=True,
        gpu_per_node=world_size,
        kernel_type=mori_mod.ops.EpDispatchCombineKernelType.IntraNode,
        rdma_block_num=0,
        num_qp_per_pe=1,
        quant_type="none",
    )


def worker_function(
    rank: int,
    world_size: int,
    token_num_per_rank: List[int],
    max_tokens_on_rank: int,
    dist_port: int,
):
    mori_mod: Any = None
    _moriep_log(rank, "worker started")
    _moriep_log(rank, "before _ensure_moriep_symbols_loaded (importlib moriep_wrapper)")
    _ensure_moriep_symbols_loaded()
    _moriep_log(rank, "after _ensure_moriep_symbols_loaded")

    random.seed(rank)
    torch.cuda.set_device(rank)

    try:
        _moriep_log(rank, "before `import mori`")
        import mori as mori_mod

        _moriep_log(rank, "after `import mori`")
        _init_dist(rank, world_size, dist_port)

        _moriep_log(rank, "building MoriEPWrapperConfig")
        mori_cfg = _build_mori_ep_config(
            rank=rank,
            world_size=world_size,
            expert_num=16,
            hidden_size=1024,
            max_tokens_on_rank=max_tokens_on_rank,
            num_experts_per_token=16,
            data_dtype=torch.bfloat16,
            mori_mod=mori_mod,
        )
        _moriep_log(rank, "calling _init_moriep_shmem_and_wrapper")
        _init_moriep_shmem_and_wrapper(mori_cfg, mori_mod, shmem_group_name="default")

        _moriep_log(rank, "MoriEPWrapper.get_instance()")
        wrapper = MoriEPWrapper.get_instance()
        assert wrapper is not None

        expert_num = 16
        top_k = expert_num
        current_device = torch.device(f"cuda:{rank}")

        # Instantiate the production router with a minimal mock config
        _moriep_log(rank, "instantiating production MoriEpIntranodeRouter")
        from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
            CombineForwardPayload,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )

        mock_config = _MockMoEConfig(
            ep_size=world_size, ep_rank=rank, expert_num=expert_num
        )
        router = MoriEpIntranodeRouter(mock_config, FusedMoEQuantConfig())

        for it in range(5):
            _moriep_log(rank, f"iter {it + 1}/5: alloc tensors + randn")
            token_num = token_num_per_rank[rank]
            a1 = torch.randn([token_num, 1024]).to(current_device).to(torch.bfloat16)
            topk_weights = torch.ones([token_num, top_k], device=current_device)
            topk_ids = torch.arange(
                expert_num, device=current_device, dtype=torch.int32
            ).repeat(token_num, 1)

            _moriep_log(rank, f"iter {it + 1}/5: router.prepare (mori dispatch)")
            payload = router.prepare(a1, None, None, topk_weights, topk_ids)
            combine_x = payload.expert_x
            combine_payload = CombineForwardPayload(fused_expert_output=combine_x)
            _moriep_log(rank, f"iter {it + 1}/5: router.finalize (mori combine)")
            a2 = router.finalize(
                combine_payload,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input=False,
                extra_finalize_args={"original_num_tokens": token_num},
            )

            _moriep_log(rank, f"iter {it + 1}/5: assert_close")
            ref_a2 = a1 * world_size
            try:
                torch.testing.assert_close(ref_a2, a2, rtol=1.3e-2, atol=1.3e-2)
            except AssertionError as exc:
                diff = (ref_a2.float() - a2.float()).abs().max().item()
                _moriep_log(
                    rank,
                    f"iter {it + 1}/5: assert_close FAILED max_abs_diff={diff} exc={exc}",
                )
                raise
            print("pass test", flush=True)
            _moriep_log(rank, f"iter {it + 1}/5: OK")
    finally:
        _moriep_log(rank, "finally: cleanup mori + dist")
        # reset_op 是实例方法，必须在 get_instance() 上调用；reset() 是类方法清空单例
        if MoriEPWrapper is not None and MoriEPWrapper.is_initialized():
            inst = MoriEPWrapper.get_instance()
            if inst is not None:
                inst.reset_op()
            MoriEPWrapper.reset()
        if (
            mori_mod is not None
            and hasattr(mori_mod, "shmem")
            and hasattr(mori_mod.shmem, "shmem_finalize")
        ):
            mori_mod.shmem.shmem_finalize()
        _cleanup_dist()


def test_single(world_size: int, dist_port: int):
    _moriep_log(
        None,
        f"[parent] test_single world_size={world_size} dist_port={dist_port} pid={os.getpid()}",
    )
    # 各 rank 必须使用相同 token 数：Ep dispatch/combine 为全体 rank 集体通信，
    # 若部分 rank assert 失败先进入 cleanup barrier，其余 rank 仍卡在下一轮 collective → 死锁。
    # 且 ref_a2=a1*world_size 的恒等检查仅在各 rank 对称 token 时成立。
    token_num_aligned = random.randint(4, 12) // 4 * 4
    token_num_per_rank = [token_num_aligned] * world_size
    max_tokens_on_rank = token_num_aligned
    _moriep_log(
        None,
        f"[parent] token_num_per_rank={token_num_per_rank} (same per rank) max_tokens_on_rank={max_tokens_on_rank}",
    )

    processes = []
    for rank in range(world_size):
        _moriep_log(None, f"[parent] spawning worker rank={rank}")
        p = mp.Process(
            target=worker_function,
            args=(
                rank,
                world_size,
                token_num_per_rank,
                max_tokens_on_rank,
                dist_port,
            ),
            kwargs={},
        )
        processes.append(p)
        p.start()
        _moriep_log(None, f"[parent] worker rank={rank} started pid={p.pid}")

    _moriep_log(None, f"[parent] all {world_size} workers spawned; joining...")
    for i, p in enumerate(processes):
        _moriep_log(None, f"[parent] join worker[{i}] pid={p.pid} ...")
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
            raise RuntimeError("Process timeout")
        if p.exitcode != 0:
            raise RuntimeError(f"子进程异常退出，退出码: {p.exitcode}")
        _moriep_log(
            None,
            f"[parent] join worker[{i}] done exitcode={p.exitcode}",
        )


_MAX_GPUS = int(os.environ.get("MORIEP_TEST_MAX_GPUS", "0"))


class MoriEpIntranodeRouterTest(unittest.TestCase):
    """Bazel-schedulable wrapper for moriep_intranode_router multi-GPU tests.

    MORIEP_TEST_MAX_GPUS env var controls which tests run:
    - "2": only test_world_size_2
    - "4": test_world_size_2 and test_world_size_4
    - unset/0: run all tests (original behavior)
    """

    def _check_skip(self):
        if importlib.util.find_spec("mori") is None:
            self.skipTest("mori is not available")
        max_gpu_count, _ = _gpu_count_without_parent_cuda_init()
        if max_gpu_count < 2:
            self.skipTest(f"Need at least 2 GPUs, got {max_gpu_count}")

    def test_world_size_2(self):
        self._check_skip()
        max_gpu_count, _ = _gpu_count_without_parent_cuda_init()
        if max_gpu_count < 2:
            self.skipTest("Need at least 2 GPUs")
        mp.set_start_method("spawn", force=True)
        test_single(world_size=2, dist_port=_get_free_port())

    def test_world_size_4(self):
        if _MAX_GPUS > 0 and _MAX_GPUS < 4:
            self.skipTest(f"MORIEP_TEST_MAX_GPUS={_MAX_GPUS}, skipping 4-GPU test")
        self._check_skip()
        max_gpu_count, _ = _gpu_count_without_parent_cuda_init()
        if max_gpu_count < 4:
            self.skipTest("Need at least 4 GPUs")
        mp.set_start_method("spawn", force=True)
        test_single(world_size=4, dist_port=_get_free_port())


if __name__ == "__main__":
    unittest.main()
