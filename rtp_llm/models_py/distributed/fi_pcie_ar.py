"""TP all-reduce over the FlashInfer PCIe IPC kernels (upstream main, vendored as ``fi_up``).

Opt-in with DSV4_PCIE_ONESHOT_AR=1 DSV4_PCIE_AR_BACKEND=fi (the default backend stays b12x). The engine's
``collective_torch.pcie_oneshot_all_reduce`` hook is the single entry point for every small TP all-reduce
(attention output, TP-MoE partial sums), so swapping the backend there covers all of them.

Contract kept from the hook:
  * built lazily on the first eager TP all-reduce, which every rank reaches in lockstep during warm-up;
  * every launch configuration a CUDA-graph capture can hit is resolved eagerly (``prepare``), because
    resolving one is a collective with a host read-back;
  * returns None (caller falls back to NCCL) for any shape the workspace does not admit -- this decision is
    a pure function of shape/dtype, so all ranks fall back together.
Environment:
  DSV4_FI_PCIE_PATH      directory containing the vendored ``fi_up`` package (see comm/largeep_a2a/README.md "FlashInfer PCIe IPC"; required)
  DSV4_FI_PCIE_MAX_ROWS  largest row count reduced through the workspace (default 256); hidden is taken from
                         the first call (4096 for DSV4)
  DSV4_FI_PCIE_TUNE_DIR  directory of the workspace's tune_cache file (default <path>/tunecache); not used to
                         pick launch configurations
  DSV4_FI_PCIE_CONFIGS   pinned launch configurations "rows:variant:blocks:threads,..." (default: the TP4
                         hidden-4096 winners measured on an RTX PRO 5000 TP4 group); each prepared row count uses the largest
                         entry <= rows
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

import torch

_logger = logging.getLogger(__name__)

# graph capture sizes are token counts; MTP verify runs (gen+1) rows per stream
_PREPARE_ROWS = sorted(
    {
        1,
        2,
        3,
        4,
        6,
        8,
        9,
        12,
        16,
        18,
        24,
        27,
        32,
        36,
        40,
        48,
        54,
        64,
        72,
        96,
        120,
        128,
        144,
        192,
        256,
    }
)


class FiPcieAllReduce:
    def __init__(
        self, group: torch.distributed.ProcessGroup, hidden: int, dtype: torch.dtype
    ):
        path = os.environ.get("DSV4_FI_PCIE_PATH", "")
        if not path:
            raise RuntimeError(
                "DSV4_PCIE_AR_BACKEND=fi needs DSV4_FI_PCIE_PATH (directory of the vendored fi_up package)"
            )
        if path not in sys.path:
            sys.path.insert(0, path)
        with _fi_env(path):
            self._build(path, group, hidden, dtype)

    def _build(self, path, group, hidden, dtype):
        from fi_up.comm.pcie_ipc_ar import PcieIpcAllReduceWorkspace

        self.hidden = int(hidden)
        self.dtype = dtype
        self.max_rows = int(os.environ.get("DSV4_FI_PCIE_MAX_ROWS", "256"))
        tune_dir = os.environ.get(
            "DSV4_FI_PCIE_TUNE_DIR", os.path.join(path, "tunecache")
        )
        self.group = group
        self.world = torch.distributed.get_world_size(group)
        ranks = torch.distributed.get_process_group_ranks(group)
        # one cache file per TP group: two TP groups on one host must not write the same file concurrently
        cache = os.path.join(
            tune_dir, f"ar_ws{self.world}_r{ranks[0]}-{ranks[-1]}.json"
        )
        if torch.distributed.get_rank(group) == 0:
            os.makedirs(tune_dir, exist_ok=True)
        self.ws = PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=self.max_rows * self.hidden,
            dtype=dtype,
            tune_cache=cache,
        )
        # Launch configurations are pinned from an explicit table (DSV4_FI_PCIE_CONFIGS, "rows:variant:blocks:
        # threads,..."; the default is the measured winner table of the TP4 hidden-4096 tune on
        # RTX PRO 5000 (PCIe, no NVLink): AR 16.1us vs NCCL 34.3us at 48 rows).
        # Two reasons not to rely on the library's tune cache: in-engine tune() deadlocked (r77, gloo timing
        # reduction vs the engine's other collectives), and the offline tune's cache file was written with
        # metadata only (the winners stayed in a store-partitioned in-memory cache), so the engine silently ran
        # the seed config, which on this fabric is 2-5x slower than NCCL (r78c: 81us/call vs NCCL 32us).
        from fi_up.comm.pcie_ipc_policy import IpcLaunchConfig, IpcVariant

        table = os.environ.get(
            "DSV4_FI_PCIE_CONFIGS",
            "1:0:96:128,2:0:16:64,4:1:64:64,8:1:128:128,16:1:128:128,32:1:96:512,64:1:128:256,128:1:128:1024,"
            "192:1:128:1024",
        )
        pinned = {}
        for item in table.split(","):
            r, v, b, t = (int(x) for x in item.split(":"))
            pinned[r] = IpcLaunchConfig(blocks=b, threads=t, variant=IpcVariant(v))
        self._configs = {}
        for r in _PREPARE_ROWS:
            if r > self.max_rows:
                continue
            probe = torch.empty(r, self.hidden, dtype=dtype, device="cuda")
            if not self.ws.supports(probe):
                continue
            # floor bucket of the tuned ladder (1,2,4,..,256), same mapping the library uses
            key = max(k for k in pinned if k <= r)
            self._configs[r] = pinned[key]
        self._shapes = set(self._configs)
        self._out = torch.empty(self.max_rows * self.hidden, dtype=dtype, device="cuda")
        # prepare()/tune() above ran on the current stream and bound the workspace to it
        self._stream = torch.cuda.current_stream()
        _logger.warning(
            "FI PCIe IPC all-reduce up: ranks=%s hidden=%d cache=%s",
            ranks,
            self.hidden,
            cache,
        )

    def try_all_reduce(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if tensor.dtype != self.dtype or tensor.dim() < 1 or not tensor.is_contiguous():
            return None
        if tensor.shape[-1] != self.hidden:
            return None
        rows = tensor.numel() // self.hidden
        if rows not in self._shapes:
            return None
        if not torch.cuda.is_current_stream_capturing():
            # eager calls must all use one stream (the workspace's epoch/arrival state is stream-ordered);
            # the first eager stream wins and any other one (rank-uniform: same code path on every rank)
            # falls back to NCCL. Captured calls replay on the capture stream, ordered by the engine.
            cur = torch.cuda.current_stream()
            if self._stream is None:
                self._stream = cur
            elif cur != self._stream:
                return None
        flat = tensor.view(rows, self.hidden)
        # one persistent output buffer: every call is stream-ordered and copied back before the next one
        out = self._out[: rows * self.hidden].view(rows, self.hidden)
        self.ws.all_reduce(flat, out=out, config=self._configs[rows])
        flat.copy_(out)
        return tensor


class _fi_env:
    """Point the vendored upstream JIT at its own cache/arch while it builds or loads its extensions, then
    restore the engine's environment (the engine's own flashinfer must not see these).
    """

    _KEYS = ("FLASHINFER_WORKSPACE_BASE", "FLASHINFER_CUDA_ARCH_LIST", "CUDA_HOME")

    def __init__(self, path):
        self.vals = {
            "FLASHINFER_WORKSPACE_BASE": path,
            "FLASHINFER_CUDA_ARCH_LIST": os.environ.get("DSV4_FI_PCIE_ARCH", "12.0f"),
            "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2"),
        }

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self._KEYS}
        os.environ.update(self.vals)

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_GLOO: dict = {}


def _gloo_subgroup(ranks: List[int]):
    """gloo twin of a TP group. new_group is collective over WORLD: every rank must create every TP group's
    twin in the same order, so all groups are created here, keyed by their rank tuple.
    """
    key = tuple(ranks)
    if key in _GLOO:
        return _GLOO[key]
    world = torch.distributed.get_world_size()
    tp = len(ranks)
    mine = None
    for base in range(0, world, tp):
        rs = list(range(base, base + tp))
        g = torch.distributed.new_group(ranks=rs, backend="gloo")
        _GLOO[tuple(rs)] = g
        if tuple(rs) == key:
            mine = g
    if mine is None:
        raise RuntimeError(f"TP group {ranks} is not a contiguous block of the world")
    return mine
