# SPDX-License-Identifier: Apache-2.0
"""Isolated, explicitly pinned FlashInfer SM120 Prefill runner.

Calling convention: FlashInfer v0.7.0 flashinfer/fused_moe/core.py,
get_cutlass_fused_moe_module/MoERunner (Apache-2.0),
https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/fused_moe/core.py
No installed Python package or Decode backend is replaced.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=8)
def _load_runner(library, sha256, device_index):
    import tvm_ffi

    path = Path(library)
    if not sha256 or hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError("SM120 Prefill MoE library SHA256 mismatch")
    module = tvm_ffi.load_module(str(path))
    with torch.cuda.device(device_index):
        runner = module.init(
            torch.bfloat16,
            torch.int64,
            torch.bfloat16,
            False,
            False,
            False,
            False,
            True,
            False,
        )
    # Keep both the module and runner alive until all captured/native users end.
    return module, runner


class Sm120PrefillMoE:
    def __init__(self, library, sha256, device):
        self.module, self.runner = _load_runner(library, sha256, device.index)

    def __call__(
        self, x, ids, router, w1, w2, scales, *, tp_size, tp_rank, ep_size, ep_rank
    ):
        output = torch.empty(
            (x.shape[0], w2.shape[1]), device=x.device, dtype=torch.bfloat16
        )
        # Match the deployed runner: global IDs, output-side router weights,
        # [up,gate], block16 NVFP4, separate activation/weight scales, default
        # GEMM tactics, fused-finalize option and explicitly disabled PDL.
        self.runner.run_moe(
            output,
            x,
            ids,
            router,
            w1,
            None,
            w2,
            None,
            scales,
            None,
            None,
            None,
            None,
            None,
            None,
            True,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
            1,
            0,
            False,
            False,
            [-1, -1],
            False,
            3,
            None,
        )
        return output
