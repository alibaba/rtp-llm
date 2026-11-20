"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



import os
from functools import lru_cache
import torch
from flashinfer.jit.core import logger


@lru_cache(maxsize=1)
def is_confidential_compute() -> bool:
    """Whether the GPU is running in NVIDIA Confidential Computing (CC) mode.

    Detected once via NVML and cached.
    Overridable with ``FLASHINFER_CONFIDENTIAL_COMPUTE=1/0``.
    """
    forced = os.environ.get("FLASHINFER_CONFIDENTIAL_COMPUTE")
    if forced is not None:
        if forced not in ("0", "1"):
            raise ValueError(
                f"FLASHINFER_CONFIDENTIAL_COMPUTE must be '0' or '1', got {forced!r}"
            )
        return forced == "1"
    if not torch.cuda.is_available():
        return False
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            state = pynvml.nvmlSystemGetConfComputeState()
            # ccFeature != 0 means CC is enabled (ON or devtools).
            return int(getattr(state, "ccFeature", 0)) != 0
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("[Flashinfer]: Confidential-compute detection failed: %r", e)
        return False


@lru_cache(maxsize=1)
def get_globaltimer_kernel():
    """Lazily JIT-build the %globaltimer kernel."""

    _GLOBALTIMER_KERNEL_CU = r"""
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    __global__ void _get_globaltimer_timestamp(uint64_t* timestamp) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(*timestamp));
    }
    void get_globaltimer_timestamp(torch::Tensor timestamp) {
        _get_globaltimer_timestamp<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint64_t*>(timestamp.data_ptr<int64_t>()));
    }
    """

    try:
        from torch.utils.cpp_extension import load_inline

        mod = load_inline(
            name="rtp_sparse_dsv4_globaltimer",
            cpp_sources="void get_globaltimer_timestamp(torch::Tensor);",
            cuda_sources=_GLOBALTIMER_KERNEL_CU,
            functions=["get_globaltimer_timestamp"],
            verbose=False,
        )
        return mod.get_globaltimer_timestamp
    except Exception as e:
        logger.warning(
            f"[Flashinfer]: %globaltimer stamp kernel build failed ({e}); "
            f"falling back to cudaEvent timing."
        )
    return None
