#pragma once
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_util.h"

#if USING_CUDA
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

void invokeFusedCopy(const FusedD2DCopyParams& params, cudaStream_t stream);

void invokeFusedStridedCopy(const FusedStridedCopyParams& params, cudaStream_t stream);

}  // namespace rtp_llm
