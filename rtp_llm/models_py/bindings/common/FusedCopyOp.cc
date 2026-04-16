#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

void fusedCopy(const FusedD2DCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
#else
    throw std::runtime_error("No supported GPU backend found for fusedCopy");
    return;
#endif
    invokeFusedCopy(params, stream);
}

void fusedStridedCopy(const FusedStridedCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
#else
    throw std::runtime_error("No supported GPU backend found for fusedStridedCopy");
    return;
#endif
    invokeFusedStridedCopy(params, stream);
}

}  // namespace rtp_llm
