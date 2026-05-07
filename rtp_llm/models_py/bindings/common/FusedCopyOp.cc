#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

void fusedCopy(const FusedD2DCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    invokeFusedCopy(params, stream);
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
    invokeFusedCopy(params, stream);
#else
    // TODO: Ascend - Add Ascend support
    throw std::runtime_error("No supported GPU backend found for fusedCopy");
#endif
}

void fusedStridedCopy(const FusedStridedCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    invokeFusedStridedCopy(params, stream);
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
    invokeFusedStridedCopy(params, stream);
#else
    throw std::runtime_error("No supported GPU backend found for fusedStridedCopy");
#endif
}

}  // namespace rtp_llm
