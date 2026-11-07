#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/common/FusedCopyHostSource.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

torch::Tensor prepareFusedCopyHostSource(const torch::Tensor& source) {
#if USING_CUDA
    TORCH_CHECK(source.device().is_cpu(), "fused H2D source must be a CPU tensor");
    if (!source.numel()) return source;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto record = [&stream](const torch::Tensor& tensor) {
        return at::cuda::CachingHostAllocator_recordEvent(
            tensor.data_ptr(), tensor.storage().data_ptr().get_context(), stream);
    };
    // fusedCopy reads mapped host memory directly, so PyTorch's normal
    // nonblocking-copy allocator hook does not run. Register its stream here;
    // the allocator records completion when the last host reference is freed.
    // A fixed number of TensorHolder rounds cannot bound GPU queue depth.
    if (record(source)) return source;
    auto owned = torch::empty(source.sizes(), source.options().pinned_memory(true));
    owned.copy_(source);
    TORCH_CHECK(record(owned), "failed to track owned fused H2D source");
    return owned;
#else
    return source;
#endif
}

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
