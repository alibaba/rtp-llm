#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

#if USING_CUDA
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include <hip/hip_runtime.h>
#endif
#if USING_XPU
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <cstring>
#endif

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

void fusedCopy(const FusedD2DCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    invokeFusedCopy(params, stream);
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
    invokeFusedCopy(params, stream);
#elif USING_XPU
    // XPU fallback: sequential async memcpy via SYCL queue
    RTP_LLM_CHECK(params.num_copies >= 0);
    sycl::queue& queue = c10::xpu::getCurrentXPUStream();
    for (int i = 0; i < params.num_copies; ++i) {
        RTP_LLM_CHECK(params.dst[i] != nullptr && params.src[i] != nullptr);
        queue.memcpy(params.dst[i], params.src[i], params.size[i]);
    }
#else
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
#elif USING_XPU
    // XPU fallback: sequential async strided memcpy via SYCL queue
    RTP_LLM_CHECK(params.num_copies >= 0);
    sycl::queue& queue = c10::xpu::getCurrentXPUStream();
    for (int i = 0; i < params.num_copies; ++i) {
        RTP_LLM_CHECK(params.dst[i] != nullptr && params.src[i] != nullptr);
        const char* src_base = static_cast<const char*>(params.src[i]);
        char*       dst_base = static_cast<char*>(params.dst[i]);
        for (size_t row = 0; row < params.num_rows[i]; ++row) {
            queue.memcpy(
                dst_base + row * params.dst_row_stride[i],
                src_base + row * params.src_row_stride[i],
                params.row_bytes[i]);
        }
    }
#else
    throw std::runtime_error("No supported GPU backend found for fusedStridedCopy");
#endif
}

}  // namespace rtp_llm
