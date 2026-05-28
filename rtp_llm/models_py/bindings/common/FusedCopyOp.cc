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
#if USING_ASCEND
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "rtp_llm/models_py/bindings/ascend/ascend_types_hdr.h"
#endif

namespace rtp_llm {

void fusedCopy(const FusedD2DCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    invokeFusedCopy(params, stream);
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
    invokeFusedCopy(params, stream);
#elif USING_ASCEND
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    for (int i = 0; i < params.num_copies; ++i) {
        ASCEND_CHECK(aclrtMemcpyAsync(params.dst[i], params.size[i],
                                       params.src[i], params.size[i],
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));
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
#elif USING_ASCEND
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    for (int i = 0; i < params.num_copies; ++i) {
        const char* src_base = static_cast<const char*>(params.src[i]);
        char*       dst_base = static_cast<char*>(params.dst[i]);
        for (size_t row = 0; row < params.num_rows[i]; ++row) {
            const void* src_ptr = src_base + row * params.src_row_stride[i];
            void*       dst_ptr = dst_base + row * params.dst_row_stride[i];
            ASCEND_CHECK(aclrtMemcpyAsync(dst_ptr, params.row_bytes[i],
                                           src_ptr, params.row_bytes[i],
                                           ACL_MEMCPY_HOST_TO_DEVICE, stream));
        }
    }
#else
    throw std::runtime_error("No supported GPU backend found for fusedStridedCopy");
#endif
}

}  // namespace rtp_llm
