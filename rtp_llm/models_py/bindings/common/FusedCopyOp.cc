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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#pragma GCC diagnostic pop
#include "rtp_llm/models_py/bindings/ascend/ascend_types_hdr.h"
#endif

namespace rtp_llm {

#if USING_ASCEND
// The CUDA kernel is direction-agnostic (raw deref); the aclrt memcpy API is
// not, so pick the kind from the actual source pointer location. All current
// Ascend call sites flush host-pinned staging buffers into device tensors
// (H2D); the device-resident branch future-proofs D2D reuse (e.g. graph mode).
static inline aclrtMemcpyKind memcpyKindFor(const void* src) {
    aclrtPtrAttributes attr{};
    if (aclrtPointerGetAttributes(src, &attr) == ACL_SUCCESS
        && attr.location.type != ACL_MEM_LOCATION_TYPE_HOST
        && attr.location.type != ACL_MEM_LOCATION_TYPE_UNREGISTERED) {
        return ACL_MEMCPY_DEVICE_TO_DEVICE;
    }
    return ACL_MEMCPY_HOST_TO_DEVICE;
}
#endif

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
                                       memcpyKindFor(params.src[i]), stream));
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
                                           memcpyKindFor(src_ptr), stream));
        }
    }
#else
    throw std::runtime_error("No supported GPU backend found for fusedStridedCopy");
#endif
}

}  // namespace rtp_llm
