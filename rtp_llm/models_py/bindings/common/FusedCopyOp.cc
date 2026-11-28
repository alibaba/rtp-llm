#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/common/FusedCopyOp.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"
#include <c10/core/DeviceGuard.h>
#include <cstdint>

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

static bool overlapsContiguousDestination(const torch::Tensor& src, const torch::Tensor& dst) {
    const auto src_begin = reinterpret_cast<uintptr_t>(src.data_ptr());
    const auto dst_begin = reinterpret_cast<uintptr_t>(dst.data_ptr());
    return src_begin < dst_begin + dst.nbytes() && dst_begin < src_begin + src.nbytes();
}

bool tryFusedMultimodalCopy(torch::Tensor&                    dst,
                            const std::vector<torch::Tensor>& srcs,
                            const std::vector<int64_t>&       row_offsets) {
    if (!dst.is_cuda() || dst.dim() != 2 || !dst.is_contiguous() || srcs.size() != row_offsets.size()
        || srcs.size() > MAX_FUSED_D2D_COPIES) {
        return false;
    }
    FusedD2DCopyParams params;
    int64_t            previous_end = 0;
    for (size_t i = 0; i < srcs.size(); ++i) {
        const auto& src = srcs[i];
        if (src.numel() == 0) {
            continue;
        }
        const auto offset = row_offsets[i];
        if (src.device() != dst.device() || src.scalar_type() != dst.scalar_type() || src.dim() != 2
            || src.size(1) != dst.size(1) || !src.is_contiguous() || offset < previous_end || offset > dst.size(0)
            || src.size(0) > dst.size(0) - offset || overlapsContiguousDestination(src, dst)) {
            return false;
        }
        previous_end  = offset + src.size(0);
        auto* dst_ptr = static_cast<char*>(dst.data_ptr()) + offset * dst.size(1) * dst.element_size();
        params.add(src.data_ptr(), dst_ptr, src.nbytes());
    }
    // A rejected batch must leave dst untouched for the ordered Python fallback.
    const c10::DeviceGuard guard(dst.device());
    fusedCopy(params);
    return true;
}

void fusedMultimodalCopy(torch::Tensor&                    dst,
                         const std::vector<torch::Tensor>& srcs,
                         const std::vector<int64_t>&       row_offsets) {
    TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA/HIP tensor");
    TORCH_CHECK(dst.dim() == 2, "dst must be 2D [tokens, hidden]");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(srcs.size() == row_offsets.size(), "srcs and row_offsets must have the same length");

    const int64_t hidden_size  = dst.size(1);
    bool          ordered_copy = false;
    int64_t       previous_end = 0;
    for (size_t i = 0; i < srcs.size(); ++i) {
        const auto& src = srcs[i];
        TORCH_CHECK(src.is_cuda(), "src[", i, "] must be a CUDA/HIP tensor");
        TORCH_CHECK(src.device() == dst.device(), "src[", i, "] must be on dst device");
        TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "src[", i, "] dtype must match dst");
        TORCH_CHECK(src.dim() == 2 && src.size(1) == hidden_size, "src[", i, "] must be [tokens, ", hidden_size, "]");
        TORCH_CHECK(src.is_contiguous(), "src[", i, "] must be contiguous");
        const int64_t offset = row_offsets[i];
        TORCH_CHECK(offset >= 0 && offset <= dst.size(0) && src.size(0) <= dst.size(0) - offset,
                    "src[",
                    i,
                    "] row range is outside dst");
        if (src.numel() == 0) {
            continue;
        }
        ordered_copy |= offset < previous_end || overlapsContiguousDestination(src, dst);
        previous_end = offset + src.size(0);
    }
    const c10::DeviceGuard guard(dst.device());
    FusedD2DCopyParams     params;
    for (size_t i = 0; i < srcs.size(); ++i) {
        const auto& src    = srcs[i];
        const auto  offset = row_offsets[i];
        if (src.numel() == 0) {
            continue;
        }
        if (ordered_copy) {
            auto target = dst.narrow(0, offset, src.size(0));
            target.copy_(overlapsContiguousDestination(src, target) ? src.clone() : src);
            continue;
        }
        if (params.num_copies == MAX_FUSED_D2D_COPIES) {
            fusedCopy(params);
            params.clear();
        }
        auto* dst_ptr = static_cast<char*>(dst.data_ptr()) + offset * hidden_size * dst.element_size();
        params.add(src.data_ptr(), dst_ptr, src.nbytes());
    }
    fusedCopy(params);
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
