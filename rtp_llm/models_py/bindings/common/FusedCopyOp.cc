#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"
#include "rtp_llm/cpp/utils/ExecutionStageRecorder.h"
#include <algorithm>
#include <cstdint>
#include <sstream>

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

namespace {

constexpr int max_trace_copies = 8;

void annotateCopyParams(ExecutionStageScope& scope, const FusedD2DCopyParams& params) noexcept {
    if (scope.id() == 0) {
        return;
    }
    try {
        std::ostringstream out;
        out << "request_id=unknown dtype=unknown shape=unknown copies=" << params.num_copies
            << " truncated=" << (params.num_copies > max_trace_copies) << " regions=[";
        for (int index = 0; index < std::min(params.num_copies, max_trace_copies); ++index) {
            if (index != 0) {
                out << ",";
            }
            out << "{index=" << index << " bytes=" << params.size[index] << "}";
        }
        out << "]";
        scope.annotate(out.str());
    } catch (...) {}
}

void annotateCopyParams(ExecutionStageScope& scope, const FusedStridedCopyParams& params) noexcept {
    if (scope.id() == 0) {
        return;
    }
    try {
        std::ostringstream out;
        out << "request_id=unknown dtype=unknown shape=unknown copies=" << params.num_copies
            << " truncated=" << (params.num_copies > max_trace_copies) << " regions=[";
        for (int index = 0; index < std::min(params.num_copies, max_trace_copies); ++index) {
            if (index != 0) {
                out << ",";
            }
            out << "{index=" << index << " rows=" << params.num_rows[index] << " row_bytes=" << params.row_bytes[index]
                << " src_row_stride=" << params.src_row_stride[index]
                << " dst_row_stride=" << params.dst_row_stride[index] << "}";
        }
        out << "]";
        scope.annotate(out.str());
    } catch (...) {}
}

void annotateCopyStream(ExecutionStageScope& scope, uintptr_t stream) noexcept {
    if (scope.id() == 0) {
        return;
    }
    try {
        scope.annotate("stream=" + std::to_string(stream) + " completion=host_return_only");
    } catch (...) {}
}

}  // namespace

void fusedCopy(const FusedD2DCopyParams& params) {
    ExecutionStageScope copy_trace("fused_copy");
    annotateCopyParams(copy_trace, params);
    ExecutionStageScope stream_trace("fused_copy.stream_lookup");
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
#else
    throw std::runtime_error("No supported GPU backend found for fusedCopy");
    return;
#endif
    annotateCopyStream(stream_trace, reinterpret_cast<uintptr_t>(stream));
    stream_trace.finish("host_returned");
    ExecutionStageScope launch_trace("fused_copy.launch");
    annotateCopyStream(launch_trace, reinterpret_cast<uintptr_t>(stream));
    invokeFusedCopy(params, stream);
    launch_trace.finish("host_returned");
}

void fusedStridedCopy(const FusedStridedCopyParams& params) {
    ExecutionStageScope copy_trace("fused_strided_copy");
    annotateCopyParams(copy_trace, params);
    ExecutionStageScope stream_trace("fused_strided_copy.stream_lookup");
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
#else
    throw std::runtime_error("No supported GPU backend found for fusedStridedCopy");
    return;
#endif
    annotateCopyStream(stream_trace, reinterpret_cast<uintptr_t>(stream));
    stream_trace.finish("host_returned");
    ExecutionStageScope launch_trace("fused_strided_copy.launch");
    annotateCopyStream(launch_trace, reinterpret_cast<uintptr_t>(stream));
    invokeFusedStridedCopy(params, stream);
    launch_trace.finish("host_returned");
}

}  // namespace rtp_llm
