#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>
#include <cstdlib>

namespace rtp_llm {

namespace {

cudaError_t copyPpuResult(const char* phase, cudaError_t result) {
    if (auto probe = no_block_copy_detail::cuda_result_for_test) {
        return static_cast<cudaError_t>(probe(phase, static_cast<int>(result)));
    }
    return result;
}

void drainPpuCopy(cudaStream_t stream, const char* phase) noexcept {
    const auto result = copyPpuResult(phase, cudaStreamSynchronize(stream));
    if (result != cudaSuccess) {
        // Completion cannot be established. Do not let an exception release
        // buffers which an earlier asynchronous copy may still be accessing.
        std::fprintf(
            stderr, "PPU copy completion unconfirmed phase=%s error_code=%d\n", phase, static_cast<int>(result));
        std::fflush(stderr);
        std::abort();
    }
}

at::cuda::CUDAStream& getNoBlockCopyStream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    return stream;
}

}  // namespace

std::shared_ptr<DeviceHostCopyStreams> acquireDeviceHostCopyStreams(int device_index) {
    return std::make_shared<DeviceHostCopyStreams>(device_index, 0, 0);
}

void execNoBlockCopy(const MultiCopyParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    auto stream = getNoBlockCopyStream().stream();

    bool attempted = false;
    try {
        for (size_t i = 0; i < params.multi_src.size(); ++i) {
            attempted = true;
            check_cuda_value(copyPpuResult("generic_submit",
                                           cudaMemcpyAsync(params.multi_dst[i].data_ptr(),
                                                           params.multi_src[i].data_ptr(),
                                                           params.multi_src[i].nbytes(),
                                                           cudaMemcpyDefault,
                                                           stream)));
        }
        drainPpuCopy(stream, "generic_completion");
        check_cuda_error();
    } catch (...) {
        if (attempted) {
            drainPpuCopy(stream, "generic_exception_drain");
        }
        throw;
    }
}

void execNoBlockCopy(const MultiCopyParams& params, const DeviceHostCopyExecutionContext&) {
    execNoBlockCopy(params);
}

void warmupNoBlockCopy() {}

// PPU has no batched/staged fast path; callers use per-item copies.
BatchedMemoryCopyStatus execBatchedMemoryCopy(const BatchedMemoryCopyParams& params,
                                              const DeviceHostCopyExecutionContext&) {
    return params.tiles.empty() ? BatchedMemoryCopyStatus::SUCCESS : BatchedMemoryCopyStatus::NOT_SUPPORTED;
}

bool execStagedMemoryCopy(const StagedMemoryCopyParams& params, StagedMemoryCopyScratch*) {
    return params.tiles.empty();
}

void releaseStagedMemoryCopyScratch(StagedMemoryCopyScratch&) {}

}  // namespace rtp_llm
