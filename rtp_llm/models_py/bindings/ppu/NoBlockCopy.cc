#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

namespace {

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

    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        check_cuda_value(cudaMemcpyAsync(params.multi_dst[i].data_ptr(),
                                         params.multi_src[i].data_ptr(),
                                         params.multi_src[i].nbytes(),
                                         cudaMemcpyDefault,
                                         stream));
    }
    check_cuda_value(cudaStreamSynchronize(stream));
    check_cuda_error();
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
