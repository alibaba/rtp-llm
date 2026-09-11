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

void warmupNoBlockCopy() {}

// Staged/batched memory-copy fast paths are CUDA-only; returning "nothing to
// do" makes KVCacheMemoryConnector fall back to the generic per-item copy.
bool execBatchedMemoryCopy(const BatchedMemoryCopyParams& params) {
    return params.tiles.empty();
}

bool execStagedMemoryCopy(const StagedMemoryCopyParams& params, StagedMemoryCopyScratch*) {
    return params.tiles.empty();
}

void releaseStagedMemoryCopyScratch(StagedMemoryCopyScratch&) {}

}  // namespace rtp_llm
