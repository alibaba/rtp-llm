#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/SplitKvCacheCopy.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <unordered_map>

namespace rtp_llm {

namespace {

int getCopyDevice(const MultiCopyParams& params) {
    if (params.multi_dst.empty()) {
        return static_cast<int>(at::cuda::current_device());
    }
    if (params.multi_dst[0].is_cuda()) {
        return static_cast<int>(params.multi_dst[0].get_device());
    }
    if (params.multi_src[0].is_cuda()) {
        return static_cast<int>(params.multi_src[0].get_device());
    }
    return static_cast<int>(at::cuda::current_device());
}

at::cuda::CUDAStream getNoBlockCopyStream(int device_id) {
    static thread_local std::unordered_map<int, at::cuda::CUDAStream> streams;
    auto                                                              stream = streams.find(device_id);
    if (stream == streams.end()) {
        stream = streams.emplace(device_id, at::cuda::getStreamFromPool(/*isHighPriority=*/false, device_id)).first;
    }
    return stream->second;
}

}  // namespace

void execNoBlockCopy(const MultiCopyParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    const int copy_device = getCopyDevice(params);
    check_cuda_value(cudaSetDevice(copy_device));

    auto stream = getNoBlockCopyStream(copy_device).stream();

    if (params.split_kv_layer_num > 0 && copy_device >= 0) {
        if (splitKvMultiCopy(params.multi_src,
                             params.multi_dst,
                             params.split_kv_layer_num,
                             static_cast<int64_t>(params.split_kv_cache_stride_bytes),
                             static_cast<int64_t>(params.split_kv_scale_stride_bytes),
                             stream)) {
            check_cuda_value(cudaStreamSynchronize(stream));
            check_cuda_error();
            return;
        }
    }

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

void warmupNoBlockCopy() {
    if (!warmupSplitKvCopyKernels(at::cuda::getCurrentCUDAStream().stream())) {
        RTP_LLM_LOG_WARNING("warmupSplitKvCopyKernels failed; split-KV copy may JIT on first use");
    }
}

}  // namespace rtp_llm
