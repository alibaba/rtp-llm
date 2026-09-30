#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/FeatureHash.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "rtp_llm/cpp/utils/Logger.h"

#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/cpp/multimodal_processor/FeatureHashKernel.h"
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif
#endif

namespace rtp_llm {

#if USING_CUDA
namespace {

bool traceFeatureHash() {
    static const bool enabled = [] {
        const char* value = std::getenv("VIT_HANG_DEBUG");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

long featureHashThreadId() {
#if defined(__linux__)
    return syscall(SYS_gettid);
#else
    return 0;
#endif
}

std::atomic<uint64_t> feature_hash_trace_id{0};

}  // namespace
#endif

torch::Tensor getMultimodalFeatureHash(const torch::Tensor& embedding) {
    TORCH_CHECK(embedding.defined() && embedding.dim() >= 1 && embedding.size(0) > 0,
                "multimodal feature tensor is empty");
    const int64_t rows      = embedding.size(0);
    const int64_t row_bytes = embedding.numel() / rows * embedding.element_size();
    TORCH_CHECK(row_bytes > 0, "multimodal feature row is empty");
#if USING_CUDA
    if (embedding.is_cuda()) {
        const bool     trace    = traceFeatureHash();
        const uint64_t trace_id = trace ? feature_hash_trace_id.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
        const long     tid      = trace ? featureHashThreadId() : 0;
        const auto     started  = trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (trace) {
            RTP_LLM_LOG_INFO("vit_feature_hash id=%lu tid=%ld stage=begin device=%d rows=%ld row_bytes=%ld",
                             trace_id,
                             tid,
                             embedding.get_device(),
                             rows,
                             row_bytes);
        }
        const c10::cuda::CUDAGuard guard(embedding.device());
        if (trace) {
            RTP_LLM_LOG_INFO("vit_feature_hash id=%lu tid=%ld stage=contiguous_begin", trace_id, tid);
        }
        auto emb = embedding.contiguous();
        if (trace) {
            RTP_LLM_LOG_INFO("vit_feature_hash id=%lu tid=%ld stage=gpu_hash_alloc_begin", trace_id, tid);
        }
        auto gpu_hashes = torch::empty({rows}, emb.options().dtype(torch::kInt32));
        if (trace) {
            RTP_LLM_LOG_INFO("vit_feature_hash id=%lu tid=%ld stage=pinned_alloc_begin", trace_id, tid);
        }
        auto hashes =
            torch::empty({rows}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
        if (trace) {
            RTP_LLM_LOG_INFO("vit_feature_hash id=%lu tid=%ld stage=get_stream_begin", trace_id, tid);
        }
        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(emb.get_device());
        if (trace) {
            RTP_LLM_LOG_INFO("vit_feature_hash id=%lu tid=%ld stage=kernel_launch_begin stream=%p",
                             trace_id,
                             tid,
                             static_cast<void*>(stream));
        }
        auto error = invokeFeatureHash(emb.data_ptr(), rows, row_bytes, gpu_hashes.data_ptr<int32_t>(), stream);
        if (error == cudaSuccess) {
            if (trace) {
                RTP_LLM_LOG_INFO("vit_feature_hash id=%lu tid=%ld stage=d2h_begin bytes=%ld pinned=%d",
                                 trace_id,
                                 tid,
                                 rows * static_cast<int64_t>(sizeof(int32_t)),
                                 hashes.is_pinned());
            }
            error = cudaMemcpyAsync(hashes.data_ptr<int32_t>(),
                                    gpu_hashes.data_ptr<int32_t>(),
                                    rows * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost,
                                    stream);
        }
        if (error == cudaSuccess) {
            if (trace) {
                RTP_LLM_LOG_INFO("vit_feature_hash id=%lu tid=%ld stage=stream_sync_begin", trace_id, tid);
            }
            error = cudaStreamSynchronize(stream);
        }
        if (trace) {
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started)
                    .count();
            RTP_LLM_LOG_INFO("vit_feature_hash id=%lu tid=%ld stage=done elapsed_ms=%ld cuda_error=%d",
                             trace_id,
                             tid,
                             elapsed_ms,
                             static_cast<int>(error));
        }
        TORCH_CHECK(error == cudaSuccess, "failed to hash multimodal features on GPU: ", cudaGetErrorString(error));
        return hashes;
    }
#endif
    auto        hashes = torch::empty({rows}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto        emb    = embedding.to(torch::kCPU).contiguous();
    const auto* bytes  = static_cast<const uint8_t*>(emb.data_ptr());
    auto*       output = hashes.data_ptr<int32_t>();
    for (int64_t row = 0; row < rows; ++row) {
        output[row] = featureHashToTokenId(hashFeatureRowCpu(bytes + row * row_bytes, row_bytes));
    }
    return hashes;
}

}  // namespace rtp_llm
