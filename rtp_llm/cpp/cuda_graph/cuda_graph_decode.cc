#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include <limits>

namespace rtp_llm {
std::vector<int> CudaGraphRunner::defaultDecodeBatchSizesToCapture(int max_batch_size) {
    std::vector<int> buckets;
    for (int batch : {1, 8, 16, 24, 32}) {
        if (batch <= max_batch_size) {
            buckets.push_back(batch);
        }
    }
    for (int64_t batch = 48; batch <= max_batch_size; batch += 16) {
        buckets.push_back(static_cast<int>(batch));
    }
    if (max_batch_size > 0 && (buckets.empty() || buckets.back() != max_batch_size)) {
        buckets.push_back(max_batch_size);
    }
    return buckets;
}

void CudaGraphRunner::replayDecode(int bs) {
    replayGraph(bs);
}

std::vector<int> CudaGraphRunner::getDecodeBatchSizesToCapture() {
    // Also validate direct C++ callers before allocating or slicing capture buffers.
    RTP_LLM_CHECK_WITH_INFO(max_bs_ > 0 && max_bs_ < std::numeric_limits<int>::max(),
                            "Decode CUDA Graph capacity must be positive and below INT_MAX");
    // If decode_capture_batch_sizes_ is provided from Python, use it directly
    if (!decode_capture_batch_sizes_.empty()) {
        for (int bucket : decode_capture_batch_sizes_) {
            RTP_LLM_CHECK_WITH_INFO(bucket > 0 && static_cast<size_t>(bucket) <= max_bs_,
                                    "DECODE_CAPTURE_CONFIG bucket %d exceeds Decode CUDA Graph capacity %zu "
                                    "or is non-positive",
                                    bucket,
                                    max_bs_);
        }
        RTP_LLM_LOG_INFO("Using decode capture batch sizes from Python: %zu sizes", decode_capture_batch_sizes_.size());
        // Sort in ascending order (from small to large)
        std::sort(decode_capture_batch_sizes_.begin(), decode_capture_batch_sizes_.end());
        decode_capture_batch_sizes_.erase(
            std::unique(decode_capture_batch_sizes_.begin(), decode_capture_batch_sizes_.end()),
            decode_capture_batch_sizes_.end());
        return decode_capture_batch_sizes_;
    }

    RTP_LLM_LOG_INFO("max_generate_batch_size for cuda graph: %d", static_cast<int>(max_bs_));
    return defaultDecodeBatchSizesToCapture(static_cast<int>(max_bs_));
}

void CudaGraphRunner::captureDecodeOneBatchSize(int bs) {
    captureOneGraphInstance(bs, "batch size");
}

void CudaGraphRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    // Pre-initialize all graph instances with keep_graph based on debug mode
    for (int bs : capture_range_) {
        graph_instances_.try_emplace(bs, enable_cuda_graph_debug_mode_);
    }
    int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, bs, bs * num_tokens_per_bs_);

        // calculate context_total_kv_length
        int max_input_len  = inputs.attention_inputs.input_lengths.max().item<int>();
        int max_prefix_len = 0;
        if (inputs.attention_inputs.prefix_lengths.defined() && inputs.attention_inputs.prefix_lengths.numel() > 0) {
            max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
        }
        inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);
        // capture-specific metadata above was written after prepareCaptureInputs synchronized the tag map.
        refreshTaggedAttentionInputs(inputs);

        auto& mem_hold = graph_instances_[bs].mem_hold_;
        mem_hold       = createCaptureMemoryHold(inputs, bs * num_tokens_per_bs_);
        mem_hold.attn_pyobj_ = prepareFmhaImpl(mem_hold.py_model_inputs_, true);
        captureDecodeOneBatchSize(bs);
        cuda_graph::finish_capture_session();
        replayAndSyncCheck(bs, "batch size");
        // Retain graph-owned buffers until the first replay is known complete.
        // A launch/synchronization failure must take the same fail-closed path
        // as a partially captured graph instead of deleting in-flight storage.
        capture_session_may_be_dirty_.store(false, std::memory_order_release);
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}
}  // namespace rtp_llm
