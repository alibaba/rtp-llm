#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#if USING_CUDA
#include <cuda_runtime.h>
#endif

namespace rtp_llm {
int CudaGraphRunner::decodeGraphKey(int batch_size, int tokens_per_request) const {
    RTP_LLM_CHECK_WITH_INFO(batch_size > 0 && batch_size <= static_cast<int>(max_bs_) && tokens_per_request > 0
                                && tokens_per_request <= num_tokens_per_bs_,
                            "invalid adaptive graph dimensions: batch=%d tokens_per_request=%d",
                            batch_size,
                            tokens_per_request);
    return batch_size + (num_tokens_per_bs_ - tokens_per_request) * (static_cast<int>(max_bs_) + 1);
}

void CudaGraphRunner::replayDecode(int bs) {
    replayGraph(bs);
}

std::vector<int> CudaGraphRunner::getDecodeBatchSizesToCapture() {
    // If decode_capture_batch_sizes_ is provided from Python, use it directly
    if (!decode_capture_batch_sizes_.empty()) {
        RTP_LLM_LOG_INFO("Using decode capture batch sizes from Python: %zu sizes", decode_capture_batch_sizes_.size());
        // Sort in ascending order (from small to large)
        std::sort(decode_capture_batch_sizes_.begin(), decode_capture_batch_sizes_.end());
        return decode_capture_batch_sizes_;
    }

    // Otherwise, use default logic
    std::vector<int> capture_bs;
    int              max_generate_batch_size = max_bs_;
    RTP_LLM_LOG_INFO("max_generate_batch_size for cuda graph: %d", max_generate_batch_size);
    // Add key batch sizes up to 32
    for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32}) {
        if (i <= max_generate_batch_size) {
            capture_bs.push_back(i);
        }
    }
    // Add range from 48 to max_generate_batch_size, stepping by 16
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (capture_bs[capture_bs.size() - 1] != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

void CudaGraphRunner::captureDecodeOneBatchSize(int bs, bool needs_distributed_warmup) {
    captureOneGraphInstance(bs, "batch size", needs_distributed_warmup);
}

void CudaGraphRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    RTP_LLM_CHECK_WITH_INFO(!adaptive_verification_ || (is_target_verify_ && num_tokens_per_bs_ > 1),
                            "adaptive verification graphs require a multi-token target model");
    // Pre-initialize all graph instances with keep_graph based on debug mode
    const int min_tokens = adaptive_verification_ ? 1 : num_tokens_per_bs_;
    for (int tokens = num_tokens_per_bs_; tokens >= min_tokens; --tokens) {
        for (int bs : capture_range_) {
            graph_instances_.try_emplace(decodeGraphKey(bs, tokens), enable_cuda_graph_debug_mode_);
        }
    }
    int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int           bs = capture_range_[i];
        for (int tokens = num_tokens_per_bs_; tokens >= min_tokens; --tokens) {
            const int     key = decodeGraphKey(bs, tokens);
            PyModelInputs inputs;
            // Prepare common inputs using shared function
            prepareCaptureInputs(inputs, bs, bs * tokens);
            if (adaptive_verification_) {
                inputs.attention_inputs.input_lengths.fill_(tokens);
                inputs.attention_inputs.prefix_lengths.fill_(max_seq_len_ - tokens);
                auto offsets = torch::arange(bs + 1, options_cpu_int32_);
                inputs.attention_inputs.cu_seqlens_host.copy_(offsets * tokens);
                inputs.attention_inputs.cu_seqlens.copy_(offsets * tokens);
                inputs.attention_inputs.cu_kv_seqlens.copy_(offsets * max_seq_len_);
            }

            // calculate context_total_kv_length
            int max_input_len  = inputs.attention_inputs.input_lengths.max().item<int>();
            int max_prefix_len = 0;
            if (inputs.attention_inputs.prefix_lengths.defined()
                && inputs.attention_inputs.prefix_lengths.numel() > 0) {
                max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
            }
            inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);

            graph_instances_[key].mem_hold_ = createCaptureMemoryHold(inputs, bs * tokens);
            graph_instances_[key].mem_hold_.attn_pyobj_ =
                py_attn_pyobj_method_(graph_instances_[key].mem_hold_.py_model_inputs_, true);
            try {
                // The largest graph is captured first. Fold the one distributed
                // Warmup forward into its first existing warmup execution so
                // multi-rank MegaMoE still synchronizes before its first JIT/
                // kernel launch without restoring a standalone full forward.
                captureDecodeOneBatchSize(key, i == capture_range_size - 1 && tokens == num_tokens_per_bs_);
                replayAndSyncCheck(key, "batch size");
                RTP_LLM_LOG_INFO(
                    "capture success for batch size: %d, tokens per request: %d, key: %d", bs, tokens, key);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("CUDA graph capture failed for decode batch size %d: %s", bs, e.what());
#if USING_CUDA
            cudaGetLastError();
            auto sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess) {
                RTP_LLM_LOG_ERROR("cudaDeviceSynchronize after capture failure returned: %s",
                                  cudaGetErrorString(sync_err));
                cudaGetLastError();
            }
#endif
            throw;
        }
        }
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}
}  // namespace rtp_llm
