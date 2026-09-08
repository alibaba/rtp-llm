#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include <optional>

namespace rtp_llm {
void CudaGraphRunner::capturePrefill() {
    RTP_LLM_LOG_INFO("Capture Prefill Start");
    // Pre-initialize all graph instances with keep_graph based on debug mode
    for (int seq_len : capture_range_) {
        graph_instances_.try_emplace(seq_len, enable_cuda_graph_debug_mode_);
    }
    int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int seq_len = capture_range_[i];
        RTP_LLM_LOG_INFO("capture range for seq len: %d", seq_len);
        buildBucketInstance(seq_len);
        capturePrefillOneSeqLen(seq_len);
        cuda_graph::finish_capture_session();
        replayAndSyncCheck(seq_len, "seq len");
        RTP_LLM_LOG_INFO("capture success for seq_len: %d", seq_len);
    }
    RTP_LLM_LOG_INFO("Capture Prefill End");
}

std::vector<int> CudaGraphRunner::getPrefillSequenceLengthsToCapture() {
    // MTP draft prefill: capture at multiples of num_tokens_per_bs_
    if (isMtpDraftPrefillCudaGraph()) {
        std::vector<int> result;
        for (int i = 1; i <= max_bs_; ++i) {
            result.push_back(i * num_tokens_per_bs_);
        }
        RTP_LLM_LOG_INFO(
            "Draft model prefill: capture seq_lens at %d intervals, %zu total (max_bs=%d, num_tokens_per_bs=%d)",
            num_tokens_per_bs_,
            result.size(),
            max_bs_,
            num_tokens_per_bs_);
        return result;
    }

    // Embedding model prefill: use Python-provided capture seq_lens
    RTP_LLM_CHECK_WITH_INFO(!prefill_capture_seq_lens_.empty(),
                            "prefill_capture_seq_lens_ must be provided from Python and cannot be empty");

    RTP_LLM_LOG_INFO("Using prefill capture sequence lengths from Python: %zu lengths",
                     prefill_capture_seq_lens_.size());

    // Sort and remove duplicates
    std::vector<int> result = prefill_capture_seq_lens_;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

    RTP_LLM_LOG_INFO(
        "Total sequence lengths to capture: %zu (min: %d, max: %d)", result.size(), result.front(), result.back());
    return result;
}

void CudaGraphRunner::capturePrefillOneSeqLen(int seq_len) {
    try {
        captureOneGraphInstance(seq_len, "seq len");
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Exception in capturePrefillOneSeqLen for seq_len %d: %s", seq_len, e.what());
        throw;
    } catch (...) {
        RTP_LLM_LOG_ERROR("Unknown exception in capturePrefillOneSeqLen for seq_len %d", seq_len);
        throw;
    }
}

void CudaGraphRunner::replayPrefill(int seq_len) {
    replayGraph(seq_len);
}
}  // namespace rtp_llm
