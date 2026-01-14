#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"

namespace rtp_llm {
void CudaGraphRunner::replayDecode(int bs) {
    replayGraph(bs);
}

std::vector<int> CudaGraphRunner::getDecodeBatchSizesToCapture() {
    // If decode_capture_batch_sizes_ is provided from Python, use it directly
    if (!decode_capture_batch_sizes_.empty()) {
        RTP_LLM_LOG_INFO("Using decode capture batch sizes from Python: %zu sizes", decode_capture_batch_sizes_.size());
        return decode_capture_batch_sizes_;
    }

    // Otherwise, use default logic
    std::vector<int> capture_bs;
    int              max_generate_batch_size = max_bs_;
    RTP_LLM_LOG_INFO("max_generate_batch_size for cuda graph: %d", max_generate_batch_size);
    // Add range 1 to 32 (inclusive)
    for (int i = 1; i <= std::min(32, max_generate_batch_size); i += 1) {
        capture_bs.push_back(i);
    }
    // Add range from 48 to max_generate_batch_size (exclusive), stepping by 16
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (capture_bs[capture_bs.size() - 1] != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

void CudaGraphRunner::captureDecodeOneBatchSize(int bs) {
    captureOneGraphInstance(bs, "batch size");
}

void CudaGraphRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, bs, bs * num_tokens_per_bs_);

        // calculate context_total_kv_length
        int max_input_len                               = inputs.attention_inputs.input_lengths.max().item<int>();
        int max_prefix_len                              = inputs.attention_inputs.prefix_lengths.max().item<int>();
        inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);

        graph_instances_[bs].mem_hold_ = createCaptureMemoryHold(inputs, bs * num_tokens_per_bs_);
        captureDecodeOneBatchSize(bs);
        replayAndSyncCheck(bs, "batch size");
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}
}  // namespace rtp_llm