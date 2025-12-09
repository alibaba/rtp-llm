#include "rtp_llm/cpp/devices/GraphBase.h"

namespace rtp_llm {

void GraphBase::replayDecode(int bs) {
    replayGraphImpl(bs);
}

std::vector<int> GraphBase::getDecodeBatchSizesToCapture() {
    // If decode_capture_batch_sizes_ is provided from Python, use it directly
    if (!decode_capture_batch_sizes_.empty()) {
        RTP_LLM_LOG_INFO("Using decode capture batch sizes from Python: %zu sizes", decode_capture_batch_sizes_.size());
        return decode_capture_batch_sizes_;
    }

    // Otherwise, use default logic
    std::vector<int> capture_bs;
    int              max_generate_batch_size = max_bs_;
    RTP_LLM_LOG_INFO("max_generate_batch_size for graph: %d", max_generate_batch_size);
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

void GraphBase::captureDecodeOneBatchSize(int bs) {
    captureOneGraphInstance(bs, "batch size");
}

void GraphBase::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, bs, bs * num_tokens_per_bs_);

        graph_mem_holds_[bs] = createCaptureMemoryHold(inputs, bs * num_tokens_per_bs_);
        captureDecodeOneBatchSize(bs);
        replayAndSyncCheck(bs, "batch size");
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}

void GraphBase::tryGetRealGraphDecodeBatchSize(PyModelInputs& inputs) {
    int graph_bs              = inputs.attention_inputs.input_lengths.size(0);
    state_.current_batch_size = graph_bs;
    RTP_LLM_LOG_INFO("canRun judge for batch size: %d", graph_bs);
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state_.current_batch_size);
    state_.current_real_graph_bs = *it;
    RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", state_.current_real_graph_bs);
    state_.seq_len_sum = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    RTP_LLM_LOG_INFO("can run graph for decode");
}

}  // namespace rtp_llm
