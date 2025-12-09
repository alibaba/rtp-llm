#include "rtp_llm/cpp/devices/GraphBase.h"
#include <optional>
#include <algorithm>

namespace rtp_llm {

void GraphBase::capturePrefill() {
    RTP_LLM_LOG_INFO("Capture Prefill Start");
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
        int seq_len = capture_range_[i];
        RTP_LLM_LOG_INFO("capture range for seq len: %d", seq_len);
        PyModelInputs inputs;
        // for attention, it always run the max_bs, so when we run `forward`, the real batch size is not sure
        // we will transfer a `batch size tensor(int)` for `copy kernel`.
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, max_bs_, seq_len);
        // Prefill-specific settings
        inputs.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = seq_len;
        inputs.attention_inputs.input_lengths.data_ptr<int>()[0] = seq_len;
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            capture_mem_hold_.py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params;
        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            inputs.bert_embedding_inputs.combo_position_ids =
                inputs.bert_embedding_inputs.combo_position_ids.slice(0, 0, seq_len);
            inputs.bert_embedding_inputs.combo_tokens_type_ids =
                inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, 0, seq_len);
        }
        graph_mem_holds_[seq_len] = createCaptureMemoryHold(inputs, max_bs_ * num_tokens_per_bs_);
        graph_mem_holds_[seq_len].decoder_layer_hidden_states_ =
            graph_mem_holds_[seq_len].decoder_layer_hidden_states_.slice(0, 0, seq_len);
        capturePrefillOneSeqLen(seq_len);
        replayAndSyncCheck(seq_len, "seq len");
        RTP_LLM_LOG_INFO("capture success for seq_len: %d", seq_len);
    }
    RTP_LLM_LOG_INFO("Capture Prefill End");
}

std::vector<int> GraphBase::getPrefillSequenceLengthsToCapture() {
    // prefill_capture_seq_lens_ must be provided from Python and cannot be empty
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

void GraphBase::capturePrefillOneSeqLen(int seq_len) {
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

void GraphBase::replayPrefill(int seq_len) {
    replayGraphImpl(seq_len);
}

void GraphBase::tryGetRealGraphPrefillSeqLen(PyModelInputs& inputs) {
    state_.current_seq_len = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), state_.current_seq_len);
    state_.current_real_graph_seq_len = *it;
    state_.current_batch_size         = inputs.attention_inputs.input_lengths.size(0);
}

}  // namespace rtp_llm
