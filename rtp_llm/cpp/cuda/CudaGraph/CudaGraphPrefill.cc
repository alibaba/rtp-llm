#include "rtp_llm/cpp/cuda/CudaGraph/CudaGraphRunner.h"
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
        PyModelInputs inputs;
        // for attention, it always run the max_bs, so when we run `forward`, the real batch size is not sure
        // we will transfer a `batch size tensor(int)` for `copy kernel`.
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, max_bs_, seq_len);
        // Prefill-specific settings, one the first seq is valid, the post ones are all empty
        if (is_prefill_cuda_graph_mode_ && num_tokens_per_bs_ == max_seq_len_) {
            // embedding model, without kv cache
            inputs.attention_inputs.prefix_lengths.fill_(0);
            // Must set cu_seqlens/cu_kv_seqlens/input_lengths to match actual seq_len,
            // otherwise FlashInfer plans for max_seq_len tokens but q/k/v only have seq_len tokens
            inputs.attention_inputs.cu_seqlens.data_ptr<int>()[0]    = 0;
            inputs.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = seq_len;
            inputs.attention_inputs.input_lengths.data_ptr<int>()[0] = seq_len;
        } else {
            inputs.attention_inputs.cu_seqlens.fill_(seq_len);
            inputs.attention_inputs.input_lengths.fill_(0);
            int kv_len     = max_seq_len_ + seq_len;
            int prefix_len = kv_len;
            inputs.attention_inputs.cu_kv_seqlens.fill_(kv_len);
            inputs.attention_inputs.prefix_lengths.fill_(prefix_len);
            inputs.attention_inputs.cu_seqlens.data_ptr<int>()[0]    = 0;
            inputs.attention_inputs.cu_kv_seqlens.data_ptr<int>()[0] = 0;
            inputs.attention_inputs.input_lengths.data_ptr<int>()[0] = seq_len;
        }

        inputs.attention_inputs.context_total_kv_length = seq_len;
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            capture_mem_hold_.py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params;
        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            inputs.bert_embedding_inputs.combo_position_ids =
                inputs.bert_embedding_inputs.combo_position_ids.slice(0, 0, seq_len);
            inputs.bert_embedding_inputs.combo_tokens_type_ids =
                inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, 0, seq_len);
        }
        graph_instances_[seq_len].mem_hold_ = createCaptureMemoryHold(inputs, max_bs_ * num_tokens_per_bs_);
        graph_instances_[seq_len].mem_hold_.attn_pyobj_ =
            py_attn_pyobj_method_(graph_instances_[seq_len].mem_hold_.py_model_inputs_, true);
        graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_ =
            graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_.slice(0, 0, seq_len);
        capturePrefillOneSeqLen(seq_len);
        replayAndSyncCheck(seq_len, "seq len");
        RTP_LLM_LOG_INFO("capture success for seq_len: %d", seq_len);
    }
    RTP_LLM_LOG_INFO("Capture Prefill End");
}

std::vector<int> CudaGraphRunner::getPrefillSequenceLengthsToCapture() {
    // Draft model prefill (num_tokens_per_bs_ != max_seq_len_): generate range 1 ~ max_bs_ * num_tokens_per_bs_
    if (num_tokens_per_bs_ != max_seq_len_) {
        int              max_seq = max_bs_ * num_tokens_per_bs_;
        std::vector<int> result;
        for (int i = 1; i <= max_seq; ++i) {
            result.push_back(i);
        }
        RTP_LLM_LOG_INFO("Draft model prefill: capture seq_lens 1~%d (max_bs=%d, num_tokens_per_bs=%d)",
                         max_seq,
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
