#include "rtp_llm/cpp/devices/graph_common/GraphBaseRunner.h"
#include <algorithm>

namespace rtp_llm {

void GraphBaseRunner::replayPrefill(int seq_len) {
    replayGraph(seq_len);
}

std::vector<int> GraphBaseRunner::getPrefillSequenceLengthsToCapture() const {
    RTP_LLM_CHECK_WITH_INFO(!prefill_capture_seq_lens_.empty(),
                            "prefill_capture_seq_lens_ must be provided from Python and cannot be empty");
    std::vector<int> result = prefill_capture_seq_lens_;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

void GraphBaseRunner::capturePrefillOneSeqLen(int seq_len) {
    captureOneGraphInstance(seq_len, "seq len");
}

void GraphBaseRunner::capturePrefill() {
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
        int           seq_len = capture_range_[i];
        PyModelInputs inputs;
        prepareCaptureInputs(inputs, max_bs_, seq_len);
        inputs.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = seq_len;
        inputs.attention_inputs.cu_kv_seqlens.data_ptr<int>()[1] = seq_len;
        inputs.attention_inputs.input_lengths.data_ptr<int>()[0] = seq_len;
        inputs.attention_inputs.context_total_kv_length          = seq_len;
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
    }
}

}  // namespace rtp_llm
