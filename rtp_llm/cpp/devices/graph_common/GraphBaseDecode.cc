#include "rtp_llm/cpp/devices/graph_common/GraphBaseRunner.h"
#include <algorithm>

namespace rtp_llm {

void GraphBaseRunner::replayDecode(int bs) {
    replayGraph(bs);
}

std::vector<int> GraphBaseRunner::getDecodeBatchSizesToCapture() const {
    if (!decode_capture_batch_sizes_.empty()) {
        return decode_capture_batch_sizes_;
    }
    std::vector<int> capture_bs;
    int              max_generate_batch_size = max_bs_;
    for (int i = 1; i <= std::min(32, max_generate_batch_size); i += 1) {
        capture_bs.push_back(i);
    }
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (!capture_bs.empty() && capture_bs.back() != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

void GraphBaseRunner::captureDecodeOneBatchSize(int bs) {
    captureOneGraphInstance(bs, "batch size");
}

void GraphBaseRunner::captureDecode() {
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        prepareCaptureInputs(inputs, bs, bs * num_tokens_per_bs_);
        int max_input_len                               = inputs.attention_inputs.input_lengths.max().item<int>();
        int max_prefix_len                              = inputs.attention_inputs.prefix_lengths.max().item<int>();
        inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);
        graph_instances_[bs].mem_hold_                  = createCaptureMemoryHold(inputs, bs * num_tokens_per_bs_);
        graph_instances_[bs].mem_hold_.attn_pyobj_ =
            py_attn_pyobj_method_(graph_instances_[bs].mem_hold_.py_model_inputs_, true);
        captureDecodeOneBatchSize(bs);
        replayAndSyncCheck(bs, "batch size");
    }
}

}  // namespace rtp_llm
