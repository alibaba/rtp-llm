#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"

namespace rtp_llm {
void CudaGraphDecodeRunner::replayDecode(int bs) {
    replayGraph(bs);
}

void CudaGraphDecodeRunner::captureDecodeOneBatchSize(int bs) {
    captureOneGraphInstance(bs, "batch size");
}

void CudaGraphDecodeRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    // Pre-initialize all graph instances with keep_graph based on debug mode
    for (int bs : capture_dispatcher_.captureRange()) {
        graph_instances_.try_emplace(bs, graph_params_.enable_cuda_graph_debug_mode);
    }
    const auto& range              = capture_dispatcher_.captureRange();
    int         capture_range_size = static_cast<int>(range.size());
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int           bs = range[static_cast<size_t>(i)];
        PyModelInputs inputs;
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, bs, bs * graph_params_.num_tokens_per_bs);

        // calculate context_total_kv_length
        int max_input_len  = inputs.attention_inputs.input_lengths.max().item<int>();
        int max_prefix_len = 0;
        if (inputs.attention_inputs.prefix_lengths.defined()) {
            max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
        }
        inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);

        graph_instances_[bs].mem_hold_ = createCaptureMemoryHold(inputs, bs * graph_params_.num_tokens_per_bs);
        graph_instances_[bs].mem_hold_.attn_pyobj_ =
            py_attn_pyobj_method_(graph_instances_[bs].mem_hold_.py_model_inputs_, true);
        captureDecodeOneBatchSize(bs);
        replayAndSyncCheck(bs, "batch size");
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}
}  // namespace rtp_llm
