#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"

namespace rtp_llm {
void CudaGraphRunner::replayDecode(int bs) {
    graph_instances_[bs].graph_.replay();
}

std::vector<int> CudaGraphRunner::getDecodeBatchSizesToCapture() {
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
    auto inputs = graph_instances_[bs].mem_hold_.py_model_inputs_;
    // WarmUp twice
    RTP_LLM_LOG_INFO("WarmUp for batch size %d start.", bs);
    py_forward_method_(inputs);

    py_forward_method_(inputs);
    RTP_LLM_LOG_INFO("WarmUp for batch size %d successfully.", bs);
    {
        CudaGraphStreamLife  stream_life(capture_stream_, device_);
        at::cuda::CUDAGraph& graph               = graph_instances_[bs].graph_;
        auto                 output_dot_filename = "";
        if (enable_cuda_graph_debug_mode_) {
            graph.enable_debug_mode();
            output_dot_filename = "cuda_graph_visualization.dot";
        }
        RTP_LLM_LOG_INFO("Capture for batch size %d begin.", bs);
        graph.capture_begin();
        CaptureCheck::in_cuda_graph_capture = true;
        auto py_outputs_obj                 = py_forward_method_(inputs);
        auto outputs                        = py_outputs_obj.cast<PyModelOutputs>();
        graph_instances_[bs].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
        graph.capture_end();
        RTP_LLM_LOG_INFO("Capture for batch size %d end.", bs);
        CaptureCheck::in_cuda_graph_capture = false;
        if (outputs.params_ptr->check_recycle()) {
            graph_instances_[bs].mem_hold_.params_ptr =
                ParamsBasePtr(outputs.params_ptr.get(), [&](ParamsBase* ptr) {});
        } else {
            graph_instances_[bs].mem_hold_.params_ptr = outputs.params_ptr;
        }

        if (enable_cuda_graph_debug_mode_) {
            graph.debug_dump(output_dot_filename);
        }
    }
}

void CudaGraphRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        inputs.input_ids = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, bs * num_tokens_per_bs_);
        inputs.attention_inputs.input_lengths =
            capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, bs);
        inputs.attention_inputs.sequence_lengths =
            capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, bs);
        // we capture the max_block_ids
        inputs.attention_inputs.kv_cache_block_id_device =
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, bs);
        inputs.attention_inputs.kv_cache_block_id_host =
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, bs);
        // pinned memory
        inputs.attention_inputs.cu_seqlens =
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, bs + 1);
        inputs.attention_inputs.prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
        inputs.attention_inputs.dtype          = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
        inputs.attention_inputs.padding_offset =
            capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, bs * num_tokens_per_bs_);
        // Copy BertEmbeddingInputs from capture_mem_hold_
        inputs.bert_embedding_inputs = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;

        graph_instances_[bs].mem_hold_ =
            CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, bs * num_tokens_per_bs_),
                              inputs,
                              kv_cache_block_offset_,
                              is_prefill_cuda_graph_mode_);
        captureDecodeOneBatchSize(bs);
        RTP_LLM_LOG_INFO("replay start check for %d", bs);
        replayDecode(bs);
        cudaDeviceSynchronize();
        RTP_LLM_LOG_INFO("replay end check for %d", bs);
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}
}  // namespace rtp_llm