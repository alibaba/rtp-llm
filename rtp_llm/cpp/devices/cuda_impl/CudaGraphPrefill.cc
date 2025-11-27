#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include <optional>

namespace rtp_llm {
void CudaGraphRunner::capturePrefill() {
    RTP_LLM_LOG_INFO("Capture Prefill Start");
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
        int seq_len = capture_range_[i];
        RTP_LLM_LOG_INFO("capture range for seq len: %d", seq_len);
        PyModelInputs inputs;
        inputs.input_ids = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, seq_len);
        // for attention, it always run the max_bs, so when we run `forward`, the real batch size is not sure
        // we will transfer a `batch size tensor(int)` for `copy kernel`.
        inputs.attention_inputs.input_lengths =
            capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, max_bs_);
        inputs.attention_inputs.sequence_lengths =
            capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, max_bs_);
        // we capture the max_block_ids
        inputs.attention_inputs.kv_cache_block_id_device =
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, max_bs_);
        inputs.attention_inputs.kv_cache_block_id_host =
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, max_bs_);
        // pinned memory
        inputs.attention_inputs.cu_seqlens =
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, max_bs_ + 1);
        inputs.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = seq_len;
        inputs.attention_inputs.input_lengths.data_ptr<int>()[0] = seq_len;
        inputs.attention_inputs.prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
        inputs.attention_inputs.dtype          = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
        inputs.attention_inputs.padding_offset =
            capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, seq_len);
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            capture_mem_hold_.py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params;
        if (inputs.attention_inputs.prefill_cuda_graph_copy_params) {
            inputs.attention_inputs.prefill_cuda_graph_copy_params->compact_attn_buf =
                inputs.attention_inputs.prefill_cuda_graph_copy_params->compact_attn_buf.slice(0, 0, seq_len);
        }
        // Copy BertEmbeddingInputs from capture_mem_hold_
        inputs.bert_embedding_inputs = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            inputs.bert_embedding_inputs.combo_position_ids =
                inputs.bert_embedding_inputs.combo_position_ids.slice(0, 0, seq_len);
            inputs.bert_embedding_inputs.combo_tokens_type_ids =
                inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, 0, seq_len);
        }
        graph_instances_[seq_len].mem_hold_ =
            CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, max_bs_ * num_tokens_per_bs_),
                              inputs,
                              kv_cache_block_offset_,
                              is_prefill_cuda_graph_mode_);
        graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_ =
            graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_.slice(0, 0, seq_len);
        capturePrefillOneSeqLen(seq_len);
        RTP_LLM_LOG_INFO("replay start check seq len for %d", seq_len);
        replayPrefill(seq_len);
        cudaDeviceSynchronize();
        RTP_LLM_LOG_INFO("replay end check seq len for %d", seq_len);
        RTP_LLM_LOG_INFO("capture success for seq_len: %d", seq_len);
    }
    RTP_LLM_LOG_INFO("Capture Prefill End");
}

std::vector<int> CudaGraphRunner::getPrefillSequenceLengthsToCapture() {
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

void CudaGraphRunner::capturePrefillOneSeqLen(int seq_len) {
    try {
        RTP_LLM_LOG_INFO("WarmUp for seq len %d start.", seq_len);
        auto inputs = graph_instances_[seq_len].mem_hold_.py_model_inputs_;
        // WarmUp twice
        py_forward_method_(inputs);

        py_forward_method_(inputs);

        RTP_LLM_LOG_INFO("WarmUp for seq len %d successfully.", seq_len);
        {
            CudaGraphStreamLife  stream_life(capture_stream_, device_);
            at::cuda::CUDAGraph& graph               = graph_instances_[seq_len].graph_;
            auto                 output_dot_filename = "";
            if (enable_cuda_graph_debug_mode_) {
                graph.enable_debug_mode();
                output_dot_filename = "cuda_graph_visualization.dot";
            }
            RTP_LLM_LOG_INFO("Capture for seq len %d begin.", seq_len);
            graph.capture_begin();
            CaptureCheck::in_cuda_graph_capture = true;
            auto py_outputs_obj                 = py_forward_method_(inputs);
            auto outputs                        = py_outputs_obj.cast<PyModelOutputs>();
            graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
            graph.capture_end();
            RTP_LLM_LOG_INFO("Capture for seq len %d end.", seq_len);
            CaptureCheck::in_cuda_graph_capture = false;
            if (outputs.params_ptr->check_recycle()) {
                graph_instances_[seq_len].mem_hold_.params_ptr =
                    ParamsBasePtr(outputs.params_ptr.get(), [&](ParamsBase* ptr) {});
            } else {
                graph_instances_[seq_len].mem_hold_.params_ptr = outputs.params_ptr;
            }

            if (enable_cuda_graph_debug_mode_) {
                graph.debug_dump(output_dot_filename);
            }
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Exception in capturePrefillOneSeqLen for seq_len %d: %s", seq_len, e.what());
        // Ensure we reset the capture state even if an exception occurs
        CaptureCheck::in_cuda_graph_capture = false;
        throw;
    } catch (...) {
        RTP_LLM_LOG_ERROR("Unknown exception in capturePrefillOneSeqLen for seq_len %d", seq_len);
        // Ensure we reset the capture state even if an exception occurs
        CaptureCheck::in_cuda_graph_capture = false;
        throw;
    }
}

void CudaGraphRunner::replayPrefill(int seq_len) {
    graph_instances_[seq_len].graph_.replay();
}
}  // namespace rtp_llm
