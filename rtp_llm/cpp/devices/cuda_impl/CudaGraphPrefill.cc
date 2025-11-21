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
        // inputs.attention_inputs.padding_offset =
        //     capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(
        //         0, 0, max_bs_ * num_tokens_per_bs_);
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
    std::vector<int> capture_seq_lens;

    // Calculate the maximum possible sequence length
    int max_possible_length = max_seq_len_ * max_bs_;
    RTP_LLM_LOG_INFO("Maximum possible sequence length: %d (max_seq_len_: %d * max_bs_: %d)",
                     max_possible_length,
                     max_seq_len_,
                     max_bs_);

    // Optimized for max sequence length of 16384
    // 1. Small sequence lengths (10-500): step size 5, covering most short sequences
    for (int i = 10; i <= std::min(500, max_possible_length); i += 5) {
        capture_seq_lens.push_back(i);
    }

    // 2. Medium sequence lengths (500-2000): step size 20
    for (int i = 500; i <= std::min(2000, max_possible_length); i += 20) {
        capture_seq_lens.push_back(i);
    }

    // 3. Large sequence lengths (2000-5000): step size 50
    for (int i = 2000; i <= std::min(5000, max_possible_length); i += 50) {
        capture_seq_lens.push_back(i);
    }

    // 4. Extra large sequence lengths (5000-10000): step size 100
    for (int i = 5000; i <= std::min(10000, max_possible_length); i += 100) {
        capture_seq_lens.push_back(i);
    }

    // 5. Very large sequence lengths (10000-16384): step size 128 (more granular for 16K range)
    for (int i = 10000; i <= std::min(16384, max_possible_length); i += 128) {
        capture_seq_lens.push_back(i);
    }

    // 6. Add common power-of-2 lengths and multiples of 512 (common in transformer models)
    // Optimized for 16K max length
    std::vector<int> common_lengths = {
        256,   512,   1024,  1536,  2048,  2560,  3072,  3584,  4096,  4608,  5120,
        5632,  6144,  6656,  7168,  7680,  8192,  8704,  9216,  9728,  10240, 10752,
        11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384  // Max 16K
    };

    // Add common lengths
    for (int len : common_lengths) {
        if (len <= max_possible_length) {
            capture_seq_lens.push_back(len);
        }
    }

    std::vector<int> key_lengths = {
        8915,  703,  2130,  2740,  1389, 1497,  697,   7945,  1097, 14,   10866, 399,  9145, 1580, 1384, 1172, 1992,
        2238,  629,  8948,  9591,  1919, 10218, 856,   11328, 673,  9227, 2842,  837,  4544, 1644, 1132, 1358, 1240,
        1120,  1214, 10380, 9130,  115,  240,   11489, 2535,  7437, 1343, 797,   1854, 2454, 9416, 2945, 321,  1220,
        1726,  1582, 697,   2582,  3149, 8711,  1225,  576,   77,   356,  252,   1283, 685,  248,  1685, 1709, 793,
        10570, 889,  9517,  10486, 1378, 1406,  2231,  720,   2208, 6614, 7536,  317,  1966, 2377, 7757, 411,  844,
        1887,  311,  8300,  1780,  971,  7481,  9136,  1293,  10,   1352, 8844,  2081, 2274, 125,  1560, 6};

    // Add key lengths from test data (only those <= 16384)
    for (int len : key_lengths) {
        if (len <= std::min(16384, max_possible_length)) {
            capture_seq_lens.push_back(len);
        }
    }

    capture_seq_lens.erase(std::remove_if(capture_seq_lens.begin(),
                                          capture_seq_lens.end(),
                                          [&](int len) { return len > max_perfill_cuda_graph_len_; }),
                           capture_seq_lens.end());

    // Remove duplicates and sort
    std::sort(capture_seq_lens.begin(), capture_seq_lens.end());
    capture_seq_lens.erase(std::unique(capture_seq_lens.begin(), capture_seq_lens.end()), capture_seq_lens.end());

    if (max_possible_length <= max_perfill_cuda_graph_len_
        && capture_seq_lens[capture_seq_lens.size() - 1] != max_possible_length) {
        capture_seq_lens.push_back(max_possible_length);
    }

    RTP_LLM_LOG_INFO("Total sequence lengths to capture: %zu", capture_seq_lens.size());
    RTP_LLM_LOG_INFO("Min length: %d, Max length: %d", capture_seq_lens.front(), capture_seq_lens.back());

    return capture_seq_lens;
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
