#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"

namespace rtp_llm {
void CudaGraphRunner::capturePrefill() {
    RTP_LLM_LOG_INFO("Capture Prefill Start");
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
        int           seq_len = capture_range_[i];
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
        inputs.attention_inputs.prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
        inputs.attention_inputs.dtype          = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
        inputs.attention_inputs.padding_offset =
            capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(
                0, 0, max_bs_ * num_tokens_per_bs_);
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            capture_mem_hold_.py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params;
        // Copy BertEmbeddingInputs from capture_mem_hold_
        inputs.bert_embedding_inputs = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;

        graph_instances_[seq_len].mem_hold_ =
            CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, max_bs_ * num_tokens_per_bs_),
                              inputs,
                              kv_cache_block_offset_,
                              is_prefill_cuda_graph_mode_);
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

    // 5. Very large sequence lengths (10000-20000): step size 200
    for (int i = 10000; i <= std::min(20000, max_possible_length); i += 200) {
        capture_seq_lens.push_back(i);
    }

    // 6. Extremely large sequence lengths (20000-max_possible_length): step size 500
    for (int i = 20000; i <= max_possible_length; i += 500) {
        capture_seq_lens.push_back(i);
    }

    // 7. Add key length points from your test data (frequently occurring lengths)
    std::vector<int> key_lengths = {
        697,  703,  720,  793,  797,  837,  844,  856,  889,  971,   1097,  1120, 1132, 1138, 1172,
        1214, 1220, 1225, 1240, 1250, 1280, 1283, 1293, 1300, 1352,  1358,  1378, 1384, 1389, 1406,
        1497, 1560, 1580, 1582, 1644, 1685, 1709, 1726, 1780, 1854,  1887,  1919, 1966, 1992, 2081,
        2130, 2208, 2231, 2238, 2274, 2377, 2454, 2535, 2582, 2740,  2842,  2945, 3149, 3170, 3210,
        3300, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 11489  // Maximum value from test data
    };

    // // 8. Add common power-of-2 lengths and multiples of 512 (common in transformer models)
    // std::vector<int> common_lengths = {
    //     256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120,
    //     5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240,
    //     10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360,
    //     15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480,
    //     20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600,
    //     26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720,
    //     31232, 31744, 32256, 32768  // 64 * 512
    // };

    // Add key lengths from test data
    for (int len : key_lengths) {
        if (len <= max_possible_length) {
            capture_seq_lens.push_back(len);
        }
    }

    // // Add common lengths
    // for (int len : common_lengths) {
    //     if (len <= max_possible_length) {
    //         capture_seq_lens.push_back(len);
    //     }
    // }

    // Remove duplicates and sort
    std::sort(capture_seq_lens.begin(), capture_seq_lens.end());
    capture_seq_lens.erase(std::unique(capture_seq_lens.begin(), capture_seq_lens.end()), capture_seq_lens.end());
    if (capture_seq_lens[capture_seq_lens.size() - 1] != max_possible_length) {
        capture_seq_lens.push_back(max_possible_length);
    }
    RTP_LLM_LOG_INFO("Total sequence lengths to capture: %zu", capture_seq_lens.size());
    RTP_LLM_LOG_INFO("Min length: %d, Max length: %d", capture_seq_lens.front(), capture_seq_lens.back());

    return capture_seq_lens;
}

void CudaGraphRunner::capturePrefillOneSeqLen(int seq_len) {
    auto inputs = graph_instances_[seq_len].mem_hold_.py_model_inputs_;
    // WarmUp twice
    RTP_LLM_LOG_INFO("WarmUp for seq len %d start.", seq_len);
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
}

void CudaGraphRunner::replayPrefill(int seq_len) {
    graph_instances_[seq_len].graph_.replay();
}
}  // namespace rtp_llm
