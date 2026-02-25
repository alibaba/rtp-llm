#include "rtp_llm/cpp/devices/graph_common/GraphBaseRunner.h"

#include <algorithm>

namespace rtp_llm {

GraphBaseRunner::GraphBaseRunner(const DeviceInitParams& params,
                                 py::object              py_instance,
                                 c10::ScalarType         model_data_type,
                                 int                     num_tokens_per_bs,
                                 bool                    is_prefill_graph_mode,
                                 GraphBackendCallbacks   backend_callbacks):
    backend_callbacks_(std::move(backend_callbacks)),
    py_instance_(std::move(py_instance)),
    enable_graph_(params.hw_kernel_config.enable_cuda_graph),
    is_prefill_graph_mode_(is_prefill_graph_mode),
    enable_graph_debug_mode_(params.hw_kernel_config.enable_cuda_graph_debug_mode),
    num_tokens_per_bs_(num_tokens_per_bs),
    max_seq_len_(params.max_seq_len),
    seq_size_per_block_(params.tokens_per_block),
    hidden_size_(params.hidden_size),
    prefill_capture_seq_lens_(params.hw_kernel_config.prefill_capture_seq_lens),
    decode_capture_batch_sizes_(params.hw_kernel_config.decode_capture_batch_sizes),
    model_data_type_(model_data_type),
    options_device_int32_(torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false)),
    options_cpu_int32_(torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false)),
    options_device_float_(torch::TensorOptions().dtype(model_data_type).device(torch::kCUDA).requires_grad(false)),
    forward_event_(backend_callbacks_.event_device_type) {
    py::gil_scoped_acquire gil;
    RTP_LLM_CHECK_WITH_INFO(py_instance_ && !py_instance_.is_none(), "GraphBaseRunner constructor py_instance is null");
    py_attn_pyobj_method_ = py_instance_.attr("prepare_fmha_impl");
    py_forward_method_    = py_instance_.attr("forward");

    if (is_prefill_graph_mode_) {
        max_bs_ = params.runtime_config.fifo_scheduler_config.max_context_batch_size;
    } else {
        max_bs_ = params.concurrency_config.concurrency_limit;
    }
}

py::object GraphBaseRunner::normalForward(PyModelInputs& inputs) {
    auto attn_pyobj = py_attn_pyobj_method_(inputs, false);
    return py_forward_method_(inputs, attn_pyobj);
}

void GraphBaseRunner::copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor) {
    RTP_LLM_CHECK_WITH_INFO(source_tensor.dim() == target_tensor.dim(), "source and target dim mismatch");
    for (int i = 0; i < source_tensor.dim(); ++i) {
        RTP_LLM_CHECK_WITH_INFO(source_tensor.size(i) <= target_tensor.size(i),
                                "target dim[%d]=%d smaller than source dim=%d",
                                i,
                                target_tensor.size(i),
                                source_tensor.size(i));
    }
    torch::Tensor target_slice = target_tensor;
    for (int i = 0; i < source_tensor.dim(); ++i) {
        target_slice = target_slice.slice(i, 0, source_tensor.size(i));
    }
    target_slice.copy_(source_tensor);
}

void GraphBaseRunner::prepareInputs(PyModelInputs& inputs) {
    forward_event_.synchronize();
    if (!is_prefill_graph_mode_) {
        auto& py_model_inputs_ = graph_instances_[state_.current_real_graph_bs].mem_hold_.py_model_inputs_;
        py_model_inputs_.attention_inputs.kv_cache_block_id_device.fill_(0);

        backend_callbacks_.memcpy_async(inputs.attention_inputs.prefix_lengths,
                                        py_model_inputs_.attention_inputs.prefix_lengths,
                                        state_.current_batch_size * sizeof(int));
        backend_callbacks_.memcpy_async(inputs.attention_inputs.input_lengths,
                                        py_model_inputs_.attention_inputs.input_lengths,
                                        state_.current_batch_size * sizeof(int));
        backend_callbacks_.memcpy_async(inputs.attention_inputs.cu_seqlens,
                                        py_model_inputs_.attention_inputs.cu_seqlens,
                                        (state_.current_batch_size + 1) * sizeof(int));
        backend_callbacks_.memcpy_async(inputs.attention_inputs.cu_kv_seqlens,
                                        py_model_inputs_.attention_inputs.cu_kv_seqlens,
                                        (state_.current_batch_size + 1) * sizeof(int));

        py_model_inputs_.input_ids.fill_(0);
        backend_callbacks_.memcpy_async(
            inputs.input_ids, py_model_inputs_.input_ids, inputs.input_ids.size(0) * sizeof(int));
        backend_callbacks_.memcpy_async(inputs.input_hiddens,
                                        py_model_inputs_.input_hiddens,
                                        inputs.input_hiddens.numel() * inputs.input_hiddens.element_size());
        backend_callbacks_.memcpy_async(inputs.attention_inputs.sequence_lengths,
                                        py_model_inputs_.attention_inputs.sequence_lengths,
                                        state_.current_batch_size * sizeof(int));

        copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                              py_model_inputs_.attention_inputs.kv_cache_block_id_device);

        backend_callbacks_.memcpy_async(inputs.attention_inputs.sequence_lengths_plus_1_d,
                                        py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d,
                                        state_.current_batch_size * sizeof(int));
        backend_callbacks_.memcpy_async(inputs.attention_inputs.decode_cu_seqlens_d,
                                        py_model_inputs_.attention_inputs.decode_cu_seqlens_d,
                                        (state_.current_batch_size + 1) * sizeof(int));
        auto attn_pyobj = graph_instances_[state_.current_real_graph_bs].mem_hold_.attn_pyobj_;
        attn_pyobj.attr("prepare")(py_model_inputs_.attention_inputs);
    } else {
        auto& py_model_inputs_ = graph_instances_[state_.current_real_graph_seq_len].mem_hold_.py_model_inputs_;
        py_model_inputs_.attention_inputs.kv_cache_block_id_device.fill_(0);
        backend_callbacks_.memcpy_async(inputs.attention_inputs.input_lengths,
                                        py_model_inputs_.attention_inputs.input_lengths,
                                        state_.current_batch_size * sizeof(int));
        backend_callbacks_.memcpy_async(inputs.attention_inputs.cu_seqlens,
                                        py_model_inputs_.attention_inputs.cu_seqlens,
                                        (state_.current_batch_size + 1) * sizeof(int));
        backend_callbacks_.memcpy_async(inputs.attention_inputs.cu_kv_seqlens,
                                        py_model_inputs_.attention_inputs.cu_kv_seqlens,
                                        (state_.current_batch_size + 1) * sizeof(int));
        backend_callbacks_.memcpy_async(
            inputs.input_ids, py_model_inputs_.input_ids, state_.current_seq_len * sizeof(int));
        backend_callbacks_.memcpy_async(inputs.attention_inputs.padding_offset,
                                        py_model_inputs_.attention_inputs.padding_offset,
                                        state_.current_seq_len * sizeof(int));
        if (py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params) {
            (*(py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params->cuda_graph_prefill_batch_size
                   .data_ptr<int>())) = state_.current_batch_size;
        }
        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            backend_callbacks_.memcpy_async(inputs.bert_embedding_inputs.combo_position_ids,
                                            py_model_inputs_.bert_embedding_inputs.combo_position_ids,
                                            state_.current_seq_len * sizeof(int));
            backend_callbacks_.memcpy_async(inputs.bert_embedding_inputs.combo_tokens_type_ids,
                                            py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids,
                                            state_.current_seq_len * sizeof(int));
        }
    }
}

PyModelOutputs GraphBaseRunner::forward(PyModelInputs& inputs) {
    PyModelOutputs outputs;
    prepareInputs(inputs);
    if (is_prefill_graph_mode_) {
        replayPrefill(state_.current_real_graph_seq_len);
        outputs.hidden_states =
            graph_instances_[state_.current_real_graph_seq_len].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state_.current_seq_len);
    } else {
        replayDecode(state_.current_real_graph_bs);
        outputs.hidden_states =
            graph_instances_[state_.current_real_graph_bs].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state_.seq_len_sum);
    }
    backend_callbacks_.record_forward_event(forward_event_);
    backend_callbacks_.synchronize_forward_stream();
    return outputs;
}

void GraphBaseRunner::tryGetRealGraphPrefillSeqLen(PyModelInputs& inputs) {
    state_.current_seq_len = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), state_.current_seq_len);
    state_.current_real_graph_seq_len = *it;
    state_.current_batch_size         = inputs.attention_inputs.input_lengths.size(0);
}

void GraphBaseRunner::tryGetRealGraphDecodeBatchSize(PyModelInputs& inputs) {
    int graph_bs              = inputs.attention_inputs.input_lengths.size(0);
    state_.current_batch_size = graph_bs;
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state_.current_batch_size);
    state_.current_real_graph_bs = *it;
    RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", state_.current_real_graph_bs);
    state_.seq_len_sum =
        inputs.attention_inputs.is_prefill ? inputs.attention_inputs.input_lengths.sum(0).item<int>() : graph_bs;
}

bool GraphBaseRunner::canRun(PyModelInputs& inputs) {
    if (!is_prefill_graph_mode_ && inputs.attention_inputs.prefix_lengths.defined()
        && inputs.attention_inputs.prefix_lengths.numel() > 0
        && inputs.attention_inputs.prefix_lengths.data_ptr<int>()[0] > 0) {
        auto input_lengths_cpu = inputs.attention_inputs.input_lengths;
        bool all_same          = true;
        for (int i = 0; i < input_lengths_cpu.size(0); i++) {
            if (input_lengths_cpu[i].item<int>() != num_tokens_per_bs_) {
                all_same = false;
                break;
            }
        }
        if (all_same && num_tokens_per_bs_ > 1) {
            tryGetRealGraphDecodeBatchSize(inputs);
            return true;
        }
    }
    if (!enable_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_graph_mode_)) {
        return false;
    }
    if (is_prefill_graph_mode_) {
        tryGetRealGraphPrefillSeqLen(inputs);
        if (state_.current_seq_len > max_prefill_graph_len_) {
            return false;
        }
    } else {
        tryGetRealGraphDecodeBatchSize(inputs);
    }
    return true;
}

void GraphBaseRunner::initKernelInternalMemory() {
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens =
        torch::zeros({int(max_bs_ + 1)}, options_cpu_int32_).pin_memory();
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens =
        torch::zeros({int(max_bs_ + 1)}, options_cpu_int32_).pin_memory();
}

int GraphBaseRunner::getCurrentRealGraphBs() const {
    return state_.current_real_graph_bs;
}

void GraphBaseRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    inputs.input_ids = torch::zeros({max_num_token_}, options_device_int32_);
    inputs.attention_inputs.input_lengths =
        torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_).pin_memory();
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32_).pin_memory();
    int kv_cols                              = backend_callbacks_.kv_block_cols(max_seq_len_, seq_size_per_block_);
    inputs.attention_inputs.kv_cache_block_id_device = torch::zeros({int(max_bs_), kv_cols}, options_device_int32_);
    if (num_tokens_per_bs_ > 1 && !is_prefill_graph_mode_) {
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, max_seq_len_ + num_tokens_per_bs_, options_cpu_int32_).pin_memory();
    } else {
        inputs.attention_inputs.prefix_lengths = torch::zeros({int(max_bs_)}, options_cpu_int32_).pin_memory();
    }
    inputs.attention_inputs.kv_cache_block_id_host = torch::zeros({int(max_bs_), kv_cols}, options_cpu_int32_);
    inputs.attention_inputs.padding_offset =
        torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32_).pin_memory();
    inputs.attention_inputs.dtype       = model_data_type_;
    inputs.attention_inputs.is_s_padded = true;
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        backend_callbacks_.sequence_lengths_plus_one_tensor(int(max_bs_), options_device_int32_);
    inputs.attention_inputs.decode_cu_seqlens_d = torch::zeros({int(max_bs_)}, options_device_int32_);
}

void GraphBaseRunner::initCaptureAttentionInputsPost() {
    auto&         inputs                        = capture_mem_hold_.py_model_inputs_;
    torch::Tensor cuda_graph_prefill_batch_size = torch::zeros({1}, options_cpu_int32_).pin_memory();
    cuda_graph_prefill_batch_size.fill_(1);
    inputs.attention_inputs.prefill_cuda_graph_copy_params =
        PyPrefillCudaGaphCopyParams{cuda_graph_prefill_batch_size, max_seq_len_, int(max_bs_)};
}

void GraphBaseRunner::initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token) {
    (void)max_num_token;
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    inputs.bert_embedding_inputs.combo_position_ids     = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);
    inputs.bert_embedding_inputs.position_encoding      = position_encoding_;
    inputs.bert_embedding_inputs.combo_tokens_type_ids  = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);
    inputs.bert_embedding_inputs.token_type_embedding   = token_type_embedding_;
    inputs.bert_embedding_inputs.input_embedding_scalar = input_embedding_scalar_;
}

void GraphBaseRunner::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = position_encoding;
}

void GraphBaseRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = token_type_embedding;
}

void GraphBaseRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
}

void GraphBaseRunner::setMaxPrefillGraphLen(int max_prefill_graph_len) {
    max_prefill_graph_len_ = max_prefill_graph_len;
}

void GraphBaseRunner::initCapture() {
    if (enable_graph_) {
        if (backend_callbacks_.should_skip_decode_capture(py_instance_, is_prefill_graph_mode_)) {
            initKernelInternalMemory();
            return;
        }
        max_num_token_ = max_bs_ * num_tokens_per_bs_;
        if (is_prefill_graph_mode_) {
            capture_range_ = getPrefillSequenceLengthsToCapture();
            if (!capture_range_.empty()) {
                max_prefill_graph_len_ = *std::max_element(capture_range_.begin(), capture_range_.end());
            }
        } else {
            capture_range_ = getDecodeBatchSizesToCapture();
        }

        PyModelInputs inputs;
        inputs.input_ids     = torch::zeros({max_num_token_}, options_device_int32_);
        inputs.input_hiddens = torch::zeros({max_num_token_, hidden_size_}, options_device_float_);
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, is_prefill_graph_mode_);
        initKernelInternalMemory();
        auto attn_pyobj = py_attn_pyobj_method_(capture_mem_hold_.py_model_inputs_, true);
        attn_pyobj.attr("prepare")(capture_mem_hold_.py_model_inputs_.attention_inputs);
        py_forward_method_(capture_mem_hold_.py_model_inputs_, attn_pyobj);
        output = torch::zeros({max_num_token_, hidden_size_}, options_device_float_);
        capture_mem_hold_.setHiddenStates(output);
        initCaptureAttentionInputsPost();

        if (is_prefill_graph_mode_) {
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.data_ptr<int>()[1] = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.data_ptr<int>()[0] = max_num_token_;
            py_forward_method_(capture_mem_hold_.py_model_inputs_);
            capturePrefill();
        } else {
            captureDecode();
        }
    } else {
        initKernelInternalMemory();
    }
}

void GraphBaseRunner::replayGraph(int key) {
    graph_instances_[key].graph_.replay();
}

void GraphBaseRunner::captureOneGraphInstance(int key, const char* key_type) {
    auto inputs     = graph_instances_[key].mem_hold_.py_model_inputs_;
    auto attn_pyobj = graph_instances_[key].mem_hold_.attn_pyobj_;
    attn_pyobj.attr("prepare")(inputs.attention_inputs);
    py_forward_method_(inputs, attn_pyobj);
    py_forward_method_(inputs, attn_pyobj);

    backend_callbacks_.device_synchronize();
    backend_callbacks_.before_capture_stream(py_instance_, key, key_type);

    backend_callbacks_.with_capture_stream([&]() {
        at::cuda::CUDAGraph& graph = graph_instances_[key].graph_;
        std::string          output_dot_filename;
        if (enable_graph_debug_mode_) {
            graph.enable_debug_mode();
            std::string key_type_str = std::string(key_type);
            std::replace(key_type_str.begin(), key_type_str.end(), ' ', '_');
            output_dot_filename = std::string(backend_callbacks_.debug_file_prefix) + std::to_string(num_tokens_per_bs_)
                                  + "_" + key_type_str + "_" + std::to_string(key) + "_visualization.dot";
        }

        backend_callbacks_.enter_capture(py_instance_);
        graph.capture_begin();
        auto py_outputs_obj = py_forward_method_(inputs, attn_pyobj);
        auto outputs        = py_outputs_obj.cast<PyModelOutputs>();
        graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
        graph.capture_end();
        backend_callbacks_.exit_capture(py_instance_);

        if (enable_graph_debug_mode_) {
            graph.debug_dump(output_dot_filename.c_str());
        }
    });
}

void GraphBaseRunner::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("replay start check for %s %d", key_type, key);
    replayGraph(key);
    backend_callbacks_.device_synchronize();
    RTP_LLM_LOG_INFO("replay end check for %s %d", key_type, key);
}

void GraphBaseRunner::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens) {
    inputs.attention_inputs.is_prefill = is_prefill_graph_mode_ || num_tokens_per_bs_ > 1;
    inputs.input_ids                   = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, seq_len_or_tokens);
    inputs.input_hiddens = capture_mem_hold_.py_model_inputs_.input_hiddens.slice(0, 0, seq_len_or_tokens);
    inputs.attention_inputs.input_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.padding_offset =
        capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, seq_len_or_tokens);
    inputs.attention_inputs.prefix_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.sequence_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, batch_size);
    inputs.attention_inputs.cu_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_kv_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.decode_cu_seqlens_d.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d.slice(0, 0, batch_size);
    inputs.attention_inputs.dtype       = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
    inputs.bert_embedding_inputs        = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
    inputs.attention_inputs.is_s_padded = true;
}

CaptureMemoryHold GraphBaseRunner::createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count) {
    return CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, tokens_count),
                             inputs,
                             is_prefill_graph_mode_ || num_tokens_per_bs_ > 1);
}

}  // namespace rtp_llm
