#include <torch/torch.h>
#include <algorithm>
#include <cstring>
#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "rtp_llm/cpp/devices/GraphBase.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/models_py/bindings/OpDefsUtils.h"

namespace rtp_llm {

// column dimension
void GraphBase::copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor) {
    if (source_tensor.dim() != target_tensor.dim()) {
        throw std::runtime_error("Error: Source and target tensors must have the same number of dimensions.");
    }

    for (int i = 0; i < source_tensor.dim(); ++i) {
        if (source_tensor.size(i) > target_tensor.size(i)) {
            std::string error_msg =
                "Error: Target tensor dimension " + std::to_string(i) + " (" + std::to_string(target_tensor.size(i))
                + ")" + " is smaller than source tensor dimension " + std::to_string(i) + " ("
                + std::to_string(source_tensor.size(i)) + "). " + "This violates the function's guarantee.";
            throw std::runtime_error(error_msg);
        }
    }

    torch::Tensor target_slice = target_tensor;

    for (int i = 0; i < source_tensor.dim(); ++i) {
        target_slice = target_slice.slice(i, 0, source_tensor.size(i));
    }

    target_slice.copy_(source_tensor);
}

void GraphBase::prepareInputs(PyModelInputs& inputs) {
    if (!is_prefill_graph_mode_) {
        auto& py_model_inputs_ = graph_mem_holds_[state_.current_real_graph_bs].py_model_inputs_;

        // Use virtual optimizedCopy method
        optimizedCopy(inputs.attention_inputs.input_lengths,
                      py_model_inputs_.attention_inputs.input_lengths,
                      state_.current_batch_size * sizeof(int));

        optimizedCopy(inputs.attention_inputs.cu_seqlens,
                      py_model_inputs_.attention_inputs.cu_seqlens,
                      (state_.current_batch_size + 1) * sizeof(int));

        py_model_inputs_.input_ids.fill_(0);
        optimizedCopy(inputs.input_ids, py_model_inputs_.input_ids, inputs.input_ids.size(0) * sizeof(int));

        optimizedCopy(inputs.attention_inputs.sequence_lengths,
                      py_model_inputs_.attention_inputs.sequence_lengths,
                      state_.current_batch_size * sizeof(int));

        copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                              py_model_inputs_.attention_inputs.kv_cache_block_id_device);

        optimizedCopy(inputs.attention_inputs.padding_offset,
                      py_model_inputs_.attention_inputs.padding_offset,
                      inputs.attention_inputs.padding_offset.size(0) * sizeof(int));

        graph_mem_holds_[state_.current_real_graph_bs].params_ptr->fillParams(
            inputs.attention_inputs.sequence_lengths,
            inputs.attention_inputs.input_lengths,
            inputs.attention_inputs.kv_cache_block_id_host,
            state_.current_batch_size,
            seq_size_per_block_);
    } else {
        auto& py_model_inputs_ = graph_mem_holds_[state_.current_real_graph_seq_len].py_model_inputs_;

        // Use virtual optimizedCopy method
        optimizedCopy(inputs.attention_inputs.input_lengths,
                      py_model_inputs_.attention_inputs.input_lengths,
                      state_.current_batch_size * sizeof(int));

        optimizedCopy(inputs.attention_inputs.cu_seqlens,
                      py_model_inputs_.attention_inputs.cu_seqlens,
                      (state_.current_batch_size + 1) * sizeof(int));

        optimizedCopy(inputs.input_ids, py_model_inputs_.input_ids, state_.current_seq_len * sizeof(int));

        optimizedCopy(inputs.attention_inputs.padding_offset,
                      py_model_inputs_.attention_inputs.padding_offset,
                      state_.current_seq_len * sizeof(int));

        if (py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params) {
            (*(py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params->cuda_graph_prefill_batch_size
                   .data_ptr<int>())) = state_.current_batch_size;
        }

        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            optimizedCopy(inputs.bert_embedding_inputs.combo_position_ids,
                          py_model_inputs_.bert_embedding_inputs.combo_position_ids,
                          state_.current_seq_len * sizeof(int));

            optimizedCopy(inputs.bert_embedding_inputs.combo_tokens_type_ids,
                          py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids,
                          state_.current_seq_len * sizeof(int));
        }
    }
}

bool GraphBase::canRun(PyModelInputs& inputs) {
    if (!enable_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_graph_mode_)) {
        return false;
    }
    if (is_prefill_graph_mode_) {
        tryGetRealGraphPrefillSeqLen(inputs);
        if (state_.current_seq_len > max_prefill_graph_len_) {
            return false;
        }
        RTP_LLM_CHECK_WITH_INFO(std::any_of(capture_range_.begin(),
                                            capture_range_.end(),
                                            [&](int x) { return x == state_.current_real_graph_seq_len; }),
                                "seqlen used in replay: %d",
                                state_.current_real_graph_seq_len);
        RTP_LLM_LOG_INFO("can run graph for prefill, current_real_graph_seq_len: %d",
                         state_.current_real_graph_seq_len);
    } else {
        tryGetRealGraphDecodeBatchSize(inputs);
    }
    return true;
}

void GraphBase::initKernelInternalMemory() {
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens =
        torch::zeros({int(max_bs_ + 1)}, options_cpu_int32_).pin_memory();
    RTP_LLM_CHECK_WITH_INFO(capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.is_pinned(),
                            "capture_mem_hold_ cu_seqlens is not pinned memory");
}

int GraphBase::getCurrentRealGraphBs() {
    return state_.current_real_graph_bs;
}

void GraphBase::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    inputs.attention_inputs.is_prefill = is_prefill_graph_mode_;
    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_device_int32_);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_);
    // sequence_lengths [batch_size, int32] (decode only)
    // sequence_length should in pinned memory
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
    // kv_cache_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_device_int32_);
    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    inputs.attention_inputs.prefix_lengths = torch::zeros({int(max_bs_)}, options_cpu_int32_);

    inputs.attention_inputs.kv_cache_block_id_host = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cpu_int32_);
    // padding_offset [max_num_token_, int32] (for attention padding)
    inputs.attention_inputs.padding_offset = torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.padding_offset = inputs.attention_inputs.padding_offset.pin_memory();
    inputs.attention_inputs.dtype          = model_data_type_;
    inputs.attention_inputs.is_s_padded    = true;
}

void GraphBase::initCaptureAttentionInputsPost() {
    auto&         inputs                   = capture_mem_hold_.py_model_inputs_;
    torch::Tensor graph_prefill_batch_size = torch::zeros({1}, options_cpu_int32_).pin_memory();
    // as one batch to capture
    graph_prefill_batch_size.fill_(1);
    RTP_LLM_CHECK_WITH_INFO(graph_prefill_batch_size.is_pinned(),
                            "capture_mem_hold_ graph_prefill_batch_size is not pinned memory");

    inputs.attention_inputs.prefill_cuda_graph_copy_params =
        PyPrefillCudaGaphCopyParams{graph_prefill_batch_size, max_seq_len_, int(max_bs_)};
}

void GraphBase::setMaxPrefillGraphLen(int max_prefill_graph_len) {
    max_prefill_graph_len_ = max_prefill_graph_len;
    RTP_LLM_LOG_INFO("Set max_prefill_graph_len_ to %d", max_prefill_graph_len_);
}

void GraphBase::initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token) {
    // Initialize BertEmbeddingInputs for capture
    // combo_position_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_position_ids = torch::zeros({max_seq_len_ * max_bs}, options_device_int32_);

    // position_encoding: from weights
    inputs.bert_embedding_inputs.position_encoding = position_encoding_;

    // combo_tokens_type_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_tokens_type_ids = torch::zeros({max_seq_len_ * max_bs}, options_device_int32_);

    // token_type_embedding: from weights
    inputs.bert_embedding_inputs.token_type_embedding = token_type_embedding_;

    // input_embedding_scalar: fixed value
    inputs.bert_embedding_inputs.input_embedding_scalar = input_embedding_scalar_;
}

void GraphBase::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("replay start check for %s %d", key_type, key);
    replayGraphImpl(key);
    syncDevice();
    RTP_LLM_LOG_INFO("replay end check for %s %d", key_type, key);
}

void GraphBase::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens) {
    // Common slice operations for input_ids and padding_offset
    inputs.input_ids = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, seq_len_or_tokens);
    inputs.attention_inputs.input_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.padding_offset =
        capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, seq_len_or_tokens);

    // Common slice operations for attention inputs
    inputs.attention_inputs.sequence_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, batch_size);
    inputs.attention_inputs.cu_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, batch_size + 1);

    // Common direct assignments (no slice needed)
    inputs.attention_inputs.prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
    inputs.attention_inputs.dtype          = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
    inputs.bert_embedding_inputs           = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
    inputs.attention_inputs.is_s_padded    = true;
}

CaptureMemoryHold GraphBase::createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count) {
    return CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, tokens_count),
                             inputs,
                             kv_cache_block_offset_,
                             is_prefill_graph_mode_);
}

py::object GraphBase::normalForward(PyModelInputs& inputs) {
    return py_forward_method_(inputs);
}

PyModelOutputs GraphBase::forward(PyModelInputs& inputs) {
    PyModelOutputs outputs;
    // decode or embedding model only
    if (canRun(inputs)) {
        RTP_LLM_LOG_INFO("Replay Start");
        prepareInputs(inputs);
        if (is_prefill_graph_mode_) {
            replayPrefill(state_.current_real_graph_seq_len);
            outputs.hidden_states =
                graph_mem_holds_[state_.current_real_graph_seq_len].decoder_layer_hidden_states_.slice(
                    0, 0, state_.current_seq_len);
        } else {
            replayDecode(state_.current_real_graph_bs);
            outputs.hidden_states = graph_mem_holds_[state_.current_real_graph_bs].decoder_layer_hidden_states_.slice(
                0, 0, state_.seq_len_sum);
        }
        RTP_LLM_LOG_INFO("Replay End");
    } else {
        RTP_LLM_LOG_INFO("Normal Graph Start");
        auto py_outputs_obj = normalForward(inputs);
        // Cast the Python object to PyModelOutputs and extract hidden states
        outputs = py_outputs_obj.cast<PyModelOutputs>();
    }

    return outputs;
}

void GraphBase::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = position_encoding;
}

void GraphBase::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = token_type_embedding;
}

void GraphBase::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
}

void GraphBase::setModelDataType(caffe2::TypeMeta data_type) {
    model_data_type_ = data_type;
}

void GraphBase::initCapture() {
    if (enable_graph_) {
        RTP_LLM_LOG_INFO("Graph capture is enabled");
        if (is_prefill_graph_mode_) {
            RTP_LLM_LOG_INFO("Graph capture for embedding");
            // for embedding model which is prefill-only, the `input_ids` shape should be: [bs, max_seq_len_].
            // we will do mask for extra tokens in attention mechanism.
            num_tokens_per_bs_ = max_seq_len_;
            RTP_LLM_LOG_INFO("num_tokens_per_bs_ set to %d (max_seq_len_)", num_tokens_per_bs_);
        }
        max_num_token_ = max_bs_ * num_tokens_per_bs_;

        if (is_prefill_graph_mode_) {
            capture_range_ = getPrefillSequenceLengthsToCapture();
            // Set max_prefill_graph_len_ to the maximum value from capture_range_
            if (!capture_range_.empty()) {
                int max_seq_len        = *std::max_element(capture_range_.begin(), capture_range_.end());
                max_prefill_graph_len_ = max_seq_len;
                RTP_LLM_LOG_INFO("Set max_prefill_graph_len_ to %d (max from capture_range_)", max_prefill_graph_len_);
            }
        } else {
            capture_range_ = getDecodeBatchSizesToCapture();
        }

        PyModelInputs inputs;
        // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids = torch::zeros({max_num_token_}, options_device_int32_);
        // input_lengths [batch_size, int32] (decode only)
        // Setup attention inputs using the extracted function
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);

        // Setup BertEmbedding inputs using the extracted function
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, kv_cache_block_offset_, is_prefill_graph_mode_);
        initKernelInternalMemory();

        // get real output data type
        RTP_LLM_LOG_INFO("initCapture forward for output datatype start");
        auto py_outputs_obj = py_forward_method_(capture_mem_hold_.py_model_inputs_);
        RTP_LLM_LOG_INFO("initCapture forward for output datatype end");
        auto outputs          = py_outputs_obj.cast<PyModelOutputs>();
        options_device_float_ = torch::TensorOptions()
                                    .dtype(outputs.hidden_states.dtype().toScalarType())
                                    .device(options_device_int32_.device())
                                    .requires_grad(false);
        // Infer hidden_size from output tensor shape
        int64_t hidden_size = outputs.hidden_states.sizes()[1];
        output              = torch::zeros({max_num_token_, hidden_size}, options_device_float_);
        capture_mem_hold_.setHiddenStates(output);
        initCaptureAttentionInputsPost();

        if (is_prefill_graph_mode_) {
            RTP_LLM_LOG_INFO("initCapture forward post check start for prefill");
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.data_ptr<int>()[0] = max_num_token_;
            py_forward_method_(capture_mem_hold_.py_model_inputs_);
            RTP_LLM_LOG_INFO("initCapture forward post check end for prefill");
            capturePrefill();
        } else {
            captureDecode();
        }
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("Graph capture is not enabled, skipping initialization");
    }
}

void GraphBase::warmupForCapture(int key, const char* key_type) {
    auto inputs = graph_mem_holds_[key].py_model_inputs_;
    // WarmUp twice
    RTP_LLM_LOG_INFO("WarmUp for %s %d start.", key_type, key);
    py_forward_method_(inputs);
    py_forward_method_(inputs);
    RTP_LLM_LOG_INFO("WarmUp for %s %d successfully.", key_type, key);
}

void GraphBase::captureOneGraphInstance(int key, const char* key_type) {
    warmupForCapture(key, key_type);
    performGraphCapture(key, key_type);
}

}  // namespace rtp_llm
