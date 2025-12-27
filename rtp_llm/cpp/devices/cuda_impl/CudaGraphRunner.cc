#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/OpDefsUtils.h"
using namespace torch_ext;
namespace rtp_llm {

// Helper function for optimized tensor copy
void optimizedCopy(const torch::Tensor& src, torch::Tensor& dst, size_t size) {
    if (src.is_cuda() && dst.is_cuda()) {
        check_cuda_value(cudaMemcpy(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyDeviceToDevice));
    } else if (!src.is_cuda() && !dst.is_cuda()) {
        memcpy(dst.data_ptr(), src.data_ptr(), size);
    } else if (src.is_cuda() && !dst.is_cuda()) {
        check_cuda_value(cudaMemcpy(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyDeviceToHost));
    } else {
        check_cuda_value(cudaMemcpy(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyHostToDevice));
    }
}

GraphBase* CudaDevice::getDeviceGraphRunner(const DeviceInitParams& params,
                                            py::object              py_instance,
                                            int                     kv_cache_block_offset,
                                            bool                    is_prefill_cuda_graph_mode) {
    if (!graph_runner_) {
        at::cuda::CUDAStream capture_stream = *torch_default_stream_;
        graph_runner_                       = new CudaGraphRunner(
            params, std::move(py_instance), kv_cache_block_offset, capture_stream, is_prefill_cuda_graph_mode);
    }
    return graph_runner_;
}

py::object CudaGraphRunner::normalForward(PyModelInputs& inputs) {
    return py_forward_method_(inputs);
}

// column dimension
void CudaGraphRunner::copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor) {
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

void CudaGraphRunner::prepareInputs(PyModelInputs& inputs) {
    if (!is_prefill_cuda_graph_mode_) {
        auto& py_model_inputs_ = graph_instances_[state_.current_real_graph_bs].mem_hold_.py_model_inputs_;

        // Optimized copies using cudaMemcpy/memcpy
        optimizedCopy(inputs.attention_inputs.input_lengths,
                      py_model_inputs_.attention_inputs.input_lengths,
                      state_.current_batch_size * sizeof(int));

        optimizedCopy(inputs.attention_inputs.cu_seqlens,
                      py_model_inputs_.attention_inputs.cu_seqlens,
                      (state_.current_batch_size + 1) * sizeof(int));

        optimizedCopy(inputs.attention_inputs.cu_kv_seqlens,
                      py_model_inputs_.attention_inputs.cu_kv_seqlens,
                      (state_.current_batch_size + 1) * sizeof(int));

        py_model_inputs_.input_ids.fill_(0);
        optimizedCopy(inputs.input_ids, py_model_inputs_.input_ids, inputs.input_ids.size(0) * sizeof(int));

        optimizedCopy(inputs.attention_inputs.sequence_lengths,
                      py_model_inputs_.attention_inputs.sequence_lengths,
                      state_.current_batch_size * sizeof(int));

        copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                              py_model_inputs_.attention_inputs.kv_cache_block_id_device);

        if (graph_instances_[state_.current_real_graph_bs].mem_hold_.params_ptr) {
            graph_instances_[state_.current_real_graph_bs].mem_hold_.params_ptr->fillParams(
                inputs.attention_inputs.sequence_lengths,
                inputs.attention_inputs.input_lengths,
                inputs.attention_inputs.kv_cache_block_id_host,
                state_.current_batch_size,
                seq_size_per_block_);
        }
    } else {
        auto& py_model_inputs_ = graph_instances_[state_.current_real_graph_seq_len].mem_hold_.py_model_inputs_;

        // Optimized copies using cudaMemcpy/memcpy
        optimizedCopy(inputs.attention_inputs.input_lengths,
                      py_model_inputs_.attention_inputs.input_lengths,
                      state_.current_batch_size * sizeof(int));

        optimizedCopy(inputs.attention_inputs.cu_seqlens,
                      py_model_inputs_.attention_inputs.cu_seqlens,
                      (state_.current_batch_size + 1) * sizeof(int));

        optimizedCopy(inputs.attention_inputs.cu_kv_seqlens,
                      py_model_inputs_.attention_inputs.cu_kv_seqlens,
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

PyModelOutputs CudaGraphRunner::forward(PyModelInputs& inputs) {
    PyModelOutputs outputs;
    // decode or embedding model only
    if (canRun(inputs)) {
        RTP_LLM_LOG_INFO("Replay Start");
        prepareInputs(inputs);
        if (is_prefill_cuda_graph_mode_) {
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
        RTP_LLM_LOG_INFO("Replay End");
    } else {
        RTP_LLM_LOG_INFO("Normal Cuda Graph Start");
        auto py_outputs_obj = normalForward(inputs);
        // Cast the Python object to PyModelOutputs and extract hidden states
        outputs = py_outputs_obj.cast<PyModelOutputs>();
    }

    return outputs;
}

void CudaGraphRunner::tryGetRealGraphPrefillSeqLen(PyModelInputs& inputs) {
    state_.current_seq_len = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), state_.current_seq_len);
    state_.current_real_graph_seq_len = *it;
    state_.current_batch_size         = inputs.attention_inputs.input_lengths.size(0);
}

void CudaGraphRunner::tryGetRealGraphDecodeBatchSize(PyModelInputs& inputs) {
    int cuda_graph_bs         = inputs.attention_inputs.input_lengths.size(0);
    state_.current_batch_size = cuda_graph_bs;
    RTP_LLM_LOG_INFO("canRun judge for batch size: %d", cuda_graph_bs);
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state_.current_batch_size);
    state_.current_real_graph_bs = *it;
    RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", state_.current_real_graph_bs);
    state_.seq_len_sum = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    RTP_LLM_LOG_INFO("can run cuda graph for decode");
}

bool CudaGraphRunner::canRun(PyModelInputs& inputs) {
    if (!enable_cuda_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_cuda_graph_mode_)) {
        return false;
    }
    if (is_prefill_cuda_graph_mode_) {
        tryGetRealGraphPrefillSeqLen(inputs);
        if (state_.current_seq_len > max_perfill_cuda_graph_len_) {
            return false;
        }
        RTP_LLM_CHECK_WITH_INFO(std::any_of(capture_range_.begin(),
                                            capture_range_.end(),
                                            [&](int x) { return x == state_.current_real_graph_seq_len; }),
                                "seqlen used in replay: %d",
                                state_.current_real_graph_seq_len);
        RTP_LLM_LOG_INFO("can run cuda graph for prefill, current_real_graph_seq_len: %d",
                         state_.current_real_graph_seq_len);
    } else {
        tryGetRealGraphDecodeBatchSize(inputs);
    }
    return true;
}

void CudaGraphRunner::initKernelInternalMemory() {
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens =
        torch::zeros({int(max_bs_ + 1)}, options_cpu_int32_).pin_memory();
    RTP_LLM_CHECK_WITH_INFO(capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.is_pinned(),
                            "capture_mem_hold_ cu_seqlens is not pinned memory");

    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens =
        torch::zeros({int(max_bs_ + 1)}, options_cpu_int32_).pin_memory();
    RTP_LLM_CHECK_WITH_INFO(capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.is_pinned(),
                            "capture_mem_hold_ cu_kv_seqlens is not pinned memory");

}

int CudaGraphRunner::getCurrentRealGraphBs() {
    return state_.current_real_graph_bs;
}

void CudaGraphRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    inputs.attention_inputs.is_prefill = is_prefill_cuda_graph_mode_;
    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32_);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_);
    inputs.attention_inputs.input_lengths = inputs.attention_inputs.input_lengths.pin_memory();
    // sequence_lengths [batch_size, int32] (decode only)
    // sequence_length should in pinned memory
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
    // kv_cache_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cuda_int32_);
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

void CudaGraphRunner::initCaptureAttentionInputsPost() {
    auto&         inputs                        = capture_mem_hold_.py_model_inputs_;
    torch::Tensor cuda_graph_prefill_batch_size = torch::zeros({1}, options_cpu_int32_).pin_memory();
    // as one batch to capture
    cuda_graph_prefill_batch_size.fill_(1);
    RTP_LLM_CHECK_WITH_INFO(cuda_graph_prefill_batch_size.is_pinned(),
                            "capture_mem_hold_ cuda_graph_prefill_batch_size is not pinned memory");

    inputs.attention_inputs.prefill_cuda_graph_copy_params =
        PyPrefillCudaGaphCopyParams{cuda_graph_prefill_batch_size, max_seq_len_, int(max_bs_)};
}

void CudaGraphRunner::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = position_encoding;
}

void CudaGraphRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = token_type_embedding;
}

void CudaGraphRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
}

void CudaGraphRunner::setModelDataType(caffe2::TypeMeta data_type) {
    model_data_type_ = data_type;
}

void CudaGraphRunner::setMaxPrefillCudaGraphLen(int max_prefill_cuda_graph_len) {
    max_perfill_cuda_graph_len_ = max_prefill_cuda_graph_len;
    RTP_LLM_LOG_INFO("Set max_perfill_cuda_graph_len_ to %d", max_perfill_cuda_graph_len_);
}

void CudaGraphRunner::initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token) {
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    // Initialize BertEmbeddingInputs for capture
    // combo_position_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_position_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // position_encoding: from weights
    inputs.bert_embedding_inputs.position_encoding = position_encoding_;

    // combo_tokens_type_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_tokens_type_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // token_type_embedding: from weights
    inputs.bert_embedding_inputs.token_type_embedding = token_type_embedding_;

    // input_embedding_scalar: fixed value
    inputs.bert_embedding_inputs.input_embedding_scalar = input_embedding_scalar_;
}

void CudaGraphRunner::initCapture() {
    if (enable_cuda_graph_) {
        RTP_LLM_LOG_INFO("CUDA graph capture is enabled");
        if (is_prefill_cuda_graph_mode_) {
            RTP_LLM_LOG_INFO("CUDA graph capture for embedding");
            // for embedding model which is prefill-only, the `input_ids` shape should be: [bs, max_seq_len_].
            // we will do mask for extra tokens in attention mechenism.
            num_tokens_per_bs_ = max_seq_len_;
            RTP_LLM_LOG_INFO("num_tokens_per_bs_ set to %d (max_seq_len_)", num_tokens_per_bs_);
        }
        max_num_token_ = max_bs_ * num_tokens_per_bs_;
        // Capture
        at::cuda::CUDAGraph graph;
        if (is_prefill_cuda_graph_mode_) {
            capture_range_ = getPrefillSequenceLengthsToCapture();
            // Set max_perfill_cuda_graph_len_ to the maximum value from capture_range_
            if (!capture_range_.empty()) {
                int max_seq_len             = *std::max_element(capture_range_.begin(), capture_range_.end());
                max_perfill_cuda_graph_len_ = max_seq_len;
                RTP_LLM_LOG_INFO("Set max_perfill_cuda_graph_len_ to %d (max from capture_range_)",
                                 max_perfill_cuda_graph_len_);
            }
        } else {
            capture_range_ = getDecodeBatchSizesToCapture();
        }

        PyModelInputs inputs;
        // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32_);
        // input_lengths [batch_size, int32] (decode only)
        // Setup attention inputs using the extracted function
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);

        // Setup BertEmbedding inputs using the extracted function
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, kv_cache_block_offset_, is_prefill_cuda_graph_mode_);
        initKernelInternalMemory();
        // get real output data type
        RTP_LLM_LOG_INFO("initCapture forward for output datatype start");
        auto py_outputs_obj = py_forward_method_(capture_mem_hold_.py_model_inputs_);
        RTP_LLM_LOG_INFO("initCapture forward for output datatype end");
        auto outputs        = py_outputs_obj.cast<PyModelOutputs>();
        options_cuda_float_ = torch::TensorOptions()
                                  .dtype(outputs.hidden_states.dtype().toScalarType())
                                  .device(torch::kCUDA)
                                  .requires_grad(false);
        // Infer hidden_size from output tensor shape
        int64_t hidden_size = outputs.hidden_states.sizes()[1];
        output              = torch::zeros({max_num_token_, hidden_size}, options_cuda_float_);
        capture_mem_hold_.setHiddenStates(output);
        initCaptureAttentionInputsPost();
        if (is_prefill_cuda_graph_mode_) {
            RTP_LLM_LOG_INFO("initCapture forward post check start for prefill");
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.data_ptr<int>()[1] = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.data_ptr<int>()[0] = max_num_token_;
            py_forward_method_(capture_mem_hold_.py_model_inputs_);
            RTP_LLM_LOG_INFO("initCapture forward post check end for prefill");
            capturePrefill();
        } else {
            captureDecode();
        }
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

void CudaGraphRunner::replayGraph(int key) {
    graph_instances_[key].graph_.replay();
}

void CudaGraphRunner::captureOneGraphInstance(int key, const char* key_type) {
    auto inputs = graph_instances_[key].mem_hold_.py_model_inputs_;
    // WarmUp twice
    RTP_LLM_LOG_INFO("WarmUp for %s %d start.", key_type, key);
    py_forward_method_(inputs);
    py_forward_method_(inputs);
    RTP_LLM_LOG_INFO("WarmUp for %s %d successfully.", key_type, key);

    {
        CudaGraphStreamLife  stream_life(capture_stream_);
        at::cuda::CUDAGraph& graph               = graph_instances_[key].graph_;
        auto                 output_dot_filename = "";
        if (enable_cuda_graph_debug_mode_) {
            graph.enable_debug_mode();
            output_dot_filename = "cuda_graph_visualization.dot";
        }
        RTP_LLM_LOG_INFO("Capture for %s %d begin.", key_type, key);
        PyModelOutputs outputs;
        {
            graph.capture_begin();
            CudaGraphCaptureGuard capture_guard;
            auto                  py_outputs_obj = py_forward_method_(inputs);
            outputs                              = py_outputs_obj.cast<PyModelOutputs>();
            graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
            graph.capture_end();
        }
        RTP_LLM_LOG_INFO("Capture for %s %d end.", key_type, key);
        if (outputs.params_ptr->check_recycle()) {
            graph_instances_[key].mem_hold_.params_ptr =
                ParamsBasePtr(outputs.params_ptr.get(), [&](ParamsBase* ptr) {});
        } else {
            graph_instances_[key].mem_hold_.params_ptr = outputs.params_ptr;
        }

        if (enable_cuda_graph_debug_mode_) {
            graph.debug_dump(output_dot_filename);
        }
    }
}

void CudaGraphRunner::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("replay start check for %s %d", key_type, key);
    replayGraph(key);
    check_cuda_value(cudaDeviceSynchronize());
    RTP_LLM_LOG_INFO("replay end check for %s %d", key_type, key);
}

void CudaGraphRunner::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens) {
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
    inputs.attention_inputs.cu_kv_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, batch_size + 1);

    // Common direct assignments (no slice needed)
    inputs.attention_inputs.prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
    inputs.attention_inputs.dtype          = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
    inputs.bert_embedding_inputs           = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
    inputs.attention_inputs.is_s_padded    = true;
}

CaptureMemoryHold CudaGraphRunner::createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count) {
    return CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, tokens_count),
                             inputs,
                             kv_cache_block_offset_,
                             is_prefill_cuda_graph_mode_);
}

}  // namespace rtp_llm
