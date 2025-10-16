#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include <chrono>
#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/OpDefsUtils.h"

using namespace torch_ext;
namespace rtp_llm {

GraphBase* CudaDevice::getDeviceGraphRunner(const DeviceInitParams& params,
                                            py::object              py_instance,
                                            int                     kv_cache_block_offset,
                                            bool                    is_prefill_cuda_graph_mode) {
    if (!graph_runner_) {
        graph_runner_ = new CudaGraphRunner(
            params, std::move(py_instance), kv_cache_block_offset, this, is_prefill_cuda_graph_mode);
    }
    return graph_runner_;
}

py::object CudaGraphRunner::normalForward(PyModelInputs& inputs) {
    return py_forward_method_(inputs);
}

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
    auto& py_model_inputs_ = graph_instances_[current_real_graph_bs_].mem_hold_.py_model_inputs_;
    py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, current_batch_size_) =
        inputs.attention_inputs.input_lengths;
    // pinned memory
    py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1) =
        inputs.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1);
    if (!is_prefill_cuda_graph_mode_) {
        py_model_inputs_.input_ids.fill_(0);
        py_model_inputs_.input_ids.slice(0, 0, inputs.input_ids.size(0)) = inputs.input_ids;
        py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, current_batch_size_) =
            inputs.attention_inputs.sequence_lengths;
        copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                              py_model_inputs_.attention_inputs.kv_cache_block_id_device);
        graph_instances_[current_real_graph_bs_].mem_hold_.params_ptr->fillParams(
            inputs.attention_inputs.sequence_lengths,
            inputs.attention_inputs.input_lengths,
            inputs.attention_inputs.kv_cache_block_id_host,
            current_batch_size_,
            seq_size_per_block_);
    } else {

        auto input_lengths_ptr  = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();
        auto padding_offset_ptr = py_model_inputs_.attention_inputs.padding_offset.data_ptr<int32_t>();

        int32_t cum_offset = 0;
        int32_t index      = 0;
        for (int32_t i = 0; i < current_batch_size_; i++) {
            index           = i * num_tokens_per_bs_;
            int32_t seq_len = input_lengths_ptr[i];
            for (int32_t j = 0; j < seq_len; j++) {
                padding_offset_ptr[index] = cum_offset;
                index++;
            }
            cum_offset += num_tokens_per_bs_ - seq_len;
        }

        py_model_inputs_.input_ids.fill_(0);
        auto lengths   = inputs.attention_inputs.input_lengths.data_ptr<int>();
        int  start_idx = 0;
        for (int i = 0; i < current_batch_size_; i++) {
            int dst_start = i * num_tokens_per_bs_;
            int dst_end   = dst_start + lengths[i];
            int src_start = start_idx;
            int src_end   = src_start + lengths[i];

            // Copy input_ids
            py_model_inputs_.input_ids.slice(0, dst_start, dst_end) = inputs.input_ids.slice(0, src_start, src_end);

            // Copy bert embedding data if available
            if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
                py_model_inputs_.bert_embedding_inputs.combo_position_ids.slice(0, dst_start, dst_end) =
                    inputs.bert_embedding_inputs.combo_position_ids.slice(0, src_start, src_end);
                py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids.slice(0, dst_start, dst_end) =
                    inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, src_start, src_end);
            }
            start_idx += lengths[i];
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
            // In embedding mode, extract valid parts from padded decoder_layer_hidden_states_
            auto& hidden_states = graph_instances_[current_real_graph_bs_].mem_hold_.decoder_layer_hidden_states_;
            // create output tensor
            outputs.hidden_states = hidden_states;
            auto input_lengths    = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();

            // calculate valid tokens num
            int32_t total_valid_tokens = 0;
            for (int i = 0; i < current_batch_size_; i++) {
                total_valid_tokens += input_lengths[i];
            }
            auto cloned = hidden_states.clone();
            // Extract valid hidden states using the extracted function
            extractValidHiddenStates(
                outputs.hidden_states, hidden_states, inputs.attention_inputs.input_lengths, total_valid_tokens);
        } else {
            replayDecode(current_real_graph_bs_);
            outputs.hidden_states =
                graph_instances_[current_real_graph_bs_].mem_hold_.decoder_layer_hidden_states_.slice(
                    0, 0, seq_len_sum_);
        }
        RTP_LLM_LOG_INFO("Replay End");
    } else {
        auto py_outputs_obj = normalForward(inputs);
        // Cast the Python object to PyModelOutputs and extract hidden states
        outputs = py_outputs_obj.cast<PyModelOutputs>();
    }

    return outputs;
}

bool CudaGraphRunner::tryGetRealGraphBatchSize(PyModelInputs& inputs) {
    int cuda_graph_bs   = inputs.attention_inputs.input_lengths.size(0);
    current_batch_size_ = cuda_graph_bs;
    RTP_LLM_LOG_INFO("canRun judge for batch size: %d", cuda_graph_bs);
    bool is_bs_supported   = (cuda_graph_bs <= max_bs_);
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), current_batch_size_);
    current_real_graph_bs_ = *it;
    RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", current_real_graph_bs_);
    seq_len_sum_ = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    RTP_LLM_LOG_INFO("can run cuda graph: %d", is_bs_supported);
    return is_bs_supported;
}

bool CudaGraphRunner::canRun(PyModelInputs& inputs) {
    if (!enable_cuda_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_cuda_graph_mode_)) {
        return false;
    }
    return tryGetRealGraphBatchSize(inputs);
}

void CudaGraphRunner::initKernelInternalMemory() {
    // for `FusedRopeKVCacheDecodeOp`, cached in pinned memory.
    BufferPtr cu_seqlens_buf = device_->allocateBuffer({DataType::TYPE_INT32, {max_bs_ + 1}, AllocationType::HOST});
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens = Buffer2torchTensor(cu_seqlens_buf, false);
    RTP_LLM_CHECK_WITH_INFO(capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.is_pinned(),
                            "capture_mem_hold_ sequence_lengths is not pinned memory");
}

int CudaGraphRunner::getCurrentRealGraphBs() {
    return current_real_graph_bs_;
}

void CudaGraphRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    auto options_cpu_int32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    inputs.attention_inputs.is_prefill = is_prefill_cuda_graph_mode_;
    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32);
    // sequence_lengths [batch_size, int32] (decode only)
    // sequence_length should in pinned memory
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
    // kv_cache_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cuda_int32);
    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    inputs.attention_inputs.prefix_lengths         = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32);
    inputs.attention_inputs.kv_cache_block_id_host = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cpu_int32);
    // padding_offset [max_num_token_, int32] (for attention padding)
    inputs.attention_inputs.padding_offset = torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32);
    inputs.attention_inputs.padding_offset = inputs.attention_inputs.padding_offset.pin_memory();
    inputs.attention_inputs.dtype          = model_data_type_;
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
        // Capture
        at::cuda::CUDAGraph graph;
        max_bs_        = *(std::max_element(capture_range_.begin(), capture_range_.end()));
        max_num_token_ = max_bs_ * num_tokens_per_bs_;
        if (is_prefill_cuda_graph_mode_) {
            capture_range_ = getDecodeBatchSizesToCapture(concurrency_limit_);
        } else {
            capture_range_ = getPrefillSequenceLengthsToCapture();
        }

        auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        PyModelInputs inputs;
        // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32);
        // input_lengths [batch_size, int32] (decode only)
        // Setup attention inputs using the extracted function
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);

        // Setup BertEmbedding inputs using the extracted function
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, kv_cache_block_offset_, is_prefill_cuda_graph_mode_);
        initKernelInternalMemory();
        // get real output data type
        auto py_outputs_obj     = py_forward_method_(capture_mem_hold_.py_model_inputs_);
        auto outputs            = py_outputs_obj.cast<PyModelOutputs>();
        auto options_cuda_float = torch::TensorOptions()
                                      .dtype(outputs.hidden_states.dtype().toScalarType())
                                      .device(torch::kCUDA)
                                      .requires_grad(false);
        output = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float);
        capture_mem_hold_.setHiddenStates(output);
        if (is_prefill_cuda_graph_mode_) {
            capturePrefill();
        } else {
            captureDecode();
        }
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

void CudaGraphRunner::extractValidHiddenStates(torch::Tensor& outputs,
                                               torch::Tensor& inputs,
                                               torch::Tensor& input_lengths,
                                               int32_t        total_valid_tokens) {

    auto input_lengths_int = input_lengths.data_ptr<int32_t>();
    // Verify if total_valid_tokens calculation is correct
    RTP_LLM_LOG_DEBUG("total_valid_tokens: %d, hidden_states.size(0): %d, seq_len_sum_: %d, num_tokens_per_bs_: %d",
                      total_valid_tokens,
                      inputs.size(0),
                      seq_len_sum_,
                      num_tokens_per_bs_);

    // Extract valid parts for each batch
    int32_t output_offset = 0;
    RTP_LLM_LOG_DEBUG("Extracting valid hidden states for embedding mode - batch_size: %d, total_valid_tokens: %d",
                      current_batch_size_,
                      total_valid_tokens);

    // Use direct memory copy for better performance
    auto    output_ptr   = outputs.data_ptr();
    auto    input_ptr    = inputs.data_ptr();
    int32_t hidden_size  = outputs.size(1);
    auto    element_size = outputs.element_size();  // Get actual element size

    for (int i = 0; i < current_batch_size_; i++) {
        int32_t actual_length = input_lengths_int[i];    // actual valid length
        int32_t batch_start   = i * num_tokens_per_bs_;  // start position in padded tensor

        // Direct memory copy - much faster than slice operations
        auto copy_size = actual_length * hidden_size;
        auto src_ptr   = static_cast<char*>(input_ptr) + batch_start * hidden_size * element_size;
        auto dst_ptr   = static_cast<char*>(output_ptr) + output_offset * hidden_size * element_size;

        if (outputs.is_cuda()) {
            cudaMemcpy(dst_ptr, src_ptr, copy_size * element_size, cudaMemcpyDeviceToDevice);
        } else {
            memcpy(dst_ptr, src_ptr, copy_size * element_size);
        }

        output_offset += actual_length;
    }

    // Resize output to contain only valid tokens
    outputs = outputs.slice(0, 0, total_valid_tokens);
}

}  // namespace rtp_llm
