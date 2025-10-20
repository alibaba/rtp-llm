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
        auto& py_model_inputs_ = graph_instances_[current_real_graph_bs_].mem_hold_.py_model_inputs_;
        py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, current_batch_size_) =
            inputs.attention_inputs.input_lengths;
        // pinned memory
        py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1) =
            inputs.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1);
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
        auto& py_model_inputs_ = graph_instances_[current_real_graph_seq_len_].mem_hold_.py_model_inputs_;
        py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, current_batch_size_) =
            inputs.attention_inputs.input_lengths;
        // pinned memory
        py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1) =
            inputs.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1);
        py_model_inputs_.input_ids.slice(0, 0, current_seq_len_) = inputs.input_ids.slice(0, 0, current_seq_len_);
        if (inputs.attention_inputs.prefill_cuda_graph_copy_params) {
            (*(inputs.attention_inputs.prefill_cuda_graph_copy_params->cuda_graph_prefill_batch_size.data_ptr<int>())) =
                current_batch_size_;
        }
        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            py_model_inputs_.bert_embedding_inputs.combo_position_ids.slice(0, 0, current_seq_len_) =
                inputs.bert_embedding_inputs.combo_position_ids.slice(0, 0, current_seq_len_);
            py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids.slice(0, 0, current_seq_len_) =
                inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, 0, current_seq_len_);
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
            replayPrefill(current_seq_len_);
            outputs.hidden_states =
                graph_instances_[current_seq_len_].mem_hold_.decoder_layer_hidden_states_.slice(0, 0, current_seq_len_);
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

void CudaGraphRunner::tryGetRealGraphPrefillSeqLen(PyModelInputs& inputs) {
    current_seq_len_            = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    auto it                     = std::lower_bound(capture_range_.begin(), capture_range_.end(), current_seq_len_);
    current_real_graph_seq_len_ = *it;
    current_batch_size_         = inputs.attention_inputs.input_lengths.size(0);
    RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "seqlen used in replay: %d", current_real_graph_seq_len_);
    RTP_LLM_LOG_INFO("can run cuda graph for prefill");
}

void CudaGraphRunner::tryGetRealGraphDecodeBatchSize(PyModelInputs& inputs) {
    int cuda_graph_bs   = inputs.attention_inputs.input_lengths.size(0);
    current_batch_size_ = cuda_graph_bs;
    RTP_LLM_LOG_INFO("canRun judge for batch size: %d", cuda_graph_bs);
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), current_batch_size_);
    current_real_graph_bs_ = *it;
    RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", current_real_graph_bs_);
    seq_len_sum_ = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    RTP_LLM_LOG_INFO("can run cuda graph for decode");
}

bool CudaGraphRunner::canRun(PyModelInputs& inputs) {
    if (!enable_cuda_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_cuda_graph_mode_)) {
        return false;
    }
    if (is_prefill_cuda_graph_mode_) {
        tryGetRealGraphPrefillSeqLen(inputs);
    } else {
        tryGetRealGraphDecodeBatchSize(inputs);
    }
    return true;
}

void CudaGraphRunner::initKernelInternalMemory() {
    // for `FusedRopeKVCacheDecodeOp`, cached in pinned memory.
    BufferPtr cu_seqlens_buf = device_->allocateBuffer({DataType::TYPE_INT32, {max_bs_ + 1}, AllocationType::HOST});
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens = Buffer2torchTensor(cu_seqlens_buf, false);
    RTP_LLM_CHECK_WITH_INFO(capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.is_pinned(),
                            "capture_mem_hold_ cu_seqlens is not pinned memory");
}

int CudaGraphRunner::getCurrentRealGraphBs() {
    return current_real_graph_bs_;
}

void CudaGraphRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    inputs.attention_inputs.is_prefill = is_prefill_cuda_graph_mode_;
    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32_);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_);
    // sequence_lengths [batch_size, int32] (decode only)
    // sequence_length should in pinned memory
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
    // kv_cache_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cuda_int32_);
    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    inputs.attention_inputs.prefix_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_);

    inputs.attention_inputs.kv_cache_block_id_host = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cpu_int32_);
    inputs.attention_inputs.dtype = model_data_type_;
}

void CudaGraphRunner::setQKVDim(int dim) {
    qkv_dim_ = dim;
}

void CudaGraphRunner::initCaptureAttentionInputsPost() {
    auto&         inputs                 = capture_mem_hold_.py_model_inputs_;
    BufferPtr     prefill_batch_size_buf = device_->allocateBuffer({DataType::TYPE_INT32, {1}, AllocationType::HOST});
    torch::Tensor cuda_graph_prefill_batch_size = Buffer2torchTensor(prefill_batch_size_buf, false);
    // as one batch to capture
    cuda_graph_prefill_batch_size.fill_(1);
    RTP_LLM_CHECK_WITH_INFO(cuda_graph_prefill_batch_size.is_pinned(),
                            "capture_mem_hold_ cuda_graph_prefill_batch_size is not pinned memory");

    torch::Tensor aligned_attn_buf = torch::zeros({max_num_token_, qkv_dim_}, options_cuda_float_);
    torch::Tensor compact_attn_buf = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
    inputs.attention_inputs.prefill_cuda_graph_copy_params = PyPrefillCudaGaphCopyParams{
        cuda_graph_prefill_batch_size, aligned_attn_buf, compact_attn_buf, max_seq_len_, hidden_size_, int(max_bs_)};
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
        max_num_token_ = max_bs_ * num_tokens_per_bs_;
        // Capture
        at::cuda::CUDAGraph graph;
        if (is_prefill_cuda_graph_mode_) {
            capture_range_ = getPrefillSequenceLengthsToCapture();
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
        output = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
        capture_mem_hold_.setHiddenStates(output);
        initCaptureAttentionInputsPost();
        if (is_prefill_cuda_graph_mode_) {
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
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

}  // namespace rtp_llm
