#include <cuda_runtime_api.h>
#include <torch/torch.h>
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

void CudaGraphRunner::captureOneBatchSize(int bs) {
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
        RTP_LLM_LOG_INFO("Capture for batch size %d end. params_ptr: %p, py_attn_params: %p",
                         bs,
                         outputs.params_ptr.get(),
                         outputs.py_attn_params.ptr());
        CaptureCheck::in_cuda_graph_capture = false;
        if (outputs.params_ptr) {
            if (outputs.params_ptr->check_recycle()) {
                graph_instances_[bs].mem_hold_.params_ptr =
                    ParamsBasePtr(outputs.params_ptr.get(), [&](ParamsBase* ptr) {});
            } else {
                graph_instances_[bs].mem_hold_.params_ptr = outputs.params_ptr;
            }
        } else if (outputs.py_attn_params) {
            graph_instances_[bs].mem_hold_.params_ptr     = nullptr;
            graph_instances_[bs].mem_hold_.py_attn_params = outputs.py_attn_params;
        }

        if (enable_cuda_graph_debug_mode_) {
            graph.debug_dump(output_dot_filename);
        }
    }
}

void CudaGraphRunner::capture() {
    RTP_LLM_LOG_INFO("Capture Start");
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        inputs.input_ids        = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, bs * num_tokens_per_bs_);
        auto options_cpu_int32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
        auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        // input_lengths [batch_size, int32]
        inputs.attention_inputs.input_lengths = torch::full({int(bs)}, num_tokens_per_bs_, options_cpu_int32);
        // sequence_lengths [batch_size, int32] (decode only)
        // sequence_length should in pinned memory
        inputs.attention_inputs.sequence_lengths = torch::ones({int(bs)}, options_cpu_int32);
        inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
        // kv_cache_block_id_device [batch_size, block_num]
        RTP_LLM_LOG_INFO("zeros bs=%d, max_seq_len=%d, seq_size_per_block_=%d", bs, max_seq_len_, seq_size_per_block_);
        cudaDeviceSynchronize();
        inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
            {int(bs), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cuda_int32);
        cudaDeviceSynchronize();
        inputs.attention_inputs.kv_cache_block_id_host =
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, bs);
        // pinned memory
        inputs.attention_inputs.cu_seqlens =
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, bs + 1);
        inputs.attention_inputs.cu_kv_seqlens =
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, bs + 1);
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
        captureOneBatchSize(bs);
        RTP_LLM_LOG_INFO("replay start check for %d", bs);
        replay(bs);
        cudaDeviceSynchronize();
        RTP_LLM_LOG_INFO("replay end check for %d", bs);
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture End");
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
    if (!is_prefill_cuda_graph_mode_) {
        py_model_inputs_.input_ids.fill_(0);
        py_model_inputs_.input_ids.slice(0, 0, inputs.input_ids.size(0)) = inputs.input_ids;
        py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, current_batch_size_) =
            inputs.attention_inputs.sequence_lengths;
        copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                              py_model_inputs_.attention_inputs.kv_cache_block_id_device);
        if (graph_instances_[current_real_graph_bs_].mem_hold_.params_ptr) {
            graph_instances_[current_real_graph_bs_].mem_hold_.params_ptr->fillParams(
                inputs.attention_inputs.sequence_lengths,
                inputs.attention_inputs.input_lengths,
                inputs.attention_inputs.kv_cache_block_id_host,
                current_batch_size_,
                seq_size_per_block_);
        }
    } else {
        py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1) =
            inputs.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1);
        py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, current_batch_size_ + 1) =
            inputs.attention_inputs.cu_kv_seqlens.slice(0, 0, current_batch_size_ + 1);

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
        RTP_LLM_LOG_DEBUG("Replay Start");
        prepareInputs(inputs);
        replay(current_real_graph_bs_);
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

            // Extract valid hidden states using the extracted function
            extractValidHiddenStates(outputs, inputs, total_valid_tokens);
        } else {
            outputs.hidden_states =
                graph_instances_[current_real_graph_bs_].mem_hold_.decoder_layer_hidden_states_.slice(
                    0, 0, seq_len_sum_);
        }
        RTP_LLM_LOG_DEBUG("Replay End");
    } else {
        auto py_outputs_obj = normalForward(inputs);
        // Cast the Python object to PyModelOutputs and extract hidden states
        outputs = py_outputs_obj.cast<PyModelOutputs>();
    }

    return outputs;
}

void CudaGraphRunner::replay(int bs) {
    graph_instances_[bs].graph_.replay();
}

bool CudaGraphRunner::tryGetRealGraphBatchSize(PyModelInputs& inputs) {
    int cuda_graph_bs   = inputs.attention_inputs.input_lengths.size(0);
    current_batch_size_ = cuda_graph_bs;
    RTP_LLM_LOG_DEBUG("canRun judge for batch size: %d", cuda_graph_bs);
    bool is_bs_supported   = (cuda_graph_bs <= max_bs_);
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), current_batch_size_);
    current_real_graph_bs_ = *it;
    RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", current_real_graph_bs_);
    seq_len_sum_ = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    RTP_LLM_LOG_DEBUG("can run cuda graph: %d", is_bs_supported);
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
                            "capture_mem_hold_ cu_seqlens is not pinned memory");
    BufferPtr cu_kv_seqlens_buf = device_->allocateBuffer({DataType::TYPE_INT32, {max_bs_ + 1}, AllocationType::HOST});
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens = Buffer2torchTensor(cu_kv_seqlens_buf, false);
    RTP_LLM_CHECK_WITH_INFO(capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.is_pinned(),
                            "capture_mem_hold_ cu_kv_seqlens is not pinned memory");
}

int CudaGraphRunner::getCurrentRealGraphBs() {
    return current_real_graph_bs_;
}

std::vector<int> CudaGraphRunner::getBatchSizesToCapture(int concurrency_limit) {
    std::vector<int> capture_bs;
    int              max_generate_batch_size = concurrency_limit;
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
    inputs.attention_inputs.prefix_lengths         = torch::zeros({int(max_bs_)}, options_cpu_int32);
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
        }
        // Capture
        at::cuda::CUDAGraph graph;
        capture_range_          = CudaGraphRunner::getBatchSizesToCapture(concurrency_limit_);
        max_bs_                 = *(std::max_element(capture_range_.begin(), capture_range_.end()));
        max_num_token_          = max_bs_ * num_tokens_per_bs_;
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
        capture();
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

void CudaGraphRunner::extractValidHiddenStates(PyModelOutputs&      outputs,
                                               const PyModelInputs& inputs,
                                               int32_t              total_valid_tokens) {
    auto& hidden_states = graph_instances_[current_real_graph_bs_].mem_hold_.decoder_layer_hidden_states_;
    auto  input_lengths = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();

    // Verify if total_valid_tokens calculation is correct
    RTP_LLM_LOG_DEBUG("total_valid_tokens: %d, hidden_states.size(0): %d, seq_len_sum_: %d",
                      total_valid_tokens,
                      hidden_states.size(0),
                      seq_len_sum_);

    // Extract valid parts for each batch
    int32_t output_offset = 0;
    RTP_LLM_LOG_DEBUG("Extracting valid hidden states for embedding mode - batch_size: %d, total_valid_tokens: %d",
                      current_batch_size_,
                      total_valid_tokens);

    for (int i = 0; i < current_batch_size_; i++) {
        int32_t actual_length = input_lengths[i];        // actual valid length
        int32_t batch_start   = i * num_tokens_per_bs_;  // start position in padded tensor

        RTP_LLM_LOG_DEBUG("Batch %d: actual_length=%d, batch_start=%d, output_offset=%d",
                          i,
                          actual_length,
                          batch_start,
                          output_offset);

        // Add boundary checks and validation
        if (actual_length <= 0) {
            RTP_LLM_LOG_ERROR("Batch %d: actual_length=%d <= 0, skipping", i, actual_length);
            continue;
        }

        if (batch_start >= hidden_states.size(0)) {
            RTP_LLM_LOG_ERROR(
                "Batch %d: batch_start=%d >= hidden_states.size(0)=%d", i, batch_start, hidden_states.size(0));
            continue;
        }

        if (batch_start + actual_length > hidden_states.size(0)) {
            RTP_LLM_LOG_ERROR("Batch %d: batch_start=%d + actual_length=%d > hidden_states.size(0)=%d",
                              i,
                              batch_start,
                              actual_length,
                              hidden_states.size(0));
            continue;
        }

        if (output_offset + actual_length > outputs.hidden_states.size(0)) {
            RTP_LLM_LOG_ERROR("Batch %d: output_offset=%d + actual_length=%d > outputs.hidden_states.size(0)=%d",
                              i,
                              output_offset,
                              actual_length,
                              outputs.hidden_states.size(0));
            continue;
        }

        // Extract valid parts from padded tensor
        outputs.hidden_states.slice(0, output_offset, output_offset + actual_length) =
            hidden_states.slice(0, batch_start, batch_start + actual_length);
        output_offset += actual_length;
    }

    // Resize output to contain only valid tokens
    outputs.hidden_states = hidden_states.slice(0, 0, total_valid_tokens);

    // Verify final result
    RTP_LLM_LOG_DEBUG("Final output_offset: %d, expected: %d", output_offset, total_valid_tokens);
}

}  // namespace rtp_llm
