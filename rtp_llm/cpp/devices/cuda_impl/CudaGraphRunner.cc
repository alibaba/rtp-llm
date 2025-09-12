#include <torch/torch.h>
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
using namespace torch_ext;
namespace rtp_llm {

static const int MIN_CACHE_INPUT_TOKEN_NUM = 512;
static const int MIN_CACHE_BATCH_SIZE      = 256;

GraphBase* CudaDevice::getDeviceGraphRunner(const DeviceInitParams& params,
                                            py::object              py_instance,
                                            int                     kv_cache_block_offset,
                                            bool                    is_embedding) {
    if (!graph_runner_) {
        graph_runner_ = new CudaGraphRunner(params, std::move(py_instance), kv_cache_block_offset, this, is_embedding);
    }
    return graph_runner_;
}

py::object CudaGraphRunner::normalForward(PyModelInputs& inputs) {
    return py_forward_method_(inputs);
}

void CudaGraphRunner::captureOneBatchSize(int bs) {
    auto inputs = graph_instances_[bs].mem_hold_.py_model_inputs_;
    // WarmUp twice
    py_forward_method_(inputs);

    py_forward_method_(inputs);

    {
        CudaGraphStreamLife  stream_life(capture_stream_, device_);
        at::cuda::CUDAGraph& graph               = graph_instances_[bs].graph_;
        auto                 output_dot_filename = "";
        if (enable_cuda_graph_debug_mode_) {
            graph.enable_debug_mode();
            output_dot_filename = "cuda_graph_visualization.dot";
        }
        graph.capture_begin();
        auto py_outputs_obj = py_forward_method_(inputs);
        auto outputs        = py_outputs_obj.cast<PyModelOutputs>();
        graph_instances_[bs].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
        auto fmha_type_obj            = py_fmha_type_method_(inputs.attention_inputs);
        auto fmha_type                = fmha_type_obj.cast<FMHAType>();
        graph_instances_[bs].use_xqa_ = fmha_type == FMHAType::XQA;
        graph.capture_end();
        // embedding model uses trtv2 attention
        if (!is_embedding_ && !graph_instances_[bs].use_xqa_) {
            // retrieve from private cache first
            int                                   input_token_num  = std::max(MIN_CACHE_INPUT_TOKEN_NUM, bs);
            int                                   batch_size       = std::max(MIN_CACHE_BATCH_SIZE, bs);
            std::shared_ptr<FlashInferAttnParams> param_shared_ptr = nullptr;
            auto cache = FlashInferAttnParams::isDecode(input_token_num) ? &PRIVATE_DECODE_PARAMS_CACHE :
                                                                           &PRIVATE_PREFILL_PARAMS_CACHE;
            if (!cache->empty()) {
                auto params = cache->back();
                if (batch_size <= params->batch_size && input_token_num <= params->input_token_num) {
                    param_shared_ptr = params;
                }
            }
            if (param_shared_ptr == nullptr) {
                auto param_ptr = FlashInferAttnParams::get(batch_size, input_token_num);
                FlashInferAttnParams::recycle(param_ptr);
                param_shared_ptr = std::shared_ptr<FlashInferAttnParams>(param_ptr);
                cache->push_back(param_shared_ptr);
            }
            graph_instances_[bs].mem_hold_.params_ = param_shared_ptr;
            RTP_LLM_CHECK_WITH_INFO(graph_instances_[bs].mem_hold_.params_ != nullptr,
                                    "capture params can't be nullptr");
        }

        if (enable_cuda_graph_debug_mode_) {
            graph.debug_dump(output_dot_filename);
        }
    }
}

void CudaGraphRunner::capture() {
    RTP_LLM_LOG_INFO("Capture Start");
    int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; i--) {
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
        inputs.attention_inputs.dtype          = torch::kBFloat16;
        graph_instances_[bs].mem_hold_ =
            CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, bs * num_tokens_per_bs_),
                              inputs,
                              kv_cache_block_offset_,
                              is_embedding_);
        captureOneBatchSize(bs);
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
    py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1) =
        inputs.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1);
    // for now cuda graph use padded mode, so we don't need the `padding_offset` for now.
    // int total_tokens = inputs.attention_inputs.padding_offset.size(0);
    // py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, total_tokens) =
    //     inputs.attention_inputs.padding_offset.slice(0, 0, total_tokens);
    if (!is_embedding_) {
        py_model_inputs_.input_ids.fill_(0);
        py_model_inputs_.input_ids.slice(0, 0, inputs.input_ids.size(0)) = inputs.input_ids;
        py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, current_batch_size_) =
            inputs.attention_inputs.sequence_lengths;
        copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                              py_model_inputs_.attention_inputs.kv_cache_block_id_device);
        if (!graph_instances_[current_real_graph_bs_].use_xqa_) {
            graph_instances_[current_real_graph_bs_].mem_hold_.params_->fillFlashInfer(
                nullptr,
                torchTensor2Buffer(inputs.attention_inputs.sequence_lengths),
                torchTensor2Buffer(inputs.attention_inputs.input_lengths),
                torchTensor2Buffer(inputs.attention_inputs.kv_cache_block_id_host),
                current_batch_size_,
                seq_size_per_block_);
            graph_instances_[current_real_graph_bs_].mem_hold_.params_->refreshFlashInferBuf(
                dynamic_cast<CudaDevice*>(device_), current_batch_size_, inputs.attention_inputs.input_lengths.size(0));
        }
    } else {
        py_model_inputs_.input_ids.fill_(0);
        auto lengths   = inputs.attention_inputs.input_lengths.data_ptr<int>();
        int  start_idx = 0;
        for (int i = 0; i < current_batch_size_; i++) {
            py_model_inputs_.input_ids.slice(0, i * num_tokens_per_bs_, i * num_tokens_per_bs_ + lengths[i]) =
                inputs.input_ids.slice(0, start_idx, start_idx + lengths[i]);
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
        replay(current_real_graph_bs_);
        if (is_embedding_) {
            // In embedding mode, extract valid parts from padded decoder_layer_hidden_states_
            auto& hidden_states = graph_instances_[current_real_graph_bs_].mem_hold_.decoder_layer_hidden_states_;
            auto  input_lengths = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();

            // calculate valid tokens num
            int32_t total_valid_tokens = 0;
            for (int i = 0; i < current_batch_size_; i++) {
                total_valid_tokens += input_lengths[i];
            }

            // 验证 total_valid_tokens 计算是否正确
            RTP_LLM_LOG_DEBUG(
                "total_valid_tokens: %d, hidden_states.size(0): %d", total_valid_tokens, hidden_states.size(0));

            // create output tensor
            auto options          = torch::TensorOptions().dtype(hidden_states.dtype()).device(hidden_states.device());
            outputs.hidden_states = torch::empty({total_valid_tokens, hidden_states.size(1)}, options);

            // Extract valid parts for each batch
            int32_t output_offset = 0;
            RTP_LLM_LOG_DEBUG(
                "Extracting valid hidden states for embedding mode - batch_size: %d, total_valid_tokens: %d",
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

                // 添加边界检查和验证
                if (actual_length <= 0) {
                    RTP_LLM_LOG_WARNING("Batch %d: actual_length=%d <= 0, skipping", i, actual_length);
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
                    RTP_LLM_LOG_ERROR(
                        "Batch %d: output_offset=%d + actual_length=%d > outputs.hidden_states.size(0)=%d",
                        i,
                        output_offset,
                        actual_length,
                        outputs.hidden_states.size(0));
                    continue;
                }

                // Extract valid parts from padded tensor
                auto valid_slice = hidden_states.slice(0, batch_start, batch_start + actual_length);
                outputs.hidden_states.slice(0, output_offset, output_offset + actual_length).copy_(valid_slice);
                output_offset += actual_length;
            }

            // 验证最终结果
            RTP_LLM_LOG_DEBUG("Final output_offset: %d, expected: %d", output_offset, total_valid_tokens);
        } else {
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

void CudaGraphRunner::replay(int bs) {
    graph_instances_[bs].graph_.replay();
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
    if (!enable_cuda_graph_ || (inputs.attention_inputs.is_prefill && !is_embedding_)) {
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

void CudaGraphRunner::initCapture() {
    if (enable_cuda_graph_) {
        RTP_LLM_LOG_INFO("CUDA graph capture is enabled");
        if (is_embedding_) {
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
        auto options_cpu_int32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
        auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        PyModelInputs inputs;
        inputs.attention_inputs.is_prefill = is_embedding_;
        // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32);
        // prefix_lengths [batch_size, int32] (for attention `prepare`)
        inputs.attention_inputs.prefix_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32);
        // input_lengths [batch_size, int32] (decode only)
        inputs.attention_inputs.input_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32);
        // sequence_lengths [batch_size, int32] (decode only)
        // sequence_length should in pinned memory
        inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32);
        inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
        // kv_cache_block_id_device [batch_size, block_num]
        inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
            {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cuda_int32);
        inputs.attention_inputs.kv_cache_block_id_host = torch::zeros(
            {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cpu_int32);
        // padding_offset [max_num_token_, int32] (for attention padding)
        inputs.attention_inputs.padding_offset = torch::zeros({int(max_seq_len_ * max_bs_)}, options_cuda_int32);
        inputs.attention_inputs.dtype          = torch::kBFloat16;  // py_model support `kBFloat16` as input type.
        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, kv_cache_block_offset_, is_embedding_);
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
}  // namespace rtp_llm
