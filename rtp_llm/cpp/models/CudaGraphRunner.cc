#include "rtp_llm/cpp/models/CudaGraphRunner.h"
#include <torch/torch.h>
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
// #include "rtp_llm/models_py/bindings/cuda/FusedRopeKVCacheOp.h"
using namespace torch_ext;

namespace rtp_llm {

void CudaGraphRunner::CaptureOneBatchSize(int bs) {
    auto inputs = graph_instances_[bs].mem_hold_.py_model_inputs_;
    // WarmUp twice
    py_forward_method_(inputs);

    py_forward_method_(inputs);
    {
        CudaGraphStreamLife  stream_life(capture_stream_, device_);
        at::cuda::CUDAGraph& graph = graph_instances_[bs].graph_;
        graph.enable_debug_mode();
        auto output_dot_filename = "cuda_graph_visualization.dot";
        graph.capture_begin();
        auto py_outputs_obj = py_forward_method_(inputs);
        auto outputs        = py_outputs_obj.cast<PyModelOutputs>();
        graph_instances_[bs].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
        graph.capture_end();
        graph_instances_[bs].mem_hold_.params_ =
            FlashInferAttnParams::retrieveCaptureParam(std::max(MIN_CACHE_INPUT_TOKEN_NUM, bs));
        RTP_LLM_CHECK_WITH_INFO(graph_instances_[bs].mem_hold_.params_ != nullptr, "capture params can't be nullptr");
        graph.debug_dump(output_dot_filename);
    }
}

void CudaGraphRunner::Capture() {
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
        graph_instances_[bs].mem_hold_ =
            CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, bs * num_tokens_per_bs_),
                              inputs,
                              kv_cache_block_offset_);
        CaptureOneBatchSize(bs);
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture End");
}

void CudaGraphRunner::PrepareInputs(PyModelInputs& inputs) {
    auto py_model_inputs_ = graph_instances_[current_real_graph_bs_].mem_hold_.py_model_inputs_;
    py_model_inputs_.input_ids.slice(0, 0, current_batch_size_ * num_tokens_per_bs_) = inputs.input_ids;
    py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, current_batch_size_) =
        inputs.attention_inputs.input_lengths;
    py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, current_batch_size_) =
        inputs.attention_inputs.sequence_lengths;
    py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, current_batch_size_) =
        inputs.attention_inputs.kv_cache_block_id_device;
    // pinned memory
    inputs.attention_inputs.cu_seqlens =
        py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1);

    graph_instances_[current_real_graph_bs_].mem_hold_.params_->fillFlashInfer(
        nullptr,
        torchTensor2Buffer(inputs.attention_inputs.sequence_lengths),
        torchTensor2Buffer(inputs.attention_inputs.input_lengths),
        torchTensor2Buffer(inputs.attention_inputs.kv_cache_block_id_host),
        current_batch_size_,
        seq_size_per_block_);
}

PyModelOutputs CudaGraphRunner::forward(PyModelInputs& inputs) {
    PyModelOutputs outputs;
    // decode only
    if (enable_cuda_graph_ && !inputs.attention_inputs.is_prefill && CanRun(inputs)) {
        RTP_LLM_LOG_INFO("Replay Start");
        PrepareInputs(inputs);
        replay(current_real_graph_bs_);
        outputs.hidden_states = graph_instances_[current_real_graph_bs_].mem_hold_.decoder_layer_hidden_states_;
        RTP_LLM_LOG_INFO("Replay End");
    } else {
        // we need to set `cu_seq_lengths`
        inputs.attention_inputs.cu_seqlens = capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens;
        auto py_outputs_obj                = py_forward_method_(inputs);
        // Cast the Python object to PyModelOutputs and extract hidden states
        outputs = py_outputs_obj.cast<PyModelOutputs>();
    }
    return outputs;
}

void CudaGraphRunner::replay(int bs) {
    graph_instances_[bs].graph_.replay();
}

bool CudaGraphRunner::CanRun(PyModelInputs& inputs) {
    int cuda_graph_bs   = inputs.attention_inputs.sequence_lengths.size(0);
    current_batch_size_ = cuda_graph_bs;
    RTP_LLM_LOG_INFO("CanRun judge for batch size: %d", cuda_graph_bs);
    bool is_bs_supported = (cuda_graph_bs <= max_bs_);
    if (is_bs_supported) {
        auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), current_batch_size_);
        current_real_graph_bs_ = *it;
        RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", current_real_graph_bs_);
    }
    RTP_LLM_LOG_INFO("can run cuda graph: %d", is_bs_supported);
    return is_bs_supported;
}

void CudaGraphRunner::init_kernel_internal_memory() {
    // for `FusedRopeKVCacheDecodeOp`, cached in pinned memory.
    BufferPtr cu_seqlens_buf = device_->allocateBuffer({DataType::TYPE_INT32, {max_bs_ + 1}, AllocationType::HOST});
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens = Buffer2torchTensor(cu_seqlens_buf, false);
    RTP_LLM_CHECK_WITH_INFO(capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.is_pinned(),
                            "capture_mem_hold_ sequence_lengths is not pinned memory");
}

int CudaGraphRunner::get_current_real_graph_bs() {
    return current_real_graph_bs_;
}

std::vector<int> CudaGraphRunner::get_batch_sizes_to_capture(int concurrency_limit) {
    std::vector<int> capture_bs;
    int              max_generate_batch_size = concurrency_limit;
    RTP_LLM_LOG_INFO("max_generate_batch_size for cuda graph: %d", max_generate_batch_size);
    // Add range 1 to 32 (inclusive)
    int step = in_test_ ? 2 : 1;
    for (int i = 1; i <= std::min(32, 1); i += step) {
        capture_bs.push_back(i);
    }
    // Add range from 48 to max_generate_batch_size (exclusive), stepping by 16
    for (int i = 48; i < max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    return capture_bs;
}

void CudaGraphRunner::init_capture() {
    if (enable_cuda_graph_) {
        RTP_LLM_LOG_INFO("CUDA graph capture is enabled");
        // Capture
        at::cuda::CUDAGraph graph;
        capture_range_ = CudaGraphRunner::get_batch_sizes_to_capture(concurrency_limit_);
        max_bs_        = *(std::max_element(capture_range_.begin(), capture_range_.end()));
        max_num_token_ = max_bs_ * num_tokens_per_bs_;
        auto options1  = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false);
        auto output    = torch::zeros({max_num_token_, hidden_size_}, options1);
        auto options2  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
        auto options3  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        PyModelInputs inputs;
        inputs.attention_inputs.is_prefill = false;
        // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids = torch::zeros({max_num_token_}, options3);
        // prefix_lengths [batch_size, int32] (for attention `prepare`)
        inputs.attention_inputs.prefix_lengths = torch::empty(0, options2);
        // input_lengths [batch_size, int32] (decode only)
        inputs.attention_inputs.input_lengths = torch::ones({int(max_bs_)}, options2);
        // sequence_lengths [batch_size, int32] (decode only)
        // sequence_length should in pinned memory
        inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options2);
        inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
        // kv_cache_block_id_device [batch_size, block_num]
        inputs.attention_inputs.kv_cache_block_id_device =
            torch::zeros({int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options3);
        inputs.attention_inputs.kv_cache_block_id_host =
            torch::zeros({int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options2);
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, kv_cache_block_offset_);
        init_kernel_internal_memory();
        Capture();
    } else {
        init_kernel_internal_memory();
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}
}  // namespace rtp_llm
