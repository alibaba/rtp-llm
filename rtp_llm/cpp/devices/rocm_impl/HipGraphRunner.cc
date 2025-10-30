#include <hip/hip_runtime_api.h>
#include <torch/torch.h>
#include <typeinfo>
#include "rtp_llm/cpp/devices/rocm_impl/HipGraphRunner.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"

using namespace torch_ext;
namespace rtp_llm {

GraphBase* ROCmDevice::getDeviceGraphRunner(const DeviceInitParams& params,
                                            py::object              py_instance,
                                            int                     kv_cache_block_offset,
                                            bool                    is_prefill_hip_graph_mode) {
    if (!graph_runner_) {
        graph_runner_ =
            new HipGraphRunner(params, std::move(py_instance), kv_cache_block_offset, this, is_prefill_hip_graph_mode);
    }
    return graph_runner_;
}

HipGraphRunner::HipGraphRunner(const DeviceInitParams& params,
                               py::object              py_instance,
                               int                     kv_cache_block_offset,
                               DeviceBase*             device,
                               bool                    is_prefill_hip_graph_mode):
    GraphBase(std::move(py_instance)),
    enable_hip_graph_(params.hw_kernel_config.enable_cuda_graph),
    is_prefill_hip_graph_mode_(is_prefill_hip_graph_mode),
    concurrency_limit_(params.concurrency_config.concurrency_limit),
    capture_stream_(at::hip::getStreamFromPool()),  // 这里和 CUDA 保持一致接口
    enable_hip_graph_debug_mode_(params.hw_kernel_config.enable_cuda_graph_debug_mode),
    hidden_size_(params.hidden_size),
    max_seq_len_(params.max_seq_len),
    seq_size_per_block_(params.tokens_per_block),
    kv_cache_block_offset_(kv_cache_block_offset),
    device_(device) {
    py::gil_scoped_acquire gil;
    if (!py_instance_ || py_instance_.is_none()) {
        throw std::runtime_error("HipGraphRunner constructor: Python instance is null or none.");
    }
    py_forward_method_     = py_instance_.attr("forward");
    py_fill_params_method_ = py_instance_.attr("fill_params");
    RTP_LLM_LOG_INFO("Initialize HipGraphRunner with parameters below: \n \
        enable_hip_graph_: %d, concurrency_limit_: %d, enable_hip_graph_debug_mode_: %d, hidden_size_: %d, max_seq_len_: %d, seq_size_per_block_: %d, kv_cache_block_offset_: %d, is_prefill_hip_graph_mode_: %d",
                     enable_hip_graph_,
                     concurrency_limit_,
                     enable_hip_graph_debug_mode_,
                     hidden_size_,
                     max_seq_len_,
                     seq_size_per_block_,
                     kv_cache_block_offset_,
                     is_prefill_hip_graph_mode_);
}

HipGraphRunner::~HipGraphRunner() {
    RTP_LLM_LOG_INFO("Release HipGraphRunner .....");
    py::gil_scoped_acquire gil;
    py_instance_.release();
    RTP_LLM_LOG_INFO("Release HipGraphRunner Successfully");
}

py::object HipGraphRunner::normalForward(PyModelInputs& inputs) {
    return py_forward_method_(inputs);
}

void HipGraphRunner::captureOneBatchSize(int bs) {
    auto inputs = graph_instances_[bs].mem_hold_.py_model_inputs_;
    // WarmUp twice
    py_forward_method_(inputs);
    py_forward_method_(inputs);

    {
        HipGraphStreamLife   stream_life(capture_stream_, device_);
        at::cuda::CUDAGraph& graph               = graph_instances_[bs].graph_;
        std::string          output_dot_filename = "";
        if (enable_hip_graph_debug_mode_) {
            graph.enable_debug_mode();
            // output_dot_filename = "hip_graph_visualization.dot";
            output_dot_filename =
                "/home/moudi.mou/graph_visualization/hip_graph_visualization_bs" + std::to_string(bs) + ".dot";
        }

        graph.capture_begin();
        // CaptureCheck::in_hip_graph_capture = true;
        auto py_outputs_obj = py_forward_method_(inputs);
        auto outputs        = py_outputs_obj.cast<PyModelOutputs>();
        graph_instances_[bs].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
        graph.capture_end();

        // CaptureCheck::in_hip_graph_capture = false;

        graph_instances_[bs].mem_hold_.params_ptr = outputs.params_ptr;

        if (enable_hip_graph_debug_mode_) {
            graph.debug_dump(output_dot_filename);
        }
    }
}

void HipGraphRunner::capture() {
    RTP_LLM_LOG_INFO("Capture Start");

    // RTP_LLM_LOG_INFO("capture_range_ :%d", capture_range_.size());
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
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
                              is_prefill_hip_graph_mode_);
        captureOneBatchSize(bs);
        // RTP_LLM_LOG_INFO("replay start check for %d", bs);
        // replay(bs);
        hipDeviceSynchronize();  // 注释掉可能导致死锁的同步调用
        // RTP_LLM_LOG_INFO("replay end check for %d", bs);
        // RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture End");
}

void HipGraphRunner::copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor) {
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

void HipGraphRunner::prepareInputs(PyModelInputs& inputs) {

    auto& py_model_inputs_ = graph_instances_[current_real_graph_bs_].mem_hold_.py_model_inputs_;

    if (!is_prefill_hip_graph_mode_) {
        py_model_inputs_.input_ids.fill_(0);
        py_model_inputs_.input_ids.slice(0, 0, inputs.input_ids.size(0)) = inputs.input_ids;
        py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, current_batch_size_) =
            (inputs.attention_inputs.input_lengths).cuda();
        py_model_inputs_.attention_inputs.sequence_lengths.copy_(inputs.attention_inputs.sequence_lengths.cuda());

        // py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, current_batch_size_) =
        //     inputs.attention_inputs.sequence_lengths;

        // 先将目标张量清零
        py_model_inputs_.attention_inputs.kv_cache_block_id_device.fill_(0);

        copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                              py_model_inputs_.attention_inputs.kv_cache_block_id_device);
        py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1) =
            inputs.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1);

        graph_instances_[current_real_graph_bs_].mem_hold_.params_ptr->update();

    } else {
        // Prefill 模式
        py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, current_batch_size_) =
            inputs.attention_inputs.input_lengths;

        py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1) =
            inputs.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1);

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

PyModelOutputs HipGraphRunner::forward(PyModelInputs& inputs) {
    PyModelOutputs outputs;
    // decode or embedding model only
    if (canRun(inputs)) {
        // RTP_LLM_LOG_INFO("Replay Start");

        prepareInputs(inputs);
        replay(current_real_graph_bs_);
        if (is_prefill_hip_graph_mode_) {
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
        // RTP_LLM_LOG_INFO("Replay End");
    } else {
        auto py_outputs_obj = normalForward(inputs);
        outputs             = py_outputs_obj.cast<PyModelOutputs>();
    }
    return outputs;
}

void HipGraphRunner::replay(int bs) {

    // hipDeviceSynchronize();
    auto& mem_hold = graph_instances_[bs].mem_hold_;
    if (!mem_hold.params_ptr) {
        RTP_LLM_LOG_ERROR("params_ptr is null for batch size %d", bs);
        throw std::runtime_error("Invalid params_ptr in replay");
    }

    // Check py_model_inputs_ members
    auto& inputs = mem_hold.py_model_inputs_;

    graph_instances_[bs].graph_.replay();
    // hipDeviceSynchronize();
}

bool HipGraphRunner::tryGetRealGraphBatchSize(PyModelInputs& inputs) {
    int hip_graph_bs = inputs.attention_inputs.input_lengths.size(0);

    current_batch_size_ = hip_graph_bs;
    // RTP_LLM_LOG_INFO("canRun judge for batch size: %d", hip_graph_bs);
    bool is_bs_supported   = (hip_graph_bs <= max_bs_);
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), current_batch_size_);
    current_real_graph_bs_ = *it;
    // RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", current_real_graph_bs_);
    seq_len_sum_ = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    // RTP_LLM_LOG_INFO("can run hip graph: %d", is_bs_supported);
    return is_bs_supported;
}

bool HipGraphRunner::canRun(PyModelInputs& inputs) {
    if (!enable_hip_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_hip_graph_mode_)) {
        return false;
    }
    return tryGetRealGraphBatchSize(inputs);
}

void HipGraphRunner::initKernelInternalMemory() {
    BufferPtr cu_seqlens_buf = device_->allocateBuffer({DataType::TYPE_INT32, {max_bs_ + 1}, AllocationType::HOST});
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens = Buffer2torchTensor(cu_seqlens_buf, false);
    // RTP_LLM_CHECK_WITH_INFO(capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.is_pinned(),
    //                         "capture_mem_hold_ cu_seqlens is not pinned memory");
}

int HipGraphRunner::getCurrentRealGraphBs() {
    return current_real_graph_bs_;
}

std::vector<int> HipGraphRunner::getBatchSizesToCapture(int concurrency_limit) {
    std::vector<int> capture_bs;
    int              max_generate_batch_size = concurrency_limit;
    RTP_LLM_LOG_INFO("max_generate_batch_size for hip graph: %d", max_generate_batch_size);
    for (int i = 1; i <= std::min(32, max_generate_batch_size); i += 1) {
        capture_bs.push_back(i);
    }
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (capture_bs[capture_bs.size() - 1] != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

void HipGraphRunner::initCapture() {
    if (enable_hip_graph_) {
        RTP_LLM_LOG_INFO("HIP graph capture is enabled");
        if (is_prefill_hip_graph_mode_) {
            RTP_LLM_LOG_INFO("HIP graph capture for embedding");
            num_tokens_per_bs_ = max_seq_len_;
        }
        at::cuda::CUDAGraph graph;
        capture_range_ = HipGraphRunner::getBatchSizesToCapture(concurrency_limit_);

        max_bs_                = *(std::max_element(capture_range_.begin(), capture_range_.end()));
        max_num_token_         = max_bs_ * num_tokens_per_bs_;
        auto options_cpu_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
        auto options_hip_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        PyModelInputs inputs;
        inputs.attention_inputs.is_prefill = is_prefill_hip_graph_mode_;
        inputs.input_ids                   = torch::zeros({max_num_token_}, options_hip_int32);
        // inputs.attention_inputs.prefix_lengths   = torch::full({int(max_bs_)}, num_tokens_per_bs_,
        // options_hip_int32);
        // inputs.attention_inputs.prefix_lengths = torch::zeros({int(max_bs_)}, options_hip_int32);
        inputs.attention_inputs.input_lengths    = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_hip_int32);
        inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_hip_int32);
        inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
            {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_hip_int32);
        inputs.attention_inputs.kv_cache_block_id_host = torch::zeros(
            {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cpu_int32);
        inputs.attention_inputs.dtype = torch::kBFloat16;
        torch::Tensor output;

        capture_mem_hold_ = CaptureMemoryHold(output, inputs, kv_cache_block_offset_, is_prefill_hip_graph_mode_);
        initKernelInternalMemory();

        auto py_outputs_obj    = py_forward_method_(capture_mem_hold_.py_model_inputs_);
        auto outputs           = py_outputs_obj.cast<PyModelOutputs>();
        auto options_hip_float = torch::TensorOptions()
                                     .dtype(outputs.hidden_states.dtype().toScalarType())
                                     .device(torch::kCUDA)
                                     .requires_grad(false);
        output = torch::zeros({max_num_token_, hidden_size_}, options_hip_float);
        capture_mem_hold_.setHiddenStates(output);

        capture();
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("HIP graph capture is not enabled, skipping initialization");
    }
}

void HipGraphRunner::extractValidHiddenStates(PyModelOutputs&      outputs,
                                              const PyModelInputs& inputs,
                                              int32_t              total_valid_tokens) {
    auto& hidden_states = graph_instances_[current_real_graph_bs_].mem_hold_.decoder_layer_hidden_states_;
    auto  input_lengths = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();
    RTP_LLM_LOG_DEBUG("total_valid_tokens: %d, hidden_states.size(0): %d, seq_len_sum_: %d",
                      total_valid_tokens,
                      hidden_states.size(0),
                      seq_len_sum_);

    int32_t output_offset = 0;
    RTP_LLM_LOG_DEBUG("Extracting valid hidden states for embedding mode - batch_size: %d, total_valid_tokens: %d",
                      current_batch_size_,
                      total_valid_tokens);

    for (int i = 0; i < current_batch_size_; i++) {
        int32_t actual_length = input_lengths[i];
        int32_t batch_start   = i * num_tokens_per_bs_;
        RTP_LLM_LOG_DEBUG("Batch %d: actual_length=%d, batch_start=%d, output_offset=%d",
                          i,
                          actual_length,
                          batch_start,
                          output_offset);

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
        outputs.hidden_states.slice(0, output_offset, output_offset + actual_length) =
            hidden_states.slice(0, batch_start, batch_start + actual_length);
        output_offset += actual_length;
    }
    outputs.hidden_states = hidden_states.slice(0, 0, total_valid_tokens);
    RTP_LLM_LOG_DEBUG("Final output_offset: %d, expected: %d", output_offset, total_valid_tokens);
}

}  // namespace rtp_llm