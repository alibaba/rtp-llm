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
    GraphBase(params, std::move(py_instance), kv_cache_block_offset, device, is_prefill_hip_graph_mode),
    capture_stream_(HipGraphUtils::getStreamFromPool()) {}

HipGraphRunner::~HipGraphRunner() {
    RTP_LLM_LOG_INFO("Release HipGraphRunner .....");
    py::gil_scoped_acquire gil;
    py_instance_.release();
    RTP_LLM_LOG_INFO("Release HipGraphRunner Successfully");
}

std::unique_ptr<void, std::function<void(void*)>> HipGraphRunner::createStreamLife(void* capture_stream) {
    auto* stream_life = new HipGraphStreamLife(*static_cast<at::hip::HIPStream*>(capture_stream), device_);
    return std::unique_ptr<void, std::function<void(void*)>>(
        stream_life, [](void* ptr) { delete static_cast<HipGraphStreamLife*>(ptr); });
}

void HipGraphRunner::setParamsPtr(int bs, const PyModelOutputs& outputs) {
    graph_instances_[bs].mem_hold_.params_ptr = outputs.params_ptr;
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
                              is_prefill_graph_mode_);
        captureOneBatchSize(bs);
        // RTP_LLM_LOG_INFO("replay start check for %d", bs);
        // replay(bs);
        hipDeviceSynchronize();  // 注释掉可能导致死锁的同步调用
        // RTP_LLM_LOG_INFO("replay end check for %d", bs);
        // RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture End");
}

void HipGraphRunner::initCapture() {
    if (enable_graph_) {
        RTP_LLM_LOG_INFO("HIP graph capture is enabled");
        if (is_prefill_graph_mode_) {
            RTP_LLM_LOG_INFO("HIP graph capture for embedding");
            num_tokens_per_bs_ = max_seq_len_;
        }
        capture_range_ = GraphUtils::getBatchSizesToCapture(concurrency_limit_);

        max_bs_                = *(std::max_element(capture_range_.begin(), capture_range_.end()));
        max_num_token_         = max_bs_ * num_tokens_per_bs_;
        auto options_cpu_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
        auto options_hip_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        PyModelInputs inputs;
        inputs.attention_inputs.is_prefill = is_prefill_graph_mode_;
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

        capture_mem_hold_ = CaptureMemoryHold(output, inputs, kv_cache_block_offset_, is_prefill_graph_mode_);
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

void HipGraphRunner::initKernelInternalMemory() {
    // for `FusedRopeKVCacheDecodeOp`, cached in pinned memory.
    BufferPtr cu_seqlens_buf = device_->allocateBuffer({DataType::TYPE_INT32, {max_bs_ + 1}, AllocationType::HOST});
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens = Buffer2torchTensor(cu_seqlens_buf, false);
}

}  // namespace rtp_llm