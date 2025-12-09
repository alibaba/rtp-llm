#include "rtp_llm/cpp/devices/rocm_impl/HipGraphRunner.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include "rtp_llm/cpp/devices/GraphUtils.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cstring>

namespace rtp_llm {

// GPU memory copy implementation
void HipGraphRunner::optimizedCopy(const torch::Tensor& src, torch::Tensor& dst, size_t size) {
    if (src.is_cuda() && dst.is_cuda()) {  // is_cuda() returns true for HIP tensors too
        ROCM_CHECK(hipMemcpy(dst.data_ptr(), src.data_ptr(), size, hipMemcpyDeviceToDevice));
    } else if (!src.is_cuda() && !dst.is_cuda()) {
        memcpy(dst.data_ptr(), src.data_ptr(), size);
    } else if (src.is_cuda() && !dst.is_cuda()) {
        ROCM_CHECK(hipMemcpy(dst.data_ptr(), src.data_ptr(), size, hipMemcpyDeviceToHost));
    } else {
        ROCM_CHECK(hipMemcpy(dst.data_ptr(), src.data_ptr(), size, hipMemcpyHostToDevice));
    }
}

// GPU device synchronization
void HipGraphRunner::syncDevice() {
    ROCM_CHECK(hipDeviceSynchronize());
}

void HipGraphRunner::replayGraphImpl(int key) {
    cuda_graphs_[key].replay();  // PyTorch uses same API name for HIP
}

void HipGraphRunner::performGraphCapture(int key, const char* key_type) {
    // Set capture stream for graph capture
    at::hip::HIPStream origin_stream = at::hip::getCurrentHIPStream(at::hip::current_device());
    if (capture_stream_) {
        at::hip::setCurrentHIPStream(*static_cast<at::hip::HIPStream*>(capture_stream_));
        RTP_LLM_LOG_INFO("Set HIP capture stream for graph capture");
    }

    auto                 inputs              = graph_mem_holds_[key].py_model_inputs_;
    at::cuda::CUDAGraph& graph               = cuda_graphs_[key];  // PyTorch aliases CUDAGraph for HIP
    auto                 output_dot_filename = "";

    if (enable_graph_debug_mode_) {
        graph.enable_debug_mode();
        output_dot_filename = "hip_graph_visualization.dot";
    }

    RTP_LLM_LOG_INFO("Capture for %s %d begin.", key_type, key);
    PyModelOutputs outputs;
    {
        graph.capture_begin();
        GraphCaptureGuard capture_guard;
        auto              py_outputs_obj = py_forward_method_(inputs);
        outputs                          = py_outputs_obj.cast<PyModelOutputs>();
        graph_mem_holds_[key].decoder_layer_hidden_states_.copy_(outputs.hidden_states);
        graph.capture_end();
    }
    RTP_LLM_LOG_INFO("Capture for %s %d end.", key_type, key);

    if (outputs.params_ptr->check_recycle()) {
        graph_mem_holds_[key].params_ptr = ParamsBasePtr(outputs.params_ptr.get(), [&](ParamsBase* ptr) {});
    } else {
        graph_mem_holds_[key].params_ptr = outputs.params_ptr;
    }

    if (enable_graph_debug_mode_) {
        graph.debug_dump(output_dot_filename);
    }

    // Restore original stream
    if (capture_stream_) {
        at::hip::setCurrentHIPStream(origin_stream);
    }
}

// Factory function for creating HipGraphRunner from ROCmDevice
GraphBase* ROCmDevice::getDeviceGraphRunner(const DeviceInitParams& params,
                                            py::object              py_instance,
                                            int                     kv_cache_block_offset,
                                            bool                    is_prefill_hip_graph_mode) {
    if (!graph_runner_) {
        // Get a capture stream from the HIP stream pool
        at::hip::HIPStream capture_stream = at::hip::getStreamFromPool(true);
        graph_runner_                     = new HipGraphRunner(
            params, std::move(py_instance), kv_cache_block_offset, capture_stream, is_prefill_hip_graph_mode);
    }
    return graph_runner_;
}

}  // namespace rtp_llm
