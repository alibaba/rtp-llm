#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/devices/GraphUtils.h"
#include <cuda_runtime_api.h>
#include <cstring>

namespace rtp_llm {

// GPU memory copy implementation
void CudaGraphRunner::optimizedCopy(const torch::Tensor& src, torch::Tensor& dst, size_t size) {
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

// GPU device synchronization
void CudaGraphRunner::syncDevice() {
    check_cuda_value(cudaDeviceSynchronize());
}

void CudaGraphRunner::replayGraphImpl(int key) {
    cuda_graphs_[key].replay();
}

void CudaGraphRunner::performGraphCapture(int key, const char* key_type) {
    // Set capture stream for graph capture
    at::cuda::CUDAStream origin_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device());
    if (capture_stream_) {
        at::cuda::setCurrentCUDAStream(*static_cast<at::cuda::CUDAStream*>(capture_stream_));
        RTP_LLM_LOG_INFO("Set CUDA capture stream for graph capture");
    }

    auto                 inputs              = graph_mem_holds_[key].py_model_inputs_;
    at::cuda::CUDAGraph& graph               = cuda_graphs_[key];
    auto                 output_dot_filename = "";

    if (enable_graph_debug_mode_) {
        graph.enable_debug_mode();
        output_dot_filename = "cuda_graph_visualization.dot";
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
        at::cuda::setCurrentCUDAStream(origin_stream);
    }
}

// Factory function for creating CudaGraphRunner from CudaDevice
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

}  // namespace rtp_llm
