#pragma once
#include "rtp_llm/cpp/models/GptModel.h"
#include <string>
#include <mutex>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace py = pybind11;

namespace rtp_llm {

class PyWrappedModel: public GptModel {
public:
    // py_instance is `py_model` indeedly.
    PyWrappedModel(const GptModelInitParams& params, py::object py_instance, bool is_embedding = false);
    ~PyWrappedModel();

    GptModelOutputs forward(const GptModelInputs& inputs) override;
    GptModelOutputs forwardMicroBatched(const GptModelInputs& inputs);

private:
    GraphBase*    graph_runner_{nullptr};
    py::object    py_model_;
    bool          enable_cuda_graph_{false};
    torch::Tensor k_cache_base_tensor_;
    torch::Tensor v_cache_base_tensor_;
    torch::Tensor k_scale_base_tensor_;
    torch::Tensor v_scale_base_tensor_;
};

// NOTE(wangyin): constructor can not be compiled correctly when placed in cc file.
inline PyWrappedModel::PyWrappedModel(const GptModelInitParams& params, py::object py_instance, bool is_embedding):
    GptModel(params), enable_cuda_graph_(params.device->initParams().hw_kernel_config.enable_cuda_graph) {
    if (setenv("PYTHONUNBUFFERED", "TRUE", 1) != 0) {
        RTP_LLM_LOG_WARNING("Failed to set PYTHONUNBUFFERED environment variable on POSIX.");
    } else {
        RTP_LLM_LOG_INFO("Set PYTHONUNBUFFERED=TRUE for Python interpreter.");
    }
    if (k_cache_buffer_) {
        k_cache_base_tensor_ = Buffer2torchTensor(k_cache_buffer_, false);
        v_cache_base_tensor_ = Buffer2torchTensor(v_cache_buffer_, false);
    }
    if (k_scale_buffer_) {
        k_scale_base_tensor_ = Buffer2torchTensor(k_scale_buffer_, false);
        v_scale_base_tensor_ = Buffer2torchTensor(v_scale_buffer_, false);
    }

    py::gil_scoped_acquire          gil;
    torch_ext::PyModelInitResources init_resources;
    if (k_cache_buffer_) {
        torch_ext::KVCache kv_cache;
        kv_cache.k_cache_base = k_cache_base_tensor_;
        kv_cache.v_cache_base = v_cache_base_tensor_;
        if (k_scale_buffer_) {
            kv_cache.k_scale_base = k_scale_base_tensor_;
            kv_cache.v_scale_base = v_scale_base_tensor_;
        }
        init_resources.kv_cache = kv_cache;
    }
    py::object py_init_result;
    if (enable_cuda_graph_) {
        int kv_cache_offset = is_embedding ? 0 : k_cache_buffer_->shape()[0] * k_cache_buffer_->shape()[1];
        graph_runner_ =
            device_->getDeviceGraphRunner(params.device->initParams(), py_instance, kv_cache_offset, is_embedding);
        RTP_LLM_CHECK_WITH_INFO(graph_runner_ != nullptr, "graph_runner_ can't be null");
        auto py_initialize_method = py_instance.attr("initialize");
        py_init_result            = py_initialize_method(init_resources);
        graph_runner_->initCapture();
    } else {
        py_model_                 = std::move(py_instance);
        auto py_initialize_method = py_model_.attr("initialize");
        py_init_result            = py_initialize_method(init_resources);
    }

    auto py_init_success = py_init_result.cast<bool>();
    if (!py_init_success) {
        throw std::runtime_error("PyWrappedModel constructor: Python model initialization failed.");
    }
    RTP_LLM_LOG_INFO("PyWrappedModel initialized done.");
}

}  // namespace rtp_llm
