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
    PyWrappedModel(const GptModelInitParams& params, py::object py_instance);
    ~PyWrappedModel();

    GptModelOutputs forward(const GptModelInputs& inputs) override;

private:
    std::shared_ptr<GraphBase> graph_runner_;
    py::object                 py_model_;
    torch::Tensor              k_cache_base_tensor_;
    torch::Tensor              v_cache_base_tensor_;
    torch::Tensor              k_scale_base_tensor_;
    torch::Tensor              v_scale_base_tensor_;
};

// NOTE(wangyin): constructor can not be compiled correctly when placed in cc file.
inline PyWrappedModel::PyWrappedModel(const GptModelInitParams& params, py::object py_instance):
    GptModel(params),
    graph_runner_(DeviceFactory::getDeviceGraphRunner(params.device->initParams(),
                                                      py_instance,
                                                      k_cache_buffer_->shape()[0] * k_cache_buffer_->shape()[1],
                                                      device_,
                                                      false)),
    py_model_(py_instance) {
    if (setenv("PYTHONUNBUFFERED", "TRUE", 1) != 0) {
        RTP_LLM_LOG_WARNING("Failed to set PYTHONUNBUFFERED environment variable on POSIX.");
    } else {
        RTP_LLM_LOG_INFO("Set PYTHONUNBUFFERED=TRUE for Python interpreter.");
    }

    k_cache_base_tensor_ = Buffer2torchTensor(k_cache_buffer_, false);
    v_cache_base_tensor_ = Buffer2torchTensor(v_cache_buffer_, false);
    if (k_scale_buffer_) {
        k_scale_base_tensor_ = Buffer2torchTensor(k_scale_buffer_, false);
        v_scale_base_tensor_ = Buffer2torchTensor(v_scale_buffer_, false);
    }

    py::gil_scoped_acquire          gil;
    bool                            enable_cuda_graph = params.device->initParams().hw_kernel_config.enable_cuda_graph;
    torch_ext::PyModelInitResources init_resources;
    init_resources.kv_cache.k_cache_base = k_cache_base_tensor_;
    init_resources.kv_cache.v_cache_base = v_cache_base_tensor_;
    if (k_scale_buffer_) {
        init_resources.kv_cache.k_scale_base = k_scale_base_tensor_;
        init_resources.kv_cache.v_scale_base = v_scale_base_tensor_;
    }
    py::object py_init_result;
    if (enable_cuda_graph) {
        auto py_initialize_method = graph_runner_->py_instance_.attr("initialize");
        py_init_result            = py_initialize_method(init_resources);
        graph_runner_->initCapture();
    } else {
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
