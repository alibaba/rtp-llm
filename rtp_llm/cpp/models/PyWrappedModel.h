#pragma once
#include "rtp_llm/cpp/models/GptModel.h"
#include <string>
#include <mutex>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace py = pybind11;


namespace rtp_llm {

class PyWrappedModel : public GptModel {
public:
    PyWrappedModel(const GptModelInitParams& params, py::object py_instance);
    ~PyWrappedModel();

    GptModelOutputs forward(const GptModelInputs& inputs) override;

private:
    py::object        py_instance_;

    torch::Tensor k_cache_base_tensor_;
    torch::Tensor v_cache_base_tensor_;
};


// NOTE(wangyin): constructor can not be compiled correctly when placed in cc file.
inline PyWrappedModel::PyWrappedModel(const GptModelInitParams& params, py::object py_instance)
    : GptModel(params)
    , py_instance_(std::move(py_instance))
{
    if (setenv("PYTHONUNBUFFERED", "TRUE", 1) != 0) {
        RTP_LLM_LOG_WARNING("Failed to set PYTHONUNBUFFERED environment variable on POSIX.");
    } else {
        RTP_LLM_LOG_INFO("Set PYTHONUNBUFFERED=TRUE for Python interpreter.");
    }

    k_cache_base_tensor_ = Buffer2torchTensor(k_cache_buffer_, false);
    v_cache_base_tensor_ = Buffer2torchTensor(v_cache_buffer_, false);

    py::gil_scoped_acquire gil;

    if (!py_instance_ || py_instance_.is_none()) {
        throw std::runtime_error("PyWrappedModel constructor: Python instance is null or none.");
    }

    auto py_initialize_method = py_instance_.attr("initialize");
    torch_ext::PyModelInitResources init_resources;
    init_resources.k_cache_base = k_cache_base_tensor_;
    init_resources.v_cache_base = v_cache_base_tensor_;
    auto py_init_result = py_initialize_method(init_resources);
    auto py_init_success = py_init_result.cast<bool>();

    if (!py_init_success) {
        throw std::runtime_error("PyWrappedModel constructor: Python model initialization failed.");
    }

    RTP_LLM_LOG_INFO("PyWrappedModel initialized done.");
}

}  // namespace rtp_llm
