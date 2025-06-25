#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/utils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include <stdexcept>
#include <mutex>
// pybind11/stl.h might still be needed if GptModelInputs/Outputs use STL containers
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "rtp_llm/cpp/utils/PyUtils.h"

#include <cstdlib>
#include <iostream>


namespace rtp_llm {

PyWrappedModel::PyWrappedModel(const GptModelInitParams& params, py::object py_instance)
    : GptModel(params)
    , py_instance_(std::move(py_instance)) // Take ownership of the passed py::object
{
    if (setenv("PYTHONUNBUFFERED", "TRUE", 1) != 0) {
        RTP_LLM_LOG_WARNING("Failed to set PYTHONUNBUFFERED environment variable on POSIX.");
    } else {
        RTP_LLM_LOG_INFO("Set PYTHONUNBUFFERED=TRUE for Python interpreter.");
    }

    k_cache_base_tensor_ = Buffer2torchTensor(k_cache_buffer_);
    v_cache_base_tensor_ = Buffer2torchTensor(v_cache_buffer_);

    py::gil_scoped_acquire gil; // Acquire GIL for safety, though direct operations are minimal now

    if (!py_instance_ || py_instance_.is_none()) {
        throw std::runtime_error("PyWrappedModel constructor: Python instance is null or none.");
    }

    RTP_LLM_LOG_INFO("PyWrappedModel initialized with a pre-existing Python object instance.");
}

PyWrappedModel::~PyWrappedModel() {
    try {
        py::gil_scoped_acquire gil;
        py_instance_.release(); // Release the Python object
        RTP_LLM_LOG_INFO("PyWrappedModel destroyed, Python object instance released.");
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during PyWrappedModel destruction: %s", e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during PyWrappedModel destruction: %s", e.what());
    }
}

GptModelOutputs PyWrappedModel::forward(const GptModelInputs& inputs) {

    py::gil_scoped_acquire gil;

    try {
        RTP_LLM_LOG_INFO("Calling forward method on Python object instance.");

        if (!py_instance_ || py_instance_.is_none()) {
            throw std::runtime_error("Python instance is not initialized.");
        }

        py::object py_forward_method = py_instance_.attr("forward");

        const BufferPtr combo_position_ids = inputs.combo_position_ids ? device_->clone({*inputs.combo_position_ids}): nullptr;
        auto attention_common_inputs = prepareAttentionInputs(inputs, DataType::TYPE_FP16, combo_position_ids);

        torch::Tensor token_ids = Buffer2torchTensor(inputs.combo_tokens).cuda();

        // py::dict kwargs = py::dict("attn_params"=attention_common_inputs);
        py::kwargs kwargs;
        kwargs["k_cache"] = k_cache_base_tensor_;
        kwargs["v_cache"] = v_cache_base_tensor_;
        kwargs["attn_params"] = attention_common_inputs;
        // py::object py_outputs_obj = py_forward_method(token_ids, k_cache, v_cache, attention_common_inputs); //

        py::object py_outputs_obj = py_forward_method(token_ids, **kwargs);

        auto hidden_states_tensor = convertPyObjectToTensor(py_outputs_obj);
        auto hidden_states = torchTensor2Buffer(hidden_states_tensor);

        RTP_LLM_LOG_INFO("Python object instance forward method called successfully.");

        return forwardPostLayers(hidden_states,
                                 inputs.input_lengths->shape()[0] != inputs.sequence_lengths->shape()[0],
                                 false,
                                 inputs.lm_output_indexes,
                                 false,
                                 inputs.combo_tokens->shape()[0],
                                 inputs,
                                 nullptr);

    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during forward call on Python instance: %s", e.what());
        throw std::runtime_error(std::string("pybind11 error during forward call on Python instance: ") + e.what());
    }
    catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during forward call on Python instance: %s", e.what());
        throw std::runtime_error(std::string("C++ error during forward call on Python instance: ") + e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("An unknown error occurred during forward call on Python instance.");
        throw std::runtime_error("An unknown error occurred during forward call on Python instance.");
    }
}


}  // namespace rtp_llm
