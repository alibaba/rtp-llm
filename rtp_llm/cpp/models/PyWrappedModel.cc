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
    // py::scoped_ostream_redirect stream_redirect(
    //     std::cout, py::module_::import("sys").attr("stdout"));
    // py::scoped_ostream_redirect err_redirect(
    //     std::cerr, py::module_::import("sys").attr("stderr"));
    try {
        RTP_LLM_LOG_INFO("Calling forward method on Python object instance.");

        if (!py_instance_ || py_instance_.is_none()) {
            throw std::runtime_error("Python instance is not initialized.");
        }

// #define BUFFER_TO_TENSOR_IF_EXISTS(buf)
//         (buf ? Buffer2torchTensor(*buf, false) : torch::Tensor())

//         auto combo_tokens = BUFFER_TO_TENSOR_IF_EXISTS(inputs.combo_tokens);
//         auto input_lengths = BUFFER_TO_TENSOR_IF_EXISTS(inputs.input_lengths);
//         auto sequence_lengths = BUFFER_TO_TENSOR_IF_EXISTS(inputs.sequence_lengths);
//         auto attention_mask = BUFFER_TO_TENSOR_IF_EXISTS(inputs.attention_mask);
//         auto kv_cache_block_id = BUFFER_TO_TENSOR_IF_EXISTS(inputs.kv_cache_block_id);

//         // std::cout << "combo_tokens: " << combo_tokens << std::endl;
//         // std::cout << "input_lengths: " << input_lengths << std::endl;
//         // std::cout << "sequence_lengths: " << sequence_lengths << std::endl;
//         // std::cout << "attention_mask: " << attention_mask << std::endl;
//         // std::cout << "kv_cache_block_id: " << kv_cache_block_id << std::endl;

// #undef BUFFER_TO_TENSOR_IF_EXISTS

        py::object py_forward_method = py_instance_.attr("forward");

        // py::object py_outputs_obj = py_forward_method(
        //     combo_tokens, input_lengths, sequence_lengths,
        //     attention_mask, kv_cache_block_id
        // );

        // // TODO(wangyin.yx): tuple should not be used here,
        // // we need a concrete, well-defined output struct.
        // py::tuple result_tuple = py_outputs_obj.cast<py::tuple>();

        // auto logits = result_tuple[0].cast<torch::Tensor>();
        // auto hidden_states = result_tuple[1].cast<torch::Tensor>();

        // // std::cout << "logits: " << logits << std::endl;
        // // std::cout << "hidden_states: " << hidden_states << std::endl;

        // auto logits_buffer = torchTensor2Buffer(logits);
        // auto hidden_states_buffer = torchTensor2Buffer(hidden_states);

        // GptModelOutputs outputs{
        //     logits_buffer,
        //     hidden_states_buffer,
        // };
        // return outputs;
        // Call the Python forward method
        const BufferPtr combo_position_ids = inputs.combo_position_ids ? device_->clone({*inputs.combo_position_ids}): nullptr;

        auto attention_common_inputs = prepareAttentionInputs(inputs, DataType::TYPE_FP16, combo_position_ids);
        torch::Tensor token_ids = Buffer2torchTensor(inputs.combo_tokens).cuda();
        torch::Tensor k_cache = Buffer2torchTensor(k_cache_buffer_, false);
        torch::Tensor v_cache = Buffer2torchTensor(v_cache_buffer_, false);
        // py::dict kwargs = py::dict("attn_params"=attention_common_inputs);
        py::kwargs kwargs;
        kwargs["k_cache"] = k_cache;
        kwargs["v_cache"] = v_cache;
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
