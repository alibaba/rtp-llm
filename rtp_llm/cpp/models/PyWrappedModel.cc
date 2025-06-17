#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/utils/utils.h"
#include <stdexcept>
#include <mutex>
// pybind11/stl.h might still be needed if GptModelInputs/Outputs use STL containers
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <cstdlib>
#include <iostream>


namespace rtp_llm {

bool PyWrappedModel::s_python_initialized = false;
std::mutex PyWrappedModel::s_python_init_mutex;

void PyWrappedModel::EnsurePythonInitialized() {
    if (!s_python_initialized) {
        std::lock_guard<std::mutex> lock(s_python_init_mutex);
        if (!s_python_initialized) {

            // Set PYTHONUNBUFFERED=TRUE before initializing the interpreter
            // This helps ensure that Python's output is not overly buffered,
            // which is useful when redirecting stdout/stderr.
            if (setenv("PYTHONUNBUFFERED", "TRUE", 1) != 0) {
                RTP_LLM_LOG_WARNING("Failed to set PYTHONUNBUFFERED environment variable on POSIX.");
            } else {
                RTP_LLM_LOG_INFO("Set PYTHONUNBUFFERED=TRUE for Python interpreter.");
            }

            // try {
            //     py::initialize_interpreter(false);
            // } catch (const py::error_already_set& e) {
            //     RTP_LLM_LOG_WARNING("Failed to initialize Python interpreter: %s", e.what());
            // }
            s_python_initialized = true;
            RTP_LLM_LOG_INFO("Python interpreter initialized via pybind11.");
        }
    }
}

PyWrappedModel::PyWrappedModel(const GptModelInitParams& params, py::object py_instance)
    : GptModel(params)
    , py_instance_(std::move(py_instance)) // Take ownership of the passed py::object
{
    EnsurePythonInitialized(); // Ensure interpreter is up for any base class or immediate needs

    py::gil_scoped_acquire gil; // Acquire GIL for safety, though direct operations are minimal now

    if (!py_instance_ || py_instance_.is_none()) {
        throw std::runtime_error("PyWrappedModel constructor: Python instance is null or none.");
    }

    RTP_LLM_LOG_INFO("PyWrappedModel initialized with a pre-existing Python object instance.");
}

PyWrappedModel::~PyWrappedModel() {
    if (s_python_initialized && py_instance_) {
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
}

GptModelOutputs PyWrappedModel::forward(const GptModelInputs& inputs) {
    EnsurePythonInitialized();

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

        py::object py_forward_method = py_instance_.attr("forward");

        // Convert GptModelInputs to a Python-compatible format.
        // This is highly dependent on GptModelInputs and what the Python 'forward' expects.
        // Example: convert to a py::dict.
        // You might need to include <pybind11/stl.h> for map/vector conversions
        // and define conversions for custom types if GptModelInputs contains them.
        py::dict py_inputs_dict;
        // Example: Assuming GptModelInputs has a 'tokens' field (std::vector<int>)
        // if (inputs.combo_tokens) { // Assuming combo_tokens is a pointer or optional
        //    py_inputs_dict["combo_tokens"] = py::cast(inputs.combo_tokens->data); // Example
        // }
        // This part needs to be adapted to the actual structure of GptModelInputs
        // and the Python method's signature.
        // For now, passing an empty dict as a placeholder.
        // py_inputs_dict["some_input_key"] = py::cast(inputs.some_field);


        // Call the Python forward method
        py::object py_outputs_obj = py_forward_method(py_inputs_dict); // Pass converted inputs
        RTP_LLM_LOG_INFO("Python object instance forward method called successfully.");

        // Convert py_outputs_obj back to GptModelOutputs
        // This is also highly dependent on the structure of GptModelOutputs
        // and what the Python 'forward' method returns.
        GptModelOutputs outputs{};
        // Example: if Python returns a dict
        // if (py::isinstance<py::dict>(py_outputs_obj)) {
        //    py::dict py_outputs_dict = py_outputs_obj.cast<py::dict>();
        //    if (py_outputs_dict.contains("output_tokens")) {
        //        outputs.output_ids = std::make_shared<Tensor>(py_outputs_dict["output_tokens"].cast<std::vector<int>>());
        //    }
        // }
        return outputs; // Return converted outputs

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
