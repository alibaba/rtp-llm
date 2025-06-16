#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/utils/utils.h"
#include <stdexcept>
#include <mutex>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <cstdlib>
#include <libgen.h>
#include <vector>
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

            py::initialize_interpreter(false);
            s_python_initialized = true;
            RTP_LLM_LOG_INFO("Python interpreter initialized via pybind11.");
        }
    }
}

// Helper function to get module name from file path
// e.g., "/path/to/my_module.py" -> "my_module"
std::string PyWrappedModel::GetPythonModuleNameFromFile(const std::string& file_path) {
    std::string path_copy = file_path; // Copy since basename/dirname might modify input
    char*       path_c_str = &path_copy[0];

    char* base = basename(path_c_str);
    std::string base_name_str(base);

    size_t dot_pos = base_name_str.rfind('.');
    if (dot_pos != std::string::npos) {
        return base_name_str.substr(0, dot_pos);
    }
    return base_name_str; // Should not happen if it's a .py file
}


PyWrappedModel::PyWrappedModel(const GptModelInitParams& params, const std::string& py_path, const std::string& class_name)
    : GptModel(params)
    , py_path_(py_path)
    , class_name_(class_name)
{
    EnsurePythonInitialized();

    py::gil_scoped_acquire gil;

    try {
        RTP_LLM_LOG_INFO("Attempting to load Python class '%s' from file: %s",
                         class_name_.c_str(), py_path_.c_str());

        // Extract directory from py_path_ to add to sys.path
        std::string dir_path_str;
        // Use a mutable copy for dirname
        std::vector<char> py_path_copy(py_path_.begin(), py_path_.end());
        py_path_copy.push_back('\0');

        char* d_name = ::dirname(py_path_copy.data());
        if (d_name) {
            dir_path_str = d_name;
        }

        if (!dir_path_str.empty() && dir_path_str != ".") {
            py::module_ sys = py::module_::import("sys");
            py::list sys_path = sys.attr("path").cast<py::list>();
            bool path_exists = false;
            for (const auto& path_item : sys_path) {
                if (py::str(path_item).cast<std::string>() == dir_path_str) {
                    path_exists = true;
                    break;
                }
            }
            if (!path_exists) {
                sys_path.append(dir_path_str.c_str());
                RTP_LLM_LOG_INFO("Added directory to sys.path: %s", dir_path_str.c_str());
            } else {
                RTP_LLM_LOG_INFO("Directory already in sys.path: %s", dir_path_str.c_str());
            }
        } else {
            RTP_LLM_LOG_INFO("No directory path extracted or using CWD for py_path ('%s'), relying on existing sys.path.", py_path_.c_str());
        }

        std::string python_module_name = GetPythonModuleNameFromFile(py_path_);
        RTP_LLM_LOG_INFO("Derived Python module name: %s", python_module_name.c_str());

        py::module_ loaded_py_module = py::module_::import(python_module_name.c_str());
        RTP_LLM_LOG_INFO("Successfully imported Python module: %s", python_module_name.c_str());

        py::object py_class = loaded_py_module.attr(class_name_.c_str());
        RTP_LLM_LOG_INFO("Successfully retrieved class '%s' from module '%s'", class_name_.c_str(), python_module_name.c_str());

        py_instance_ = py_class(); // Call constructor with no arguments
        RTP_LLM_LOG_INFO("Instantiated Python class '%s' with no parameters.", class_name_.c_str());

        RTP_LLM_LOG_INFO("PyWrappedModel initialized. Python class instance '%s' created from file: %s",
                         class_name_.c_str(), py_path_.c_str());

    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during module loading: %s", e.what());
        // e.what() already contains a formatted Python traceback.
        throw std::runtime_error("pybind11 error during Python module loading for '" + class_name_ + "': " + e.what());
    } catch (const std::exception& e) {
        throw std::runtime_error("C++ error during Python module loading for '" + class_name_ + "': " + e.what());
    } catch (...) {
        throw std::runtime_error("An unknown error occurred during Python module loading for '" + class_name_ + "' from file.");
    }
}

PyWrappedModel::~PyWrappedModel() {
    if (s_python_initialized && py_instance_) {
        try {
            py::gil_scoped_acquire gil;
            py_instance_.release();
            RTP_LLM_LOG_INFO("PyWrappedModel destroyed, Python class instance for '%s' released.", class_name_.c_str());
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_ERROR("Python error during PyWrappedModel destruction for class '%s': %s", class_name_.c_str(), e.what());
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("C++ error during PyWrappedModel destruction for class '%s': %s", class_name_.c_str(), e.what());
        }
    }
    // Note: py::finalize_interpreter() is generally not called here.
    // It should be called once when the application exits, if at all.
}

GptModelOutputs PyWrappedModel::forward(const GptModelInputs& inputs) {
    EnsurePythonInitialized();

    py::gil_scoped_acquire gil;
    try {
        RTP_LLM_LOG_INFO("Calling forward method on Python class instance '%s'.", class_name_.c_str());

        if (!py_instance_) {
            throw std::runtime_error("Python instance is not initialized for class: " + class_name_);
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
        RTP_LLM_LOG_INFO("Python class '%s' forward method called successfully.", class_name_.c_str());

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
        RTP_LLM_LOG_ERROR("Python error during forward call on class '%s': %s", class_name_.c_str(), e.what());
        throw std::runtime_error("pybind11 error during forward call on Python class '" + class_name_ + "': " + e.what());
    }
    catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during forward call on class '%s': %s", class_name_.c_str(), e.what());
        throw std::runtime_error("C++ error during forward call on Python class '" + class_name_ + "': " + e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("An unknown error occurred during forward call on class '%s'.", class_name_.c_str());
        throw std::runtime_error("An unknown error occurred during forward call on Python class '" + class_name_ + "'.");
    }
}

}  // namespace rtp_llm
