#pragma once
#include "rtp_llm/cpp/models/GptModel.h"
#include <string>
#include <mutex>
#include <map> // Required for GptModelInitParams if it uses std::map

#include <pybind11/pybind11.h> // Core pybind11 functionality
#include <pybind11/embed.h>   // For py::initialize_interpreter, etc.
#include <pybind11/stl.h>     // For automatic conversion of std::map

namespace py = pybind11;


namespace rtp_llm {

class PyWrappedModel : public GptModel {
public:
    // Renamed module_name to class_name for clarity
    PyWrappedModel(const GptModelInitParams& params, const std::string& py_path, const std::string& class_name);
    ~PyWrappedModel();

    GptModelOutputs forward(const GptModelInputs& inputs) override;

private:
    const std::string py_path_;
    const std::string class_name_; // Stores the Python class name
    py::object        py_instance_; // Stores the instance of the Python class

    static void EnsurePythonInitialized();
    static bool s_python_initialized;
    static std::mutex s_python_init_mutex;

    // Helper to extract module name from file path
    static std::string GetPythonModuleNameFromFile(const std::string& file_path);
};

}  // namespace rtp_llm
