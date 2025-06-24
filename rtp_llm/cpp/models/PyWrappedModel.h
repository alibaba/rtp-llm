#pragma once
#include "rtp_llm/cpp/models/GptModel.h"
#include <string>
#include <mutex>

#include <pybind11/pybind11.h> // Core pybind11 functionality
#include <pybind11/embed.h>   // For py::initialize_interpreter, etc.

namespace py = pybind11;


namespace rtp_llm {

class PyWrappedModel : public GptModel {
public:
    // Constructor now takes an initialized Python object instance
    PyWrappedModel(const GptModelInitParams& params, py::object py_instance);
    ~PyWrappedModel();

    GptModelOutputs forward(const GptModelInputs& inputs) override;

private:
    py::object        py_instance_; // Stores the instance of the Python class

};

}  // namespace rtp_llm
