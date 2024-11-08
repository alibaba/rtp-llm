#pragma once

#include <torch/python.h>
#include <torch/all.h>

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace rtp_llm {

namespace py = pybind11;

std::unordered_map<std::string, py::handle> convertPyObjectToDict(py::handle obj); 

std::vector<py::handle> convertPyObjectToVec(py::handle obj);

inline torch::Tensor convertPyObjectToTensor(py::handle obj) {
    // pybind11 can not use isinstance to check py object is torch tensor.
    // invoke this function need to ensure this obj is a torch tensor.
    return py::cast<torch::Tensor>(obj);
}


}  // namespace rtp_llm