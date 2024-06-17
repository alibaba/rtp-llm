#pragma once
#include "torch/all.h"
#include <torch/python.h>
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace fastertransformer {

namespace py = pybind11;

std::unordered_map<std::string, py::handle> convertPyObjectToDict(py::handle obj);

std::vector<py::handle> convertPyObjectToVec(py::handle obj);

torch::Tensor convertPyObjectToTensor(py::handle obj);

}