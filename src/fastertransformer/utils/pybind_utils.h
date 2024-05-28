#pragma once
#include "torch/all.h"
#include <torch/python.h>
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace fastertransformer {

namespace py = pybind11;

// Dict[str, torch.Tensor] -> std::unordered_map<std::string, torch::Tensor>
std::unordered_map<std::string, torch::Tensor> convertPyObjectToDict(py::handle obj);

// List[Dict[str, torch.Tensor]] -> std::vector<std::unordered_map<std::string, torch::Tensor>>
std::vector<std::unordered_map<std::string, torch::Tensor>> convertPyobjectToVectorDict(py::object obj);

}