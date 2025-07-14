#pragma once

#include <torch/python.h>
#include <torch/all.h>

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace rtp_llm {

__attribute__((visibility("default"))) std::unordered_map<std::string, pybind11::handle>
                                       convertPyObjectToDict(pybind11::handle obj);

__attribute__((visibility("default"))) std::vector<pybind11::handle> convertPyObjectToVec(pybind11::handle obj);

inline py::object convertTensorToObject(torch::Tensor t) {
    return py::cast(t);
}

__attribute__((visibility("default"))) py::object
convertTensorMapVectorToObject(const std::vector<std::map<std::string, torch::Tensor>>& tensor_map_vec);

inline torch::Tensor convertPyObjectToTensor(pybind11::handle obj) {
    // pybind11 can not use isinstance to check py object is torch tensor.
    // invoke this function need to ensure this obj is a torch tensor.
    return pybind11::cast<torch::Tensor>(obj);
}

__attribute__((visibility("default"))) std::vector<std::map<std::string, torch::Tensor>>
                                       pyListToTensorMapVec(py::list pyList);

}  // namespace rtp_llm