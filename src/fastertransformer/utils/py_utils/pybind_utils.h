#pragma once
#include "torch/all.h"
#include <torch/python.h>
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace fastertransformer {

namespace py = pybind11;

inline std::unordered_map<std::string, py::handle> convertPyObjectToDict(py::handle obj) {
    if (!py::isinstance<py::dict>(obj)) {
        throw std::runtime_error("Expected a dict, but get " + py::cast<std::string>(py::str(obj)));
    }
    py::dict py_dict = py::reinterpret_borrow<py::dict>(obj);
    std::unordered_map<std::string, py::handle> map;
    for (auto kv : py_dict) {   
        if (!py::isinstance<py::str>(kv.first)) {
            throw std::runtime_error("Expected a str, but get " + py::cast<std::string>(py::str(obj)));
        }
        map[py::cast<std::string>(kv.first)] = kv.second;
    }
    return map;
}

inline std::vector<py::handle> convertPyObjectToVec(py::handle obj) {
    if (!py::isinstance<py::list>(obj)) {
        throw std::runtime_error("Expected a list, but get " + py::cast<std::string>(py::str(obj)));
    }
    py::list py_list = py::reinterpret_borrow<py::list>(obj);
    std::vector<py::handle> vec;    
    for (auto item : py_list) {
        vec.push_back(item);
    }
    return vec;
}

inline torch::Tensor convertPyObjectToTensor(py::handle obj) {
    // pybind11 can not use isinstance to check py object is torch tensor.
    // invoke this function need to ensure this obj is a torch tensor.
    return py::cast<torch::Tensor>(obj);
}

}
