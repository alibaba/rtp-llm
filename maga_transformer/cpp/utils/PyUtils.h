#pragma once

#include <torch/python.h>
#include <torch/all.h>

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace rtp_llm {

inline std::unordered_map<std::string, pybind11::handle> convertPyObjectToDict(pybind11::handle obj) {
    if (!pybind11::isinstance<pybind11::dict>(obj)) {
        throw std::runtime_error("Expected a dict, but get " + pybind11::cast<std::string>(pybind11::str(obj)));
    }
    pybind11::dict py_dict = pybind11::reinterpret_borrow<pybind11::dict>(obj);
    std::unordered_map<std::string, pybind11::handle> map;
    for (auto kv : py_dict) {   
        if (!pybind11::isinstance<pybind11::str>(kv.first)) {
            throw std::runtime_error("Expected a str, but get " + pybind11::cast<std::string>(pybind11::str(obj)));
        }
        map[pybind11::cast<std::string>(kv.first)] = kv.second;
    }
    return map;
}

inline std::vector<pybind11::handle> convertPyObjectToVec(pybind11::handle obj) {
    if (!pybind11::isinstance<pybind11::list>(obj)) {
        throw std::runtime_error("Expected a list, but get " + pybind11::cast<std::string>(pybind11::str(obj)));
    }
    pybind11::list py_list = pybind11::reinterpret_borrow<pybind11::list>(obj);
    std::vector<pybind11::handle> vec;    
    for (auto item : py_list) {
        vec.push_back(item);
    }
    return vec;
}

inline torch::Tensor convertPyObjectToTensor(pybind11::handle obj) {
    // pybind11 can not use isinstance to check py object is torch tensor.
    // invoke this function need to ensure this obj is a torch tensor.
    return pybind11::cast<torch::Tensor>(obj);
}


}  // namespace rtp_llm