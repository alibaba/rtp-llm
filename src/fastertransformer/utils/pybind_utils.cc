#include "src/fastertransformer/utils/pybind_utils.h"

namespace fastertransformer {

std::unordered_map<std::string, torch::Tensor> convertPyObjectToDict(py::handle obj) {
    if (!py::isinstance<py::dict>(obj)) {
        throw std::runtime_error("Expected a dicts");
    }
    py::dict py_dict = py::reinterpret_borrow<py::dict>(obj);
    std::unordered_map<std::string, torch::Tensor> map;
    for (auto kv : py_dict) {
        std::string   key   = py::cast<std::string>(kv.first);
        torch::Tensor value = py::cast<torch::Tensor>(kv.second);        
        map[key] = value;
    }
    return map;
}

std::vector<std::unordered_map<std::string, torch::Tensor>> convertPyobjectToVectorDict(py::object obj) {
    if (!py::isinstance<py::list>(obj)) {
        throw std::runtime_error("Expected a list");
    }
    py::list py_list = py::reinterpret_borrow<py::list>(obj);
    std::vector<std::unordered_map<std::string, torch::Tensor>> vec;    
    for (auto item : py_list) {
        vec.push_back(std::move(convertPyObjectToDict(item)));
    }
    return vec;
}

}