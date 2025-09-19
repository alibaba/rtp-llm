#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace rtp_llm {

std::unordered_map<std::string, pybind11::handle> convertPyObjectToDict(pybind11::handle obj) {
    if (!pybind11::isinstance<pybind11::dict>(obj)) {
        throw std::runtime_error("Expected a dict, but get " + pybind11::cast<std::string>(pybind11::str(obj)));
    }
    pybind11::dict                                    py_dict = pybind11::reinterpret_borrow<pybind11::dict>(obj);
    std::unordered_map<std::string, pybind11::handle> map;
    for (auto kv : py_dict) {
        if (!pybind11::isinstance<pybind11::str>(kv.first)) {
            throw std::runtime_error("Expected a str, but get " + pybind11::cast<std::string>(pybind11::str(obj)));
        }
        map[pybind11::cast<std::string>(kv.first)] = kv.second;
    }
    return map;
}

std::vector<pybind11::handle> convertPyObjectToVec(pybind11::handle obj) {
    if (!pybind11::isinstance<pybind11::list>(obj)) {
        throw std::runtime_error("Expected a list, but get " + pybind11::cast<std::string>(pybind11::str(obj)));
    }
    pybind11::list                py_list = pybind11::reinterpret_borrow<pybind11::list>(obj);
    std::vector<pybind11::handle> vec;
    for (auto item : py_list) {
        vec.push_back(item);
    }
    return vec;
}

std::vector<std::map<std::string, torch::Tensor>> pyListToTensorMapVec(py::list pyList) {
    std::vector<std::map<std::string, torch::Tensor>> out;
    auto                                              torch_type = py::module::import("torch").attr("Tensor");
    for (auto& item : pyList) {
        std::map<std::string, torch::Tensor> single_out;
        if (!pybind11::isinstance<py::dict>(item)) {
            throw std::runtime_error("Input is not a dictionary");
        }
        py::dict py_dict = py::cast<py::dict>(item);
        for (auto item : py_dict) {
            std::string key   = py::str(item.first);
            py::handle  value = item.second;
            if (!py::isinstance(value, torch_type)) {
                throw std::runtime_error("Non-tensor value found in dictionary");
            }
            single_out[key] = value.cast<torch::Tensor>();
        }
        out.emplace_back(std::move(single_out));
    }
    return out;
}

py::object convertTensorMapVectorToObject(const std::vector<std::map<std::string, torch::Tensor>>& tensor_map_vec) {
    py::list result_list;
    for (const auto& map : tensor_map_vec) {
        py::dict tensor_dict;
        for (const auto& item : map) {
            tensor_dict[py::cast(item.first)] = py::cast(item.second);
        }
        result_list.append(tensor_dict);
    }
    return result_list;
}

}  // namespace rtp_llm