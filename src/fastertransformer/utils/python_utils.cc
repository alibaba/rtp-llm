#include "src/fastertransformer/utils/python_utils.h"

namespace rtp_llm {

// ensure gil is acquired before enter this function
py::object ConvertTensorMapVectorToObject(const std::vector<std::map<std::string, torch::Tensor>>& tensor_map_vec) {
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

// ensure gil is acquired before enter this functionn
py::object ConvertTensorToObject(torch::Tensor t) {
    return py::cast(t);
}

std::vector<std::map<std::string, torch::Tensor>> pyListToTensorMapVec(py::list pyList) {
    std::vector<std::map<std::string, torch::Tensor>> out;
    auto torch_type = py::module::import("torch").attr("Tensor");
    for (auto& item: pyList) {
        std::map<std::string, torch::Tensor> single_out;
        if (!pybind11::isinstance<py::dict>(item)) {
            throw std::runtime_error("Input is not a dictionary");
        }
        py::dict py_dict = py::cast<py::dict>(item);
        for (auto item : py_dict) {
            std::string key = py::str(item.first);
            py::handle value = item.second;
            if (!py::isinstance(value, torch_type)) {
                throw std::runtime_error("Non-tensor value found in dictionary");
            }
            single_out[key] = value.cast<torch::Tensor>();
        }
        out.emplace_back(std::move(single_out));
    }
    return out;
}

}  // namespace rtp_llm
