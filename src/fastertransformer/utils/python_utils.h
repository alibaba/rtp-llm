#include <torch/python.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/all.h>

namespace rtp_llm {

// all functions below need to hold gil before
py::object ConvertTensorMapVectorToObject(const std::vector<std::map<std::string, torch::Tensor>>& tensor_map_vec);

py::object ConvertTensorToObject(torch::Tensor t);

std::vector<std::map<std::string, torch::Tensor>> pyListToTensorMapVec(py::list pyList);

} // rtp_llm