#include "common.h"
#include <string>
#include <stdexcept>

namespace py = pybind11;

namespace tipc {

py::bytes export_tensor_ipc(const torch::Tensor& t);

torch::Tensor import_tensor_ipc(py::bytes b);

}