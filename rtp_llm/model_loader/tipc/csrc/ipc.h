#include "common.h"
#include <string>
#include <stdexcept>

namespace py = pybind11;

namespace tipc {

py::bytes export_tensor_ipc(const torch::Tensor& t);

torch::Tensor import_tensor_ipc(py::bytes b);

// Versioned RL transport: allocation offset and physical device UUID are
// independent of the exporting process's CUDA ordinal. Legacy APIs stay intact.
py::bytes     export_tensor_ipc_v2(const torch::Tensor& t);
torch::Tensor import_tensor_ipc_v2(py::bytes b);

}  // namespace tipc
