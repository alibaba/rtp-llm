#include <pybind11/pybind11.h>
#include "rtp_llm/tools/tipc/cpp/IpcOp.h"

namespace py = pybind11;

PYBIND11_MODULE(tipc_lib, m) {
    m.doc() = "tipc cpp library";
    m.def("import_tensor_ipc", &torch_ext::import_tensor_ipc, py::arg("encoded"));
    m.def("export_tensor_ipc", &torch_ext::export_tensor_ipc, py::arg("tensor"));
}