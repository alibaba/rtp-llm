# include "common.h"
# include "ipc.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("export_tensor_ipc", &tipc::export_tensor_ipc,
          py::arg("tensor"),
          "Export CUDA tensor to IPC bytes");
    m.def("import_tensor_ipc", &tipc::import_tensor_ipc,
          py::arg("bytes"),
          "Import CUDA tensor from IPC bytes");
}