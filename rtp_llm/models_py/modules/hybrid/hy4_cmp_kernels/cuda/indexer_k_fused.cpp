#include <torch/extension.h>
#include <pybind11/stl.h>
namespace hy4_indexer_k_host { void bind(pybind11::module_&); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) { hy4_indexer_k_host::bind(module); }
