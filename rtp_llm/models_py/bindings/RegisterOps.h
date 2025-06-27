#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <torch/extension.h>


namespace torch_ext {

void registerPyModuleOps(pybind11::module &m);

#ifdef __aarch64__
inline void registerPyModuleOps(pybind11::module &m) {}
#endif

}

