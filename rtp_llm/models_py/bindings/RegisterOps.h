#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <torch/extension.h>

namespace torch_ext {

void registerPyModuleOps(pybind11::module& m);

}
