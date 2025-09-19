#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <torch/extension.h>

namespace rtp_llm {

void registerPyModuleOps(pybind11::module& m);

}
