#pragma once

#include <pybind11/pybind11.h>

namespace rtp_llm {

void registerCommPybindings(pybind11::module& m);

}  // namespace rtp_llm
