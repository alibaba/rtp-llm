#pragma once

#include <pybind11/pybind11.h>

namespace rtp_llm {
void registerVllmCustomAllReduce(pybind11::module& m);
}
