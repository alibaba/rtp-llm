#pragma once

#include <pybind11/pybind11.h>

namespace rtp_llm {
void registerRocmQuickReduce(pybind11::module& m);
}
