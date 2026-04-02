#pragma once

#include "rtp_llm/cpp/cuda/cufmha/TRTAttn.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

void registerTRTAttn(const py::module& m);

}  // namespace rtp_llm
