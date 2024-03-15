#pragma once

#include "src/fastertransformer/devices/OpData.h"


#include <optional>
#include <functional>
#include <sstream>

namespace fastertransformer {

void GemmParams::check() const {
    // check dim
    RUNTIME_ASSERT_OP_ARG((A.dim() >= 2) && (B.dim() >= 2),
        "gemm params dim is must greater than 2!");

}

}  // namespace fastertransformer
