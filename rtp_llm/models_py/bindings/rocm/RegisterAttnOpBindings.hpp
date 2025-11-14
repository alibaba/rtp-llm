#pragma once

#include "rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.h"
#include "rtp_llm/models_py/bindings/rocm/PagedAttn.h"

namespace rtp_llm {

void registerAttnOpBindings(py::module& rtp_ops_m) {
    rtp_llm::registerFusedRopeKVCacheOp(rtp_ops_m);
    rtp_llm::registerPagedAttnDecodeOp(rtp_ops_m);
}

}  // namespace rtp_llm
