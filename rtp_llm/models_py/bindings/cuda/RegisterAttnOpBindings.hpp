#pragma once

#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/models_py/bindings/cuda/SparseMlaParams.h"

#if USING_CUDA && !defined(USE_PPU)
#include "rtp_llm/models_py/bindings/cuda/XQAAttnOp.h"
#endif

namespace torch_ext {

void registerAttnOpBindings(py::module& rtp_ops_m) {
    rtp_llm::registerPyFlashInferMlaParams(rtp_ops_m);
    rtp_llm::registerPySparseMlaParams(rtp_ops_m);
#if USING_CUDA && !defined(USE_PPU)
    rtp_llm::registerXQAAttnOp(rtp_ops_m);
#endif
}

}  // namespace torch_ext
