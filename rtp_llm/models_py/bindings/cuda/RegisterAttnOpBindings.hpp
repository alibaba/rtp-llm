#pragma once

#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/models_py/bindings/cuda/SparseMlaParams.h"

// XQA compiles in every CUDA build: `@//:using_cuda12` selects CudaXqa.cc and
// //3rdparty/xqa, and every real config inherits --config=cuda12 (cuda13_base
// --config=cuda12); cuda13 only undefs the USING_CUDA12 C++ macro. Guarding by
// USING_CUDA keeps the C++ condition aligned with the compiled build graph.
// USE_PPU stays excluded (PPU has its own attention path).
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
