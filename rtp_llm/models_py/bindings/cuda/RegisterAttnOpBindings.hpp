#pragma once

#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/models_py/bindings/cuda/SparseMlaParams.h"

// XQA is available on NVIDIA CUDA 12 and CUDA 13 builds; PPU has its own path.
#if defined(USING_CUDA12) || defined(USING_CUDA13)
#include "rtp_llm/models_py/bindings/cuda/XQAAttnOp.h"
#endif

namespace torch_ext {

void registerAttnOpBindings(py::module& rtp_ops_m) {
    rtp_llm::registerPyFlashInferMlaParams(rtp_ops_m);
    rtp_llm::registerPySparseMlaParams(rtp_ops_m);
#if defined(USING_CUDA12) || defined(USING_CUDA13)
    rtp_llm::registerXQAAttnOp(rtp_ops_m);
#endif
}

}  // namespace torch_ext
