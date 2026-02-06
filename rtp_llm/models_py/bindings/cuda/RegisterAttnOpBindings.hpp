#pragma once

#include "rtp_llm/models_py/bindings/cuda/FlashInferOp.h"
#include "rtp_llm/models_py/bindings/cuda/TRTAttnOp.h"
#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/models_py/bindings/cuda/IndexerParams.h"
#include "rtp_llm/models_py/bindings/cuda/FusedRopeKVCacheOp.h"

#ifdef USING_CUDA12
#include "rtp_llm/models_py/bindings/cuda/XQAAttnOp.h"
#endif

namespace torch_ext {

void registerAttnOpBindings(py::module& rtp_ops_m) {
    rtp_llm::registerFusedRopeKVCacheOp(rtp_ops_m);
    rtp_llm::registerFlashInferOp(rtp_ops_m);
    rtp_llm::registerTRTAttnOp(rtp_ops_m);
    rtp_llm::registerPyFlashInferMlaParams(rtp_ops_m);
    rtp_llm::registerPyIndexerParams(rtp_ops_m);
#ifdef USING_CUDA12
    rtp_llm::registerXQAAttnOp(rtp_ops_m);
#endif
}

}  // namespace torch_ext
