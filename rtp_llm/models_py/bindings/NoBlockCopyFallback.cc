#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

#if defined(__GNUC__)
#define RTP_LLM_WEAK __attribute__((weak))
#else
#define RTP_LLM_WEAK
#endif

// Optional fast paths added after the original NoBlockCopy device interface.
// Weak fallbacks let downstream backends keep their existing implementation;
// returning false for non-empty work routes the caller to the generic copy path.
RTP_LLM_WEAK bool execBatchedMemoryCopy(const BatchedMemoryCopyParams& params) {
    return params.tiles.empty();
}

RTP_LLM_WEAK bool execStagedMemoryCopy(const StagedMemoryCopyParams& params, StagedMemoryCopyScratch*) {
    return params.tiles.empty();
}

RTP_LLM_WEAK void releaseStagedMemoryCopyScratch(StagedMemoryCopyScratch&) {}

RTP_LLM_WEAK bool prewarmStagedMemoryCopyScratch(StagedMemoryCopyScratch&, int, size_t, size_t) {
    return false;
}

#undef RTP_LLM_WEAK

}  // namespace rtp_llm
