#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

void execNoBlockCopy(const MultiCopyParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        params.multi_dst[i].copy_(params.multi_src[i]);
    }
}

void warmupNoBlockCopy() {}

}  // namespace rtp_llm
