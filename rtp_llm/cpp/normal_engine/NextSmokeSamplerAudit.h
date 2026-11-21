#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>

namespace rtp_llm::next_smoke_audit {
inline bool enabled() {
    static const bool value = [] {
        const char* mode = std::getenv("RTP_NEXT_SAMPLER_DIAGNOSTIC");
        const char* target = std::getenv("TEST_TARGET");
        const char* tmp = std::getenv("TEST_TMPDIR");
        if (!mode || std::string_view(mode) != "1" || !target || !tmp || tmp[0] != '/') return false;
        const std::string_view t(target);
        return t == "//rtp_llm/test/smoke:next_moe_nvfp4_cudagraph_tp2_sm100_sample_diag" ||
               t == "//rtp_llm/test/smoke:next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100_sample_diag";
    }();
    return value;
}

inline int slot(int64_t stream_id) {
    if (!enabled() || stream_id < 0) return -1;
    const uint64_t tag = static_cast<uint64_t>(stream_id) * UINT64_C(11400714819323198485);
    std::ifstream in(std::string(std::getenv("TEST_TMPDIR")) + "/next-smoke-auth-" + std::to_string(tag));
    const int value = in.get();
    return value >= '0' && value <= '3' && in.get() == std::char_traits<char>::eof() ? value - '0' : -1;
}

// Called only for the isolated, fully authorized fixture batch. Read-only
// pre-sampler snapshot. A GPU synchronization is intentional in diagnostic
// siblings, never in the ordinary golden or for a mixed/unknown request batch.
inline std::pair<at::Tensor, at::Tensor> top2(const at::Tensor& logits) {
    auto top = logits.topk(2, -1, true, true);
    return {std::get<0>(top).to(at::kFloat).cpu().contiguous(),
            std::get<1>(top).to(at::kLong).cpu().contiguous()};
}
}  // namespace rtp_llm::next_smoke_audit
