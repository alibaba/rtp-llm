#include "rtp_llm/cpp/normal_engine/speculative/MtpPrefillChunkPlan.h"
#undef NDEBUG
#include <cassert>
#include <stdexcept>

int main() {
    using namespace rtp_llm;
    assert(nativeMtpPrefillChunkTopologySupported(8, 8, 1));
    assert(!nativeMtpPrefillChunkTopologySupported(8, 16, 2));
    assert(!nativeMtpPrefillChunkTopologySupported(1, 1, 1));
    // PREFILL role metadata is true even for local, non-PD requests.
    assert(nativeMtpPrefillChunkPdSupported(true, true, false));
    assert(!nativeMtpPrefillChunkPdSupported(true, true, true));
    assert(!nativeMtpPrefillChunkPdSupported(true, false, false));
    assert(nativeMtpPrefillChunkPdSupported(false, false, false));
    const auto rounds = planMtpPrefillChunks({13, 37, 2}, {7, 9, 101}, 16, 8);
    std::vector<int64_t> seen(52), terminal(3);
    for (const auto& round : rounds) {
        int64_t packed = 0;
        for (const auto& slice : round) {
            assert(slice.length > 0);
            assert(slice.prefix == std::vector<int64_t>({7, 9, 101})[slice.request] + slice.request_offset);
            for (int64_t i = 0; i < slice.length; ++i) ++seen[slice.source_offset + i];
            if (slice.terminal) ++terminal[slice.request];
            else assert((slice.request_offset + slice.length) % 8 == 0);
            packed += slice.length;
        }
        assert(packed <= 16);
    }
    for (auto count : seen) assert(count == 1);
    for (auto count : terminal) assert(count == 1);
    assert(planMtpPrefillChunks({65536}, {0}, 8192, 8).size() == 8);
    assert(planMtpPrefillChunks(std::vector<int64_t>(64, 1024), std::vector<int64_t>(64), 8192, 8).size() == 8);
    assert(planMtpPrefillChunks({8, 8}, {3, 11}, 16, 8).size() == 1);
    bool rejected = false;
    try { planMtpPrefillChunks({8}, {0}, 7, 8); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
