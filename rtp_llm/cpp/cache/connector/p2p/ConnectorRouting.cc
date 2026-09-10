#include "rtp_llm/cpp/cache/connector/p2p/ConnectorRouting.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

void checkTiledRanges(const std::vector<StageLayerRange>& ranges, const char* side) {
    RTP_LLM_CHECK_WITH_INFO(!ranges.empty(), "PD routing: %s stage layer ranges are empty", side);
    uint32_t expected_begin = 0;
    for (size_t s = 0; s < ranges.size(); ++s) {
        RTP_LLM_CHECK_WITH_INFO(ranges[s].size > 0, "PD routing: %s stage %zu has an empty layer range", side, s);
        RTP_LLM_CHECK_WITH_INFO(ranges[s].begin == expected_begin,
                                "PD routing: %s stage %zu layer range [%u, %u) does not continue from %u",
                                side,
                                s,
                                ranges[s].begin,
                                ranges[s].end(),
                                expected_begin);
        expected_begin = ranges[s].end();
    }
}

}  // namespace

std::vector<P2PRoutingEntry> ConnectorRouting::buildRoutingTable(const std::vector<StageLayerRange>& prefill_ranges,
                                                                 const std::vector<StageLayerRange>& decode_ranges) {
    checkTiledRanges(prefill_ranges, "prefill");
    checkTiledRanges(decode_ranges, "decode");
    RTP_LLM_CHECK_WITH_INFO(decode_ranges.back().end() == prefill_ranges.back().end(),
                            "PD routing: layer coverage mismatch: decode covers %u layers but prefill has %u",
                            decode_ranges.back().end(),
                            prefill_ranges.back().end());

    // Both sides are contiguous tilings, so every (p, d) intersection is a
    // single interval; a two-pointer sweep enumerates them in layer order.
    std::vector<P2PRoutingEntry> entries;
    size_t                       p = 0;
    size_t                       d = 0;
    while (p < prefill_ranges.size() && d < decode_ranges.size()) {
        const uint32_t begin = std::max(prefill_ranges[p].begin, decode_ranges[d].begin);
        const uint32_t end   = std::min(prefill_ranges[p].end(), decode_ranges[d].end());
        if (begin < end) {
            entries.push_back({static_cast<int>(p), static_cast<int>(d), begin, end - begin});
        }
        if (prefill_ranges[p].end() < decode_ranges[d].end()) {
            ++p;
        } else {
            ++d;
        }
    }
    return entries;
}

std::vector<P2PRoutingEntry> ConnectorRouting::entriesFromPrefillStage(const std::vector<P2PRoutingEntry>& entries,
                                                                       int prefill_stage) {
    std::vector<P2PRoutingEntry> filtered;
    for (const auto& entry : entries) {
        if (entry.prefill_stage == prefill_stage) {
            filtered.push_back(entry);
        }
    }
    return filtered;
}

std::vector<P2PRoutingEntry> ConnectorRouting::entriesToDecodeStage(const std::vector<P2PRoutingEntry>& entries,
                                                                    int                                 decode_stage) {
    std::vector<P2PRoutingEntry> filtered;
    for (const auto& entry : entries) {
        if (entry.decode_stage == decode_stage) {
            filtered.push_back(entry);
        }
    }
    return filtered;
}

}  // namespace rtp_llm
