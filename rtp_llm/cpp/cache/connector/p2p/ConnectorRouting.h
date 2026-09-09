#pragma once

#include <cstdint>
#include <vector>

namespace rtp_llm {

// Global layer range [begin, begin + size) owned by one PP stage.
struct StageLayerRange {
    uint32_t begin = 0;
    uint32_t size  = 0;

    uint32_t end() const {
        return begin + size;
    }
};

// One PD transfer channel: a prefill stage sends the global layers
// [layer_begin, layer_begin + layer_size) to a decode stage.
struct P2PRoutingEntry {
    int      prefill_stage = 0;
    int      decode_stage  = 0;
    uint32_t layer_begin   = 0;
    uint32_t layer_size    = 0;

    uint32_t layer_end() const {
        return layer_begin + layer_size;
    }
};

/* PD routing table over arbitrary pp_P x pp_D: each entry is the layer-range
   intersection of one prefill stage and one decode stage; symmetric/converge/
   broadcast shapes are plain intersection cases. Input ranges on each side
   must tile contiguously from layer 0 and both sides must cover the same
   total layer count; violations fail loudly at table-build time. */
class ConnectorRouting {
public:
    static std::vector<P2PRoutingEntry> buildRoutingTable(const std::vector<StageLayerRange>& prefill_ranges,
                                                          const std::vector<StageLayerRange>& decode_ranges);

    static std::vector<P2PRoutingEntry> entriesFromPrefillStage(const std::vector<P2PRoutingEntry>& entries,
                                                                int                                 prefill_stage);

    static std::vector<P2PRoutingEntry> entriesToDecodeStage(const std::vector<P2PRoutingEntry>& entries,
                                                             int                                 decode_stage);
};

}  // namespace rtp_llm
