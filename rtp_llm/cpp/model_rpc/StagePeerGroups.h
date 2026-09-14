#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/connector/p2p/ConnectorRouting.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

// One PP stage's main-model layer range plus the peer subset serving it,
// carried by the prefill -> decode allocate request.
struct StagePeerGroup {
    StageLayerRange          range;
    std::vector<std::string> peer_addrs;
};

// One load unit from a single prefill peer: which peer of the stage group to
// read, how to slice the local destination block and the remote source block.
struct StagePeerSlice {
    size_t peer_index;
    int    dst_partition_count;
    int    dst_partition_id;
    int    src_partition_count;
    int    src_partition_id;
};

// Maps one decode lane onto the prefill stage's TP lanes by head-sliced KV
// ownership: equal widths copy whole-to-whole, a finer prefill TP assembles
// tp_P/tp_D consecutive peer slices, a finer decode TP reads one sub-slice of
// a single peer block. replicated_kv (MLA) keeps one whole-block peer per
// stage since every lane owns the full block.
std::vector<StagePeerSlice> planStagePeerSlices(int prefill_tp, int decode_tp, int decode_tp_rank, bool replicated_kv);

// Splits rank-ordered workers (stage-major, tp_size per stage) into per-stage
// groups; empty when pp_size <= 1 so the flat peer_addrs path is kept.
std::vector<StagePeerGroup> buildStagePeerGroups(const ParallelismConfig&        parallelism_config,
                                                 const std::vector<std::string>& workers);

// Fails loudly unless the groups tile [0, total_layers) contiguously with
// non-empty layer ranges and peers.
void validateStagePeerGroups(const std::vector<StagePeerGroup>& groups, int64_t total_layers);

}  // namespace rtp_llm
