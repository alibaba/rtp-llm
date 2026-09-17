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
    bool                     is_last_stage = false;
};

// One load unit from a single prefill peer: which peer of the stage group to
// read, how to slice the local destination block and the remote source block.
struct StagePeerSlice {
    // Index into the group's peer_addrs: the peer this read pulls from; also
    // the peer_idx passed to the shouldLoad*FromPeer checks and
    // sliceCpDestinationForPeer.
    size_t peer_index;
    // Local side of the cut: with MHA and a finer prefill TP (tp_p > tp_d),
    // one local block is cut into local_piece_count = tp_p / tp_d pieces, one
    // per contributing peer, and this read fills local_piece_id.
    int local_piece_count;
    int local_piece_id;
    // Peer side of the cut: the serving peer cuts its block into
    // peer_piece_count pieces and sends only peer_piece_id.
    int peer_piece_count;
    int peer_piece_id;
};

// Routing inputs for one stage group: everything the per-group A/B/C decision
// reads, so dispatch can be planned without touching cache internals.
struct StageGroupLoadParams {
    int  group_peer_count;
    int  prefill_cp_size;
    bool prefill_cp_enabled;
    int  decode_tp_size;
    int  decode_tp_rank;
    bool replicated_kv;
    bool opaque_kv_store;
};

// One stage group's load plan. loads reuses StagePeerSlice; page_level_rr is the
// group-level routing mode shared by every load; a non-empty error maps to
// LOAD_KV_CACHE_FAILED on the caller side.
struct StageGroupLoadPlan {
    std::vector<StagePeerSlice> loads;
    bool                        page_level_rr = false;
    std::string                 error;
};

// Maps one decode lane onto the prefill stage's TP lanes by head-sliced KV
// ownership: equal widths copy whole-to-whole, a finer prefill TP assembles
// tp_P/tp_D consecutive peer slices, a finer decode TP reads one sub-slice of
// a single peer block. replicated_kv (MLA) keeps one whole-block peer per
// stage since every lane owns the full block.
std::vector<StagePeerSlice> planStagePeerSlices(int prefill_tp, int decode_tp, int decode_tp_rank, bool replicated_kv);

// Plans the per-group loads for one stage group: CP-sharded whole-block reads
// (MLA/opaque only), CP full-replication lane reads, or planStagePeerSlices.
StageGroupLoadPlan planStageGroupLoads(const StageGroupLoadParams& params);

// Splits rank-ordered workers (stage-major, tp_size per stage) into per-stage
// groups; empty when pp_size <= 1 so the flat peer_addrs path is kept.
std::vector<StagePeerGroup> buildStagePeerGroups(const ParallelismConfig&        parallelism_config,
                                                 const std::vector<std::string>& workers);

// Fails loudly unless the groups tile [0, total_layers) contiguously with
// non-empty layer ranges and peers.
void validateStagePeerGroups(const std::vector<StagePeerGroup>& groups, int64_t total_layers);

}  // namespace rtp_llm
