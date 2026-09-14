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

// Splits rank-ordered workers (stage-major, tp_size per stage) into per-stage
// groups; empty when pp_size <= 1 so the flat peer_addrs path is kept.
std::vector<StagePeerGroup> buildStagePeerGroups(const ParallelismConfig&        parallelism_config,
                                                 const std::vector<std::string>& workers);

// Fails loudly unless the groups tile [0, total_layers) contiguously with
// non-empty layer ranges and peers.
void validateStagePeerGroups(const std::vector<StagePeerGroup>& groups, int64_t total_layers);

}  // namespace rtp_llm
