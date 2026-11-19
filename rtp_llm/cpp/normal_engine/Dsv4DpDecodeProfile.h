#pragma once

#include "rtp_llm/cpp/normal_engine/DecodeReplicaCoordinator.h"

namespace rtp_llm {

std::unique_ptr<DecodeReplicaCoordinator>
createDsv4DpDecodeCoordinator(const EngineInitParams& params, std::string_view resolved_moe_strategy, bool has_draft);

}  // namespace rtp_llm
