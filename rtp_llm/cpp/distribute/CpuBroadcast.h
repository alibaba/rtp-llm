#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

// CPU metadata broadcast for TP sync paths. Uses the UDS CpuBroadcaster when it
// is initialized; otherwise allow_fallback controls whether regular c10d
// execBroadcast fallback is used. All ranks must call with identical tensor
// counts and byte sizes.
void execBroadcastCpu(const BroadcastParams& params, bool allow_fallback = true);

}  // namespace rtp_llm
