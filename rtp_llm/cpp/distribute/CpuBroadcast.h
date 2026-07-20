#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

// CPU metadata broadcast for TP sync paths. Before UDS initialization,
// allow_fallback controls whether regular c10d execBroadcast may be used. Once
// UDS is initialized, a runtime transport failure is terminal for that
// CpuBroadcaster instance: this function fails fast instead of falling back to
// c10d/NCCL, because switching collectives independently could diverge TP
// ranks. Recovery requires a group-wide destroy/re-init or process restart.
// The blocking UDS call conditionally releases the Python GIL when invoked
// from a Python-owned thread; native engine threads do not hold the GIL.
// All ranks must call with identical tensor counts and byte sizes.
void execBroadcastCpu(const BroadcastParams& params, bool allow_fallback = true);

}  // namespace rtp_llm
