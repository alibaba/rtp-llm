#pragma once

#include "rtp_llm/cpp/comm/CollectiveTypes.h"

namespace rtp_llm {

// ===================================================================
// Collective communication ops
//
// Currently dispatched to Python callbacks registered via
// `register_comm_ops` (see registerCommPybindings). The callbacks
// live as static-storage globals; this file MUST link into exactly
// one .so (librtp_compute_ops.so) so that all callers share the same
// callback table.
// ===================================================================

void            execBroadcast(const BroadcastParams& params);
void            execBroadcastCpu(const BroadcastParams& params);
bool            isCpuTpBroadcasterInitialized();
AllReduceOutput execAllReduce(const AllReduceParams& params);
void            execAllGather(const AllGatherParams& params);
void            execSyncCommunication(bool timeout = true);
void            execSyncCommunication(ParallelMode mode, bool timeout = true);

}  // namespace rtp_llm
