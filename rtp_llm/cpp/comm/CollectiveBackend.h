#pragma once

#include <pybind11/pybind11.h>

#include "rtp_llm/models_py/bindings/core/OpData.h"

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
AllReduceOutput execAllReduce(const AllReduceParams& params);
void            execAllGather(const AllGatherParams& params);
void            execSyncCommunication(bool timeout = true);
void            execSyncCommunication(ParallelMode mode, bool timeout = true);

// Pybind registration: exposes `register_comm_ops` / `clear_comm_ops`.
void registerCommPybindings(pybind11::module& m);

}  // namespace rtp_llm
