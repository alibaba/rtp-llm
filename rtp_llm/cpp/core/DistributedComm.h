#pragma once

#include "rtp_llm/cpp/core/OpData.h"
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <unordered_map>
#include <mutex>

namespace rtp_llm {

// Same pattern as RuntimeState in ExecOps.h:
// globalPgState() is defined in DistributedComm.cc (NOT inline) so
// that the function-local static lives in exactly one translation unit.
// The containing .so (librtp_compute_ops.so) owns the single instance;
// libth_transformer.so resolves the symbol at load-time.
namespace detail {

struct ParallelModeHash {
    std::size_t operator()(ParallelMode m) const {
        return std::hash<int>()(static_cast<int>(m));
    }
};

struct ProcessGroupEntry {
    c10::intrusive_ptr<c10d::ProcessGroup> pg;
    int                                    rank       = 0;
    int                                    world_size = 1;
    int                                    device_id  = 0;
};

struct PgMapState {
    std::mutex                                                            mutex;
    std::unordered_map<ParallelMode, ProcessGroupEntry, ParallelModeHash> map;
};

PgMapState& globalPgState();  // defined in DistributedComm.cc

}  // namespace detail

// Registry for torch.distributed process groups, keyed by ParallelMode.
// Groups are created in Python (collective_torch.py) and registered here
// so C++ communication ops can use them without custom NCCL infrastructure.

void registerProcessGroup(ParallelMode mode, c10::intrusive_ptr<c10d::ProcessGroup> pg, int device_id = 0);
c10::intrusive_ptr<c10d::ProcessGroup> getProcessGroup(ParallelMode mode);
bool                                   hasProcessGroup(ParallelMode mode);
void                                   clearProcessGroups();

// High-level communication ops using c10d ProcessGroup.
// These replace the former NCCL-based implementations in CudaOps.cc.

void            c10dBroadcast(const BroadcastParams& params);
AllReduceOutput c10dAllReduce(const AllReduceParams& params);
void            c10dAllGather(const AllGatherParams& params);
void            c10dSyncCommunication(bool timeout = true);
void            c10dSyncCommunication(ParallelMode mode, bool timeout = true);

}  // namespace rtp_llm
