#pragma once

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include <string>
#include <vector>

namespace rtp_llm {

// All state transitions are scheduler-owned. Execution holds protect every
// child through process()/async output. A captured lease independently retains
// the checkpoint while the scheduler accepts or discards the producer result.
struct SharedPrefixRequest {
    enum class Phase { QUEUED, PRODUCING, BRANCHING, DRAINING };
    std::string parent_id;
    std::vector<GenerateStreamPtr> streams;
    int boundary_tokens = 0;
    Phase phase = Phase::QUEUED;
    PrefixCheckpointLeasePtr checkpoint;
    size_t next_branch = 0;
    bool producer_guard = false;

    void fail(const std::string& reason) {
        for (const auto& stream : streams) {
            if (!stream->isFinished() && !stream->hasError()) {
                stream->reportError(ErrorCode::MALLOC_FAILED, reason);
            }
        }
        phase = Phase::DRAINING;
    }
};

}  // namespace rtp_llm
