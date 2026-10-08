#pragma once

#include <cstdint>
#include <map>
#include <memory>

#include "rtp_llm/cpp/cache/KVCacheAllocationLease.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

// Scheduler-owned dispatch cursor, distinct from the worker's committed cursor.
// Prepare on a copy and publish that copy only after the entire gather succeeds.
// All access must serialize with stream allocation, mutation and release.
class V41CacheDispatch {
public:
    void prepare(const StreamGroups& streams, GptModelInputs& inputs);

private:
    struct Request {
        std::weak_ptr<GenerateStream>       stream;
        int64_t                             incarnation;
        int64_t                             generation;
        int64_t                             next_chunk;
        int64_t                             next_position;
        KVCacheAllocationLease::BlockTables tables;
    };
    std::map<int64_t, Request> requests_;
    int64_t                    generation_ = 0;
};

}  // namespace rtp_llm
