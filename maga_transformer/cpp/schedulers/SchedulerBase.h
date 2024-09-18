#pragma once

#include <list>
#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/stream/StreamGroups.h"

namespace rtp_llm {

class SchedulerBase {
public:
    virtual ~SchedulerBase() {}
    virtual absl::Status                                 enqueue(const GenerateStreamPtr& stream) = 0;
    virtual absl::StatusOr<std::list<GenerateStreamPtr>> schedule(size_t reserve_step = 0)           = 0;
    virtual absl::Status                                 stop()                                   = 0;
    virtual bool                                         empty()                                  = 0;
};

}  // namespace rtp_llm
