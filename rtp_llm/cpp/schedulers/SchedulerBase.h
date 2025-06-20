#pragma once

#include <list>
#include <vector>
#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/dataclass/MergedQuery.h"
#include "rtp_llm/cpp/dataclass/Query.h"
#include "rtp_llm/cpp/stream/StreamGroups.h"

namespace rtp_llm {

class SchedulerBase {
public:
    virtual ~SchedulerBase() {}
    virtual absl::Status                                 enqueue(const GenerateStreamPtr& stream) = 0;
    virtual absl::Status                                 batchEnqueue(const std::vector<GenerateStreamPtr>& streams) = 0;
    virtual absl::StatusOr<std::list<GenerateStreamPtr>> schedule(size_t reserve_step = 0)        = 0;
    virtual absl::Status                                 stop()                                   = 0;
    virtual bool                                         empty()                                  = 0;
    virtual int64_t                                      lastScheduleTime()                       = 0;
    virtual int64_t                                      onflightStreams()                        = 0;
    virtual bool                                         canLoadBalance() { return true; }
    virtual void                                         updateSchedulerInfo(const std::string& scheduler_info) {}
};

}  // namespace rtp_llm
