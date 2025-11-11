#pragma once

#include <cstdint>
#include <mutex>
#include <queue>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"

namespace rtp_llm {

struct WorkerStatusInfo {
    std::string        role;
    EngineScheduleInfo engine_schedule_info;
    int64_t            status_version;
    bool               alive;
    int32_t            dp_size;
    int32_t            tp_size;
    int32_t            dp_rank;
    std::string        precision;
};

}  // namespace rtp_llm
