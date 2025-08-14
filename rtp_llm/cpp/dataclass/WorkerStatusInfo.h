#pragma once

#include <cstdint>
#include <mutex>
#include <queue>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include "rtp_llm/cpp/dataclass/LoadBalance.h"
#include "rtp_llm/cpp/dataclass/EngineScheduleInfo.h"


namespace rtp_llm {

struct WorkerStatusInfo {
    std::string role;
    LoadBalanceInfo load_balance_info;
    EngineScheduleInfo engine_schedule_info;
    int64_t     status_version;
    bool        alive;
    int32_t     dp_size;
    int32_t     tp_size;
    int32_t     version;
    int32_t     dp_rank;
    std::string precision;
};

void registerWorkerStatusInfo(const pybind11::module& m);

}  // namespace rtp_llm
