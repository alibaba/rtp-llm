#pragma once

#include <cstdint>
#include <string>

#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm::sleep_memory {

bool synchronizeSleepDevice(const char* stage);
bool trimSleepAllocator(int64_t local_rank, int64_t epoch, int64_t world_rank, bool graph_enabled, const char* phase);
void logSleepMemorySnapshotForRank(const std::string& phase, const ParallelismConfig& parallelism, int64_t epoch);

}  // namespace rtp_llm::sleep_memory
