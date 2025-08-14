#pragma once

#include <cstdint>
#include <mutex>
#include <queue>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include "rtp_llm/cpp/utils/PyUtils.h"
namespace rtp_llm {

struct CacheStatusInfo {
    int64_t available_kv_cache;
    int64_t total_kv_cache;
    int64_t block_size;
    int64_t version;
    std::vector<int64_t> cached_keys;
};


void registerCacheStatusInfo(const pybind11::module& m);
}  // namespace rtp_llm
