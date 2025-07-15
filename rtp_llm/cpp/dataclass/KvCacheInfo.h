#pragma once

#include <iostream>
#include <vector>
#include "rtp_llm/cpp/utils/PyUtils.h"
namespace rtp_llm {
struct KVCacheInfo {
    size_t              available_kv_cache = 0;
    size_t              total_kv_cache     = 0;
    size_t              block_size         = 0;
    std::vector<size_t> cached_keys;
    int64_t             version = -1;
};
void registerKvCacheInfo(const pybind11::module& m);

}  // namespace rtp_llm
