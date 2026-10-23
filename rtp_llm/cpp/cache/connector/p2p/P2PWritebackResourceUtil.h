#pragma once

#include "rtp_llm/cpp/cache/SingleTypeCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include <vector>

namespace rtp_llm {

bool writebackKeepsKvReserve(const std::vector<SingleTypeCacheManagerPtr>& groups, double reserve_ratio);

bool writebackKeepsKvReserveAfterRelease(const std::vector<SingleTypeCacheManagerPtr>& groups,
                                         const KVCacheResource&              source,
                                         const KVCacheResource&              retained,
                                         double                              reserve_ratio);

}  // namespace rtp_llm
