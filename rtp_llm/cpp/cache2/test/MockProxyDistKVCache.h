#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache2/ProxyDistKvCache.h"

namespace rtp_llm {

class MockProxyDistKVCache: public ProxyDistKvCache {
public:
    MockProxyDistKVCache(CacheManager*                       cache_manager,
                         const GptInitParameter&             gpt_init_params,
                         const kmonitor::MetricsReporterPtr& metrics_reporter):
        ProxyDistKvCache(cache_manager, gpt_init_params, metrics_reporter) {}

    ~MockProxyDistKVCache() override = default;

    using MapStrStr = std::map<std::string, std::string>;

    MOCK_METHOD(bool,
                getForAllRank,
                (const std::vector<int64_t>& cache_keys,
                 const std::vector<int32_t>& block_indices,
                 const LocationsMapPtr&      locations_map_ptr,
                 size_t                      ignore_block_num,
                 int64_t                     request_id,
                 MapStrStr                   extra_metas),
                (const, override));

    MOCK_METHOD(bool,
                putForAllRank,
                (const std::vector<int64_t>& cache_keys,
                 const std::vector<int32_t>& block_indices,
                 size_t                      ignore_block_num,
                 int64_t                     request_id,
                 MapStrStr                   extra_metas),
                (const, override));
};

}  // namespace rtp_llm