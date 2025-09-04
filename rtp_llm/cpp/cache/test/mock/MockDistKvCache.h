#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <map>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/DistKvCache.h"

namespace rtp_llm {

using MapStrStr = std::map<std::string, std::string>;

class MockDistKvCache: public DistKvCache {
public:
    MockDistKvCache(): DistKvCache(nullptr, GptInitParameter{}, nullptr) {}
    MockDistKvCache(CacheManager*                       cache_manager,
                    const GptInitParameter&             gpt_init_params,
                    const kmonitor::MetricsReporterPtr& metrics_reporter):
        DistKvCache(cache_manager, gpt_init_params, metrics_reporter) {}
    ~MockDistKvCache() override = default;

public:
    MOCK_METHOD(bool, init, (const DistKvCacheInitParams& init_params), (override));

    MOCK_METHOD(
        int32_t,
        matchForAllRank,
        (const std::vector<int64_t>& cache_keys, size_t ignore_block_num, int64_t request_id, MapStrStr extra_metas),
        (override));

    MOCK_METHOD(int32_t,
                match,
                (const std::vector<int64_t>&               cache_keys,
                 size_t                                    ignore_block_num,
                 int64_t                                   request_id,
                 MapStrStr                                 extra_metas,
                 const std::shared_ptr<std::atomic<bool>>& stop),
                (const, override));

    MOCK_METHOD(bool,
                getForAllRank,
                (const std::vector<int64_t>& cache_keys,
                 const std::vector<int32_t>& block_indices,
                 size_t                      ignore_block_num,
                 int64_t                     request_id,
                 MapStrStr                   extra_metas),
                (const, override));

    MOCK_METHOD(bool,
                get,
                (const std::vector<int64_t>& cache_keys,
                 const std::vector<int32_t>& block_indices,
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

    MOCK_METHOD(bool,
                put,
                (const std::vector<int64_t>& cache_keys,
                 const std::vector<int32_t>& block_indices,
                 size_t                      ignore_block_num,
                 int64_t                     request_id,
                 MapStrStr                   extra_metas),
                (const, override));
};

}  // namespace rtp_llm