#pragma once

#include "rtp_llm/cpp/cache/DistStorage.h"
#include "rtp_llm/cpp/cache/DistKvCachePlanner.h"
#include "rtp_llm/cpp/cache/DistStorage.h"
#include "rtp_llm/cpp/cache/DistStorageManager.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"

namespace rtp_llm {

class CacheManager;

struct DistKvCacheInitParams {
    DistStorageManagerInitParams manager_params;
};

/**
 * @brief Distributed KV cache manage interface.
 * wrap dist kvcache impl for rank and all kinds of storage.
 */
class DistKvCache {
public:
    DistKvCache(CacheManager*                       cache_manager,
                const GptInitParameter&             params,
                const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);
    ~DistKvCache();

    bool init(const DistKvCacheInitParams& init_params);

public:
    int32_t matchCacheForAllRank(const std::vector<int64_t>&        cache_keys,
                                 const std::vector<int32_t>&        block_indices,
                                 int64_t                            request_id,
                                 std::map<std::string, std::string> extra_metas) const;

    int32_t matchCache(const std::vector<int64_t>&        cache_keys,
                       const std::vector<int32_t>&        block_indices,
                       int64_t                            request_id,
                       std::map<std::string, std::string> extra_metas,
                       int                                tp_rank) const;

    bool getCacheForAllRank(const std::vector<int64_t>&        cache_keys,
                            const std::vector<int32_t>&        block_indices,
                            int64_t                            request_id,
                            std::map<std::string, std::string> extra_metas) const;

    bool getCache(const std::vector<int64_t>&        cache_keys,
                  const std::vector<int32_t>&        block_indices,
                  int64_t                            request_id,
                  std::map<std::string, std::string> extra_metas) const;

    bool putForAllRank(const std::vector<int64_t>&        cache_keys,
                       const std::vector<int32_t>&        block_indices,
                       int64_t                            request_id,
                       std::map<std::string, std::string> extra_metas) const;

    bool putCache(const std::vector<int64_t>&        cache_keys,
                  const std::vector<int32_t>&        block_indices,
                  int64_t                            request_id,
                  std::map<std::string, std::string> extra_metas) const;

private:
    enum OpType {
        OP_GET = 0,
        OP_PUT = 1
    };
    bool syncCallAllRank(const std::vector<int64_t>&              cache_keys,
                         const std::vector<int32_t>&              block_indices,
                         int64_t                                  request_id,
                         const std::map<std::string, std::string> extra_metas,
                         DistKvCache::OpType                      op_type) const;

private:
    CacheManager*                cache_manager_ = 0;
    const GptInitParameter       params_;
    kmonitor::MetricsReporterPtr metrics_reporter_;

    std::map<std::string, std::string> default_metas_;

    DistKvCacheInitParams               init_params_;
    std::unique_ptr<DistKvCachePlanner> planner_;
    std::unique_ptr<DistStorageManager> storage_;
    std::shared_ptr<RPCPool>            rpc_pool_;
};

}  // namespace rtp_llm