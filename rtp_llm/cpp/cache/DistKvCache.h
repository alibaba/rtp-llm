#pragma once

#include "rtp_llm/cpp/cache/DistStorage.h"
#include "rtp_llm/cpp/cache/DistKvCachePlanner.h"
#include "rtp_llm/cpp/cache/DistStorage.h"
#include "rtp_llm/cpp/cache/DistStorageManager.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"

namespace rtp_llm {

class CacheManager;

struct DistKvCacheInitParams {
    DistStorageManagerInitParams storage_manager_params;
};

/**
 * @brief Distributed KV cache manage interface.
 * wrap dist kvcache impl for rank and all kinds of storage.
 */
class DistKvCache {
public:
    DistKvCache(CacheManager*                       cache_manager,
                const GptInitParameter&             gpt_init_params,
                const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);
    ~DistKvCache();

    bool init(const DistKvCacheInitParams& init_params);

public:
    int32_t matchForAllRank(const std::vector<int64_t>&        cache_keys,
                            int64_t                            request_id,
                            std::map<std::string, std::string> extra_metas) const;

    int32_t match(const std::vector<int64_t>&        cache_keys,
                  int64_t                            request_id,
                  std::map<std::string, std::string> extra_metas) const;

    bool getForAllRank(const std::vector<int64_t>&        cache_keys,
                       const std::vector<int32_t>&        block_indices,
                       int64_t                            request_id,
                       std::map<std::string, std::string> extra_metas);

    bool get(const std::vector<int64_t>&        cache_keys,
             const std::vector<int32_t>&        block_indices,
             int64_t                            request_id,
             std::map<std::string, std::string> extra_metas) const;

    bool putForAllRank(const std::vector<int64_t>&        cache_keys,
                       const std::vector<int32_t>&        block_indices,
                       int64_t                            request_id,
                       std::map<std::string, std::string> extra_metas) const;

    bool put(const std::vector<int64_t>&        cache_keys,
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
    CacheManager*                cache_manager_{nullptr};
    const GptInitParameter       gpt_init_params_;
    kmonitor::MetricsReporterPtr metrics_reporter_;

    std::map<std::string, std::string> default_metas_;

    DistKvCacheInitParams               init_params_;
    std::unique_ptr<DistKvCachePlanner> planner_;
    std::unique_ptr<DistStorageManager> storage_;
    std::shared_ptr<RPCPool>            rpc_pool_;

    std::atomic<int64_t> total_matched_len_{0};
    std::atomic<int64_t> total_input_len_{0};
};

}  // namespace rtp_llm