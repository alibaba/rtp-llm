#pragma once

#include "rtp_llm/cpp/cache/DistKvCache.h"
#include "rtp_llm/cpp/cache2/kvcm_client_wrapper/KVCMClientWrapper.h"

namespace rtp_llm {

class ProxyDistKvCache: public DistKvCache {
public:
    ProxyDistKvCache(CacheManager*                       cache_manager,
                     const GptInitParameter&             gpt_init_params,
                     const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);
    ~ProxyDistKvCache() override = default;

    bool init(const DistKvCacheInitParams& init_params) override;

    bool get(const std::vector<int64_t>&        cache_keys,
             const std::vector<int32_t>&        block_indices,
             const kv_cache_manager::Locations& locations,
             const kv_cache_manager::BlockMask& block_mask,
             int64_t                            request_id,
             std::map<std::string, std::string> extra_metas) const override;

    bool put(const std::vector<int64_t>&        cache_keys,
             const std::vector<int32_t>&        block_indices,
             const kv_cache_manager::Locations& locations,
             const kv_cache_manager::BlockMask& block_mask,
             int64_t                            request_id,
             std::map<std::string, std::string> extra_metas) const override;

private:
    int32_t matchAllRankImpl(const std::vector<int64_t>&               cache_keys,
                             LocationsMapPtr                           locations_map_ptr,
                             size_t                                    ignore_block_num,
                             int64_t                                   request_id,
                             const std::map<std::string, std::string>& extra_metas,
                             const std::shared_ptr<std::atomic<bool>>& stop) const override;

    bool putAllRankImpl(const std::vector<int64_t>&               cache_keys,
                        const std::vector<int32_t>&               block_indices,
                        size_t                                    ignore_block_num,
                        int64_t                                   request_id,
                        const std::map<std::string, std::string>& extra_metas) const override;

    bool fillDistKvCacheRequestPB(DistKvCacheRequestPB&                     request,
                                  const std::vector<int64_t>&               cache_keys,
                                  const std::vector<int32_t>&               block_indices,
                                  const kv_cache_manager::LocationsMap&     locations_map,
                                  const kv_cache_manager::BlockMask&        block_mask,
                                  int64_t                                   request_id,
                                  const std::map<std::string, std::string>& extra_metas,
                                  DistKvCache::OpType                       op_type,
                                  int                                       rank) const override;

    bool                               initKVCMClientWrapper();
    std::map<std::string, std::string> genKVCMClientConfig() const;
    std::string                        genUniqueId(const std::map<std::string, std::string>& extra_metas) const;

private:
    std::shared_ptr<KVCMClientWrapper> kvcm_client_wrapper_;
};

}  // namespace rtp_llm