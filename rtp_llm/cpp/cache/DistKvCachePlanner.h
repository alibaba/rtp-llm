#pragma once

#include "rtp_llm/cpp/cache/DistStorage.h"
#include "rtp_llm/cpp/th_op/GptInitParameter.h"

namespace rtp_llm {

class DistKvCachePlanner {
public:
    virtual std::vector<DistStorage::Item> layout(const std::vector<int64_t>&               cache_keys,
                                                  const std::vector<int32_t>&               block_indices,
                                                  const std::map<std::string, std::string>& metas) = 0;

    virtual bool verify(const std::vector<DistStorage::Item>&     buffers,
                        const std::vector<int64_t>&               cache_keys,
                        const std::vector<int32_t>&               block_indices,
                        const std::map<std::string, std::string>& metas,
                        int32_t                                   tp_rank) = 0;
};

class CacheManager;

// use 3fs prompt wise
class DefaultDistKvCachePlanner: public DistKvCachePlanner {
public:
    DefaultDistKvCachePlanner(CacheManager*                       cache_manager,
                              const GptInitParameter&             gpt_init_params,
                              const DistStorage3FSInitParams&     storage_3fs_init_params,
                              const kmonitor::MetricsReporterPtr& metrics_reporter);

public:
    std::vector<DistStorage::Item> layout(const std::vector<int64_t>&               cache_keys,
                                          const std::vector<int32_t>&               block_indices,
                                          const std::map<std::string, std::string>& metas) override;

    bool verify(const std::vector<DistStorage::Item>&     buffers,
                const std::vector<int64_t>&               cache_keys,
                const std::vector<int32_t>&               block_indices,
                const std::map<std::string, std::string>& metas,
                int32_t                                   tp_rank) override;

private:
    std::string constructKvCacheKey(int64_t last_cache_key, int32_t rank = -1) const;
    bool        makeMetaIov(DistStorage::Iov&           meta_iov,
                            const std::vector<int64_t>& cache_keys,
                            const std::vector<int32_t>& block_indices,
                            int32_t                     tp_rank);

private:
    CacheManager*                  cache_manager_ = nullptr;
    const GptInitParameter         gpt_init_params_;
    const DistStorage3FSInitParams storage_3fs_init_params_;
    kmonitor::MetricsReporterPtr   metrics_reporter_;
};

}  // namespace rtp_llm