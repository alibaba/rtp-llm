#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/DistStorage.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"

namespace rtp_llm {

class DistKvCachePlanner {
public:
    virtual std::vector<DistStorage::Item> layout(const std::vector<size_t>&                cache_keys,
                                                  const std::vector<int32_t>&               block_indices,
                                                  size_t                                    ignore_block_num,
                                                  const std::map<std::string, std::string>& metas) = 0;

    virtual bool verify(const std::vector<DistStorage::Item>&     buffers,
                        const std::vector<size_t>&                cache_keys,
                        const std::vector<int32_t>&               block_indices,
                        const std::map<std::string, std::string>& metas) = 0;
};

class CacheManager;

// use 3fs prompt wise
class DefaultDistKvCachePlanner: public DistKvCachePlanner {
public:
    DefaultDistKvCachePlanner(CacheManager*                       cache_manager,
                              const GptInitParameter&             gpt_init_params,
                              const DistStorage3FSInitParams&     init_params_3fs,
                              const kmonitor::MetricsReporterPtr& metrics_reporter);

public:
    std::vector<DistStorage::Item> layout(const std::vector<size_t>&                cache_keys,
                                          const std::vector<int32_t>&               block_indices,
                                          size_t                                    ignore_block_num,
                                          const std::map<std::string, std::string>& metas) override;

    bool verify(const std::vector<DistStorage::Item>&     buffers,
                const std::vector<size_t>&                cache_keys,
                const std::vector<int32_t>&               block_indices,
                const std::map<std::string, std::string>& metas) override;

private:
    std::optional<std::string> generateKvCacheKey(const std::map<std::string, std::string>& metas) const;

private:
    CacheManager*                  cache_manager_ = nullptr;
    const GptInitParameter         gpt_init_params_;
    const DistStorage3FSInitParams init_params_3fs_;
    kmonitor::MetricsReporterPtr   metrics_reporter_;
};

}  // namespace rtp_llm