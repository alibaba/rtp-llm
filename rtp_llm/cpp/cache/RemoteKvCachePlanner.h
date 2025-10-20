#include "rtp_llm/cpp/cache/DistKvCachePlanner.h"

namespace rtp_llm {

class CacheManager;

class RemoteKvCachePlanner: public DistKvCachePlanner {
public:
    RemoteKvCachePlanner(CacheManager*                       cache_manager,
                         const GptInitParameter&             gpt_init_params,
                         const kmonitor::MetricsReporterPtr& metrics_reporter);

public:
    std::vector<DistStorage::Item> layout(const std::vector<int64_t>&               cache_keys,
                                          const std::vector<int32_t>&               block_indices,
                                          const kv_cache_manager::BlockMask&        block_mask,
                                          const std::map<std::string, std::string>& metas) override;

    bool verify(const std::vector<DistStorage::Item>&     buffers,
                const std::vector<int64_t>&               cache_keys,
                const std::vector<int32_t>&               block_indices,
                const std::map<std::string, std::string>& metas) override;

private:
    CacheManager*                cache_manager_ = nullptr;
    const GptInitParameter       gpt_init_params_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
};
}  // namespace rtp_llm
