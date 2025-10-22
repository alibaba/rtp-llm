#include "CacheManagerFactory.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

std::shared_ptr<CacheManager>
CacheManagerFactory::createCacheManager(const CacheConfig&                 config,
                                        rtp_llm::DeviceBase*               device,
                                        bool                               warmup,
                                        const kmonitor::MetricsReporterPtr metrics_reporter,
                                        const GptInitParameter&            params) {
    std::shared_ptr<CacheManager> cache_manager;
    if (params.kv_cache_config.enable_dist_kvcache) {
        cache_manager = std::make_shared<ProxyCacheManager>(config, device, warmup, metrics_reporter, params);
    } else {
        cache_manager = std::make_shared<CacheManager>(config, device, warmup, metrics_reporter, params);
    }

    if (params.kv_cache_config.enable_dist_kvcache || params.kv_cache_config.enable_3fs) {
        if (!cache_manager->initDistKvCache()) {
            RTP_LLM_FAIL("dist kv cache init failed");
        }
    }
    return cache_manager;
}

}  // namespace rtp_llm