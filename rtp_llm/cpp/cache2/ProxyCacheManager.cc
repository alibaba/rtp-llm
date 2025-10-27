#include "ProxyCacheManager.h"
#include "ProxyDistKvCache.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

ProxyCacheManager::ProxyCacheManager(const CacheConfig&                 config,
                                     DeviceBase*                        device,
                                     bool                               warmup,
                                     const kmonitor::MetricsReporterPtr metrics_reporter,
                                     const GptInitParameter&            params):
    CacheManager(config, device, warmup, metrics_reporter, params) {
    if (!params_.kv_cache_config.enable_dist_kvcache) {
        RTP_LLM_FAIL("not set enable_dist_kvcache");
    }
    enable_dist_kvcache_ = true;
}

bool ProxyCacheManager::initDistKvCache() {
    // TODO : use these legacy parameters?
    DistKvCacheInitParams init_params;
    init_params.match_timeout_ms         = params_.kv_cache_config.match_timeout_ms;
    init_params.rpc_get_cache_timeout_ms = params_.kv_cache_config.rpc_get_cache_timeout_ms;
    init_params.rpc_put_cache_timeout_ms = params_.kv_cache_config.rpc_put_cache_timeout_ms;
    init_params.max_block_size_per_item  = params_.kv_cache_config.max_block_size_per_item;

    auto dist_kvcache = std::make_shared<ProxyDistKvCache>(this, params_, metrics_reporter_);
    if (!dist_kvcache->init(init_params)) {
        RTP_LLM_LOG_WARNING("proxy dist kvcache init failed!!!");
        return false;
    }

    dist_kvcache_  = std::move(dist_kvcache);
    lora_info_map_ = getLoraInfo();
    return true;
}

std::map<std::string, std::string> ProxyCacheManager::genExtraMeta(const std::string& adapter_name) const {
    std::map<std::string, std::string> extra_metas;
    extra_metas["LORA_ADAPTER_NAME"] = adapter_name;
    return extra_metas;
}

bool ProxyCacheManager::dynamicEnableDist(bool manualFlag) const {
    return true;
}

}  // namespace rtp_llm