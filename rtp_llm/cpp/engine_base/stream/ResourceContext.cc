#include "rtp_llm/cpp/engine_base/stream/ResourceContext.h"

#include "autil/EnvUtil.h"

namespace rtp_llm {

namespace {

constexpr char kIgnoreRequestCacheSwitchesEnv[] = "RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES";

}  // namespace

void ResourceContext::initCacheConfig(const KVCacheConfig& kv_cache_config) {
    reuse_cache                           = kv_cache_config.reuse_cache;
    enable_prefill_partial_tail_reuse     = kv_cache_config.enable_prefill_partial_tail_reuse;
    enable_prefill_common_prefix_recovery = kv_cache_config.enable_prefill_common_prefix_recovery;
    prefill_segment_before =
        kv_cache_config.prefill_segment_before.empty() ?
            nullptr :
            std::make_shared<const std::vector<std::vector<int>>>(kv_cache_config.prefill_segment_before);
    prefill_segment_after =
        kv_cache_config.prefill_segment_after.empty() ?
            nullptr :
            std::make_shared<const std::vector<std::vector<int>>>(kv_cache_config.prefill_segment_after);
    enable_device_cache                   = kv_cache_config.enable_device_cache;
    enable_memory_cache               = kv_cache_config.enable_memory_cache;
    enable_disk_cache                 = kv_cache_config.enable_disk_cache;
    enable_remote_cache               = kv_cache_config.enable_remote_cache;
    ignore_request_cache_switches     = autil::EnvUtil::getEnv(kIgnoreRequestCacheSwitchesEnv, false);
}

}  // namespace rtp_llm
