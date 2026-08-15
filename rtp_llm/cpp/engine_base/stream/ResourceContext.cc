#include "rtp_llm/cpp/engine_base/stream/ResourceContext.h"

#include "autil/EnvUtil.h"

namespace rtp_llm {

namespace {

constexpr char kIgnoreRequestCacheSwitchesEnv[] = "RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES";

}  // namespace

void ResourceContext::initCacheConfig(const KVCacheConfig& kv_cache_config) {
    reuse_cache                   = kv_cache_config.reuse_cache;
    enable_device_cache           = kv_cache_config.enable_device_cache;
    enable_host_cache             = kv_cache_config.enable_host_cache;
    enable_disk_cache             = kv_cache_config.enable_disk_cache;
    enable_remote_cache           = kv_cache_config.enable_remote_cache;
    ignore_request_cache_switches = autil::EnvUtil::getEnv(kIgnoreRequestCacheSwitchesEnv, false);
}

}  // namespace rtp_llm
