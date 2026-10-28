#include "rtp_llm/cpp/engine_base/stream/ResourceContext.h"

#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

namespace rtp_llm {

namespace {

constexpr char kIgnoreRequestCacheSwitchesEnv[] = "RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES";

}  // namespace

void ResourceContext::initCacheConfig(const KVCacheConfig& kv_cache_config) {
    reuse_cache                   = kv_cache_config.reuse_cache;
    enable_device_cache           = kv_cache_config.enable_device_cache;
    enable_memory_cache           = kv_cache_config.enable_memory_cache;
    enable_disk_cache             = kv_cache_config.enable_disk_cache;
    enable_remote_cache           = kv_cache_config.enable_remote_cache;
    ignore_request_cache_switches = autil::EnvUtil::getEnv(kIgnoreRequestCacheSwitchesEnv, false);
    enable_decode_publication = remoteCacheDecodePublicationEnabled()
                                && !autil::EnvUtil::getEnv("RTP_LLM_STREAM_ASYNC", false);
}

}  // namespace rtp_llm
