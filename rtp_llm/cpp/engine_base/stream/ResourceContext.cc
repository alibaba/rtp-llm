#include "rtp_llm/cpp/engine_base/stream/ResourceContext.h"

namespace rtp_llm {

void ResourceContext::initCacheConfig(const KVCacheConfig& kv_cache_config) {
    reuse_cache         = kv_cache_config.reuse_cache;
    enable_device_cache = kv_cache_config.enable_device_cache;
    enable_host_cache   = kv_cache_config.enable_host_cache;
    enable_disk_cache   = kv_cache_config.enable_disk_cache;
    enable_remote_cache = kv_cache_config.enable_remote_cache;
}

}  // namespace rtp_llm
