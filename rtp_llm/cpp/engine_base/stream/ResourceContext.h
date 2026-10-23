#pragma once

#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class GenerateStream;

struct ResourceContext {
    std::shared_ptr<KVCacheManager> cache_manager;
    std::shared_ptr<SystemPrompt>   system_prompt;

    RoleType role_type{RoleType::PDFUSION};
    bool     decode_entrance{false};  // PD反转模式：Decode侧作为请求入口

    bool reuse_cache{false};
    bool enable_prefill_partial_tail_reuse{false};
    bool enable_prefill_common_prefix_recovery{false};
    bool enable_device_cache{true};
    bool enable_memory_cache{false};
    bool enable_disk_cache{false};
    bool enable_remote_cache{false};
    bool ignore_request_cache_switches{false};

    void initCacheConfig(const KVCacheConfig& kv_cache_config);
};

// Temporarily bind the warmup cache. Destruction restores the production
// manager even when preRun throws.
class ScopedWarmUpCacheManagerBinding {
public:
    ScopedWarmUpCacheManagerBinding(ResourceContext& resource_context, std::shared_ptr<KVCacheManager> manager):
        resource_context_(resource_context), previous_manager_(resource_context.cache_manager) {
        resource_context_.cache_manager = std::move(manager);
    }

    ~ScopedWarmUpCacheManagerBinding() {
        resource_context_.cache_manager = std::move(previous_manager_);
    }

    ScopedWarmUpCacheManagerBinding(const ScopedWarmUpCacheManagerBinding&) = delete;
    ScopedWarmUpCacheManagerBinding& operator=(const ScopedWarmUpCacheManagerBinding&) = delete;

private:
    ResourceContext& resource_context_;
    std::shared_ptr<KVCacheManager> previous_manager_;
};

}  // namespace rtp_llm
