#pragma once

#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/RoleTypes.h"

namespace rtp_llm {

class GenerateStream;

struct ResourceContext {
    std::shared_ptr<KVCacheManager> cache_manager;
    std::shared_ptr<SystemPrompt>   system_prompt;

    RoleType role_type{RoleType::PDFUSION};

    bool reuse_cache{false};
    bool enable_memory_cache{false};
    bool enable_remote_cache{false};
    bool enable_device_cache{true};
    bool write_cache_sync{false};
};

}  // namespace rtp_llm
