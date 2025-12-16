#pragma once

#include <memory>
#include <vector>

namespace rtp_llm {

class GenerateStream;
class KVCacheManager;
class SystemPrompt;

struct ResourceContext {
    std::shared_ptr<KVCacheManager>              cache_manager;
    std::shared_ptr<KVCacheManager>              propose_cache_manager;
    std::vector<std::shared_ptr<KVCacheManager>> mtp_cache_managers;
    std::shared_ptr<SystemPrompt>                system_prompt;

    bool reuse_cache{false};
    bool enable_3fs{false};
    bool enable_memory_block_cache{false};
};

}  // namespace rtp_llm
