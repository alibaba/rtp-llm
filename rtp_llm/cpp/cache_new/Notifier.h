#pragma once

#include "rtp_llm/cpp/cache_new/MemoryBlockCache.h"

namespace rtp_llm {

class Notifier {
public:
    Notifier() = default;
    ~Notifier() = default;

public:
    // write through
    void notify_match(const std::vector<int64_t>& cache_keys);
    // write back
    void notify_evict(const std::vector<int64_t>& cache_keys);

private:
    std::shared_ptr<MemoryBlockCache> memory_block_cache_;
};

}  // namespace rtp_llm