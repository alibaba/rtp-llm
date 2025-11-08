#pragma once

#include <vector>
#include <memory>

namespace rtp_llm {

// TODO(chanyin): for compatibility with old cache system, refactor in future
class KVCacheGroup;

class KVCacheResource {
public:
    KVCacheResource(const std::vector<int>& block_id): block_id(block_id) {}
    void            clear();
    KVCacheResource clone(KVCacheGroup& kv_cache_group) const;

public:
    // [max_block_per_seq]
    std::vector<int> block_id;
};

}  // namespace rtp_llm

