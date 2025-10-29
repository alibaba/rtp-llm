#pragma once

namespace rtp_llm {

class MemoryBlockPool {
public:
    MemoryBlockPool(const CacheConfig& config, AllocationType atype = AllocationType::DEVICE);
    ~MemoryBlockPool() = default;

public:
    bool init();
    std::vector<int> malloc(int num_blocks);
    void free(vector<int> block_ids);
    bool has(int64_t cache_key) const;

private:
    const CacheConfig cache_config_;
    const AllocationType atype_;
    std::set<int> free_block_ids;
    KVCacheBuffer kv_cache_;
};

}  // namespace rtp_llm
