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

private:
    struct KVCacheBuffer {
        rtp_llm::BufferPtr kv_blocks;
        rtp_llm::BufferPtr kv_scales;
    };

    struct BlockAddrInfo {
        void* k_addr       = nullptr;
        void* v_addr       = nullptr;
        void* k_scale_addr = nullptr;
        void* v_scale_addr = nullptr;
    };

    const CacheConfig cache_config_;
    const AllocationType atype_;
    std::set<int> free_block_ids;
    KVCacheBuffer kv_cache_;
};

}  // namespace rtp_llm
