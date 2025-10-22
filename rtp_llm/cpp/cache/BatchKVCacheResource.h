#pragma once

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <memory>

namespace rtp_llm {

struct BlockIds {
    std::vector<int> block_indices;
};

class BatchKVCacheResource {
public:
    BatchKVCacheResource() {}
    int                     batchSize() const;
    int                     blockSize(int batch_id) const;
    void                    resize(size_t batch_size);
    void                    resize(size_t batch_id, int reserver_blocks, bool clear = false);
    void                    shrink(size_t batch_id, int reserver_blocks);
    void                    pushBack(const KVCacheResource& addr);
    void                    append(size_t batch_id, const KVCacheResource& addr);
    void                    appendClone(const KVCacheResource& addr, std::shared_ptr<CacheManager>& cache_manager);
    void                    append(const std::vector<KVCacheResource>& resource);
    int                     maxBlockSize() const;
    const std::vector<int>& blocks(int batch_id) const;
    void                    clear();
    void                    check() const;

    std::string debugString() const;

public:
    // [batch_size, max_block_per_seq]
    std::vector<std::vector<int32_t>> batch_block_id;

    // batch_id -> layer_id -> block_indices
    std::vector<std::vector<std::shared_ptr<BlockIds>>> batch_cache_layer_layouts;

    // block_idx that has been cached in block_cache
    // batch_id -> group_id -> block_indices 
    std::vector<std::vector<std::shared_ptr<BlockIds>>> batch_cache_layer_cached_layouts;


    // cache_keys and batch_block_id are not consistent at all times
    std::vector<std::vector<int64_t>> cache_keys;
};

}  // namespace rtp_llm
