#pragma once

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <memory>

namespace rtp_llm {

struct BlockIds {
    std::vector<int> block_indices;
};

// struct KVCacheResource {
//     std::vector<std::shared_ptr<BlockIds>>& batch_layer_block_ids;
//     std::vector<std::shared_ptr<BlockIds>>& batch_group_block_ids;
//     KVCacheResource(std::vector<std::shared_ptr<BlockIds>>& batch_layer_block_ids,
//     std::vector<std::shared_ptr<BlockIds>>& batch_group_block_ids):
//         batch_layer_block_ids(batch_layer_block_ids), batch_group_block_ids(batch_group_block_ids) {}
// };

typedef std::vector<std::shared_ptr<BlockIds>> GroupBlockIds;
typedef std::vector<std::shared_ptr<BlockIds>> LayerBlockIds;
typedef std::vector<GroupBlockIds>             BatchGroupBlockIds;
typedef std::vector<LayerBlockIds>             BatchLayerBlockIds;

class BatchKVCacheResource {
public:
    BatchKVCacheResource() {}
    int                     batchSize() const;
    int                     blockSize(int batch_id) const;
    void                    resize(size_t batch_size);
    void                    resize(size_t batch_id, int reserver_blocks, int value);
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
    bool enable_reuse_cache = true;

    // batch_id -> block_indices
    std::vector<std::vector<int32_t>> batch_block_id;

    // batch_id -> layer_id -> block_indices
    BatchLayerBlockIds batch_layer_block_ids;

    // batch_id -> group_id -> block_indices
    BatchGroupBlockIds batch_group_block_ids;

    // cache_keys and batch_block_id are not consistent at all times
    std::vector<std::vector<size_t>> cache_keys;
};

using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

}  // namespace rtp_llm
