#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <memory>

namespace rtp_llm {

struct BlockIds {
    std::vector<int> block_indices;
};

typedef size_t  CacheKeyType;
typedef int32_t BlockIdxType;

typedef std::vector<CacheKeyType> CacheKeysType;
typedef std::vector<BlockIdxType> BlockIndicesType;

typedef std::vector<std::shared_ptr<BlockIds>> GroupBlockIds;
typedef std::vector<std::shared_ptr<BlockIds>> LayerBlockIds;
// typedef std::vector<GroupBlockIds>             BatchGroupBlockIds;
// typedef std::vector<LayerBlockIds>             BatchLayerBlockIds;
// typedef std::vector<CacheKeyType>              BatchCacheKeys;

class KVCacheResourceV1 {
public:
    // layer_id -> block_indices
    LayerBlockIds layer_block_ids;
    // group_id -> block_indices
    GroupBlockIds group_block_ids;
    // cache_keys and block_id are not consistent at all times
    CacheKeysType cache_keys;

public:
    void resize(int reserver_blocks, int value) {
        for (auto& group : group_block_ids) {
            group->block_indices.resize(reserver_blocks, value);
        }
    }
};

class BatchKVCacheResource {
public:
    BatchKVCacheResource() {}
    int                     batchSize() const;
    int                     blockSize(int batch_id) const;
    void                    resize(size_t batch_size);
    void                    resize(size_t batch_id, int reserver_blocks, int value);
    void                    shrink(size_t batch_id, int reserver_blocks);
    int                     maxBlockSize() const;
    const std::vector<int>& blocks(int batch_id) const;
    void                    clear();
    void                    check() const;

    std::string debugString() const;

public:
    bool enable_reuse_cache = true;

    // this two member will be deleted soon
    std::vector<std::vector<int32_t>> batch_block_id;
    std::vector<std::vector<size_t>>  cache_keys;

    std::vector<KVCacheResourceV1> batch_resource;
};

using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

}  // namespace rtp_llm