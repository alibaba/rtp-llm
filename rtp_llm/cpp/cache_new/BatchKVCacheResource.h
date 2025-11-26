#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <memory>

namespace rtp_llm {

struct BlockIds {
    size_t size() {
        return block_indices.size();
    }

    std::vector<int> block_indices;
};

typedef int64_t CacheKeyType;
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
    void initGroups(int group_nums) {
        for (int i = 0; i < group_nums; i++) {
            group_block_ids.push_back(std::make_shared<BlockIds>());
        }
    }

    void resizeBlocks(int reserver_blocks, int value) {
        for (auto& group : group_block_ids) {
            group->block_indices.resize(reserver_blocks, value);
        }
    }

    int blocks() const {
        return group_block_ids[0]->size();
    }
};
class BatchKVCacheResource {
public:
    BatchKVCacheResource() {}
    int                     batchSize() const;
    int                     blockSize(int batch_id) const;
    void                    initGroups(int group_nums);
    void                    resize(size_t batch_size);
    void                    resize(size_t batch_id, int reserver_blocks, int value);
    int                     maxBlockSize() const;
    const std::vector<int>& blocks(int batch_id, int group_id = 0) const;
    void                    clear();
    void                    check() const;

    void resizeBlocks(int reserver_blocks, int value) {
        for (auto& resource : batch_resource) {
            resource.resizeBlocks(reserver_blocks, value);
        }
    }
    std::string debugString() const;

public:
    bool enable_reuse_cache  = true;
    bool first_fill_finished = false;
    bool last_block_aligned  = true;

    std::vector<KVCacheResourceV1> batch_resource;
};

using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

}  // namespace rtp_llm
