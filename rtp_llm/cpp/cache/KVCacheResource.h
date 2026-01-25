#pragma once

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

using CacheKeyType = int64_t;
using BlockIdxType = int32_t;

using CacheKeysType    = std::vector<CacheKeyType>;
using BlockIndicesType = std::vector<BlockIdxType>;

class BlockIds {
public:
    size_t blocksNum() const {
        return block_indices.size();
    }

    BlockIndicesType& blocks() {
        return block_indices;
    }

    void resize(int reserver_blocks, int value) {
        block_indices.resize(reserver_blocks, value);
    }

private:
    BlockIndicesType block_indices;
};

using GroupBlockIds = std::vector<std::shared_ptr<BlockIds>>;
using LayerBlockIds = std::vector<std::shared_ptr<BlockIds>>;

class KVCacheResource {
public:
    void initGroups(int group_num, int layer_num);
    void resizeBlocks(int reserver_blocks, int value = 0);

    int               blocksNum(int group_id = 0) const;
    BlockIndicesType& blocks(int group_id = 0) const;

    int groupNums() const;

    GroupBlockIds&       groupBlocks();
    const GroupBlockIds& groupBlocks() const;

    const LayerBlockIds& layerBlocks() const;

    CacheKeysType&       cacheKeys();
    const CacheKeysType& cacheKeys() const;

    size_t reuseBlocksNum() const;
    void   setReuseBlocksNum(size_t reuse_blocks_num);

    std::string debugString() const;

private:
    // layer_id -> block_indices
    LayerBlockIds layer_block_ids;
    // group_id -> block_indices
    GroupBlockIds group_block_ids;
    CacheKeysType cache_keys;
    size_t        reuse_blocks_num_{0};
};

}  // namespace rtp_llm
