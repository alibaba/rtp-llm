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

    size_t reuseBlockNum() const;

    size_t deviceReuseBlockNum() const;
    void   setDeviceReuseBlockNum(size_t device_reuse_blocks_num);

    size_t memoryReuseBlockNum() const;
    void   setMemoryReuseBlockNum(size_t memory_reuse_blocks_num);

    size_t remoteReuseBlockNum() const;
    void   setRemoteReuseBlockNum(size_t remote_reuse_blocks_num);

    bool skipLastBlock() const;
    void setSkipLastBlock(bool skip_last_block);

    std::string debugString() const;

private:
    // layer_id -> block_indices
    LayerBlockIds layer_block_ids;
    // group_id -> block_indices
    GroupBlockIds group_block_ids;
    CacheKeysType cache_keys;

    size_t device_reuse_block_num_{0};
    size_t memory_reuse_block_num_{0};
    size_t remote_reuse_block_num_{0};
    bool   skip_last_block_{true};
};

}  // namespace rtp_llm
