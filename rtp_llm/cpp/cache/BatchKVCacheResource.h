#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <memory>

namespace rtp_llm {

typedef int64_t CacheKeyType;
typedef int32_t BlockIdxType;

typedef std::vector<CacheKeyType> CacheKeysType;
typedef std::vector<BlockIdxType> BlockIndicesType;

class BlockIds {
public:
    size_t blocksNum() {
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

typedef std::vector<std::shared_ptr<BlockIds>> GroupBlockIds;
typedef std::vector<std::shared_ptr<BlockIds>> LayerBlockIds;

class KVCacheResourceV1 {
public:
    void initGroups(int group_nums) {
        for (int i = 0; i < group_nums; i++) {
            group_block_ids.push_back(std::make_shared<BlockIds>());
        }
    }

    void resizeBlocks(int reserver_blocks, int value = 0) {
        for (auto& group : group_block_ids) {
            group->resize(reserver_blocks, value);
        }
    }

    int blocksNum(int group_id = 0) const {
        RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
        return group_block_ids[group_id]->blocksNum();
    }

    BlockIndicesType& blocks(int group_id = 0) const {
        RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
        return group_block_ids[group_id]->blocks();
    }

    int groupNums() const {
        return group_block_ids.size();
    }

    GroupBlockIds& groupBlocks() {
        return group_block_ids;
    }

    CacheKeysType& cacheKeys() {
        return cache_keys;
    }

    std::string debugString() const {
        std::stringstream debug_string;
        for (int group_id = 0; group_id < group_block_ids.size(); group_id++) {
            debug_string << "group:[" << group_id << "], block:[";
            auto& block_indices = blocks(group_id);
            for (auto& block : block_indices) {
                debug_string << block << ", ";
            }
            debug_string << "], ";
        }

        return debug_string.str();
    }

private:
    // layer_id -> block_indices
    LayerBlockIds layer_block_ids;
    // group_id -> block_indices
    GroupBlockIds group_block_ids;
    CacheKeysType cache_keys;
};

class BatchKVCacheResource {
public:
    BatchKVCacheResource() {}

    int batchSize() const {
        return static_cast<int>(batch_resource.size());
    }

    void resetBatchSize(size_t batch_size) {
        batch_resource.resize(batch_size);
    }

    void initGroups(int group_nums) {
        for (auto& batch : batch_resource) {
            batch.initGroups(group_nums);
        }
    }

    int groupNums() const {
        RTP_LLM_CHECK(!batch_resource.empty());
        return batch_resource[0].groupNums();
    }

    void resizeBlocks(int reserver_blocks, int value = 0) {
        for (auto& resource : batch_resource) {
            resource.resizeBlocks(reserver_blocks, value);
        }
    }

    int blocksNum(int batch_id, int group_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].blocksNum(group_id);
    }

    int maxBlocksNum() const {
        return batch_resource.empty() ? 0 : batch_resource[0].blocksNum();
    }

    BlockIndicesType& blocks(int batch_id, int group_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].blocks(group_id);
    }

    GroupBlockIds& groupBlocks(int batch_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].groupBlocks();
    }

    KVCacheResourceV1& cacheResource(int batch_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id];
    }

    void clearBlocks() {
        resizeBlocks(0, 0);
    }

    CacheKeysType& cacheKeys(int batch_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].cacheKeys();
    }

    void check() const {
        RTP_LLM_CHECK(!batch_resource.empty());
        size_t blocks_num = batch_resource[0].blocksNum(0);
        for (const auto& resource : batch_resource) {
            RTP_LLM_CHECK(resource.blocksNum(0) == blocks_num);
        }
    }

    std::string debugString() const {
        std::stringstream debug_string, batch_resource_string;
        for (size_t i = 0; i < batch_resource.size(); i++) {
            batch_resource_string << "batch:[" << i << "], detail info: ";
            batch_resource_string << batch_resource[i].debugString();
        }

        debug_string << "BatchKVCacheResource {" << batch_resource_string.str() << "}";
        return debug_string.str();
    }

public:
    bool                           enable_reuse_cache  = true;
    bool                           first_fill_finished = false;
    bool                           last_block_aligned  = true;
    std::vector<KVCacheResourceV1> batch_resource;
};

using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

}  // namespace rtp_llm
