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

    const GroupBlockIds& groupBlocks() const {
        return group_block_ids;
    }

    CacheKeysType& cacheKeys() {
        return cache_keys;
    }

    const CacheKeysType& cacheKeys() const {
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

    const BlockIndicesType& blocks(int batch_id, int group_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].blocks(group_id);
    }

    BlockIndicesType& mutableBlocks(int batch_id, int group_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].blocks(group_id);
    }

    const GroupBlockIds& groupBlocks(int batch_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].groupBlocks();
    }

    const KVCacheResourceV1& cacheResource(int batch_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id];
    }

    void clearBlocks() {
        resizeBlocks(0, 0);
    }

    const CacheKeysType& cacheKeys(int batch_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].cacheKeys();
    }

    void popBackCacheKey(int batch_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        auto& keys = batch_resource[batch_id].cacheKeys();
        if (!keys.empty()) {
            keys.pop_back();
        }
    }

    void popBackAllBatchCacheKeys() {
        for (auto& resource : batch_resource) {
            auto& keys = resource.cacheKeys();
            if (!keys.empty()) {
                keys.pop_back();
            }
        }
    }

    void clearCacheKeys(int batch_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].cacheKeys().clear();
    }

    void pushBackCacheKey(int batch_id, CacheKeyType key) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].cacheKeys().push_back(key);
    }

    void initBatchGroups(int batch_id, int group_nums) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].initGroups(group_nums);
    }

    void setBatchBlocks(int batch_id, int group_id, const BlockIndicesType& blocks) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].blocks(group_id) = blocks;
    }

    void setBatchCacheKeys(int batch_id, const CacheKeysType& keys) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].cacheKeys() = keys;
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

    void resetAndReturnOldResources(int new_batch_size, std::vector<KVCacheResourceV1>& old_resources) {
        old_resources = std::move(batch_resource);
        batch_resource.clear();
        batch_resource.resize(new_batch_size);
    }

    void moveBatchResource(int batch_idx, KVCacheResourceV1&& resource) {
        RTP_LLM_CHECK(batch_idx >= 0 && static_cast<size_t>(batch_idx) < batch_resource.size());
        batch_resource[batch_idx] = std::move(resource);
    }

    std::vector<BlockIndicesType> getAllBatchBlocks(int group_id = 0) const {
        std::vector<BlockIndicesType> all_blocks;
        all_blocks.reserve(batch_resource.size());
        for (const auto& resource : batch_resource) {
            all_blocks.push_back(resource.blocks(group_id));
        }
        return all_blocks;
    }

    bool hasCacheKeys() const {
        if (batch_resource.empty()) {
            return false;
        }
        for (const auto& resource : batch_resource) {
            if (!resource.cacheKeys().empty()) {
                return true;
            }
        }
        return false;
    }

public:
    bool enable_reuse_cache  = true;
    bool first_fill_finished = false;
    bool last_block_aligned  = true;

private:
    std::vector<KVCacheResourceV1> batch_resource;
};

using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

}  // namespace rtp_llm
