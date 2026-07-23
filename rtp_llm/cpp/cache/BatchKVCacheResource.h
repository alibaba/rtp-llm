#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
class BatchKVCacheResource {
public:
    BatchKVCacheResource() {}

    int batchSize() const {
        return static_cast<int>(batch_resource.size());
    }

    void resetBatchSize(size_t batch_size) {
        batch_resource.resize(batch_size);
    }

    void initGroups(std::shared_ptr<const CacheTopology> topology) {
        RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "BatchKVCacheResource::initGroups requires a topology");
        for (auto& batch : batch_resource) {
            batch.initGroups(topology);
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

    int blocksNum(int batch_id, std::string_view tag) const {
        return cacheResource(batch_id).blocksNum(tag);
    }

    int blocksNumByIndex(int batch_id, size_t group_index) const {
        return static_cast<int>(groupBlocks(batch_id).at(group_index)->blocksNum());
    }

    int curBlocksNum() const {
        if (batch_resource.empty()) {
            return 0;
        }

        auto& resource       = batch_resource[0];
        int   max_blocks_num = 0;
        for (const auto& blocks : resource.groupBlocks()) {
            max_blocks_num = std::max(max_blocks_num, static_cast<int>(blocks->blocksNum()));
        }
        return max_blocks_num;
    }

    const BlockIndicesType& blocks(int batch_id, std::string_view tag) const {
        return cacheResource(batch_id).blocks(tag);
    }

    const BlockIndicesType& blocksByIndex(int batch_id, size_t group_index) const {
        return groupBlocks(batch_id).at(group_index)->blocks();
    }

    const BlockIndicesType& blocksForLayer(int batch_id, int layer_id, std::string_view tag) const {
        return cacheResource(batch_id).blocksForLayer(layer_id, tag);
    }

    const BlockIndicesType& kernelBlocks(int batch_id, std::string_view tag) const {
        return cacheResource(batch_id).kernelBlocks(tag);
    }

    const BlockIndicesType& kernelBlocksForLayer(int batch_id, int layer_id, std::string_view tag) const {
        return cacheResource(batch_id).kernelBlocksForLayer(layer_id, tag);
    }

    BlockIds& mutableBlockIds(int batch_id, std::string_view tag) {
        return cacheResource(batch_id).mutableBlockIds(tag);
    }

    BlockIds& mutableBlockIdsByIndex(int batch_id, size_t group_index) {
        return *cacheResource(batch_id).groupBlocks().at(group_index);
    }

    BlockIds& mutableBlockIdsForLayer(int batch_id, int layer_id, std::string_view tag) {
        return cacheResource(batch_id).mutableBlockIdsForLayer(layer_id, tag);
    }

    const GroupBlockIds& groupBlocks(int batch_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].groupBlocks();
    }

    const KVCacheResource& cacheResource(int batch_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id];
    }

    KVCacheResource& cacheResource(int batch_id = 0) {
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
        auto& resource = batch_resource[batch_id];
        auto& keys     = resource.cacheKeys();
        keys.push_back(key);
    }

    void setBatchBlocks(int batch_id, std::string_view tag, const BlockIndicesType& blocks) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].mutableBlockIds(tag).assign(blocks);
    }

    void setBatchCacheKeys(int batch_id, const CacheKeysType& keys) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].cacheKeys() = keys;
    }

    void check() const {
        RTP_LLM_CHECK(!batch_resource.empty());
        const auto& expected = batch_resource[0].groupBlocks();
        const auto& tags     = batch_resource[0].groupTags();
        RTP_LLM_CHECK(expected.size() == tags.size());
        for (size_t group_index = 0; group_index < expected.size(); ++group_index) {
            const auto& tag        = tags[group_index];
            const auto  blocks_num = expected[group_index]->blocksNum();
            for (const auto& resource : batch_resource) {
                RTP_LLM_CHECK(resource.groupBlocks().size() == expected.size());
                RTP_LLM_CHECK(resource.groupTags().at(group_index) == tag);
                RTP_LLM_CHECK(resource.groupBlocks().at(group_index)->blocksNum() == blocks_num);
            }
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

    void resetAndReturnOldResources(int new_batch_size, std::vector<KVCacheResource>& old_resources) {
        old_resources = std::move(batch_resource);
        batch_resource.clear();
        batch_resource.resize(new_batch_size);
    }

    void moveBatchResource(int batch_idx, KVCacheResource&& resource) {
        RTP_LLM_CHECK(batch_idx >= 0 && static_cast<size_t>(batch_idx) < batch_resource.size());
        batch_resource[batch_idx] = std::move(resource);
    }

    std::vector<BlockIndicesType> getAllBatchBlocks(std::string_view tag) const {
        std::vector<BlockIndicesType> all_blocks;
        all_blocks.reserve(batch_resource.size());
        for (const auto& resource : batch_resource) {
            all_blocks.push_back(resource.blocks(tag));
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

    bool lastBlockAligned() const {
        for (const auto& resource : batch_resource) {
            if (!resource.lastBlockAligned()) {
                return false;
            }
        }
        return true;
    }

    void setLastBlockAligned(bool last_block_aligned) {
        for (auto& resource : batch_resource) {
            resource.setLastBlockAligned(last_block_aligned);
        }
    }

    void swapBlocks(int32_t batch_id, std::string_view tag, size_t rhs, size_t lhs) {
        batch_resource[batch_id].swapBlocks(tag, rhs, lhs);
    }

private:
    std::vector<KVCacheResource> batch_resource;  // [batch_size]
};

using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

}  // namespace rtp_llm
