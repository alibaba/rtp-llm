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
        cache_keys_initialized_ = false;
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

    int curBlocksNum() const {
        if (batch_resource.empty()) {
            return 0;
        }

        return batch_resource[0].maxBlocksNum();
    }

    const BlockIndicesType& blocks(int batch_id, std::string_view tag) const {
        return cacheResource(batch_id).blocks(tag);
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

    BlockIds& mutableBlockIdsForLayer(int batch_id, int layer_id, std::string_view tag) {
        return cacheResource(batch_id).mutableBlockIdsForLayer(layer_id, tag);
    }

    std::shared_ptr<BlockIds> groupBlockIds(int batch_id, std::string_view tag) const {
        return cacheResource(batch_id).groupBlockIds(tag);
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
        cache_keys_initialized_ = false;
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
        const auto& first = batch_resource[0];
        const auto  tags  = first.groupTags();
        for (const auto& resource : batch_resource) {
            RTP_LLM_CHECK(resource.groupNums() == first.groupNums());
            for (const auto& tag : tags) {
                RTP_LLM_CHECK(resource.blocksNum(tag) == first.blocksNum(tag));
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
        cache_keys_initialized_ = false;
    }

    void moveBatchResource(int batch_idx, KVCacheResource&& resource) {
        RTP_LLM_CHECK(batch_idx >= 0 && static_cast<size_t>(batch_idx) < batch_resource.size());
        batch_resource[batch_idx] = std::move(resource);
    }

    void swap(BatchKVCacheResource& other) noexcept {
        batch_resource.swap(other.batch_resource);
        std::swap(cache_keys_initialized_, other.cache_keys_initialized_);
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

    // Set by initCacheKeys()/updateCacheKeys() once the rolling cache keys have been computed for
    // this resource. A first malloc that fails with MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED
    // re-enters KVCacheManager::malloc() having allocated nothing, so curBlocksNum() is still zero
    // and the attempt still looks like a first malloc. This flag tells the retry apart from a
    // genuine first attempt, so the keys are not recomputed and the prefill-cache-hit metric is not
    // double-counted for the same request.
    bool cacheKeysInitialized() const {
        return cache_keys_initialized_;
    }

    void markCacheKeysInitialized() {
        cache_keys_initialized_ = true;
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

    void swapBlocks(int32_t batch_id, std::string_view group_tag, size_t rhs, size_t lhs) {
        cacheResource(batch_id).swapBlocks(group_tag, rhs, lhs);
    }

private:
    std::vector<KVCacheResource> batch_resource;  // [batch_size]
    bool                         cache_keys_initialized_{false};
};

using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

}  // namespace rtp_llm
