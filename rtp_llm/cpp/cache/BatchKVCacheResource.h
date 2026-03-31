#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

// Forward declaration for pointer type
class BatchKVCacheResource;
using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

class BatchKVCacheResource {
public:
    BatchKVCacheResource() = default;

    // Copy constructor: RAII-compliant deep copy
    BatchKVCacheResource(const BatchKVCacheResource& other) {
        initializeFrom(other);
    }

    // Copy assignment operator
    BatchKVCacheResource& operator=(const BatchKVCacheResource& other) {
        if (this != &other) {
            initializeFrom(other);
        }
        return *this;
    }

    // Move constructor
    BatchKVCacheResource(BatchKVCacheResource&& other) noexcept: batch_resource(std::move(other.batch_resource)) {}

    // Move assignment operator
    BatchKVCacheResource& operator=(BatchKVCacheResource&& other) noexcept {
        if (this != &other) {
            batch_resource = std::move(other.batch_resource);
        }
        return *this;
    }

    BatchKVCacheResourcePtr copy() const {
        return std::make_shared<BatchKVCacheResource>(*this);
    }

    int batchSize() const {
        return static_cast<int>(batch_resource.size());
    }

    void resetBatchSize(size_t batch_size) {
        batch_resource.resize(batch_size);
    }

    void initGroups(int                                group_nums,
                    int                                layer_num,
                    const std::vector<int>&            layer_to_group_id          = {},
                    size_t                             kernel_blocks_per_kv_block = 1,
                    const std::vector<CacheGroupType>& group_types                = {}) {
        for (auto& batch : batch_resource) {
            batch.initGroups(group_nums, layer_num, layer_to_group_id, kernel_blocks_per_kv_block, group_types);
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

    int curBlocksNum() const {
        if (batch_resource.empty()) {
            return 0;
        }

        auto& resource   = batch_resource[0];
        int   group_nums = resource.groupNums();

        int max_blocks_num = 0;
        for (int i = 0; i < group_nums; i++) {
            max_blocks_num = std::max(max_blocks_num, resource.blocksNum(i));
        }
        return max_blocks_num;
    }

    const BlockIndicesType& blocks(int batch_id, int group_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].blocks(group_id);
    }

    const BlockIndicesType& kernelBlocks(int batch_id, int group_id = 0) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].kernelBlocks(group_id);
    }

    BlockIds& mutableBlockIds(int batch_id, int group_id = 0) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        return batch_resource[batch_id].mutableBlockIds(group_id);
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
        batch_resource[batch_id].cacheKeys().push_back(key);
    }

    void initBatchGroups(int                                batch_id,
                         int                                group_nums,
                         int                                layer_num,
                         const std::vector<int>&            layer_to_group_id          = {},
                         size_t                             kernel_blocks_per_kv_block = 1,
                         const std::vector<CacheGroupType>& group_types                = {}) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].initGroups(
            group_nums, layer_num, layer_to_group_id, kernel_blocks_per_kv_block, group_types);
    }

    void setBatchBlocks(int batch_id, int group_id, const BlockIndicesType& blocks) {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
        batch_resource[batch_id].mutableBlockIds(group_id).assign(blocks);
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

    void resetAndReturnOldResources(int new_batch_size, std::vector<KVCacheResource>& old_resources) {
        old_resources = std::move(batch_resource);
        batch_resource.clear();
        batch_resource.resize(new_batch_size);
    }

    void moveBatchResource(int batch_idx, KVCacheResource&& resource) {
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

    void swapBlocks(int32_t batch_id, size_t group_id, size_t rhs, size_t lhs) {
        batch_resource[batch_id].swapBlocks(group_id, rhs, lhs);
    }

private:
    void initializeFrom(const BatchKVCacheResource& other) {
        resetBatchSize(other.batchSize());
        if (other.batchSize() == 0 || other.groupNums() == 0) {
            return;
        }

        // Derive layer_to_group mapping from the first batch's layer/group block pointer relationships.
        // layer_block_ids[i] is a shared_ptr alias to group_block_ids[gid], so we match by pointer identity.
        const auto&      src_layers = other.batch_resource[0].layerBlocks();
        const auto&      src_groups = other.batch_resource[0].groupBlocks();
        int              layer_num  = static_cast<int>(src_layers.size());
        std::vector<int> layer_to_group_id;
        if (layer_num > 0) {
            layer_to_group_id.resize(layer_num, 0);
            for (int l = 0; l < layer_num; ++l) {
                bool found = false;
                for (int g = 0; g < static_cast<int>(src_groups.size()); ++g) {
                    if (src_layers[l].get() == src_groups[g].get()) {
                        layer_to_group_id[l] = g;
                        found = true;
                        break;
                    }
                }
                RTP_LLM_CHECK_WITH_INFO(found, "initializeFrom: layer " + std::to_string(l) + " has no matching group");
            }
        }

        initGroups(other.groupNums(), layer_num, layer_to_group_id);

        for (int batch_id = 0; batch_id < other.batchSize(); ++batch_id) {
            for (int group_id = 0; group_id < other.groupNums(); ++group_id) {
                const auto& blocks = other.batch_resource[batch_id].blocks(group_id);
                setBatchBlocks(batch_id, group_id, blocks);
            }
            setBatchCacheKeys(batch_id, other.batch_resource[batch_id].cacheKeys());
        }
    }

    std::vector<KVCacheResource> batch_resource;  // [batch_size]
};

}  // namespace rtp_llm
