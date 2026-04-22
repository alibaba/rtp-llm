#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

// Per-batch entry: holds KVCacheResources for all models and shared cache_keys.
// Within one batch slot (one stream), all models share the same cache_keys
// since they are driven by the same token sequence.
struct ModelKVResources {
    std::vector<KVCacheResource> model_resources;  // [model_id]
    CacheKeysType                cache_keys;

    size_t modelNum() const {
        return model_resources.size();
    }
    bool empty() const {
        return model_resources.empty();
    }
    void resize(size_t model_num) {
        model_resources.resize(model_num);
    }

    KVCacheResource& at(size_t model_id) {
        RTP_LLM_CHECK(model_id < model_resources.size());
        return model_resources[model_id];
    }

    const KVCacheResource& at(size_t model_id) const {
        RTP_LLM_CHECK(model_id < model_resources.size());
        return model_resources[model_id];
    }

    std::string debugString() const {
        std::stringstream ss;
        for (size_t m = 0; m < model_resources.size(); m++) {
            ss << "model:[" << m << "], " << model_resources[m].debugString();
        }
        ss << "cache_keys:[";
        for (size_t k = 0; k < cache_keys.size(); k++) {
            if (k > 0)
                ss << ", ";
            ss << cache_keys[k];
        }
        ss << "]";
        return ss.str();
    }
};

class BatchKVCacheResource {
public:
    BatchKVCacheResource() {}

    int batchSize() const {
        return static_cast<int>(batch_resource_.size());
    }

    size_t modelNum() const {
        if (batch_resource_.empty()) {
            return 0;
        }
        return batch_resource_[0].modelNum();
    }

    void resetBatchSize(size_t batch_size, size_t model_num = 1) {
        batch_resource_.resize(batch_size);
        for (auto& entry : batch_resource_) {
            entry.resize(model_num);
        }
    }

    void initGroups(int                                group_nums,
                    int                                layer_num,
                    const std::vector<int>&            layer_to_group_id          = {},
                    size_t                             kernel_blocks_per_kv_block = 1,
                    const std::vector<CacheGroupType>& group_types                = {},
                    size_t                             model_id                   = 0) {
        for (auto& entry : batch_resource_) {
            if (model_id >= entry.modelNum()) {
                entry.resize(model_id + 1);
            }
            entry.at(model_id).initGroups(
                group_nums, layer_num, layer_to_group_id, kernel_blocks_per_kv_block, group_types);
        }
    }

    int groupNums(size_t model_id = 0) const {
        RTP_LLM_CHECK(!batch_resource_.empty());
        return batch_resource_[0].at(model_id).groupNums();
    }

    void resizeBlocks(int reserver_blocks, int value = 0, size_t model_id = 0) {
        for (auto& entry : batch_resource_) {
            entry.at(model_id).resizeBlocks(reserver_blocks, value);
        }
    }

    int blocksNum(int batch_id, int group_id = 0, size_t model_id = 0) const {
        checkBatchId(batch_id);
        return batch_resource_[batch_id].at(model_id).blocksNum(group_id);
    }

    int curBlocksNum(size_t model_id = 0) const {
        if (batch_resource_.empty() || batch_resource_[0].empty()) {
            return 0;
        }

        auto& resource   = batch_resource_[0].at(model_id);
        int   group_nums = resource.groupNums();

        int max_blocks_num = 0;
        for (int i = 0; i < group_nums; i++) {
            max_blocks_num = std::max(max_blocks_num, resource.blocksNum(i));
        }
        return max_blocks_num;
    }

    const BlockIndicesType& blocks(int batch_id, int group_id = 0, size_t model_id = 0) const {
        checkBatchId(batch_id);
        return batch_resource_[batch_id].at(model_id).blocks(group_id);
    }

    const BlockIndicesType& kernelBlocks(int batch_id, int group_id = 0, size_t model_id = 0) const {
        checkBatchId(batch_id);
        return batch_resource_[batch_id].at(model_id).kernelBlocks(group_id);
    }

    BlockIds& mutableBlockIds(int batch_id, int group_id = 0, size_t model_id = 0) {
        checkBatchId(batch_id);
        return batch_resource_[batch_id].at(model_id).mutableBlockIds(group_id);
    }

    const GroupBlockIds& groupBlocks(int batch_id = 0, size_t model_id = 0) const {
        checkBatchId(batch_id);
        return batch_resource_[batch_id].at(model_id).groupBlocks();
    }

    const KVCacheResource& cacheResource(int batch_id = 0, size_t model_id = 0) const {
        checkBatchId(batch_id);
        return batch_resource_[batch_id].at(model_id);
    }

    KVCacheResource& cacheResource(int batch_id = 0, size_t model_id = 0) {
        checkBatchId(batch_id);
        return batch_resource_[batch_id].at(model_id);
    }

    void clearBlocks() {
        for (auto& entry : batch_resource_) {
            for (auto& model_resource : entry.model_resources) {
                model_resource.resizeBlocks(0, 0);
            }
        }
    }

    // -------- cache_keys: shared across all models in the same batch entry --------

    const CacheKeysType& cacheKeys(int batch_id = 0) const {
        checkBatchId(batch_id);
        return batch_resource_[batch_id].cache_keys;
    }

    void popBackCacheKey(int batch_id = 0) {
        checkBatchId(batch_id);
        auto& keys = batch_resource_[batch_id].cache_keys;
        if (!keys.empty()) {
            keys.pop_back();
        }
    }

    void popBackAllBatchCacheKeys() {
        for (auto& entry : batch_resource_) {
            if (!entry.cache_keys.empty()) {
                entry.cache_keys.pop_back();
            }
        }
    }

    void clearCacheKeys(int batch_id = 0) {
        checkBatchId(batch_id);
        batch_resource_[batch_id].cache_keys.clear();
    }

    void pushBackCacheKey(int batch_id, CacheKeyType key) {
        checkBatchId(batch_id);
        batch_resource_[batch_id].cache_keys.push_back(key);
    }

    // -------- per-batch init / set operations --------

    void initBatchGroups(int                                batch_id,
                         int                                group_nums,
                         int                                layer_num,
                         const std::vector<int>&            layer_to_group_id          = {},
                         size_t                             kernel_blocks_per_kv_block = 1,
                         const std::vector<CacheGroupType>& group_types                = {},
                         size_t                             model_id                   = 0) {
        checkBatchId(batch_id);
        auto& entry = batch_resource_[batch_id];
        if (model_id >= entry.modelNum()) {
            entry.resize(model_id + 1);
        }
        entry.at(model_id).initGroups(
            group_nums, layer_num, layer_to_group_id, kernel_blocks_per_kv_block, group_types);
    }

    void setBatchBlocks(int batch_id, int group_id, const BlockIndicesType& blocks, size_t model_id = 0) {
        checkBatchId(batch_id);
        batch_resource_[batch_id].at(model_id).mutableBlockIds(group_id).assign(blocks);
    }

    void setBatchCacheKeys(int batch_id, const CacheKeysType& keys) {
        checkBatchId(batch_id);
        batch_resource_[batch_id].cache_keys = keys;
    }

    void check(size_t model_id = 0) const {
        RTP_LLM_CHECK(!batch_resource_.empty());
        size_t blocks_num = batch_resource_[0].at(model_id).blocksNum(0);
        for (const auto& entry : batch_resource_) {
            RTP_LLM_CHECK(entry.at(model_id).blocksNum(0) == blocks_num);
        }
    }

    std::string debugString() const {
        std::stringstream ss;
        for (size_t i = 0; i < batch_resource_.size(); i++) {
            ss << "batch:[" << i << "], " << batch_resource_[i].debugString();
        }
        return "BatchKVCacheResource {" + ss.str() + "}";
    }

    void resetAndReturnOldResources(int new_batch_size, std::vector<ModelKVResources>& old_resources) {
        old_resources = std::move(batch_resource_);
        batch_resource_.clear();
        batch_resource_.resize(new_batch_size);
    }

    void moveBatchResource(int batch_idx, ModelKVResources&& resource) {
        checkBatchId(batch_idx);
        batch_resource_[batch_idx] = std::move(resource);
    }

    const ModelKVResources& modelResources(int batch_id = 0) const {
        checkBatchId(batch_id);
        return batch_resource_[batch_id];
    }

    ModelKVResources& modelResources(int batch_id = 0) {
        checkBatchId(batch_id);
        return batch_resource_[batch_id];
    }

    std::vector<BlockIndicesType> getAllBatchBlocks(int group_id = 0, size_t model_id = 0) const {
        std::vector<BlockIndicesType> all_blocks;
        all_blocks.reserve(batch_resource_.size());
        for (const auto& entry : batch_resource_) {
            all_blocks.push_back(entry.at(model_id).blocks(group_id));
        }
        return all_blocks;
    }

    bool hasCacheKeys() const {
        for (const auto& entry : batch_resource_) {
            if (!entry.cache_keys.empty()) {
                return true;
            }
        }
        return false;
    }

    bool lastBlockAligned(size_t model_id = 0) const {
        for (const auto& entry : batch_resource_) {
            if (!entry.at(model_id).lastBlockAligned()) {
                return false;
            }
        }
        return true;
    }

    void setLastBlockAligned(bool last_block_aligned, size_t model_id = 0) {
        for (auto& entry : batch_resource_) {
            entry.at(model_id).setLastBlockAligned(last_block_aligned);
        }
    }

    void swapBlocks(int32_t batch_id, size_t group_id, size_t rhs, size_t lhs, size_t model_id = 0) {
        checkBatchId(batch_id);
        batch_resource_[batch_id].at(model_id).swapBlocks(group_id, rhs, lhs);
    }

private:
    void checkBatchId(int batch_id) const {
        RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource_.size());
    }

    std::vector<ModelKVResources> batch_resource_;  // [batch_id]
};

using BatchKVCacheResourcePtr = std::shared_ptr<BatchKVCacheResource>;

}  // namespace rtp_llm
