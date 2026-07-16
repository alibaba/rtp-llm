#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

struct GroupBase {
    KVCacheSpecPtr   spec;
    CacheGroupPolicy policy;
    std::vector<int> layer_ids;
    uint32_t         block_num             = 0;
    uint32_t         local_kv_head_num     = 1;
    size_t           kv_block_stride_bytes = 0;
    size_t           kv_scale_stride_bytes = 0;
};

struct LayerBase {
    std::vector<int>           group_ids;
    std::map<std::string, int> tag_to_gid;
};

struct CacheConfig {
    std::vector<GroupBase>               groups;
    std::vector<LayerBase>               layers;
    std::unordered_map<std::string, int> tag_to_gid;

    std::vector<int> layer_to_block_stride_bytes;
    bool             group_block_layout_initialized = false;

    rtp_llm::DataType dtype;
    uint32_t          layer_num;      // the number of main model layers
    uint32_t          layer_all_num;  // the number of all layers including mtp modules
    bool              use_mla   = false;
    bool              is_sparse = false;

    // Block configuration
    uint32_t block_num;
    size_t   seq_size_per_block        = 1;
    size_t   kernel_seq_size_per_block = 1;

    // Returns how many kernel blocks fit inside one physical (kv-manager) block.
    size_t kernelBlocksPerKvBlock() const {
        if (kernel_seq_size_per_block == 0) {
            return 1;
        }
        return std::max<size_t>(1, seq_size_per_block / kernel_seq_size_per_block);
    }

    // Block sizing information
    // ---- Per-block sizes (all layers) ----
    size_t kv_block_size_bytes = 0;
    size_t kv_scale_size_bytes = 0;
    size_t block_size_bytes    = 0;  // (kv + scales together)

    // ---- Per-block strides (one layer) ----
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride_bytes = 0;

    // Attention-specific configuration
    int linear_step     = 1;  // For Linear attention: keep one cache block every `linear_step` blocks
    int group_layer_num = 1;  // Number of layers per group for hybrid attention

    // mtp-model configurations
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    CacheConfig() {}

    static uint32_t
    mtpGlobalLayerId(uint32_t main_layer_num, int module_index, uint32_t module_layer_num, int local_layer_id) {
        constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
        if (module_index < 0 || module_layer_num == 0 || local_layer_id < 0
            || static_cast<uint32_t>(local_layer_id) >= module_layer_num) {
            return invalid;
        }
        const uint64_t global_layer_id = static_cast<uint64_t>(main_layer_num)
                                         + static_cast<uint64_t>(module_index) * module_layer_num
                                         + static_cast<uint32_t>(local_layer_id);
        return global_layer_id < invalid ? static_cast<uint32_t>(global_layer_id) : invalid;
    }

    int groupNums() const {
        return static_cast<int>(groups.size());
    }

    const KVCacheSpecPtr& specForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(
            gid < groups.size(), "CacheConfig::specForGroup invalid gid=%zu size=%zu", gid, groups.size());
        RTP_LLM_CHECK_WITH_INFO(groups[gid].spec != nullptr, "CacheConfig::specForGroup null spec gid=%zu", gid);
        return groups[gid].spec;
    }

    CacheGroupType typeForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(
            gid < groups.size(), "CacheConfig::typeForGroup invalid gid=%zu size=%zu", gid, groups.size());
        return groups[gid].policy.group_type;
    }

    const std::string& tagForGroup(size_t gid) const {
        return specForGroup(gid)->tag;
    }

    int groupIdForTag(const std::string& tag) const {
        const auto it = tag_to_gid.find(tag);
        RTP_LLM_CHECK_WITH_INFO(it != tag_to_gid.end(), "CacheConfig::groupIdForTag missing tag=%s", tag.c_str());
        return it->second;
    }

    const std::vector<int>& layerIdsForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(
            gid < groups.size(), "CacheConfig::layerIdsForGroup invalid gid=%zu size=%zu", gid, groups.size());
        return groups[gid].layer_ids;
    }

    std::vector<CacheGroupType> groupTypesSnapshot() const {
        std::vector<CacheGroupType> types;
        types.reserve(groups.size());
        for (const auto& group : groups) {
            types.push_back(group.policy.group_type);
        }
        return types;
    }

    std::vector<std::string> groupTagsSnapshot() const {
        std::vector<std::string> tags;
        tags.reserve(groups.size());
        for (const auto& group : groups) {
            RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheConfig::groupTagsSnapshot null spec");
            tags.push_back(group.spec->tag);
        }
        return tags;
    }

    std::vector<CacheGroupPolicy> groupPoliciesSnapshot() const {
        std::vector<CacheGroupPolicy> policies;
        policies.reserve(groups.size());
        for (const auto& group : groups) {
            policies.push_back(group.policy);
        }
        return policies;
    }

    std::vector<uint32_t> groupBlockNumsSnapshot() const {
        if (!group_block_layout_initialized) {
            return {};
        }
        std::vector<uint32_t> block_nums;
        block_nums.reserve(groups.size());
        for (const auto& group : groups) {
            block_nums.push_back(group.block_num);
        }
        return block_nums;
    }

    std::vector<size_t> groupBlockSizeBytesSnapshot() const {
        std::vector<size_t> result;
        result.reserve(static_cast<size_t>(groupNums()));
        for (size_t gid = 0; gid < static_cast<size_t>(groupNums()); ++gid) {
            result.push_back(blockSizeBytesForGroup(gid));
        }
        return result;
    }

    std::vector<size_t> groupKvBlockStrideBytesSnapshot() const {
        if (!group_block_layout_initialized) {
            return {};
        }
        std::vector<size_t> strides;
        strides.reserve(groups.size());
        for (const auto& group : groups) {
            strides.push_back(group.kv_block_stride_bytes);
        }
        return strides;
    }

    std::vector<size_t> groupKvScaleStrideBytesSnapshot() const {
        if (!group_block_layout_initialized) {
            return {};
        }
        std::vector<size_t> strides;
        strides.reserve(groups.size());
        for (const auto& group : groups) {
            strides.push_back(group.kv_scale_stride_bytes);
        }
        return strides;
    }

    std::vector<std::vector<int>> layerGroupIdsSnapshot() const {
        std::vector<std::vector<int>> result;
        if (!layers.empty()) {
            result.reserve(layers.size());
            for (const auto& layer : layers) {
                result.push_back(layer.group_ids);
            }
            return result;
        }
        return result;
    }

    std::vector<std::map<std::string, int>> layerTagToGroupIdSnapshot() const {
        std::vector<std::map<std::string, int>> result;
        result.reserve(layers.size());
        for (const auto& layer : layers) {
            result.push_back(layer.tag_to_gid);
        }
        return result;
    }

    uint32_t blockNumForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(
            gid < groups.size(), "CacheConfig::blockNumForGroup invalid gid=%zu size=%zu", gid, groups.size());
        if (group_block_layout_initialized && groups[gid].block_num > 0) {
            return groups[gid].block_num;
        }
        return block_num;
    }

    size_t kvBlockStrideBytesForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(),
                                "CacheConfig::kvBlockStrideBytesForGroup invalid gid=%zu size=%zu",
                                gid,
                                groups.size());
        if (group_block_layout_initialized) {
            return groups[gid].kv_block_stride_bytes;
        }
        return specForGroup(gid)->block_size_bytes();
    }

    size_t kvScaleStrideBytesForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(),
                                "CacheConfig::kvScaleStrideBytesForGroup invalid gid=%zu size=%zu",
                                gid,
                                groups.size());
        if (group_block_layout_initialized) {
            return groups[gid].kv_scale_stride_bytes;
        }
        return specForGroup(gid)->scale_block_size_bytes();
    }

    size_t blockSizeBytesForGroup(size_t gid) const {
        return layerIdsForGroup(gid).size() * (kvBlockStrideBytesForGroup(gid) + kvScaleStrideBytesForGroup(gid));
    }

    uint32_t localKvHeadNumForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(
            gid < groups.size(), "CacheConfig::localKvHeadNumForGroup invalid gid=%zu size=%zu", gid, groups.size());
        RTP_LLM_CHECK_WITH_INFO(groups[gid].local_kv_head_num > 0,
                                "CacheConfig::localKvHeadNumForGroup invalid local_kv_head_num=%u gid=%zu",
                                groups[gid].local_kv_head_num,
                                gid);
        return groups[gid].local_kv_head_num;
    }

    void setGroupPolicies(const std::vector<CacheGroupPolicy>& policies) {
        RTP_LLM_CHECK_WITH_INFO(policies.size() == groups.size(),
                                "CacheConfig::setGroupPolicies size %zu != group size %zu",
                                policies.size(),
                                groups.size());
        for (size_t gid = 0; gid < policies.size(); ++gid) {
            groups[gid].policy = policies[gid];
        }
    }

    void setGroupBlockLayout(const std::vector<uint32_t>& block_nums,
                             const std::vector<size_t>&   kv_block_stride_bytes,
                             const std::vector<size_t>&   kv_scale_stride_bytes) {
        const size_t group_num = groups.size();
        RTP_LLM_CHECK_WITH_INFO(block_nums.size() == group_num,
                                "CacheConfig::setGroupBlockLayout block_nums size %zu != group size %zu",
                                block_nums.size(),
                                group_num);
        RTP_LLM_CHECK_WITH_INFO(kv_block_stride_bytes.size() == group_num,
                                "CacheConfig::setGroupBlockLayout kv stride size %zu != group size %zu",
                                kv_block_stride_bytes.size(),
                                group_num);
        RTP_LLM_CHECK_WITH_INFO(kv_scale_stride_bytes.size() == group_num,
                                "CacheConfig::setGroupBlockLayout scale stride size %zu != group size %zu",
                                kv_scale_stride_bytes.size(),
                                group_num);
        for (size_t gid = 0; gid < group_num; ++gid) {
            groups[gid].block_num             = block_nums[gid];
            groups[gid].kv_block_stride_bytes = kv_block_stride_bytes[gid];
            groups[gid].kv_scale_stride_bytes = kv_scale_stride_bytes[gid];
        }
        group_block_layout_initialized = true;
    }

    void resizeLayerRoutes(size_t layer_count) {
        layers.resize(layer_count);
    }

    void setLayerIdsForGroup(size_t gid, const std::vector<int>& layer_ids) {
        RTP_LLM_CHECK_WITH_INFO(
            gid < groups.size(), "CacheConfig::setLayerIdsForGroup invalid gid=%zu size=%zu", gid, groups.size());
        groups[gid].layer_ids = layer_ids;
    }

    void appendLayerToGroup(size_t gid, int layer_id, const std::string& tag) {
        RTP_LLM_CHECK_WITH_INFO(
            gid < groups.size(), "CacheConfig::appendLayerToGroup invalid gid=%zu size=%zu", gid, groups.size());
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0, "CacheConfig::appendLayerToGroup invalid layer_id=%d", layer_id);
        const auto layer = static_cast<size_t>(layer_id);
        if (layer >= layers.size()) {
            resizeLayerRoutes(layer + 1);
        }
        groups[gid].layer_ids.push_back(layer_id);
        layers[layer].group_ids.push_back(static_cast<int>(gid));
        if (!tag.empty()) {
            layers[layer].tag_to_gid[tag] = static_cast<int>(gid);
        }
    }

    std::shared_ptr<CacheConfig>
    mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num);

    CacheGroupPolicy policyForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(
            gid < groups.size(), "CacheConfig::policyForGroup invalid gid=%zu size=%zu", gid, groups.size());
        return groups[gid].policy;
    }

    int groupIdForLayerTag(int layer_id, const std::string& tag) const {
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                "CacheConfig::groupIdForLayerTag invalid layer_id=%d size=%zu",
                                layer_id,
                                layers.size());
        const auto& tag_to_group = layers[static_cast<size_t>(layer_id)].tag_to_gid;
        const auto  it           = tag_to_group.find(tag);
        RTP_LLM_CHECK_WITH_INFO(it != tag_to_group.end(),
                                "CacheConfig::groupIdForLayerTag missing tag=%s for layer_id=%d",
                                tag.c_str(),
                                layer_id);
        return it->second;
    }

    int groupIdFor(int layer_id) const {
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                "CacheConfig::groupIdFor invalid layer_id=%d size=%zu",
                                layer_id,
                                layers.size());
        const auto& gids = layers[static_cast<size_t>(layer_id)].group_ids;
        RTP_LLM_CHECK_WITH_INFO(gids.size() == 1,
                                "CacheConfig::groupIdFor requires exactly one cache tag for layer_id=%d, got %zu",
                                layer_id,
                                gids.size());
        return gids.front();
    }

    const std::vector<int>& groupIdsForLayer(int layer_id) const {
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                "CacheConfig::groupIdsForLayer invalid layer_id=%d size=%zu",
                                layer_id,
                                layers.size());
        const auto& gids = layers[static_cast<size_t>(layer_id)].group_ids;
        RTP_LLM_CHECK_WITH_INFO(!gids.empty(), "CacheConfig::groupIdsForLayer missing layer_id=%d", layer_id);
        return gids;
    }

    static bool samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs);

    void        setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers);
    void        fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                                 const std::vector<std::vector<int>>& layers_by_group,
                                 const std::vector<CacheGroupType>&   types,
                                 const std::vector<std::string>&      tags     = {},
                                 const std::vector<CacheGroupPolicy>& policies = {});
    void        finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config);
    std::string debugString(size_t indent = 0) const;
};

}  // namespace rtp_llm
