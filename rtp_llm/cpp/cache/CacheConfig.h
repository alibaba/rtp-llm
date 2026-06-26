#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"
#include "rtp_llm/cpp/cache/spec/KVCacheSpec.h"
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
    size_t           kv_block_stride_bytes = 0;
    size_t           kv_scale_stride_bytes = 0;
};

struct LayerBase {
    std::vector<int>           group_ids;
    std::map<std::string, int> tag_to_gid;
    int                        legacy_single_group_id = -1;
};

struct CacheConfig {
    std::vector<GroupBase>          groups;
    std::vector<LayerBase>          layers;
    std::unordered_map<std::string, int> tag_to_gid;

    // Cache specification and layer mapping are owned by groups/layers above.
    std::vector<int>               layer_to_block_stride_bytes;
    std::vector<size_t>            group_seq_size_per_block;
    bool                           group_block_layout_initialized          = false;
    bool                           use_independent_block_pools              = false;
    bool                           use_typed_cache_regions                  = false;
    bool                           use_opaque_kv_cache_store                = false;
    bool                           disable_decode_first_malloc_device_reuse = false;

    // Model configuration
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
    size_t kv_block_size_bytes  = 0;
    size_t kv_scale_size_bytes  = 0;
    size_t block_size_bytes     = 0;  // (kv + scales together)

    // ---- Per-block strides (one layer) ----
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride_bytes = 0;

    // Bytes pre-reserved for explicitly-sized pools.
    // CacheConfigCreator deducts this from kv_cache_mem_size before computing the
    // paged block_num, so paged pools don't overcommit HBM. 0 means no reservation.
    size_t explicitly_sized_pool_reserve_bytes = 0;

    // Attention-specific configuration
    int linear_step = 1;  // For Linear attention: keep one cache block every `linear_step` blocks
    int group_layer_num  = 1;  // Number of layers per group for hybrid attention

    // mtp-model configurations
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    CacheConfig() {}

    int groupNums() const {
        return static_cast<int>(groups.size());
    }

    const KVCacheSpecPtr& specForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(), "CacheConfig::specForGroup invalid gid=%zu size=%zu", gid, groups.size());
        RTP_LLM_CHECK_WITH_INFO(groups[gid].spec != nullptr, "CacheConfig::specForGroup null spec gid=%zu", gid);
        return groups[gid].spec;
    }

    CacheGroupType typeForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(), "CacheConfig::typeForGroup invalid gid=%zu size=%zu", gid, groups.size());
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
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(), "CacheConfig::layerIdsForGroup invalid gid=%zu size=%zu", gid, groups.size());
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
        result.reserve(layers.size());
        for (const auto& layer : layers) {
            result.push_back(layer.group_ids);
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

    std::vector<int> primaryLayerGroupIdsSnapshot() const {
        std::vector<int> legacy_route;
        legacy_route.reserve(layers.size());
        for (const auto& layer : layers) {
            legacy_route.push_back(layer.legacy_single_group_id);
        }
        return legacy_route;
    }

    uint32_t blockNumForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(), "CacheConfig::blockNumForGroup invalid gid=%zu size=%zu", gid, groups.size());
        if (group_block_layout_initialized && groups[gid].block_num > 0) {
            return groups[gid].block_num;
        }
        return block_num;
    }

    size_t kvBlockStrideBytesForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(), "CacheConfig::kvBlockStrideBytesForGroup invalid gid=%zu size=%zu", gid, groups.size());
        if (group_block_layout_initialized) {
            return groups[gid].kv_block_stride_bytes;
        }
        return specForGroup(gid)->block_size_bytes();
    }

    size_t kvScaleStrideBytesForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(), "CacheConfig::kvScaleStrideBytesForGroup invalid gid=%zu size=%zu", gid, groups.size());
        if (group_block_layout_initialized) {
            return groups[gid].kv_scale_stride_bytes;
        }
        return specForGroup(gid)->scale_block_size_bytes();
    }

    size_t blockSizeBytesForGroup(size_t gid) const {
        return layerIdsForGroup(gid).size() * (kvBlockStrideBytesForGroup(gid) + kvScaleStrideBytesForGroup(gid));
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
        const size_t group_num = static_cast<size_t>(groupNums());
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
        for (auto& layer : layers) {
            layer.legacy_single_group_id = layer.group_ids.size() == 1 ? layer.group_ids.front() : -1;
        }
    }

    void setLayerIdsForGroup(size_t gid, const std::vector<int>& layer_ids) {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(),
                                "CacheConfig::setLayerIdsForGroup invalid gid=%zu size=%zu",
                                gid,
                                groups.size());
        groups[gid].layer_ids = layer_ids;
        if (groups[gid].spec != nullptr) {
            groups[gid].spec->layers = layer_ids;
        }
    }

    void appendLayerToGroup(size_t gid, int layer_id, const std::string& tag) {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(),
                                "CacheConfig::appendLayerToGroup invalid gid=%zu size=%zu",
                                gid,
                                groups.size());
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0, "CacheConfig::appendLayerToGroup invalid layer_id=%d", layer_id);
        const auto layer = static_cast<size_t>(layer_id);
        if (layer >= layers.size()) {
            layers.resize(layer + 1);
        }
        groups[gid].layer_ids.push_back(layer_id);
        if (groups[gid].spec != nullptr) {
            groups[gid].spec->layers = groups[gid].layer_ids;
        }
        layers[layer].group_ids.push_back(static_cast<int>(gid));
        layers[layer].legacy_single_group_id = layers[layer].group_ids.size() == 1 ? layers[layer].group_ids.front() : -1;
        if (!tag.empty()) {
            layers[layer].tag_to_gid[tag] = static_cast<int>(gid);
        }
    }

    size_t fullGroupId() const {
        for (size_t gid = 0; gid < static_cast<size_t>(groupNums()); ++gid) {
            if (typeForGroup(gid) == CacheGroupType::FULL) {
                return gid;
            }
        }
        return 0;
    }

    std::shared_ptr<CacheConfig> mergeMTPModule(const CacheConfig& propose_config,
                                                int                module_index,
                                                uint32_t           main_layer_num) {
        RTP_LLM_CHECK_WITH_INFO(!groups.empty(), "CacheConfig::mergeMTPModule requires destination topology views");
        RTP_LLM_CHECK_WITH_INFO(!propose_config.groups.empty(),
                                "CacheConfig::mergeMTPModule requires propose topology views");
        RTP_LLM_CHECK_WITH_INFO(module_index >= 0, "CacheConfig::mergeMTPModule invalid module_index=%d", module_index);

        auto sub_cfg           = std::make_shared<CacheConfig>(propose_config);
        sub_cfg->block_num     = block_num;
        sub_cfg->layer_all_num = sub_cfg->layer_num;

        const auto mtp_layer_num = propose_config.layer_num;
        const auto total_layers =
            static_cast<size_t>(main_layer_num) + static_cast<size_t>(module_index + 1) * mtp_layer_num;
        resizeLayerRoutes(total_layers);
        if (layer_to_block_stride_bytes.size() < total_layers) {
            layer_to_block_stride_bytes.resize(total_layers, 0);
        }

        const auto fallback_full_gid = fullGroupId();
        const auto target_group_num  = static_cast<size_t>(groupNums());
        const auto propose_group_num = static_cast<size_t>(propose_config.groupNums());
        std::vector<std::vector<int>> sub_global_layer_ids(propose_group_num);

        for (size_t gid = 0; gid < propose_group_num; ++gid) {
            const auto tag = propose_config.tagForGroup(gid);
            auto       target_gid = gid < target_group_num ? gid : fallback_full_gid;
            for (size_t candidate_gid = 0; candidate_gid < target_group_num; ++candidate_gid) {
                if (tagForGroup(candidate_gid) == tag) {
                    target_gid = candidate_gid;
                    break;
                }
            }
            for (int local_layer_id : propose_config.layerIdsForGroup(gid)) {
                if (local_layer_id < 0 || local_layer_id >= static_cast<int>(mtp_layer_num)) {
                    continue;
                }
                const auto global_layer_id = static_cast<int>(main_layer_num)
                                             + module_index * static_cast<int>(mtp_layer_num) + local_layer_id;
                const auto global_layer    = static_cast<size_t>(global_layer_id);
                sub_global_layer_ids[gid].push_back(global_layer_id);

                appendLayerToGroup(target_gid, global_layer_id, tag);

                RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(local_layer_id)
                                            < sub_cfg->layer_to_block_stride_bytes.size(),
                                        "CacheConfig::mergeMTPModule local layer stride missing layer=%d size=%zu",
                                        local_layer_id,
                                        sub_cfg->layer_to_block_stride_bytes.size());
                layer_to_block_stride_bytes[global_layer] =
                    sub_cfg->layer_to_block_stride_bytes[static_cast<size_t>(local_layer_id)];
            }
        }

        for (size_t gid = 0; gid < propose_group_num; ++gid) {
            sub_cfg->setLayerIdsForGroup(gid, sub_global_layer_ids[gid]);
        }
        return sub_cfg;
    }

    uint32_t explicitIndependentBlocks(size_t gid) const {
        return policyForGroup(gid).explicit_block_num;
    }

    bool usesExplicitIndependentBlocks(size_t gid) const {
        return explicitIndependentBlocks(gid) > 0;
    }

    CacheGroupPolicy policyForGroup(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.size(), "CacheConfig::policyForGroup invalid gid=%zu size=%zu", gid, groups.size());
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

    static CacheGroupPolicy cacheGroupPolicyForSpec(const KVCacheSpecPtr& spec, CacheGroupType group_type) {
        CacheGroupPolicy policy = defaultCacheGroupPolicy(group_type);
        if (spec && spec->is_state_cache) {
            policy.evict_policy = CacheEvictPolicy::INDEPENDENT;
        }
        if (spec && spec->skip_prefix_reuse) {
            policy.reuse_policy         = CacheReusePolicy::NON_REUSABLE;
            policy.active_tail_blocks   = 1;
            policy.validate_tail_blocks = false;
        }
        return policy;
    }

    static bool samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
        return lhs.reuse_policy == rhs.reuse_policy && lhs.evict_policy == rhs.evict_policy
               && lhs.active_tail_blocks == rhs.active_tail_blocks
               && lhs.validate_tail_blocks == rhs.validate_tail_blocks
               && lhs.explicit_block_num == rhs.explicit_block_num
               && lhs.reserve_from_paged_budget == rhs.reserve_from_paged_budget
               && lhs.prefix_reusable == rhs.prefix_reusable
               && lhs.uses_pinned_cpu_backing == rhs.uses_pinned_cpu_backing
               && lhs.is_cp_shardable == rhs.is_cp_shardable
               && lhs.has_sparse_slots == rhs.has_sparse_slots
               && lhs.has_kernel_block_subdiv == rhs.has_kernel_block_subdiv
               && lhs.cp_compact_tail_blocks == rhs.cp_compact_tail_blocks
               && lhs.is_reservable == rhs.is_reservable
               && lhs.group_type == rhs.group_type;
    }

    void setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers) {
        RTP_LLM_CHECK_WITH_INFO(!new_groups.empty(), "CacheConfig::setTopology requires at least one cache group");
        RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "CacheConfig::setTopology requires positive layer_num");
        RTP_LLM_CHECK_WITH_INFO(new_layers.size() == static_cast<size_t>(layer_num),
                                "CacheConfig::setTopology layer count %zu != layer_num %u",
                                new_layers.size(),
                                layer_num);

        std::unordered_map<std::string, int> new_tag_to_gid;
        for (size_t gid = 0; gid < new_groups.size(); ++gid) {
            auto& group = new_groups[gid];
            RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheConfig::setTopology got null spec at group %zu", gid);
            RTP_LLM_CHECK_WITH_INFO(!group.spec->tag.empty(),
                                    "CacheConfig::setTopology requires non-empty tag for group %zu",
                                    gid);
            new_tag_to_gid.emplace(group.spec->tag, static_cast<int>(gid));
            group.spec         = group.spec->clone();
            group.spec->layers = group.layer_ids;
        }

        std::vector<std::vector<bool>> group_has_layer(
            new_groups.size(), std::vector<bool>(static_cast<size_t>(layer_num), false));
        for (size_t gid = 0; gid < new_groups.size(); ++gid) {
            for (int layer_id : new_groups[gid].layer_ids) {
                RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < new_layers.size(),
                                        "CacheConfig::setTopology tag=%s has invalid layer id %d for layer_num=%u",
                                        new_groups[gid].spec->tag.c_str(),
                                        layer_id,
                                        layer_num);
                const auto layer_index = static_cast<size_t>(layer_id);
                RTP_LLM_CHECK_WITH_INFO(!group_has_layer[gid][layer_index],
                                        "CacheConfig::setTopology tag=%s has duplicate layer id %d",
                                        new_groups[gid].spec->tag.c_str(),
                                        layer_id);
                group_has_layer[gid][layer_index] = true;
            }
        }

        for (size_t layer_id = 0; layer_id < new_layers.size(); ++layer_id) {
            auto& layer = new_layers[layer_id];
            RTP_LLM_CHECK_WITH_INFO(!layer.group_ids.empty(),
                                    "CacheConfig::setTopology missing group mapping for layer %zu",
                                    layer_id);
            std::map<int, bool> seen_gids;
            for (int gid : layer.group_ids) {
                RTP_LLM_CHECK_WITH_INFO(gid >= 0 && static_cast<size_t>(gid) < new_groups.size(),
                                        "CacheConfig::setTopology layer %zu has invalid gid %d",
                                        layer_id,
                                        gid);
                RTP_LLM_CHECK_WITH_INFO(seen_gids.emplace(gid, true).second,
                                        "CacheConfig::setTopology layer %zu has duplicate gid %d",
                                        layer_id,
                                        gid);
                RTP_LLM_CHECK_WITH_INFO(group_has_layer[static_cast<size_t>(gid)][layer_id],
                                        "CacheConfig::setTopology layer %zu gid %d is missing reverse group layer id",
                                        layer_id,
                                        gid);
            }

            for (const auto& [tag, gid] : layer.tag_to_gid) {
                RTP_LLM_CHECK_WITH_INFO(gid >= 0 && static_cast<size_t>(gid) < new_groups.size(),
                                        "CacheConfig::setTopology layer %zu tag=%s has invalid gid %d",
                                        layer_id,
                                        tag.c_str(),
                                        gid);
                RTP_LLM_CHECK_WITH_INFO(tag == new_groups[static_cast<size_t>(gid)].spec->tag,
                                        "CacheConfig::setTopology layer %zu tag=%s does not match gid %d tag=%s",
                                        layer_id,
                                        tag.c_str(),
                                        gid,
                                        new_groups[static_cast<size_t>(gid)].spec->tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(std::find(layer.group_ids.begin(), layer.group_ids.end(), gid)
                                            != layer.group_ids.end(),
                                        "CacheConfig::setTopology layer %zu tag=%s gid %d is not in layer groups",
                                        layer_id,
                                        tag.c_str(),
                                        gid);
            }

            layer.legacy_single_group_id = layer.group_ids.size() == 1 ? layer.group_ids.front() : -1;
        }

        groups                         = std::move(new_groups);
        layers                         = std::move(new_layers);
        tag_to_gid                     = std::move(new_tag_to_gid);
        group_block_layout_initialized = false;
    }

    void fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&    specs,
                          const std::vector<std::vector<int>>& layers_by_group,
                          const std::vector<CacheGroupType>&   types,
                          const std::vector<std::string>&      tags = {}) {
        const size_t group_num = specs.size();
        RTP_LLM_CHECK_WITH_INFO(group_num > 0, "CacheConfig::fromGroupedSpecs requires at least one cache spec");
        RTP_LLM_CHECK_WITH_INFO(layers_by_group.size() == group_num,
                                "CacheConfig::fromGroupedSpecs layer group count %zu != spec count %zu",
                                layers_by_group.size(),
                                group_num);
        RTP_LLM_CHECK_WITH_INFO(types.size() == group_num,
                                "CacheConfig::fromGroupedSpecs group type count %zu != spec count %zu",
                                types.size(),
                                group_num);
        RTP_LLM_CHECK_WITH_INFO(tags.empty() || tags.size() == group_num,
                                "CacheConfig::fromGroupedSpecs tag count %zu != spec count %zu",
                                tags.size(),
                                group_num);
        RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "CacheConfig::fromGroupedSpecs requires positive layer_num");

        std::vector<GroupBase> new_groups;
        std::vector<LayerBase> new_layers(static_cast<size_t>(layer_num));
        new_groups.reserve(group_num);

        for (size_t gid = 0; gid < group_num; ++gid) {
            const auto& spec = specs[gid];
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheConfig::fromGroupedSpecs got null spec at group %zu", gid);
            std::string tag = tags.empty() ? spec->tag : tags[gid];
            if (tag.empty() && group_num == 1) {
                tag = "default";
            }
            RTP_LLM_CHECK_WITH_INFO(!tag.empty(),
                                    "CacheConfig::fromGroupedSpecs requires non-empty tag for cache spec %zu",
                                    gid);
            auto stored_spec = spec->clone();
            stored_spec->tag = tag;

            GroupBase group;
            group.spec      = stored_spec;
            group.policy    = cacheGroupPolicyForSpec(stored_spec, types[gid]);
            group.layer_ids = layers_by_group[gid];
            new_groups.push_back(group);

            for (int layer_id : layers_by_group[gid]) {
                RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < new_layers.size(),
                                        "CacheConfig::fromGroupedSpecs tag=%s has invalid layer id %d for layer_num=%u",
                                        tag.c_str(),
                                        layer_id,
                                        layer_num);
                auto& layer = new_layers[static_cast<size_t>(layer_id)];
                layer.group_ids.push_back(static_cast<int>(gid));
                const auto [it, inserted] = layer.tag_to_gid.emplace(tag, static_cast<int>(gid));
                RTP_LLM_CHECK_WITH_INFO(inserted || it->second == static_cast<int>(gid),
                                        "CacheConfig::fromGroupedSpecs layer %d tag %s maps to both group %d and %zu",
                                        layer_id,
                                        tag.c_str(),
                                        inserted ? static_cast<int>(gid) : it->second,
                                        gid);
            }
        }

        setTopology(std::move(new_groups), std::move(new_layers));
    }

    void finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
        (void)runtime_config;
        if (!use_independent_block_pools || !group_block_layout_initialized || groups.empty()) {
            explicitly_sized_pool_reserve_bytes = 0;
            return;
        }

        size_t reserve = 0;
        for (size_t gid = 0; gid < groups.size(); ++gid) {
            const auto explicit_independent_blocks = explicitIndependentBlocks(gid);
            const auto rule_blocks = explicit_independent_blocks > 0 ? explicit_independent_blocks : global_block_num;
            groups[gid].block_num = rule_blocks;

            // Explicit independent pools are allocated outside the paged pool budget.
            if (explicit_independent_blocks > 0) {
                reserve += static_cast<size_t>(rule_blocks) * blockSizeBytesForGroup(gid);
            }
        }
        explicitly_sized_pool_reserve_bytes = reserve;
    }

    std::string debugString(size_t indent = 0) const {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";
        const std::string indent2    = indent_str + "    ";

        std::ostringstream os;
        os << indent_str << "CacheConfig{\n";

// Macro to simplify repeated field output and eliminate duplicate field names
#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"

// Helper macro for complex expressions
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

        // Model configuration section
        os << indent1 << "# Model Configuration:\n";
        OUTPUT_FIELD_EXPR("dtype", static_cast<int>(dtype));
        OUTPUT_FIELD(layer_num);
        OUTPUT_FIELD(layer_all_num);
        OUTPUT_FIELD_EXPR("use_mla", (use_mla ? "true" : "false"));
        os << "\n";

        // Block configuration section
        os << indent1 << "# Block Configuration:\n";
        OUTPUT_FIELD(block_num);
        OUTPUT_FIELD(seq_size_per_block);
        OUTPUT_FIELD(kernel_seq_size_per_block);
        os << "\n";

        // Block sizing information section
        os << indent1 << "# Block Sizing Information:\n";
        OUTPUT_FIELD(kv_block_size_bytes);
        OUTPUT_FIELD(kv_scale_size_bytes);
        OUTPUT_FIELD(block_size_bytes);
        OUTPUT_FIELD(kv_block_stride_bytes);
        OUTPUT_FIELD(kv_scale_stride_bytes);
        os << "\n";

        const auto group_policies      = groupPoliciesSnapshot();
        const auto group_block_nums    = groupBlockNumsSnapshot();
        const auto group_layer_ids     = layerGroupIdsSnapshot();
        const auto group_tags          = groupTagsSnapshot();
        std::vector<std::vector<int>> layers_by_group;
        layers_by_group.reserve(groups.size());
        for (const auto& group : groups) {
            layers_by_group.push_back(group.layer_ids);
        }

        // Attention-specific configuration section
        os << indent1 << "# Attention Configuration:\n";
        OUTPUT_FIELD(linear_step);
        OUTPUT_FIELD(group_layer_num);
        OUTPUT_FIELD_EXPR("full_group_num",
                          std::count_if(group_policies.begin(), group_policies.end(),
                                        [](const CacheGroupPolicy& p) { return p.group_type == CacheGroupType::FULL; }));
        OUTPUT_FIELD_EXPR("linear_group_num",
                          std::count_if(group_policies.begin(), group_policies.end(),
                                        [](const CacheGroupPolicy& p) { return p.group_type == CacheGroupType::LINEAR; }));
        OUTPUT_FIELD_EXPR("swa_group_num",
                          std::count_if(group_policies.begin(), group_policies.end(),
                                        [](const CacheGroupPolicy& p) { return p.group_type == CacheGroupType::SWA; }));
        OUTPUT_FIELD(use_independent_block_pools);
        OUTPUT_FIELD(use_typed_cache_regions);
        OUTPUT_FIELD(use_opaque_kv_cache_store);
        OUTPUT_FIELD(disable_decode_first_malloc_device_reuse);
        os << indent1 << "group_block_nums=" << rtp_llm::vectorToString(group_block_nums) << "\n";
        os << "\n";

        // Cache specification section
        os << indent1 << "# Cache Specifications:\n";
        OUTPUT_FIELD_EXPR("cache_specs.size()", groups.size());
        for (size_t i = 0; i < groups.size(); ++i) {
            const auto& spec = groups[i].spec;
            if (!spec) {
                os << indent1 << "cache_specs[" << i << "]=null\n";
                continue;
            }

            os << indent1 << "cache_specs[" << i << "] {\n";
            os << spec->debugString(indent + 2);
            os << indent1 << "}\n";
        }
        os << "\n";

        // Layer mapping section
        os << indent1 << "# Layer Mapping:\n";
        OUTPUT_FIELD_EXPR("layer_ids.size()", layers_by_group.size());
        os << indent1 << "layer_ids=" << rtp_llm::vectorsToString(layers_by_group) << "\n";
        OUTPUT_FIELD_EXPR("group_types.size()", group_policies.size());
        os << indent1 << "group_types=[";
        for (size_t i = 0; i < group_policies.size(); ++i) {
            os << static_cast<int>(group_policies[i].group_type);
            if (i + 1 < group_policies.size()) {
                os << ",";
            }
        }
        os << "]\n";
        OUTPUT_FIELD_EXPR("group_tags.size()", group_tags.size());
        os << indent1 << "group_tags=[";
        for (size_t i = 0; i < group_tags.size(); ++i) {
            os << group_tags[i];
            if (i + 1 < group_tags.size()) {
                os << ",";
            }
        }
        os << "]\n";
        OUTPUT_FIELD_EXPR("layer_to_group_ids.size()", group_layer_ids.size());
        os << indent1 << "layer_to_group_ids=" << rtp_llm::vectorsToString(group_layer_ids) << "\n";
        os << "\n";

        // mtp configurations section
        os << indent1 << "# MTP Configurations:\n";
        OUTPUT_FIELD_EXPR("mtp_sub_configs.size()", mtp_sub_configs.size());
        for (size_t i = 0; i < mtp_sub_configs.size(); ++i) {
            const auto& sub = mtp_sub_configs[i];
            if (!sub) {
                os << indent1 << "mtp_sub_configs[" << i << "]=null\n";
                continue;
            }
            os << indent1 << "mtp_sub_configs[" << i << "]:\n";
            os << sub->debugString(indent + 4);
        }
        os << "\n";

#undef OUTPUT_FIELD
#undef OUTPUT_FIELD_EXPR

        os << indent_str << "}\n";
        return os.str();
    }
};

}  // namespace rtp_llm
