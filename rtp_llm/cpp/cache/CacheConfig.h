#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"
#include "rtp_llm/cpp/cache/spec/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/spec/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

class GroupBase {
public:
    int size() const {
        return static_cast<int>(specs_.size());
    }

    const KVCacheSpecPtr& spec(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < specs_.size(), "GroupBase::spec invalid gid=%zu size=%zu", gid, specs_.size());
        return specs_[gid];
    }

    CacheGroupType type(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < policies_.size(), "GroupBase::type invalid gid=%zu size=%zu", gid, policies_.size());
        return policies_[gid].group_type;
    }

    const std::string& tag(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < specs_.size(), "GroupBase::tag invalid gid=%zu size=%zu", gid, specs_.size());
        return specs_[gid]->tag;
    }

    int groupIdForTag(const std::string& tag) const {
        const auto it = tag_to_gid_.find(tag);
        RTP_LLM_CHECK_WITH_INFO(it != tag_to_gid_.end(), "GroupBase::groupIdForTag missing tag=%s", tag.c_str());
        return it->second;
    }

    const CacheGroupPolicy& policy(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(
            gid < policies_.size(), "GroupBase::policy invalid gid=%zu size=%zu", gid, policies_.size());
        return policies_[gid];
    }

    const std::vector<int>& layerIds(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(
            gid < layer_ids_.size(), "GroupBase::layerIds invalid gid=%zu size=%zu", gid, layer_ids_.size());
        return layer_ids_[gid];
    }

    uint32_t blockNum(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(
            gid < block_nums_.size(), "GroupBase::blockNum invalid gid=%zu size=%zu", gid, block_nums_.size());
        return block_nums_[gid];
    }

    size_t kvBlockStrideBytes(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < kv_block_stride_bytes_.size(),
                                "GroupBase::kvBlockStrideBytes invalid gid=%zu size=%zu",
                                gid,
                                kv_block_stride_bytes_.size());
        return kv_block_stride_bytes_[gid];
    }

    size_t kvScaleStrideBytes(size_t gid) const {
        RTP_LLM_CHECK_WITH_INFO(gid < kv_scale_stride_bytes_.size(),
                                "GroupBase::kvScaleStrideBytes invalid gid=%zu size=%zu",
                                gid,
                                kv_scale_stride_bytes_.size());
        return kv_scale_stride_bytes_[gid];
    }

    bool empty() const {
        return specs_.empty();
    }

private:
    std::vector<KVCacheSpecPtr>          specs_;
    std::vector<CacheGroupPolicy>        policies_;
    std::vector<std::vector<int>>        layer_ids_;
    std::vector<uint32_t>                block_nums_;
    std::vector<size_t>                  kv_block_stride_bytes_;
    std::vector<size_t>                  kv_scale_stride_bytes_;
    std::unordered_map<std::string, int> tag_to_gid_;

    friend struct CacheConfig;
};

class LayerBase {
public:
    int groupIdFor(int layer_id) const {
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < group_ids_.size(),
                                "LayerBase::groupIdFor invalid layer_id=%d size=%zu",
                                layer_id,
                                group_ids_.size());
        const auto& gids = group_ids_[static_cast<size_t>(layer_id)];
        RTP_LLM_CHECK_WITH_INFO(gids.size() == 1,
                                "LayerBase::groupIdFor requires exactly one cache tag for layer_id=%d, got %zu",
                                layer_id,
                                gids.size());
        return gids.front();
    }

    int groupIdForTag(int layer_id, const std::string& tag) const {
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < tag_to_gid_.size(),
                                "LayerBase::groupIdForTag invalid layer_id=%d size=%zu",
                                layer_id,
                                tag_to_gid_.size());
        const auto& tag_to_gid = tag_to_gid_[static_cast<size_t>(layer_id)];
        const auto  it         = tag_to_gid.find(tag);
        RTP_LLM_CHECK_WITH_INFO(it != tag_to_gid.end(),
                                "LayerBase::groupIdForTag missing tag=%s for layer_id=%d",
                                tag.c_str(),
                                layer_id);
        return it->second;
    }

    const std::vector<int>& groupIds(int layer_id) const {
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < group_ids_.size(),
                                "LayerBase::groupIds invalid layer_id=%d size=%zu",
                                layer_id,
                                group_ids_.size());
        const auto& gids = group_ids_[static_cast<size_t>(layer_id)];
        RTP_LLM_CHECK_WITH_INFO(!gids.empty(), "LayerBase::groupIds missing layer_id=%d", layer_id);
        return gids;
    }

    bool empty() const {
        return group_ids_.empty();
    }

private:
    std::vector<std::vector<int>>           group_ids_;
    std::vector<std::map<std::string, int>> tag_to_gid_;

    friend struct CacheConfig;
};

struct CacheConfig {
    GroupBase groups;
    LayerBase layers;

    // Cache specification and layer mapping are owned by groups/layers above.
    std::vector<int>               layer_to_block_stride_bytes;
    std::vector<size_t>            group_seq_size_per_block;
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
        return groups.size();
    }

    const KVCacheSpecPtr& specForGroup(size_t gid) const {
        return groups.spec(gid);
    }

    CacheGroupType typeForGroup(size_t gid) const {
        return groups.type(gid);
    }

    const std::string& tagForGroup(size_t gid) const {
        return groups.tag(gid);
    }

    const std::vector<int>& layerIdsForGroup(size_t gid) const {
        return groups.layerIds(gid);
    }

    std::vector<CacheGroupType> groupTypesSnapshot() const {
        std::vector<CacheGroupType> types;
        types.reserve(groups.policies_.size());
        for (const auto& p : groups.policies_) {
            types.push_back(p.group_type);
        }
        return types;
    }

    std::vector<std::string> groupTagsSnapshot() const {
        std::vector<std::string> tags;
        tags.reserve(groups.specs_.size());
        for (const auto& s : groups.specs_) {
            tags.push_back(s->tag);
        }
        return tags;
    }

    std::vector<CacheGroupPolicy> groupPoliciesSnapshot() const {
        return groups.policies_;
    }

    std::vector<uint32_t> groupBlockNumsSnapshot() const {
        return groups.block_nums_;
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
        return groups.kv_block_stride_bytes_;
    }

    std::vector<size_t> groupKvScaleStrideBytesSnapshot() const {
        return groups.kv_scale_stride_bytes_;
    }

    std::vector<std::vector<int>> layerGroupIdsSnapshot() const {
        return layers.group_ids_;
    }

    std::vector<std::map<std::string, int>> layerTagToGroupIdSnapshot() const {
        return layers.tag_to_gid_;
    }

    std::vector<int> primaryLayerGroupIdsSnapshot() const {
        std::vector<int> primary;
        primary.reserve(layers.group_ids_.size());
        for (const auto& gids : layers.group_ids_) {
            primary.push_back(gids.size() == 1 ? gids.front() : -1);
        }
        return primary;
    }

    uint32_t blockNumForGroup(size_t gid) const {
        if (gid < groups.block_nums_.size() && groups.block_nums_[gid] > 0) {
            return groups.blockNum(gid);
        }
        return block_num;
    }

    size_t kvBlockStrideBytesForGroup(size_t gid) const {
        if (gid < groups.kv_block_stride_bytes_.size() && groups.kv_block_stride_bytes_[gid] > 0) {
            return groups.kvBlockStrideBytes(gid);
        }
        return specForGroup(gid)->block_size_bytes();
    }

    size_t kvScaleStrideBytesForGroup(size_t gid) const {
        if (gid < groups.kv_scale_stride_bytes_.size()) {
            return groups.kvScaleStrideBytes(gid);
        }
        return specForGroup(gid)->scale_block_size_bytes();
    }

    size_t blockSizeBytesForGroup(size_t gid) const {
        return layerIdsForGroup(gid).size() * (kvBlockStrideBytesForGroup(gid) + kvScaleStrideBytesForGroup(gid));
    }

    void setGroupPolicies(const std::vector<CacheGroupPolicy>& policies) {
        RTP_LLM_CHECK_WITH_INFO(policies.size() == groups.specs_.size(),
                                "CacheConfig::setGroupPolicies size %zu != group size %zu",
                                policies.size(),
                                groups.specs_.size());
        groups.policies_ = policies;
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
        groups.block_nums_            = block_nums;
        groups.kv_block_stride_bytes_ = kv_block_stride_bytes;
        groups.kv_scale_stride_bytes_ = kv_scale_stride_bytes;
    }

    void resizeLayerRoutes(size_t layer_count) {
        layers.group_ids_.resize(layer_count);
        layers.tag_to_gid_.resize(layer_count);
    }

    void setLayerIdsForGroup(size_t gid, const std::vector<int>& layer_ids) {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.layer_ids_.size(),
                                "CacheConfig::setLayerIdsForGroup invalid gid=%zu size=%zu",
                                gid,
                                groups.layer_ids_.size());
        groups.layer_ids_[gid] = layer_ids;
    }

    void appendLayerToGroup(size_t gid, int layer_id, const std::string& tag) {
        RTP_LLM_CHECK_WITH_INFO(gid < groups.layer_ids_.size(),
                                "CacheConfig::appendLayerToGroup invalid gid=%zu size=%zu",
                                gid,
                                groups.layer_ids_.size());
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0, "CacheConfig::appendLayerToGroup invalid layer_id=%d", layer_id);
        const auto layer = static_cast<size_t>(layer_id);
        if (layer >= layers.group_ids_.size()) {
            layers.group_ids_.resize(layer + 1);
            layers.tag_to_gid_.resize(layer + 1);
        }
        groups.layer_ids_[gid].push_back(layer_id);
        layers.group_ids_[layer].push_back(static_cast<int>(gid));
        if (!tag.empty()) {
            layers.tag_to_gid_[layer][tag] = static_cast<int>(gid);
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
        return groups.policy(gid);
    }

    int groupIdForLayerTag(int layer_id, const std::string& tag) const {
        return layers.groupIdForTag(layer_id, tag);
    }

    int groupIdFor(int layer_id) const {
        return layers.groupIdFor(layer_id);
    }

    const std::vector<int>& groupIdsForLayer(int layer_id) const {
        return layers.groupIds(layer_id);
    }

    static CacheGroupType inferGroupType(const KVCacheSpecPtr& spec) {
        if (spec && spec->lifecycle != CacheGroupType::FULL) {
            return spec->lifecycle;
        }
        return spec->type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
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

    static CacheGroupPolicy policyFromSpecDesc(const KVCacheSpecDesc& desc) {
        CacheGroupPolicy policy = defaultCacheGroupPolicy(SpecBuilder::groupType(desc));
        if (desc.is_state_cache) {
            policy.evict_policy = CacheEvictPolicy::INDEPENDENT;
        }
        if (desc.skip_prefix_reuse) {
            policy.reuse_policy         = CacheReusePolicy::NON_REUSABLE;
            policy.active_tail_blocks   = 1;
            policy.validate_tail_blocks = false;
        }
        if (desc.has_reuse_policy) {
            policy.reuse_policy = desc.reuse_policy;
        }
        if (desc.has_evict_policy) {
            policy.evict_policy = desc.evict_policy;
        }
        if (desc.has_active_tail_blocks) {
            policy.active_tail_blocks = desc.active_tail_blocks;
        }
        if (desc.has_validate_tail_blocks) {
            policy.validate_tail_blocks = desc.validate_tail_blocks;
        }
        policy.explicit_block_num        = desc.extra.explicit_block_num;
        policy.reserve_from_paged_budget = desc.extra.reserve_from_paged_budget;
        if (desc.has_prefix_reusable) {
            policy.prefix_reusable = desc.prefix_reusable;
        }
        policy.uses_pinned_cpu_backing   = desc.uses_pinned_cpu_backing;
        if (desc.has_is_cp_shardable) {
            policy.is_cp_shardable = desc.is_cp_shardable;
        }
        if (desc.has_sparse_slots) {
            policy.has_sparse_slots = desc.sparse_slots;
        }
        if (desc.has_kernel_block_subdiv) {
            policy.has_kernel_block_subdiv = desc.kernel_block_subdiv;
        }
        if (desc.has_cp_compact_tail_blocks) {
            policy.cp_compact_tail_blocks = desc.cp_compact_tail_blocks;
        }
        if (desc.has_is_reservable) {
            policy.is_reservable = desc.is_reservable;
        }
        return policy;
    }

    static CacheGroupPolicy cacheGroupPolicyForDesc(const KVCacheSpecDesc& desc, const KVCacheSpecPtr& /*spec*/) {
        return policyFromSpecDesc(desc);
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

    void fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&             specs,
                          const std::vector<std::vector<int>>&          layers_by_group,
                          const std::vector<CacheGroupType>&            types,
                          const std::vector<std::string>&               tags    = {}) {
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

        groups.specs_.clear();
        groups.policies_.clear();
        groups.layer_ids_.clear();
        groups.block_nums_.clear();
        groups.kv_block_stride_bytes_.clear();
        groups.kv_scale_stride_bytes_.clear();
        groups.tag_to_gid_.clear();

        groups.specs_.reserve(group_num);
        groups.policies_.reserve(group_num);
        groups.layer_ids_.reserve(group_num);

        layers.group_ids_.assign(layer_num, std::vector<int>());
        layers.tag_to_gid_.assign(layer_num, std::map<std::string, int>());

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
            stored_spec->layers = layers_by_group[gid];

            groups.specs_.push_back(stored_spec);
            groups.layer_ids_.push_back(layers_by_group[gid]);
            groups.policies_.push_back(cacheGroupPolicyForSpec(stored_spec, types[gid]));
            groups.tag_to_gid_.emplace(tag, static_cast<int>(gid));

            std::vector<bool> seen_layer(static_cast<size_t>(layer_num), false);
            for (int layer_id : layers_by_group[gid]) {
                RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layer_num,
                                        "CacheConfig::fromGroupedSpecs tag=%s has invalid layer id %d for layer_num=%u",
                                        tag.c_str(),
                                        layer_id,
                                        layer_num);
                const auto layer = static_cast<size_t>(layer_id);
                RTP_LLM_CHECK_WITH_INFO(!seen_layer[layer],
                                        "CacheConfig::fromGroupedSpecs tag=%s has duplicate layer id %d",
                                        tag.c_str(),
                                        layer_id);
                seen_layer[layer] = true;

                layers.group_ids_[layer].push_back(static_cast<int>(gid));
                const auto current_tag_gid = layers.tag_to_gid_[layer].find(tag);
                RTP_LLM_CHECK_WITH_INFO(current_tag_gid == layers.tag_to_gid_[layer].end()
                                            || current_tag_gid->second == static_cast<int>(gid),
                                        "CacheConfig::fromGroupedSpecs layer %d tag %s maps to both group %d and %zu",
                                        layer_id,
                                        tag.c_str(),
                                        current_tag_gid == layers.tag_to_gid_[layer].end() ? -1 :
                                                                                             current_tag_gid->second,
                                        gid);
                layers.tag_to_gid_[layer][tag] = static_cast<int>(gid);
            }
        }

        for (size_t layer = 0; layer < static_cast<size_t>(layer_num); ++layer) {
            RTP_LLM_CHECK_WITH_INFO(!layers.group_ids_[layer].empty(),
                                    "CacheConfig::fromGroupedSpecs missing group mapping for layer %zu",
                                    layer);
        }

    }

    void fromLayerSpecs(const std::map<int64_t, std::vector<KVCacheSpecPtr>>& layer_specs) {
        RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "CacheConfig::fromLayerSpecs requires positive layer_num");
        RTP_LLM_CHECK_WITH_INFO(layer_specs.size() == static_cast<size_t>(layer_num),
                                "CacheConfig::fromLayerSpecs layer map size %zu != layer_num %u",
                                layer_specs.size(),
                                layer_num);

        std::vector<KVCacheSpecPtr> specs;
        std::vector<std::vector<int>> layers_by_group;
        std::vector<CacheGroupType> types;
        std::vector<std::string> tags;
        std::map<std::string, size_t> tag_to_group;
        std::map<std::string, std::string> tag_to_fingerprint;

        for (uint32_t layer_id = 0; layer_id < layer_num; ++layer_id) {
            const auto layer_it = layer_specs.find(static_cast<int64_t>(layer_id));
            RTP_LLM_CHECK_WITH_INFO(layer_it != layer_specs.end(),
                                    "CacheConfig::fromLayerSpecs missing specs for layer %u",
                                    layer_id);
            RTP_LLM_CHECK_WITH_INFO(!layer_it->second.empty(),
                                    "CacheConfig::fromLayerSpecs layer %u has no specs",
                                    layer_id);
            std::map<std::string, bool> layer_seen_tags;
            for (const auto& spec : layer_it->second) {
                RTP_LLM_CHECK_WITH_INFO(spec != nullptr,
                                        "CacheConfig::fromLayerSpecs layer %u has null spec",
                                        layer_id);
                std::string tag = spec->tag;
                if (tag.empty() && layer_it->second.size() == 1) {
                    tag = "default";
                }
                RTP_LLM_CHECK_WITH_INFO(!tag.empty(),
                                        "CacheConfig::fromLayerSpecs layer %u has empty cache spec tag",
                                        layer_id);
                RTP_LLM_CHECK_WITH_INFO(layer_seen_tags.emplace(tag, true).second,
                                        "CacheConfig::fromLayerSpecs layer %u has duplicate tag=%s",
                                        layer_id,
                                        tag.c_str());

                const auto fingerprint = spec->fingerprint();
                auto       fp_it       = tag_to_fingerprint.find(tag);
                if (fp_it == tag_to_fingerprint.end()) {
                    tag_to_fingerprint.emplace(tag, fingerprint);
                } else {
                    RTP_LLM_CHECK_WITH_INFO(fp_it->second == fingerprint,
                                            "CacheConfig::fromLayerSpecs tag=%s has multiple physical prototypes",
                                            tag.c_str());
                }

                auto group_it = tag_to_group.find(tag);
                if (group_it == tag_to_group.end()) {
                    const size_t gid = specs.size();
                    tag_to_group.emplace(tag, gid);
                    specs.push_back(spec);
                    layers_by_group.emplace_back();
                    types.push_back(inferGroupType(spec));
                    tags.push_back(tag);
                    group_it = tag_to_group.find(tag);
                }
                layers_by_group[group_it->second].push_back(static_cast<int>(layer_id));
            }
        }

        fromGroupedSpecs(specs, layers_by_group, types, tags);
    }

    void fromLayerDescs(const std::map<int64_t, std::vector<KVCacheSpecDesc>>& layer_descs,
                        const SpecBuildContext&                                ctx) {
        RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "CacheConfig::fromLayerDescs requires positive layer_num");
        RTP_LLM_CHECK_WITH_INFO(layer_descs.size() == static_cast<size_t>(layer_num),
                                "CacheConfig::fromLayerDescs layer map size %zu != layer_num %u",
                                layer_descs.size(),
                                layer_num);

        struct GroupBuildState {
            KVCacheSpecPtr    spec;
            std::string       fingerprint;
            CacheGroupType    type;
            CacheGroupPolicy  policy;
            std::vector<int>  layers;
            uint64_t          order = 0;
        };
        std::map<std::string, GroupBuildState> group_by_tag;
        std::map<uint64_t, std::string>        explicit_order_to_tag;
        uint64_t next_first_seen_order = 0;
        bool     has_explicit_group_order = false;
        uint64_t max_explicit_group_order = 0;

        for (uint32_t layer = 0; layer < layer_num; ++layer) {
            const auto it = layer_descs.find(static_cast<int64_t>(layer));
            RTP_LLM_CHECK_WITH_INFO(it != layer_descs.end(),
                                    "CacheConfig::fromLayerDescs missing descs for layer %u",
                                    layer);
            RTP_LLM_CHECK_WITH_INFO(!it->second.empty(),
                                    "CacheConfig::fromLayerDescs layer %u has no descs",
                                    layer);
            std::set<std::string> layer_tags;
            for (const auto& desc : it->second) {
                auto spec = SpecBuilder::build(desc, ctx);
                RTP_LLM_CHECK_WITH_INFO(layer_tags.insert(spec->tag).second,
                                        "CacheConfig::fromLayerDescs layer %u has duplicate tag=%s",
                                        layer,
                                        spec->tag.c_str());
                const auto policy = cacheGroupPolicyForDesc(desc, spec);
                const auto type = inferGroupType(spec);
                auto       group_it = group_by_tag.find(spec->tag);
                if (group_it == group_by_tag.end()) {
                    GroupBuildState state;
                    state.spec        = spec;
                    state.fingerprint = spec->fingerprint();
                    state.type        = type;
                    state.policy      = policy;
                    state.order       = desc.has_group_order ? desc.group_order : (UINT64_C(1) << 32) + next_first_seen_order++;
                    if (desc.has_group_order) {
                        has_explicit_group_order = true;
                        max_explicit_group_order = std::max<uint64_t>(max_explicit_group_order, desc.group_order);
                        const auto [order_it, inserted] = explicit_order_to_tag.emplace(desc.group_order, spec->tag);
                        RTP_LLM_CHECK_WITH_INFO(inserted || order_it->second == spec->tag,
                                                "CacheConfig::fromLayerDescs group order %u maps to both tag=%s and tag=%s",
                                                desc.group_order,
                                                order_it->second.c_str(),
                                                spec->tag.c_str());
                    }
                    group_it          = group_by_tag.emplace(spec->tag, std::move(state)).first;
                } else {
                    RTP_LLM_CHECK_WITH_INFO(group_it->second.fingerprint == spec->fingerprint(),
                                            "CacheConfig::fromLayerDescs tag=%s has multiple physical prototypes",
                                            spec->tag.c_str());
                    RTP_LLM_CHECK_WITH_INFO(group_it->second.type == type,
                                            "CacheConfig::fromLayerDescs tag=%s has inconsistent group type",
                                            spec->tag.c_str());
                    RTP_LLM_CHECK_WITH_INFO(samePolicy(group_it->second.policy, policy),
                                            "CacheConfig::fromLayerDescs tag=%s has inconsistent policy",
                                            spec->tag.c_str());
                    if (desc.has_group_order) {
                        RTP_LLM_CHECK_WITH_INFO(group_it->second.order == desc.group_order,
                                                "CacheConfig::fromLayerDescs tag=%s has inconsistent group order",
                                                spec->tag.c_str());
                    }
                }
                group_it->second.layers.push_back(static_cast<int>(layer));
            }
        }

        if (has_explicit_group_order) {
            for (uint64_t order = 0; order <= max_explicit_group_order; ++order) {
                if (explicit_order_to_tag.find(order) != explicit_order_to_tag.end()) {
                    continue;
                }
                KVCacheSpecDesc placeholder;
                placeholder.tag                = "__empty_group_order_" + std::to_string(order);
                placeholder.cache_type         = CacheType::FIXED_STATE;
                placeholder.seq_size_per_block = ctx.seq_size_per_block == 0 ? 1 : ctx.seq_size_per_block;
                placeholder.dtype              = DataType::TYPE_UINT8;
                placeholder.entry_elems        = 1;
                placeholder.entries_per_block  = 1;
                placeholder.store_dtype        = DataType::TYPE_UINT8;
                auto spec                      = SpecBuilder::build(placeholder, ctx);
                GroupBuildState state;
                state.spec        = spec;
                state.fingerprint = spec->fingerprint();
                state.type        = inferGroupType(spec);
                state.policy      = cacheGroupPolicyForDesc(placeholder, spec);
                state.order       = order;
                group_by_tag.emplace(spec->tag, std::move(state));
            }
        }

        std::vector<std::string> ordered_tags;
        ordered_tags.reserve(group_by_tag.size());
        for (const auto& [tag, _] : group_by_tag) {
            ordered_tags.push_back(tag);
        }
        std::sort(ordered_tags.begin(), ordered_tags.end(), [&](const std::string& lhs, const std::string& rhs) {
            const auto lhs_order = group_by_tag.at(lhs).order;
            const auto rhs_order = group_by_tag.at(rhs).order;
            return lhs_order == rhs_order ? lhs < rhs : lhs_order < rhs_order;
        });
        std::vector<KVCacheSpecPtr>    specs;
        std::vector<std::vector<int>>  layers_by_group;
        std::vector<CacheGroupType>    types;
        std::vector<CacheGroupPolicy> policies;
        specs.reserve(ordered_tags.size());
        layers_by_group.reserve(ordered_tags.size());
        types.reserve(ordered_tags.size());
        policies.reserve(ordered_tags.size());
        for (const auto& tag : ordered_tags) {
            const auto& state = group_by_tag.at(tag);
            specs.push_back(state.spec);
            layers_by_group.push_back(state.layers);
            types.push_back(state.type);
            policies.push_back(state.policy);
        }
        fromGroupedSpecs(specs, layers_by_group, types, ordered_tags);
        setGroupPolicies(policies);
    }

    void finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
        (void)runtime_config;
        auto& block_nums = groups.block_nums_;
        if (!use_independent_block_pools || block_nums.empty()) {
            explicitly_sized_pool_reserve_bytes = 0;
            return;
        }

        size_t reserve = 0;
        for (size_t gid = 0; gid < block_nums.size(); ++gid) {
            const auto explicit_independent_blocks = explicitIndependentBlocks(gid);
            const auto rule_blocks = explicit_independent_blocks > 0 ? explicit_independent_blocks : global_block_num;
            block_nums[gid] = rule_blocks;

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

        // Attention-specific configuration section
        os << indent1 << "# Attention Configuration:\n";
        OUTPUT_FIELD(linear_step);
        OUTPUT_FIELD(group_layer_num);
        OUTPUT_FIELD_EXPR("full_group_num",
                          std::count_if(groups.policies_.begin(), groups.policies_.end(),
                                        [](const CacheGroupPolicy& p) { return p.group_type == CacheGroupType::FULL; }));
        OUTPUT_FIELD_EXPR("linear_group_num",
                          std::count_if(groups.policies_.begin(), groups.policies_.end(),
                                        [](const CacheGroupPolicy& p) { return p.group_type == CacheGroupType::LINEAR; }));
        OUTPUT_FIELD_EXPR("swa_group_num",
                          std::count_if(groups.policies_.begin(), groups.policies_.end(),
                                        [](const CacheGroupPolicy& p) { return p.group_type == CacheGroupType::SWA; }));
        OUTPUT_FIELD(use_independent_block_pools);
        OUTPUT_FIELD(use_typed_cache_regions);
        OUTPUT_FIELD(use_opaque_kv_cache_store);
        OUTPUT_FIELD(disable_decode_first_malloc_device_reuse);
        os << indent1 << "group_block_nums=" << rtp_llm::vectorToString(groups.block_nums_) << "\n";
        os << "\n";

        // Cache specification section
        os << indent1 << "# Cache Specifications:\n";
        OUTPUT_FIELD_EXPR("cache_specs.size()", groups.specs_.size());
        for (size_t i = 0; i < groups.specs_.size(); ++i) {
            const auto& spec = groups.specs_[i];
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
        OUTPUT_FIELD_EXPR("layer_ids.size()", groups.layer_ids_.size());
        os << indent1 << "layer_ids=" << rtp_llm::vectorsToString(groups.layer_ids_) << "\n";
        OUTPUT_FIELD_EXPR("group_types.size()", groups.policies_.size());
        os << indent1 << "group_types=[";
        for (size_t i = 0; i < groups.policies_.size(); ++i) {
            os << static_cast<int>(groups.policies_[i].group_type);
            if (i + 1 < groups.policies_.size()) {
                os << ",";
            }
        }
        os << "]\n";
        OUTPUT_FIELD_EXPR("group_tags.size()", groups.specs_.size());
        os << indent1 << "group_tags=[";
        for (size_t i = 0; i < groups.specs_.size(); ++i) {
            os << groups.specs_[i]->tag;
            if (i + 1 < groups.specs_.size()) {
                os << ",";
            }
        }
        os << "]\n";
        OUTPUT_FIELD_EXPR("layer_to_group_ids.size()", layers.group_ids_.size());
        os << indent1 << "layer_to_group_ids=" << rtp_llm::vectorsToString(layers.group_ids_) << "\n";
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
