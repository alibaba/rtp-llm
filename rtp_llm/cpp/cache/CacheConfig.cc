#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace rtp_llm {

namespace {

CacheGroupType groupTypeForSpec(const KVCacheSpec& spec) {
    return spec.type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
}

}  // namespace

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.group_type == rhs.group_type && lhs.enable_prefix_reuse == rhs.enable_prefix_reuse
           && lhs.evict_policy == rhs.evict_policy && lhs.reservable == rhs.reservable
           && lhs.explicit_block_num == rhs.explicit_block_num
           && lhs.charge_to_paged_budget == rhs.charge_to_paged_budget && lhs.memory_placement == rhs.memory_placement
           && lhs.active_tail_blocks == rhs.active_tail_blocks && lhs.validate_tail_blocks == rhs.validate_tail_blocks
           && lhs.cp_mapping == rhs.cp_mapping && lhs.cp_slice == rhs.cp_slice;
}

void CacheConfig::setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers) {
    RTP_LLM_CHECK_WITH_INFO(!new_groups.empty(), "CacheConfig::setTopology requires at least one cache group");
    RTP_LLM_CHECK_WITH_INFO(!new_layers.empty(), "CacheConfig::setTopology requires at least one cache layer");
    const auto expected_layers = layer_all_num > 0 ? layer_all_num : layer_num;
    RTP_LLM_CHECK_WITH_INFO(expected_layers == 0 || new_layers.size() == static_cast<size_t>(expected_layers),
                            "CacheConfig::setTopology layer count %zu != expected %u",
                            new_layers.size(),
                            expected_layers);

    for (size_t gid = 0; gid < new_groups.size(); ++gid) {
        auto& group = new_groups[gid];
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheConfig::setTopology got null spec at group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheConfig::setTopology requires tag for group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(group.spec->tag == group.tag,
                                "CacheConfig::setTopology tag=%s does not match spec tag=%s",
                                group.tag.c_str(),
                                group.spec->tag.c_str());

        const auto expected_group_type = groupTypeForSpec(*group.spec);
        RTP_LLM_CHECK_WITH_INFO(expected_group_type != CacheGroupType::LINEAR
                                    || group.policy.group_type == CacheGroupType::LINEAR,
                                "CacheConfig::setTopology group %zu tag=%s policy type %s does not match spec type %d",
                                gid,
                                group.tag.c_str(),
                                cacheGroupTypeName(group.policy.group_type),
                                static_cast<int>(group.spec->type));

        group.spec = group.spec->clone();
        if (group.block_num == 0) {
            group.block_num = block_num;
        }
        if (group.seq_size_per_block == 0) {
            group.seq_size_per_block = group.spec->seq_size_per_block > 0 ? group.spec->seq_size_per_block :
                                                                            std::max<size_t>(1, seq_size_per_block);
        }
        if (group.kernel_seq_size_per_block == 0) {
            group.kernel_seq_size_per_block =
                group.policy.group_type == CacheGroupType::FULL && kernel_seq_size_per_block > 0 ?
                    std::min(kernel_seq_size_per_block, group.seq_size_per_block) :
                    group.seq_size_per_block;
        }
        if (group.kv_block_stride_bytes == 0) {
            group.kv_block_stride_bytes = group.spec->block_size_bytes();
        }
        if (group.kv_scale_stride_bytes == 0) {
            group.kv_scale_stride_bytes = group.spec->scale_block_size_bytes();
        }
    }

    cache_topology = CacheTopology::create(std::move(new_groups), std::move(new_layers));
}

void CacheConfig::setGroupPolicies(const std::vector<CacheGroupPolicy>& policies) {
    RTP_LLM_CHECK_WITH_INFO(policies.size() == topology().groups().size(),
                            "CacheConfig::setGroupPolicies size %zu != group size %zu",
                            policies.size(),
                            topology().groups().size());
    auto groups = topology().groups();
    for (size_t gid = 0; gid < policies.size(); ++gid) {
        groups[gid].policy                    = policies[gid];
        groups[gid].kernel_seq_size_per_block = 0;
    }
    setTopology(std::move(groups), topology().layers());
}

void CacheConfig::setGroupBlockLayout(const std::vector<uint32_t>& block_nums,
                                      const std::vector<size_t>&   kv_block_stride_bytes,
                                      const std::vector<size_t>&   kv_scale_stride_bytes) {
    const size_t group_num = topology().groups().size();
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
    auto groups = topology().groups();
    for (size_t gid = 0; gid < group_num; ++gid) {
        groups[gid].block_num             = block_nums[gid];
        groups[gid].kv_block_stride_bytes = kv_block_stride_bytes[gid];
        groups[gid].kv_scale_stride_bytes = kv_scale_stride_bytes[gid];
    }
    group_block_layout_initialized = true;
    setTopology(std::move(groups), topology().layers());
}

std::shared_ptr<CacheConfig>
CacheConfig::mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(groupNums() > 0, "CacheConfig::mergeMTPModule requires destination topology");
    RTP_LLM_CHECK_WITH_INFO(propose_config.groupNums() > 0, "CacheConfig::mergeMTPModule requires propose topology");
    RTP_LLM_CHECK_WITH_INFO(module_index >= 0, "CacheConfig::mergeMTPModule invalid module_index=%d", module_index);

    auto sub_cfg           = std::make_shared<CacheConfig>(propose_config);
    sub_cfg->block_num     = block_num;
    sub_cfg->layer_all_num = sub_cfg->layer_num;

    const auto mtp_layer_num = propose_config.layer_num;
    const auto total_layers =
        static_cast<size_t>(main_layer_num) + static_cast<size_t>(module_index + 1) * mtp_layer_num;
    auto target_groups = topology().groups();
    auto target_layers = topology().layers();
    target_layers.resize(total_layers);
    for (size_t layer_id = 0; layer_id < target_layers.size(); ++layer_id) {
        target_layers[layer_id].layer_id = static_cast<int>(layer_id);
    }
    if (layer_to_block_stride_bytes.size() < total_layers) {
        layer_to_block_stride_bytes.resize(total_layers, 0);
    }

    const auto                              target_group_num = target_groups.size();
    std::unordered_map<std::string, size_t> propose_gid_by_tag;
    for (size_t gid = 0; gid < propose_config.topology().groups().size(); ++gid) {
        propose_gid_by_tag.emplace(propose_config.tagForGroup(gid), gid);
    }

    const auto          invalid_gid = std::numeric_limits<size_t>::max();
    std::vector<size_t> propose_gid_for_target(target_group_num, invalid_gid);
    std::vector<bool>   propose_gid_used(propose_config.topology().groups().size(), false);

    // Tags identify cache groups within one model, but independently trained
    // target and draft models are not required to use the same tag names.
    // Preserve exact matches first, then align an unmatched draft group only
    // when its cache role has one unambiguous target group.
    for (size_t target_gid = 0; target_gid < target_group_num; ++target_gid) {
        const auto propose_it = propose_gid_by_tag.find(tagForGroup(target_gid));
        if (propose_it == propose_gid_by_tag.end()) {
            continue;
        }
        propose_gid_for_target[target_gid]   = propose_it->second;
        propose_gid_used[propose_it->second] = true;
    }
    for (size_t propose_gid = 0; propose_gid < propose_config.topology().groups().size(); ++propose_gid) {
        if (propose_gid_used[propose_gid]
            || propose_config.topology().groupById(propose_gid).layer_ids.empty()) {
            continue;
        }

        const auto propose_type       = propose_config.typeForGroup(propose_gid);
        size_t     matched_target_gid = invalid_gid;
        size_t     candidate_count    = 0;
        for (size_t target_gid = 0; target_gid < target_group_num; ++target_gid) {
            if (propose_gid_for_target[target_gid] != invalid_gid || typeForGroup(target_gid) != propose_type) {
                continue;
            }
            matched_target_gid = target_gid;
            ++candidate_count;
        }

        RTP_LLM_CHECK_WITH_INFO(
            candidate_count == 1,
            "CacheConfig::mergeMTPModule cannot align propose tag=%s type=%s: found %zu unmatched target groups",
            propose_config.tagForGroup(propose_gid).c_str(),
            cacheGroupTypeName(propose_type),
            candidate_count);
        propose_gid_for_target[matched_target_gid] = propose_gid;
        propose_gid_used[propose_gid]              = true;
        RTP_LLM_LOG_INFO(
            "CacheConfig::mergeMTPModule aligned propose tag=%s gid=%zu to target tag=%s gid=%zu by type=%s",
            propose_config.tagForGroup(propose_gid).c_str(),
            propose_gid,
            tagForGroup(matched_target_gid).c_str(),
            matched_target_gid,
            cacheGroupTypeName(propose_type));
    }

    std::vector<GroupBase> sub_groups;
    std::vector<LayerBase> sub_layers(static_cast<size_t>(mtp_layer_num));
    sub_groups.reserve(target_group_num);

    for (size_t target_gid = 0; target_gid < target_group_num; ++target_gid) {
        const auto&  target_tag        = tagForGroup(target_gid);
        const auto   propose_gid       = propose_gid_for_target[target_gid];
        const bool   has_propose_group = propose_gid != invalid_gid;
        const auto&  source_config     = has_propose_group ? propose_config : *this;
        const size_t source_gid        = has_propose_group ? propose_gid : target_gid;
        const auto&  source_group      = source_config.topology().groupById(source_gid);
        const auto&  source_tag        = source_config.tagForGroup(source_gid);

        if (has_propose_group) {
            RTP_LLM_CHECK_WITH_INFO(
                source_group.policy.group_type == target_groups[target_gid].policy.group_type,
                "CacheConfig::mergeMTPModule target tag=%s type=%s does not match propose tag=%s type=%s",
                target_tag.c_str(),
                cacheGroupTypeName(target_groups[target_gid].policy.group_type),
                source_tag.c_str(),
                cacheGroupTypeName(source_group.policy.group_type));
            RTP_LLM_CHECK_WITH_INFO(
                source_group.layer_ids.size() == static_cast<size_t>(mtp_layer_num),
                "CacheConfig::mergeMTPModule tag=%s must cover every module layer, got=%zu expected=%u",
                source_tag.c_str(),
                source_group.layer_ids.size(),
                mtp_layer_num);
            for (size_t local_layer_id = 0; local_layer_id < source_group.layer_ids.size(); ++local_layer_id) {
                RTP_LLM_CHECK_WITH_INFO(
                    source_group.layer_ids[local_layer_id] == static_cast<int>(local_layer_id),
                    "CacheConfig::mergeMTPModule tag=%s source layers must be ordered 0..%u, index=%zu value=%d",
                    source_tag.c_str(),
                    mtp_layer_num - 1,
                    local_layer_id,
                    source_group.layer_ids[local_layer_id]);
            }

            const size_t expected_existing_layers =
                static_cast<size_t>(group_layer_num) + static_cast<size_t>(module_index) * mtp_layer_num;
            RTP_LLM_CHECK_WITH_INFO(target_groups[target_gid].layer_ids.size() == expected_existing_layers,
                                    "CacheConfig::mergeMTPModule tag=%s gid=%zu physical group alignment mismatch: "
                                    "existing_layers=%zu expected=%zu module=%d group_layer_num=%d module_layers=%u",
                                    target_tag.c_str(),
                                    target_gid,
                                    target_groups[target_gid].layer_ids.size(),
                                    expected_existing_layers,
                                    module_index,
                                    group_layer_num,
                                    mtp_layer_num);
        }

        GroupBase sub_group = source_group;
        sub_group.layer_ids.clear();

        if (!has_propose_group) {
            sub_groups.push_back(std::move(sub_group));
            continue;
        }

        for (int local_layer_id : propose_config.layerIdsForGroup(source_gid)) {
            if (local_layer_id < 0 || local_layer_id >= static_cast<int>(mtp_layer_num)) {
                continue;
            }
            const auto global_layer_id = mtpGlobalLayerId(main_layer_num, module_index, mtp_layer_num, local_layer_id);
            RTP_LLM_CHECK_WITH_INFO(global_layer_id != std::numeric_limits<uint32_t>::max(),
                                    "CacheConfig::mergeMTPModule invalid global layer: main=%u module=%d "
                                    "module_layers=%u local=%d",
                                    main_layer_num,
                                    module_index,
                                    mtp_layer_num,
                                    local_layer_id);
            const auto global_layer = static_cast<size_t>(global_layer_id);

            sub_group.layer_ids.push_back(local_layer_id);
            auto& sub_layer    = sub_layers[static_cast<size_t>(local_layer_id)];
            sub_layer.layer_id = local_layer_id;
            sub_layer.group_tags.push_back(source_tag);

            target_groups[target_gid].layer_ids.push_back(static_cast<int>(global_layer_id));
            target_layers[global_layer].group_tags.push_back(target_tag);

            RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(local_layer_id) < sub_cfg->layer_to_block_stride_bytes.size(),
                                    "CacheConfig::mergeMTPModule local layer stride missing layer=%d size=%zu",
                                    local_layer_id,
                                    sub_cfg->layer_to_block_stride_bytes.size());
            layer_to_block_stride_bytes[global_layer] =
                sub_cfg->layer_to_block_stride_bytes[static_cast<size_t>(local_layer_id)];
        }

        sub_groups.push_back(std::move(sub_group));
    }

    RTP_LLM_CHECK_WITH_INFO(sub_groups.size() == target_group_num,
                            "CacheConfig::mergeMTPModule sub group count %zu != target group count %zu",
                            sub_groups.size(),
                            target_group_num);
    for (size_t layer_id = 0; layer_id < sub_layers.size(); ++layer_id) {
        RTP_LLM_CHECK_WITH_INFO(!sub_layers[layer_id].group_tags.empty(),
                                "CacheConfig::mergeMTPModule missing group mapping for sub layer %zu",
                                layer_id);
    }

    sub_cfg->group_block_layout_initialized = group_block_layout_initialized;
    sub_cfg->setTopology(std::move(sub_groups), std::move(sub_layers));
    layer_all_num = static_cast<uint32_t>(total_layers);
    setTopology(std::move(target_groups), std::move(target_layers));
    return sub_cfg;
}

void CacheConfig::fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                                   const std::vector<std::vector<int>>& layers_by_group,
                                   const std::vector<CacheGroupType>&   types,
                                   const std::vector<std::string>&      tags,
                                   const std::vector<CacheGroupPolicy>& policies) {
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
    RTP_LLM_CHECK_WITH_INFO(policies.empty() || policies.size() == group_num,
                            "CacheConfig::fromGroupedSpecs policy count %zu != spec count %zu",
                            policies.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "CacheConfig::fromGroupedSpecs requires positive layer_num");

    const bool             has_explicit_policies = !policies.empty();
    std::vector<GroupBase> new_groups;
    std::vector<LayerBase> new_layers(static_cast<size_t>(layer_num));
    new_groups.reserve(group_num);
    for (size_t layer_id = 0; layer_id < new_layers.size(); ++layer_id) {
        new_layers[layer_id].layer_id = static_cast<int>(layer_id);
    }

    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto& spec = specs[gid];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheConfig::fromGroupedSpecs got null spec at group %zu", gid);
        std::string tag = tags.empty() ? spec->tag : tags[gid];
        if (tag.empty() && group_num == 1) {
            tag = "default";
        }
        RTP_LLM_CHECK_WITH_INFO(
            !tag.empty(), "CacheConfig::fromGroupedSpecs requires non-empty tag for cache spec %zu", gid);
        auto stored_spec = spec->clone();
        stored_spec->tag = tag;

        GroupBase group;
        group.tag    = tag;
        group.spec   = stored_spec;
        group.policy = has_explicit_policies ? policies[gid] : defaultCacheGroupPolicy(types[gid]);
        RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == types[gid],
                                "CacheConfig::fromGroupedSpecs policy type mismatch gid=%zu policy=%d type=%d",
                                gid,
                                static_cast<int>(group.policy.group_type),
                                static_cast<int>(types[gid]));
        group.layer_ids = layers_by_group[gid];
        new_groups.push_back(group);

        for (int layer_id : layers_by_group[gid]) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < new_layers.size(),
                                    "CacheConfig::fromGroupedSpecs tag=%s has invalid layer id %d for layer_num=%u",
                                    tag.c_str(),
                                    layer_id,
                                    layer_num);
            auto& layer = new_layers[static_cast<size_t>(layer_id)];
            layer.group_tags.push_back(tag);
        }
    }

    group_block_layout_initialized = false;
    setTopology(std::move(new_groups), std::move(new_layers));
}

void CacheConfig::finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
    // TODO: use RuntimeConfig when group-level block sizing needs runtime parallelism context.
    (void)runtime_config;
    if (global_block_num > 0) {
        block_num = global_block_num;
        for (auto& sub_cfg : mtp_sub_configs) {
            if (sub_cfg != nullptr) {
                sub_cfg->finalizeBlockNums(global_block_num, runtime_config);
            }
        }
    }

    if (!use_independent_block_pools || !group_block_layout_initialized || groupNums() == 0) {
        explicitly_sized_pool_reserve_bytes = 0;
        if (groupNums() > 0) {
            auto groups = topology().groups();
            for (auto& group : groups) {
                group.block_num = global_block_num;
            }
            setTopology(std::move(groups), topology().layers());
        }
        return;
    }

    size_t     reserve = 0;
    const auto step    = static_cast<uint32_t>(std::max(1, linear_step));
    auto       groups  = topology().groups();
    for (size_t gid = 0; gid < groups.size(); ++gid) {
        const auto explicit_independent_blocks = groups[gid].policy.explicit_block_num;
        uint32_t   rule_blocks                 = global_block_num;
        if (explicit_independent_blocks > 0) {
            rule_blocks = explicit_independent_blocks;
        } else if (groups[gid].policy.group_type == CacheGroupType::SWA) {
            rule_blocks = global_block_num / step + (global_block_num % step != 0 ? 1u : 0u);
        }
        groups[gid].block_num = rule_blocks;

        // Only groups that opt in reserve paged-pool budget for explicit blocks.
        if (explicit_independent_blocks > 0 && groups[gid].policy.charge_to_paged_budget) {
            reserve += static_cast<size_t>(rule_blocks) * groups[gid].layer_ids.size()
                       * (groups[gid].kv_block_stride_bytes + groups[gid].kv_scale_stride_bytes);
        }
    }
    explicitly_sized_pool_reserve_bytes = reserve;
    setTopology(std::move(groups), topology().layers());
}

std::string CacheConfig::debugString(size_t indent) const {
    const std::string indent_str = std::string(indent, ' ');
    const std::string indent1    = indent_str + "  ";

    std::ostringstream os;
    os << indent_str << "CacheConfig{\n";

#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

    os << indent1 << "# Model Configuration:\n";
    OUTPUT_FIELD_EXPR("dtype", static_cast<int>(dtype));
    OUTPUT_FIELD(layer_num);
    OUTPUT_FIELD(layer_all_num);
    OUTPUT_FIELD_EXPR("use_mla", (use_mla ? "true" : "false"));
    os << "\n";

    os << indent1 << "# Block Configuration:\n";
    OUTPUT_FIELD(block_num);
    OUTPUT_FIELD(seq_size_per_block);
    OUTPUT_FIELD(kernel_seq_size_per_block);
    os << "\n";

    os << indent1 << "# Block Sizing Information:\n";
    OUTPUT_FIELD(kv_block_size_bytes);
    OUTPUT_FIELD(kv_scale_size_bytes);
    OUTPUT_FIELD(block_size_bytes);
    OUTPUT_FIELD(kv_block_stride_bytes);
    OUTPUT_FIELD(kv_scale_stride_bytes);
    os << "\n";

    const auto                    group_policies   = groupPoliciesSnapshot();
    const auto                    group_block_nums = groupBlockNumsSnapshot();
    const auto                    group_layer_ids  = layerGroupIdsSnapshot();
    const auto                    group_tags       = groupTagsSnapshot();
    const auto&                   topology_groups  = topology().groups();
    std::vector<std::vector<int>> layers_by_group;
    layers_by_group.reserve(topology_groups.size());
    for (const auto& group : topology_groups) {
        layers_by_group.push_back(group.layer_ids);
    }

    os << indent1 << "# Attention Configuration:\n";
    OUTPUT_FIELD(linear_step);
    OUTPUT_FIELD(group_layer_num);
    OUTPUT_FIELD_EXPR("full_group_num",
                      std::count_if(group_policies.begin(), group_policies.end(), [](const CacheGroupPolicy& p) {
                          return p.group_type == CacheGroupType::FULL;
                      }));
    OUTPUT_FIELD_EXPR("linear_group_num",
                      std::count_if(group_policies.begin(), group_policies.end(), [](const CacheGroupPolicy& p) {
                          return p.group_type == CacheGroupType::LINEAR;
                      }));
    os << indent1 << "group_block_nums=" << rtp_llm::vectorToString(group_block_nums) << "\n";
    os << "\n";

    os << indent1 << "# Cache Specifications:\n";
    OUTPUT_FIELD_EXPR("groups.size()", topology_groups.size());
    for (size_t i = 0; i < topology_groups.size(); ++i) {
        const auto& spec = topology_groups[i].spec;
        if (!spec) {
            os << indent1 << "groups[" << i << "].spec=null\n";
            continue;
        }

        os << indent1 << "groups[" << i << "] {\n";
        os << spec->debugString(indent + 2);
        os << indent1 << "}\n";
    }
    os << "\n";

    os << indent1 << "# Layer Mapping:\n";
    OUTPUT_FIELD_EXPR("layers_by_group.size()", layers_by_group.size());
    os << indent1 << "layers_by_group=" << rtp_llm::vectorsToString(layers_by_group) << "\n";
    OUTPUT_FIELD_EXPR("group_policies.size()", group_policies.size());
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

}  // namespace rtp_llm
