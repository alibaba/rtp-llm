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

    for (size_t group_index = 0; group_index < new_groups.size(); ++group_index) {
        auto& group = new_groups[group_index];
        RTP_LLM_CHECK_WITH_INFO(
            group.spec != nullptr, "CacheConfig::setTopology got null spec at group %zu", group_index);
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheConfig::setTopology requires tag for group %zu", group_index);
        RTP_LLM_CHECK_WITH_INFO(group.spec->tag == group.tag,
                                "CacheConfig::setTopology tag=%s does not match spec tag=%s",
                                group.tag.c_str(),
                                group.spec->tag.c_str());

        const auto expected_group_type = groupTypeForSpec(*group.spec);
        RTP_LLM_CHECK_WITH_INFO(expected_group_type != CacheGroupType::LINEAR
                                    || group.policy.group_type == CacheGroupType::LINEAR,
                                "CacheConfig::setTopology group %zu tag=%s policy type %s does not match spec type %d",
                                group_index,
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
    for (size_t group_index = 0; group_index < policies.size(); ++group_index) {
        groups[group_index].policy                    = policies[group_index];
        groups[group_index].kernel_seq_size_per_block = 0;
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
    for (size_t group_index = 0; group_index < group_num; ++group_index) {
        groups[group_index].block_num             = block_nums[group_index];
        groups[group_index].kv_block_stride_bytes = kv_block_stride_bytes[group_index];
        groups[group_index].kv_scale_stride_bytes = kv_scale_stride_bytes[group_index];
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

    const auto                      target_group_num = target_groups.size();
    std::unordered_set<std::string> propose_tags;
    for (const auto& group : propose_config.topology().groups()) {
        propose_tags.emplace(group.tag);
    }

    std::vector<GroupBase> sub_groups;
    std::vector<LayerBase> sub_layers(static_cast<size_t>(mtp_layer_num));
    sub_groups.reserve(target_group_num);

    for (size_t target_group_index = 0; target_group_index < target_group_num; ++target_group_index) {
        const auto& tag               = target_groups[target_group_index].tag;
        const bool  has_propose_group = propose_tags.find(tag) != propose_tags.end();
        const auto& source_group      = has_propose_group ? propose_config.group(tag) : group(tag);

        if (has_propose_group) {
            RTP_LLM_CHECK_WITH_INFO(
                source_group.layer_ids.size() == static_cast<size_t>(mtp_layer_num),
                "CacheConfig::mergeMTPModule tag=%s must cover every module layer, got=%zu expected=%u",
                tag.c_str(),
                source_group.layer_ids.size(),
                mtp_layer_num);
            for (size_t local_layer_id = 0; local_layer_id < source_group.layer_ids.size(); ++local_layer_id) {
                RTP_LLM_CHECK_WITH_INFO(
                    source_group.layer_ids[local_layer_id] == static_cast<int>(local_layer_id),
                    "CacheConfig::mergeMTPModule tag=%s source layers must be ordered 0..%u, index=%zu value=%d",
                    tag.c_str(),
                    mtp_layer_num - 1,
                    local_layer_id,
                    source_group.layer_ids[local_layer_id]);
            }

            const size_t expected_existing_layers =
                static_cast<size_t>(group_layer_num) + static_cast<size_t>(module_index) * mtp_layer_num;
            RTP_LLM_CHECK_WITH_INFO(
                target_groups[target_group_index].layer_ids.size() == expected_existing_layers,
                "CacheConfig::mergeMTPModule tag=%s group_index=%zu physical group alignment mismatch: "
                "existing_layers=%zu expected=%zu module=%d group_layer_num=%d module_layers=%u",
                tag.c_str(),
                target_group_index,
                target_groups[target_group_index].layer_ids.size(),
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

        for (int local_layer_id : source_group.layer_ids) {
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
            sub_layer.group_tags.push_back(tag);

            target_groups[target_group_index].layer_ids.push_back(static_cast<int>(global_layer_id));
            target_layers[global_layer].group_tags.push_back(tag);

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

    for (size_t group_index = 0; group_index < group_num; ++group_index) {
        const auto& spec = specs[group_index];
        RTP_LLM_CHECK_WITH_INFO(
            spec != nullptr, "CacheConfig::fromGroupedSpecs got null spec at group %zu", group_index);
        std::string tag = tags.empty() ? spec->tag : tags[group_index];
        if (tag.empty() && group_num == 1) {
            tag = "default";
        }
        RTP_LLM_CHECK_WITH_INFO(
            !tag.empty(), "CacheConfig::fromGroupedSpecs requires non-empty tag for cache spec %zu", group_index);
        auto stored_spec = spec->clone();
        stored_spec->tag = tag;

        GroupBase group;
        group.tag    = tag;
        group.spec   = stored_spec;
        group.policy = has_explicit_policies ? policies[group_index] : defaultCacheGroupPolicy(types[group_index]);
        RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == types[group_index],
                                "CacheConfig::fromGroupedSpecs policy type mismatch group_index=%zu policy=%d type=%d",
                                group_index,
                                static_cast<int>(group.policy.group_type),
                                static_cast<int>(types[group_index]));
        group.layer_ids = layers_by_group[group_index];
        new_groups.push_back(group);

        for (int layer_id : layers_by_group[group_index]) {
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
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto explicit_independent_blocks = groups[group_index].policy.explicit_block_num;
        uint32_t   rule_blocks                 = global_block_num;
        if (explicit_independent_blocks > 0) {
            rule_blocks = explicit_independent_blocks;
        } else if (groups[group_index].policy.group_type == CacheGroupType::SWA) {
            rule_blocks = global_block_num / step + (global_block_num % step != 0 ? 1u : 0u);
        }
        groups[group_index].block_num = rule_blocks;

        // Only groups that opt in reserve paged-pool budget for explicit blocks.
        if (explicit_independent_blocks > 0 && groups[group_index].policy.charge_to_paged_budget) {
            reserve += static_cast<size_t>(rule_blocks) * groups[group_index].layer_ids.size()
                       * (groups[group_index].kv_block_stride_bytes + groups[group_index].kv_scale_stride_bytes);
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
    const auto                    group_tags       = groupTagsSnapshot();
    const auto&                   topology_groups  = topology().groups();
    std::vector<std::vector<int>> layers_by_group;
    layers_by_group.reserve(topology_groups.size());
    for (const auto& group : topology_groups) {
        layers_by_group.push_back(group.layer_ids);
    }
    std::vector<std::vector<std::string>> layer_group_tags;
    layer_group_tags.reserve(topology().layers().size());
    for (const auto& layer : topology().layers()) {
        layer_group_tags.push_back(layer.group_tags);
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
    OUTPUT_FIELD_EXPR("layer_to_group_tags.size()", layer_group_tags.size());
    os << indent1 << "layer_to_group_tags=[";
    for (size_t layer_index = 0; layer_index < layer_group_tags.size(); ++layer_index) {
        if (layer_index > 0) {
            os << ",";
        }
        os << "[";
        for (size_t tag_index = 0; tag_index < layer_group_tags[layer_index].size(); ++tag_index) {
            if (tag_index > 0) {
                os << ",";
            }
            os << layer_group_tags[layer_index][tag_index];
        }
        os << "]";
    }
    os << "]\n";
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
