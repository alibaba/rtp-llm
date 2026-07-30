#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

CacheGroupType groupTypeForSpec(const KVCacheSpec& spec) {
    return spec.type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
}

bool isFullAttentionSpec(KVCacheSpecType type) {
    return type == KVCacheSpecType::MultiHeadAttention || type == KVCacheSpecType::MultiHeadLatentAttention;
}

std::string cacheGroupPolicySummary(const CacheGroupPolicy& policy) {
    std::ostringstream os;
    os << "{group_type=" << cacheGroupTypeName(policy.group_type) << ", prefix_reuse=" << policy.enable_prefix_reuse
       << ", evict=" << static_cast<int>(policy.evict_policy) << ", reservable=" << policy.reservable
       << ", explicit_block_num=" << policy.explicit_block_num
       << ", charge_to_paged_budget=" << policy.charge_to_paged_budget
       << ", active_tail_blocks=" << policy.active_tail_blocks
       << ", validate_tail_blocks=" << policy.validate_tail_blocks
       << ", cp_mapping=" << static_cast<int>(policy.cp_mapping) << ", cp_slice=" << static_cast<int>(policy.cp_slice)
       << '}';
    return os.str();
}

std::string targetGroupSummary(const CacheConfig& target_config) {
    std::ostringstream os;
    os << '[';
    bool first_group = true;
    for (const auto& group : target_config.topology().groups()) {
        if (!first_group) {
            os << ", ";
        }
        first_group = false;
        os << "{tag=" << group.tag << ", group_type=" << cacheGroupTypeName(group.policy.group_type);
        if (group.spec != nullptr) {
            os << ", spec_type=" << static_cast<int>(group.spec->type)
               << ", dtype=" << static_cast<int>(group.spec->memoryLayoutDType())
               << ", block_size_bytes=" << group.spec->block_size_bytes()
               << ", scale_block_size_bytes=" << group.spec->scale_block_size_bytes();
        } else {
            os << ", spec=null";
        }
        os << ", seq_size_per_block=" << group.seq_size_per_block
           << ", kv_block_stride_bytes=" << group.kv_block_stride_bytes
           << ", kv_scale_stride_bytes=" << group.kv_scale_stride_bytes
           << ", policy=" << cacheGroupPolicySummary(group.policy) << '}';
    }
    os << ']';
    return os.str();
}

std::optional<std::string> resolveDefaultMTPGroupAlias(const CacheConfig& target_config,
                                                       const CacheConfig& propose_config) {
    if (propose_config.groupNums() != 1 || propose_config.topology().groups().front().tag != "default") {
        return std::nullopt;
    }

    for (const auto& target_group : target_config.topology().groups()) {
        if (target_group.tag == "default") {
            return std::nullopt;  // Exact tag matching remains authoritative.
        }
    }

    const auto& source_group = propose_config.topology().groups().front();
    if (source_group.policy.group_type != CacheGroupType::FULL || source_group.spec == nullptr
        || !isFullAttentionSpec(source_group.spec->type)) {
        return std::nullopt;
    }

    std::vector<std::string> candidates;
    for (const auto& target_group : target_config.topology().groups()) {
        // The aliased draft layer uses its own MTP memory layout. Group-tag APIs still expose it as the target
        // group, however, so both logical block granularity and physical block shape must remain compatible.
        if (CacheConfig::samePolicy(target_group.policy, source_group.policy) && target_group.spec != nullptr
            && target_group.spec->type == source_group.spec->type
            && target_group.spec->memoryLayoutDType() == source_group.spec->memoryLayoutDType()
            && target_group.spec->block_size_bytes() == source_group.spec->block_size_bytes()
            && target_group.spec->scale_block_size_bytes() == source_group.spec->scale_block_size_bytes()
            && target_group.seq_size_per_block == source_group.seq_size_per_block
            && target_group.kv_block_stride_bytes == source_group.kv_block_stride_bytes
            && target_group.kv_scale_stride_bytes == source_group.kv_scale_stride_bytes) {
            candidates.push_back(target_group.tag);
        }
    }

    if (candidates.empty()) {
        const auto target_summary = targetGroupSummary(target_config);
        const auto source_policy  = cacheGroupPolicySummary(source_group.policy);
        RTP_LLM_FAIL("CacheConfig::mergeMTPModule no compatible target group for sole propose tag=default: "
                     "source_spec_type=%d source_dtype=%d source_seq_size_per_block=%zu "
                     "source_block_size_bytes=%zu source_scale_block_size_bytes=%zu "
                     "source_kv_block_stride_bytes=%zu source_kv_scale_stride_bytes=%zu "
                     "source_policy=%s target_groups=%s",
                     static_cast<int>(source_group.spec->type),
                     static_cast<int>(source_group.spec->memoryLayoutDType()),
                     source_group.seq_size_per_block,
                     source_group.spec->block_size_bytes(),
                     source_group.spec->scale_block_size_bytes(),
                     source_group.kv_block_stride_bytes,
                     source_group.kv_scale_stride_bytes,
                     source_policy.c_str(),
                     target_summary.c_str());
    }
    if (candidates.size() > 1) {
        const auto target_summary = targetGroupSummary(target_config);
        RTP_LLM_FAIL("CacheConfig::mergeMTPModule ambiguous default FULL group mapping: "
                     "compatible target groups=%zu spec_type=%d target_groups=%s",
                     candidates.size(),
                     static_cast<int>(source_group.spec->type),
                     target_summary.c_str());
    }
    return candidates.front();
}

std::vector<GroupBase> copyGroups(const CacheTopology& topology) {
    const auto groups = topology.groups();
    return {groups.begin(), groups.end()};
}

}  // namespace

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.group_type == rhs.group_type && lhs.enable_prefix_reuse == rhs.enable_prefix_reuse
           && lhs.evict_policy == rhs.evict_policy && lhs.reservable == rhs.reservable
           && lhs.explicit_block_num == rhs.explicit_block_num
           && lhs.charge_to_paged_budget == rhs.charge_to_paged_budget
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
    }

    cache_topology = CacheTopology::create(std::move(new_groups), std::move(new_layers));
}

void CacheConfig::setGroupPolicies(const std::unordered_map<std::string, CacheGroupPolicy>& policies) {
    RTP_LLM_CHECK_WITH_INFO(policies.size() == topology().groups().size(),
                            "CacheConfig::setGroupPolicies size %zu != group size %zu",
                            policies.size(),
                            topology().groups().size());
    auto groups = copyGroups(topology());
    for (auto& group : groups) {
        const auto it = policies.find(group.tag);
        RTP_LLM_CHECK_WITH_INFO(
            it != policies.end(), "CacheConfig::setGroupPolicies missing tag=%s", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == it->second.group_type,
                                "CacheConfig::setGroupPolicies cannot change group type for tag=%s",
                                group.tag.c_str());
        group.policy = it->second;
    }
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
    auto target_groups = copyGroups(topology());
    auto target_layers = topology().layers();
    target_layers.resize(total_layers);
    for (size_t layer_id = 0; layer_id < target_layers.size(); ++layer_id) {
        target_layers[layer_id].layer_id = static_cast<int>(layer_id);
    }
    if (layer_to_block_stride_bytes.size() < total_layers) {
        layer_to_block_stride_bytes.resize(total_layers, 0);
    }

    const auto target_group_num  = target_groups.size();
    const auto default_alias_tag = resolveDefaultMTPGroupAlias(*this, propose_config);

    std::vector<GroupBase> sub_groups;
    std::vector<LayerBase> sub_layers(static_cast<size_t>(mtp_layer_num));
    sub_groups.reserve(target_group_num);

    for (auto& target_group : target_groups) {
        const auto& tag                = target_group.tag;
        const bool  has_exact_group    = propose_config.topology().contains(tag);
        const bool  uses_default_alias = !has_exact_group && default_alias_tag.has_value() && tag == *default_alias_tag;
        const bool  has_propose_group  = has_exact_group || uses_default_alias;
        const auto& source_group       = has_exact_group    ? propose_config.group(tag) :
                                         uses_default_alias ? propose_config.group("default") :
                                                              target_group;

        if (has_propose_group) {
            RTP_LLM_CHECK_WITH_INFO(
                source_group.layer_ids.size() == static_cast<size_t>(mtp_layer_num),
                "CacheConfig::mergeMTPModule source_tag=%s target_tag=%s must cover every module layer, "
                "got=%zu expected=%u",
                source_group.tag.c_str(),
                tag.c_str(),
                source_group.layer_ids.size(),
                mtp_layer_num);
            for (size_t local_layer_id = 0; local_layer_id < source_group.layer_ids.size(); ++local_layer_id) {
                RTP_LLM_CHECK_WITH_INFO(
                    source_group.layer_ids[local_layer_id] == static_cast<int>(local_layer_id),
                    "CacheConfig::mergeMTPModule source_tag=%s target_tag=%s source layers must be ordered 0..%u, "
                    "index=%zu value=%d",
                    source_group.tag.c_str(),
                    tag.c_str(),
                    mtp_layer_num - 1,
                    local_layer_id,
                    source_group.layer_ids[local_layer_id]);
            }

            const size_t expected_existing_layers =
                static_cast<size_t>(group_layer_num) + static_cast<size_t>(module_index) * mtp_layer_num;
            RTP_LLM_CHECK_WITH_INFO(target_group.layer_ids.size() == expected_existing_layers,
                                    "CacheConfig::mergeMTPModule source_tag=%s target_tag=%s "
                                    "physical group alignment mismatch: "
                                    "existing_layers=%zu expected=%zu module=%d group_layer_num=%d module_layers=%u",
                                    source_group.tag.c_str(),
                                    tag.c_str(),
                                    target_group.layer_ids.size(),
                                    expected_existing_layers,
                                    module_index,
                                    group_layer_num,
                                    mtp_layer_num);
        }

        GroupBase sub_group = source_group;
        sub_group.layer_ids.clear();
        if (uses_default_alias) {
            RTP_LLM_LOG_INFO("CacheConfig::mergeMTPModule aliases propose tag=default to target tag=%s: "
                             "module=%d spec_type=%d dtype=%d seq_size_per_block=%zu "
                             "kv_block_stride_bytes=%zu kv_scale_stride_bytes=%zu",
                             tag.c_str(),
                             module_index,
                             static_cast<int>(source_group.spec->type),
                             static_cast<int>(source_group.spec->memoryLayoutDType()),
                             source_group.seq_size_per_block,
                             source_group.kv_block_stride_bytes,
                             source_group.kv_scale_stride_bytes);
            auto aliased_spec = source_group.spec->clone();
            aliased_spec->tag = tag;
            sub_group.tag     = tag;
            sub_group.spec    = std::move(aliased_spec);
        }

        if (!has_propose_group) {
            sub_groups.push_back(std::move(sub_group));
            continue;
        }

        for (int local_layer_id : propose_config.layerIdsForGroup(source_group.tag)) {
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

            target_group.layer_ids.push_back(static_cast<int>(global_layer_id));
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
            auto groups = copyGroups(topology());
            for (auto& group : groups) {
                group.block_num = global_block_num;
            }
            setTopology(std::move(groups), topology().layers());
        }
        return;
    }

    size_t     reserve = 0;
    const auto step    = static_cast<uint32_t>(std::max(1, linear_step));
    auto       groups  = copyGroups(topology());
    for (auto& group : groups) {
        const auto explicit_independent_blocks = group.policy.explicit_block_num;
        uint32_t   rule_blocks                 = global_block_num;
        if (explicit_independent_blocks > 0) {
            rule_blocks = explicit_independent_blocks;
        } else if (group.policy.group_type == CacheGroupType::SWA) {
            rule_blocks = global_block_num / step + (global_block_num % step != 0 ? 1u : 0u);
        }
        group.block_num = rule_blocks;

        // Only groups that opt in reserve paged-pool budget for explicit blocks.
        if (explicit_independent_blocks > 0 && group.policy.charge_to_paged_budget) {
            reserve += static_cast<size_t>(rule_blocks) * group.layer_ids.size()
                       * (group.kv_block_stride_bytes + group.kv_scale_stride_bytes);
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

    const auto topology_groups = topology().groups();

    os << indent1 << "# Attention Configuration:\n";
    OUTPUT_FIELD(linear_step);
    OUTPUT_FIELD(group_layer_num);
    OUTPUT_FIELD_EXPR("full_group_num",
                      std::count_if(topology_groups.begin(), topology_groups.end(), [](const GroupBase& group) {
                          return group.policy.group_type == CacheGroupType::FULL;
                      }));
    OUTPUT_FIELD_EXPR("linear_group_num",
                      std::count_if(topology_groups.begin(), topology_groups.end(), [](const GroupBase& group) {
                          return group.policy.group_type == CacheGroupType::LINEAR;
                      }));
    os << indent1 << "group_block_layout=[";
    bool first_group = true;
    for (const auto& group : topology_groups) {
        if (!first_group) {
            os << ",";
        }
        first_group = false;
        os << "{" << group.tag << ":" << group.block_num << "}";
    }
    os << "]\n";
    os << "\n";

    os << indent1 << "# Cache Specifications:\n";
    OUTPUT_FIELD_EXPR("groups.size()", topology_groups.size());
    for (const auto& group : topology_groups) {
        const auto& spec = group.spec;
        if (!spec) {
            os << indent1 << "group[" << group.tag << "].spec=null\n";
            continue;
        }

        os << indent1 << "group[" << group.tag << "] {\n";
        os << spec->debugString(indent + 2);
        os << indent1 << "}\n";
    }
    os << "\n";

    os << indent1 << "# Layer Mapping:\n";
    for (const auto& group : topology_groups) {
        os << indent1 << "group[" << group.tag << "].layers=" << rtp_llm::vectorToString(group.layer_ids) << "\n";
        os << indent1 << "group[" << group.tag << "].type=" << static_cast<int>(group.policy.group_type) << "\n";
    }
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
