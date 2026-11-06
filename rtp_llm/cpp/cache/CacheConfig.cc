#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool isFullAttentionSpec(KVCacheSpecType type) {
    return type == KVCacheSpecType::MultiHeadAttention || type == KVCacheSpecType::MultiHeadLatentAttention;
}

std::string cacheGroupPolicySummary(const CacheGroupPolicy& policy) {
    std::ostringstream os;
    os << "{group_type=" << cacheGroupTypeName(policy.group_type) << ", prefix_reuse=" << policy.enable_prefix_reuse
       << ", reservable=" << policy.reservable << ", explicit_block_num=" << policy.explicit_block_num
       << ", charge_to_paged_budget=" << policy.charge_to_paged_budget
       << ", memory_placement=" << static_cast<int>(policy.memory_placement)
       << ", active_tail_blocks=" << policy.active_tail_blocks << ", shared_chunk_state=" << policy.shared_chunk_state
       << ", validate_tail_blocks=" << policy.validate_tail_blocks
       << ", cp_mapping=" << static_cast<int>(policy.cp_mapping) << ", cp_slice=" << static_cast<int>(policy.cp_slice)
       << '}';
    return os.str();
}

std::string targetGroupSummary(const CacheConfig& target_config) {
    std::ostringstream os;
    os << '[';
    bool first = true;
    for (const auto& group : target_config.topology().groups()) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << "{tag=" << group.tag << ", group_type=" << cacheGroupTypeName(group.policy.group_type);
        if (group.spec != nullptr) {
            os << ", spec_type=" << static_cast<int>(group.spec->type)
               << ", dtype=" << static_cast<int>(group.spec->memoryLayoutDType())
               << ", block_size_bytes=" << group.spec->block_size_bytes()
               << ", scale_block_size_bytes=" << group.spec->scale_block_size_bytes();
        } else {
            os << ", spec=null";
        }
        os << ", seq_size_per_block=" << group.seqSizePerBlock()
           << ", kv_block_stride_bytes=" << group.kvBlockStrideBytes()
           << ", kv_scale_stride_bytes=" << group.kvScaleStrideBytes()
           << ", policy=" << cacheGroupPolicySummary(group.policy) << '}';
    }
    os << ']';
    return os.str();
}

std::optional<size_t> resolveDefaultMTPGroupAlias(const CacheConfig& target_config, const CacheConfig& propose_config) {
    if (propose_config.groupNums() != 1 || propose_config.topology().groups().front().tag != "default") {
        return std::nullopt;
    }

    for (const auto& group : target_config.topology().groups()) {
        if (group.tag == "default") {
            return std::nullopt;  // Exact tag matching remains authoritative.
        }
    }

    const auto& source_group = propose_config.topology().groups().front();
    if (source_group.policy.group_type != CacheGroupType::FULL || source_group.spec == nullptr
        || !isFullAttentionSpec(source_group.spec->type)) {
        return std::nullopt;
    }

    std::vector<size_t> candidates;
    for (size_t target_gid = 0; target_gid < target_config.topology().groups().size(); ++target_gid) {
        const auto& target_group = target_config.topology().groups()[target_gid];
        // The aliased draft layer uses its own MTP memory layout. Group-tag APIs still expose it as the target
        // group, however, so both logical block granularity and physical block shape must remain compatible.
        if (CacheConfig::samePolicy(target_group.policy, source_group.policy) && target_group.spec != nullptr
            && target_group.spec->type == source_group.spec->type
            && target_group.spec->memoryLayoutDType() == source_group.spec->memoryLayoutDType()
            && target_group.spec->block_size_bytes() == source_group.spec->block_size_bytes()
            && target_group.spec->scale_block_size_bytes() == source_group.spec->scale_block_size_bytes()
            && target_group.seqSizePerBlock() == source_group.seqSizePerBlock()
            && target_group.kvBlockStrideBytes() == source_group.kvBlockStrideBytes()
            && target_group.kvScaleStrideBytes() == source_group.kvScaleStrideBytes()) {
            candidates.push_back(target_gid);
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
                     source_group.seqSizePerBlock(),
                     source_group.spec->block_size_bytes(),
                     source_group.spec->scale_block_size_bytes(),
                     source_group.kvBlockStrideBytes(),
                     source_group.kvScaleStrideBytes(),
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

}  // namespace

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.group_type == rhs.group_type && lhs.enable_prefix_reuse == rhs.enable_prefix_reuse
           && lhs.reservable == rhs.reservable && lhs.explicit_block_num == rhs.explicit_block_num
           && lhs.charge_to_paged_budget == rhs.charge_to_paged_budget && lhs.memory_placement == rhs.memory_placement
           && lhs.active_tail_blocks == rhs.active_tail_blocks && lhs.validate_tail_blocks == rhs.validate_tail_blocks
           && lhs.cp_mapping == rhs.cp_mapping && lhs.cp_slice == rhs.cp_slice
           && lhs.sliding_window_size == rhs.sliding_window_size && lhs.shared_chunk_state == rhs.shared_chunk_state;
}

void CacheConfig::setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers) {
    main_config.setTopology(std::move(new_groups), std::move(new_layers));
}

void CacheConfig::setGroupBlockLayout(const std::vector<std::string>& group_tags,
                                      const std::vector<uint32_t>&    block_nums,
                                      const std::vector<size_t>&      kv_block_stride_bytes,
                                      const std::vector<size_t>&      kv_scale_stride_bytes) {
    main_config.setGroupBlockLayout(group_tags, block_nums, kv_block_stride_bytes, kv_scale_stride_bytes);
}

CacheModuleConfig
CacheConfig::mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(groupNums() > 0, "CacheConfig::mergeMTPModule requires destination topology");
    RTP_LLM_CHECK_WITH_INFO(propose_config.groupNums() > 0, "CacheConfig::mergeMTPModule requires propose topology");
    RTP_LLM_CHECK_WITH_INFO(module_index >= 0, "CacheConfig::mergeMTPModule invalid module_index=%d", module_index);

    const auto default_alias_target_gid = resolveDefaultMTPGroupAlias(*this, propose_config);
    bool       any_group                = false;
    for (const auto& propose_group : propose_config.topology().groups()) {
        const auto& tag              = propose_group.tag;
        const bool  has_exact_target = std::any_of(topology().groups().begin(),
                                                  topology().groups().end(),
                                                  [&tag](const auto& group) { return group.tag == tag; });
        // Every declared draft segment needs backing in an existing target
        // pool. Dropping an unmatched tag would also omit its memory budget.
        RTP_LLM_CHECK_WITH_INFO(has_exact_target || (!any_group && default_alias_target_gid.has_value()),
                                "CacheConfig::mergeMTPModule unmapped draft cache group tag=%s",
                                tag.c_str());
        any_group = true;
    }
    // Copy every per-module field, then overwrite the topology with the sub
    // layout (local layer ids) below.
    CacheModuleConfig sub_cfg = propose_config.main_config;

    const auto mtp_layer_num = propose_config.layer_num();
    RTP_LLM_CHECK_WITH_INFO(mtp_layer_num > 0, "CacheConfig::mergeMTPModule requires module layers");
    const auto total_layers =
        static_cast<uint64_t>(main_layer_num) + (static_cast<uint64_t>(module_index) + 1) * mtp_layer_num;
    RTP_LLM_CHECK_WITH_INFO(total_layers <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
                            "CacheConfig::mergeMTPModule layer count exceeds int range");
    auto target_groups = topology().groups();
    auto target_layers = topology().layers();
    target_layers.resize(total_layers);
    for (size_t layer_id = 0; layer_id < target_layers.size(); ++layer_id) {
        target_layers[layer_id].layer_id = static_cast<int>(layer_id);
    }
    const auto                              target_group_num = target_groups.size();
    std::unordered_map<std::string, size_t> propose_gid_by_tag;
    for (size_t gid = 0; gid < propose_config.topology().groups().size(); ++gid) {
        propose_gid_by_tag.emplace(propose_config.topology().groups()[gid].tag, gid);
    }
    std::vector<GroupBase> sub_groups;
    std::vector<LayerBase> sub_layers(static_cast<size_t>(mtp_layer_num));
    sub_groups.reserve(target_group_num);

    for (size_t target_gid = 0; target_gid < target_group_num; ++target_gid) {
        const auto& tag             = target_groups[target_gid].tag;
        const auto  propose_it      = propose_gid_by_tag.find(tag);
        const bool  has_exact_group = propose_it != propose_gid_by_tag.end();
        const bool  uses_default_alias =
            !has_exact_group && default_alias_target_gid.has_value() && target_gid == *default_alias_target_gid;

        const CacheConfig* source_config = this;
        size_t             source_gid    = target_gid;
        if (has_exact_group) {
            source_config = &propose_config;
            source_gid    = propose_it->second;
        } else if (uses_default_alias) {
            source_config = &propose_config;
            source_gid    = 0;
        }
        const bool  has_propose_group = has_exact_group || uses_default_alias;
        const auto& source_group      = source_config->topology().groups()[source_gid];
        const auto  source_layer_ids  = source_config->layerIdsForGroup(source_group.tag);

        if (has_propose_group) {
            RTP_LLM_CHECK_WITH_INFO(samePolicy(target_groups[target_gid].policy, source_group.policy),
                                    "CacheConfig::mergeMTPModule incompatible pool policy for tag=%s",
                                    tag.c_str());
            // MTP reuses the target stream's physical/kernel block tables. Its
            // own backing may have different byte strides, but table entries
            // must address the same token spans in both model segments.
            RTP_LLM_CHECK_WITH_INFO(target_groups[target_gid].seqSizePerBlock() == source_group.seqSizePerBlock()
                                        && target_groups[target_gid].kernelSeqSizePerBlock()
                                               == source_group.kernelSeqSizePerBlock(),
                                    "CacheConfig::mergeMTPModule incompatible block token spans for tag=%s",
                                    tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(
                source_layer_ids.size() == static_cast<size_t>(mtp_layer_num),
                "CacheConfig::mergeMTPModule source_tag=%s target_tag=%s must cover every module layer, "
                "got=%zu expected=%u",
                source_group.tag.c_str(),
                tag.c_str(),
                source_layer_ids.size(),
                mtp_layer_num);
            for (size_t local_layer_id = 0; local_layer_id < source_layer_ids.size(); ++local_layer_id) {
                RTP_LLM_CHECK_WITH_INFO(
                    source_layer_ids[local_layer_id] == static_cast<int>(local_layer_id),
                    "CacheConfig::mergeMTPModule source_tag=%s target_tag=%s source layers must be ordered 0..%u, "
                    "index=%zu value=%d",
                    source_group.tag.c_str(),
                    tag.c_str(),
                    mtp_layer_num - 1,
                    local_layer_id,
                    source_layer_ids[local_layer_id]);
            }
        }

        GroupBase sub_group = source_group;
        if (uses_default_alias) {
            RTP_LLM_LOG_INFO("CacheConfig::mergeMTPModule aliases propose tag=default to target tag=%s: "
                             "module=%d spec_type=%d dtype=%d seq_size_per_block=%zu "
                             "kv_block_stride_bytes=%zu kv_scale_stride_bytes=%zu",
                             tag.c_str(),
                             module_index,
                             static_cast<int>(source_group.spec->type),
                             static_cast<int>(source_group.spec->memoryLayoutDType()),
                             source_group.seqSizePerBlock(),
                             source_group.kvBlockStrideBytes(),
                             source_group.kvScaleStrideBytes());
            auto aliased_spec = source_group.spec->clone();
            aliased_spec->tag = tag;
            sub_group.tag     = tag;
            sub_group.spec    = std::move(aliased_spec);
        }

        if (!has_propose_group) {
            sub_groups.push_back(std::move(sub_group));
            continue;
        }

        for (int local_layer_id : source_layer_ids) {
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

            auto& sub_layer    = sub_layers[static_cast<size_t>(local_layer_id)];
            sub_layer.layer_id = local_layer_id;
            sub_layer.group_tags.push_back(tag);

            target_layers[global_layer].group_tags.push_back(tag);
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

    sub_cfg.setTopology(std::move(sub_groups), std::move(sub_layers));
    setTopology(std::move(target_groups), std::move(target_layers));
    return sub_cfg;
}

const GroupBase& CacheConfig::physicalGroupForLayer(int layer_id, const std::string& tag) const {
    const auto& logical_group = topology().groupForLayer(layer_id, tag);
    if (layer_id < static_cast<int>(main_config.layer_num)) {
        return logical_group;
    }

    for (size_t module_index = 0; module_index < mtp_sub_configs.size(); ++module_index) {
        const auto& sub_config = mtp_sub_configs[module_index];
        const auto  first_layer_id =
            mtpGlobalLayerId(main_config.layer_num, static_cast<int>(module_index), sub_config.layer_num, 0);
        if (first_layer_id != std::numeric_limits<uint32_t>::max() && static_cast<uint64_t>(layer_id) >= first_layer_id
            && static_cast<uint64_t>(layer_id) - first_layer_id < sub_config.layer_num) {
            return sub_config.topology().groupForLayer(layer_id - static_cast<int>(first_layer_id), tag);
        }
    }

    RTP_LLM_FAIL("CacheConfig cannot resolve physical owner for layer=%d tag=%s", layer_id, tag.c_str());
}

void CacheConfig::fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                                   const std::vector<std::vector<int>>& layers_by_group,
                                   const std::vector<CacheGroupType>&   types,
                                   const std::vector<std::string>&      tags,
                                   const std::vector<CacheGroupPolicy>& policies) {
    main_config.fromGroupedSpecs(specs, layers_by_group, types, tags, policies);
}

void CacheConfig::finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
    RTP_LLM_CHECK_WITH_INFO(global_block_num > 0, "cache configuration requires positive baseline block count");
    for (auto& sub_cfg : mtp_sub_configs) {
        sub_cfg.finalizeBlockNums(global_block_num, runtime_config);
    }

    main_config.finalizeBlockNums(global_block_num, runtime_config);
}

std::string CacheConfig::debugString(size_t indent) const {
    const std::string indent_str = std::string(indent, ' ');
    const std::string indent1    = indent_str + "  ";

    std::ostringstream os;
    os << indent_str << "CacheConfig{\n";

#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

    os << indent1 << "# Shared Configuration:\n";
    OUTPUT_FIELD(cache_min_replay_tokens);
    OUTPUT_FIELD(cache_key_hash_seed);
    OUTPUT_FIELD(seq_size_per_block);
    os << "\n";

#undef OUTPUT_FIELD
#undef OUTPUT_FIELD_EXPR

    os << indent1 << "main_config:\n";
    os << main_config.debugString(indent + 2);
    os << "\n";

    os << indent1 << "mtp_sub_configs.size()=" << mtp_sub_configs.size() << "\n";
    for (size_t i = 0; i < mtp_sub_configs.size(); ++i) {
        os << indent1 << "mtp_sub_configs[" << i << "]:\n";
        os << mtp_sub_configs[i].debugString(indent + 4);
    }
    os << "\n";

    os << indent_str << "}\n";
    return os.str();
}

}  // namespace rtp_llm
