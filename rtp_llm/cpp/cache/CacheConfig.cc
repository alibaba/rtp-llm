#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

namespace {

CacheGroupType groupTypeForSpec(const KVCacheSpec& spec) {
    return spec.type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
}

}  // namespace

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.policyFingerprint() == rhs.policyFingerprint();
}

void CacheConfig::setTopology(size_t module_id, std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers) {
    auto& record = moduleRecord(module_id);
    RTP_LLM_CHECK_WITH_INFO(!new_groups.empty(), "CacheConfig::setTopology requires at least one cache group");

    for (size_t gid = 0; gid < new_groups.size(); ++gid) {
        auto& group = new_groups[gid];
        RTP_LLM_CHECK_WITH_INFO(
            group.spec != nullptr, "CacheConfig::setTopology got null spec at group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheConfig::setTopology requires tag for group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(group.spec->tag == group.tag,
                                "CacheConfig::setTopology tag=%s does not match spec tag=%s",
                                group.tag.c_str(),
                                group.spec->tag.c_str());

        const auto expected_group_type = groupTypeForSpec(*group.spec);
        RTP_LLM_CHECK_WITH_INFO(expected_group_type != CacheGroupType::LINEAR
                                    || group.policy.group_type == CacheGroupType::LINEAR,
                                "CacheConfig::setTopology group %zu tag=%s policy type %s does not match "
                                "spec type %d",
                                gid,
                                group.tag.c_str(),
                                cacheGroupTypeName(group.policy.group_type),
                                static_cast<int>(group.spec->type));

        group.spec = group.spec->clone();
        RTP_LLM_CHECK_WITH_INFO(group.spec->seq_size_per_block > 0,
                                "CacheConfig::setTopology tag=%s has zero seq_size_per_block",
                                group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.spec->kernel_seq_size_per_block > 0,
                                "CacheConfig::setTopology tag=%s has zero kernel_seq_size_per_block",
                                group.tag.c_str());
    }

    auto topology = CacheTopology::create(std::move(new_groups), std::move(new_layers));
    record.topology = std::move(topology);
}

void CacheConfig::setGroupBlockLayout(const std::vector<std::string>& group_tags,
                                      const std::vector<uint32_t>&    block_nums,
                                      const std::vector<size_t>&      kv_block_stride_bytes,
                                      const std::vector<size_t>&      kv_scale_stride_bytes) {
    const auto& old_topology = topology();
    const size_t group_num  = old_topology.groups().size();
    RTP_LLM_CHECK_WITH_INFO(group_tags.size() == group_num,
                            "CacheConfig::setGroupBlockLayout tags size %zu != group size %zu",
                            group_tags.size(),
                            group_num);
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
    auto                            groups = old_topology.groups();
    std::unordered_set<std::string> seen_tags;
    for (size_t row = 0; row < group_num; ++row) {
        const auto& tag = group_tags[row];
        RTP_LLM_CHECK_WITH_INFO(
            seen_tags.emplace(tag).second, "CacheConfig::setGroupBlockLayout duplicate tag=%s", tag.c_str());
        auto group = std::find_if(
            groups.begin(), groups.end(), [&](const GroupBase& candidate) { return candidate.tag == tag; });
        RTP_LLM_CHECK_WITH_INFO(
            group != groups.end(), "CacheConfig::setGroupBlockLayout unknown tag=%s", tag.c_str());
        group->block_num = block_nums[row];
        RTP_LLM_CHECK_WITH_INFO(group->kvBlockStrideBytes() == kv_block_stride_bytes[row]
                                    && group->kvScaleStrideBytes() == kv_scale_stride_bytes[row],
                                "CacheConfig layout must match Spec for tag=%s",
                                tag.c_str());
    }
    // Layout replacement reuses the already-frozen Specs and publishes a new snapshot.
    auto layers  = old_topology.layers();
    moduleRecord(0).topology = CacheTopology::create(std::move(groups), std::move(layers));
}

void CacheConfig::validateCrossModuleInvariants() const {
    RTP_LLM_CHECK_WITH_INFO(moduleCount() > 0, "cache config requires at least one module");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "cache-key block span must be positive");
    std::vector<std::string> seen_tags;
    for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
        const auto& module_tags = topology(module_id).groupTags();
        for (const auto& tag : module_tags) {
            RTP_LLM_CHECK_WITH_INFO(
                std::find(seen_tags.begin(), seen_tags.end(), tag) == seen_tags.end(),
                "cache tag '%s' (module %zu) collides with the tag of another module; the Python model "
                "description must declare module-scoped tags (e.g. mtp_<tag>)",
                tag.c_str(),
                module_id);
            seen_tags.push_back(tag);
        }
        for (size_t earlier_id = 0; earlier_id < module_id; ++earlier_id) {
            for (const auto& group : topology(module_id).groups()) {
                if (!group.policy.enable_prefix_reuse) {
                    continue;
                }
                for (const auto& earlier : topology(earlier_id).groups()) {
                    if (earlier.policy.enable_prefix_reuse && earlier.policy.group_type == group.policy.group_type) {
                        RTP_LLM_CHECK_WITH_INFO(
                            earlier.policy.explicit_block_num == group.policy.explicit_block_num,
                            "jointly reusable groups %s and %s must use the same explicit block capacity",
                            earlier.tag.c_str(),
                            group.tag.c_str());
                    }
                }
            }
        }
    }
}

void CacheConfig::fromGroupedSpecs(size_t                              module_id,
                                   const std::vector<KVCacheSpecPtr>&   specs,
                                   uint32_t                             layer_num,
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
        new_groups.push_back(group);

        for (int layer_id : layers_by_group[gid]) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < new_layers.size(),
                                    "CacheConfig::fromGroupedSpecs tag=%s has invalid layer id %d for "
                                    "layer_num=%u",
                                    tag.c_str(),
                                    layer_id,
                                    layer_num);
            auto& layer = new_layers[static_cast<size_t>(layer_id)];
            layer.group_tags.push_back(tag);
        }
    }

    setTopology(module_id, std::move(new_groups), std::move(new_layers));
}

void CacheConfig::addMtpModule(std::shared_ptr<const CacheTopology> topology, CacheModuleMetadata metadata) {
    RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "CacheConfig::addMtpModule requires a published topology");
    ModuleRecord record;
    record.topology = std::move(topology);
    record.metadata = std::move(metadata);
    modules_.push_back(std::move(record));
}

uint32_t CacheConfig::blockWeight(const std::string& tag) const {
    const auto it = pool_block_weights.find(tag);
    return it == pool_block_weights.end() ? 1u : it->second;
}

uint32_t CacheConfig::blockNumForGroup(const GroupBase& group, uint32_t baseline) const {
    const auto weight = blockWeight(group.tag);
    RTP_LLM_CHECK_WITH_INFO(weight > 0, "cache pool '%s' block weight must be positive", group.tag.c_str());
    // Main's #1574 rule: a group either pins its own explicit block count or
    // follows the global baseline; pool weights scale the baseline share.
    const uint64_t blocks =
        group.policy.explicit_block_num > 0 ?
            group.policy.explicit_block_num :
            static_cast<uint64_t>(baseline) * weight;
    RTP_LLM_CHECK_WITH_INFO(
        blocks <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
        "cache pool '%s' block count exceeds int32 block ID range: baseline=%u weight=%u blocks=%lu",
        group.tag.c_str(),
        baseline,
        weight,
        blocks);
    return static_cast<uint32_t>(blocks);
}

uint32_t CacheConfig::maxBaselineBlockNum() const {
    uint64_t maximum = std::numeric_limits<uint32_t>::max();
    for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
        const auto& topology = topologyPtr(module_id);
        if (topology == nullptr) {
            continue;
        }
        for (const auto& group : topology->groups()) {
            if (topology->layerIdsForGroup(group.tag).empty()) {
                continue;
            }
            if (group.policy.explicit_block_num > 0) {
                (void)blockNumForGroup(group, 1);
                continue;
            }
            const auto weight = blockWeight(group.tag);
            RTP_LLM_CHECK_WITH_INFO(weight > 0, "cache pool '%s' block weight must be positive", group.tag.c_str());
            // Baseline * weight must stay inside the int32 block-ID range.
            const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / weight;
            maximum = std::min(maximum, limit);
        }
    }
    return static_cast<uint32_t>(maximum);
}

void CacheConfig::finalizeBlockNums(uint32_t                   global_block_num,
                                    const RuntimeConfig&       runtime_config,
                                    const PPBlockNumOverrides* pp_overrides) {
    RTP_LLM_CHECK_WITH_INFO(global_block_num > 0, "cache configuration requires positive baseline block count");
    (void)runtime_config;
    const auto block_num_for_group = [&](const GroupBase& group) {
        if (pp_overrides == nullptr) {
            return blockNumForGroup(group, global_block_num);
        }
        const auto it = pp_overrides->find(group.tag);
        RTP_LLM_CHECK_WITH_INFO(
            it != pp_overrides->end(), "cache group [%s] is missing from PP block overrides", group.tag.c_str());
        return it->second;
    };
    const auto finalize_module = [&](size_t module_id) {
        auto& record = moduleRecord(module_id);
        if (record.topology == nullptr) {
            return;
        }
        auto groups = record.topology->groups();
        for (auto& group : groups) {
            group.block_num = block_num_for_group(group);
            RTP_LLM_CHECK_WITH_INFO(
                group.block_num > 0, "cache group [%s] requires positive block count", group.tag.c_str());
        }
        // The published topology already owns frozen Specs; the capacity
        // replacement reuses them and publishes a new snapshot.
        auto layers = record.topology->layers();
        auto topology = CacheTopology::create(std::move(groups), std::move(layers));
        record.topology = std::move(topology);
    };
    finalize_module(0);
    for (size_t module_id = 1; module_id < moduleCount(); ++module_id) {
        // Draft groups without local layers receive their capacity from the owning PP stage.
        const auto& topology = moduleRecord(module_id).topology;
        if ((topology != nullptr && !topology->layers().empty()) || pp_overrides != nullptr) {
            finalize_module(module_id);
        }
    }
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
    OUTPUT_FIELD_EXPR("local_kv_cache_sharded", (local_kv_cache_sharded ? "true" : "false"));
    OUTPUT_FIELD(global_layer_begin);
    os << "\n";

#undef OUTPUT_FIELD
#undef OUTPUT_FIELD_EXPR

    for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
        os << indent1 << "module[" << module_id << "]:\n";
        const auto& metadata = moduleMetadata(module_id);
        os << indent1 << "  dtype=" << static_cast<int>(metadata.dtype) << "\n";
        os << indent1 << "  use_mla=" << (metadata.use_mla ? "true" : "false") << "\n";
        os << indent1 << "  reuse_cache_tail_blocks=" << metadata.reuse_cache_tail_blocks << "\n";
        const auto& topology = topologyPtr(module_id);
        os << indent1 << "  layer_num=" << (topology == nullptr ? 0u : topology->layers().size()) << "\n";
        if (topology == nullptr) {
            os << indent1 << "  topology=<unpublished>\n";
            continue;
        }
        std::vector<uint32_t>         group_block_nums;
        std::vector<std::vector<int>> layers_by_group;
        layers_by_group.reserve(topology->groups().size());
        for (const auto& group : topology->groups()) {
            layers_by_group.push_back(topology->layerIdsForGroup(group.tag));
            group_block_nums.push_back(group.block_num);
        }
        os << indent1 << "  group_block_nums=" << rtp_llm::vectorToString(group_block_nums) << "\n";
        os << indent1 << "  group_tags=[";
        for (size_t i = 0; i < topology->groups().size(); ++i) {
            os << topology->groups()[i].tag;
            if (i + 1 < topology->groups().size()) {
                os << ",";
            }
        }
        os << "]\n";
        os << indent1 << "  layers_by_group=" << rtp_llm::vectorsToString(layers_by_group) << "\n";
        os << "\n";
    }

    os << indent_str << "}\n";
    return os.str();
}

}  // namespace rtp_llm
