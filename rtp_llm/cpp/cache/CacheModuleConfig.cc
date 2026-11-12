#include "rtp_llm/cpp/cache/CacheModuleConfig.h"
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

void CacheModuleConfig::setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers) {
    RTP_LLM_CHECK_WITH_INFO(!new_groups.empty(), "CacheModuleConfig::setTopology requires at least one cache group");
    RTP_LLM_CHECK_WITH_INFO(!new_layers.empty(), "CacheModuleConfig::setTopology requires at least one cache layer");
    RTP_LLM_CHECK_WITH_INFO(new_layers.size() >= layer_num,
                            "CacheModuleConfig::setTopology layer count %zu is less than layer count %u",
                            new_layers.size(),
                            layer_num);

    for (size_t gid = 0; gid < new_groups.size(); ++gid) {
        auto& group = new_groups[gid];
        RTP_LLM_CHECK_WITH_INFO(
            group.spec != nullptr, "CacheModuleConfig::setTopology got null spec at group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheModuleConfig::setTopology requires tag for group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(group.spec->tag == group.tag,
                                "CacheModuleConfig::setTopology tag=%s does not match spec tag=%s",
                                group.tag.c_str(),
                                group.spec->tag.c_str());

        const auto expected_group_type = groupTypeForSpec(*group.spec);
        RTP_LLM_CHECK_WITH_INFO(expected_group_type != CacheGroupType::LINEAR
                                    || group.policy.group_type == CacheGroupType::LINEAR,
                                "CacheModuleConfig::setTopology group %zu tag=%s policy type %s does not match "
                                "spec type %d",
                                gid,
                                group.tag.c_str(),
                                cacheGroupTypeName(group.policy.group_type),
                                static_cast<int>(group.spec->type));

        group.spec = group.spec->clone();
        RTP_LLM_CHECK_WITH_INFO(group.spec->seq_size_per_block > 0,
                                "CacheModuleConfig::setTopology tag=%s has zero seq_size_per_block",
                                group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.spec->kernel_seq_size_per_block > 0,
                                "CacheModuleConfig::setTopology tag=%s has zero kernel_seq_size_per_block",
                                group.tag.c_str());
    }

    cache_topology = CacheTopology::create(std::move(new_groups), std::move(new_layers));
}

void CacheModuleConfig::setGroupBlockLayout(const std::vector<std::string>& group_tags,
                                            const std::vector<uint32_t>&    block_nums,
                                            const std::vector<size_t>&      kv_block_stride_bytes,
                                            const std::vector<size_t>&      kv_scale_stride_bytes) {
    const size_t group_num = topology().groups().size();
    RTP_LLM_CHECK_WITH_INFO(group_tags.size() == group_num,
                            "CacheModuleConfig::setGroupBlockLayout tags size %zu != group size %zu",
                            group_tags.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(block_nums.size() == group_num,
                            "CacheModuleConfig::setGroupBlockLayout block_nums size %zu != group size %zu",
                            block_nums.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(kv_block_stride_bytes.size() == group_num,
                            "CacheModuleConfig::setGroupBlockLayout kv stride size %zu != group size %zu",
                            kv_block_stride_bytes.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(kv_scale_stride_bytes.size() == group_num,
                            "CacheModuleConfig::setGroupBlockLayout scale stride size %zu != group size %zu",
                            kv_scale_stride_bytes.size(),
                            group_num);
    auto                            groups = topology().groups();
    std::unordered_set<std::string> seen_tags;
    for (size_t row = 0; row < group_num; ++row) {
        const auto& tag = group_tags[row];
        RTP_LLM_CHECK_WITH_INFO(
            seen_tags.emplace(tag).second, "CacheModuleConfig::setGroupBlockLayout duplicate tag=%s", tag.c_str());
        auto group = std::find_if(
            groups.begin(), groups.end(), [&](const GroupBase& candidate) { return candidate.tag == tag; });
        RTP_LLM_CHECK_WITH_INFO(
            group != groups.end(), "CacheModuleConfig::setGroupBlockLayout unknown tag=%s", tag.c_str());
        group->block_num = block_nums[row];
        RTP_LLM_CHECK_WITH_INFO(group->kvBlockStrideBytes() == kv_block_stride_bytes[row]
                                    && group->kvScaleStrideBytes() == kv_scale_stride_bytes[row],
                                "CacheModuleConfig layout must match Spec for tag=%s",
                                tag.c_str());
    }
    cache_topology = CacheTopology::create(std::move(groups), topology().layers());
}

void CacheModuleConfig::fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                                         const std::vector<std::vector<int>>& layers_by_group,
                                         const std::vector<CacheGroupType>&   types,
                                         const std::vector<std::string>&      tags,
                                         const std::vector<CacheGroupPolicy>& policies) {
    const size_t group_num = specs.size();
    RTP_LLM_CHECK_WITH_INFO(group_num > 0, "CacheModuleConfig::fromGroupedSpecs requires at least one cache spec");
    RTP_LLM_CHECK_WITH_INFO(layers_by_group.size() == group_num,
                            "CacheModuleConfig::fromGroupedSpecs layer group count %zu != spec count %zu",
                            layers_by_group.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(types.size() == group_num,
                            "CacheModuleConfig::fromGroupedSpecs group type count %zu != spec count %zu",
                            types.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(tags.empty() || tags.size() == group_num,
                            "CacheModuleConfig::fromGroupedSpecs tag count %zu != spec count %zu",
                            tags.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(policies.empty() || policies.size() == group_num,
                            "CacheModuleConfig::fromGroupedSpecs policy count %zu != spec count %zu",
                            policies.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "CacheModuleConfig::fromGroupedSpecs requires positive layer_num");

    const bool             has_explicit_policies = !policies.empty();
    std::vector<GroupBase> new_groups;
    std::vector<LayerBase> new_layers(static_cast<size_t>(layer_num));
    new_groups.reserve(group_num);
    for (size_t layer_id = 0; layer_id < new_layers.size(); ++layer_id) {
        new_layers[layer_id].layer_id = static_cast<int>(layer_id);
    }

    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto& spec = specs[gid];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheModuleConfig::fromGroupedSpecs got null spec at group %zu", gid);
        std::string tag = tags.empty() ? spec->tag : tags[gid];
        if (tag.empty() && group_num == 1) {
            tag = "default";
        }
        RTP_LLM_CHECK_WITH_INFO(
            !tag.empty(), "CacheModuleConfig::fromGroupedSpecs requires non-empty tag for cache spec %zu", gid);
        auto stored_spec = spec->clone();
        stored_spec->tag = tag;

        GroupBase group;
        group.tag    = tag;
        group.spec   = stored_spec;
        group.policy = has_explicit_policies ? policies[gid] : defaultCacheGroupPolicy(types[gid]);
        RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == types[gid],
                                "CacheModuleConfig::fromGroupedSpecs policy type mismatch gid=%zu policy=%d type=%d",
                                gid,
                                static_cast<int>(group.policy.group_type),
                                static_cast<int>(types[gid]));
        new_groups.push_back(group);

        for (int layer_id : layers_by_group[gid]) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < new_layers.size(),
                                    "CacheModuleConfig::fromGroupedSpecs tag=%s has invalid layer id %d for "
                                    "layer_num=%u",
                                    tag.c_str(),
                                    layer_id,
                                    layer_num);
            auto& layer = new_layers[static_cast<size_t>(layer_id)];
            layer.group_tags.push_back(tag);
        }
    }

    setTopology(std::move(new_groups), std::move(new_layers));
}

void CacheModuleConfig::finalizeBlockNums(uint32_t                       global_block_num,
                                           const RuntimeConfig&           runtime_config,
                                           const LinearPrefillPoolRatio*  linear_prefill_pool_ratio) {
    RTP_LLM_CHECK_WITH_INFO(global_block_num > 0, "cache configuration requires positive baseline block count");
    // TODO: use RuntimeConfig when group-level block sizing needs runtime parallelism context.
    (void)runtime_config;

    if (groupNums() == 0) {
        return;
    }

    auto groups = topology().groups();
    for (auto& group : groups) {
        const auto explicit_independent_blocks = group.policy.explicit_block_num;
        uint32_t   rule_blocks                 = global_block_num;
        if (explicit_independent_blocks > 0) {
            rule_blocks = explicit_independent_blocks;
        } else if (linear_prefill_pool_ratio != nullptr && linear_prefill_pool_ratio->reference_pages > 0
                   && group.policy.memory_placement == CacheMemoryPlacement::DEVICE
                   && group.supportsBorrowedLinearPrefill(seq_size_per_block)) {
            const auto linear_blocks = linear_prefill_pool_ratio->physicalBlocks(global_block_num);
            RTP_LLM_CHECK_WITH_INFO(linear_blocks <= LinearPrefillPoolRatio::kMaxPhysicalBlocks,
                                    "Prefill LINEAR pool block count exceeds the block index range");
            rule_blocks = static_cast<uint32_t>(linear_blocks);
        }
        group.block_num = rule_blocks;
    }
    // The published topology already owns frozen Specs; changing capacity does
    // not require cloning their immutable byte layouts again.
    cache_topology = CacheTopology::create(std::move(groups), topology().layers());
}

size_t CacheModuleConfig::blockSizeBytesForGroup(std::string_view group_tag) const {
    size_t      total = 0;
    const auto& tag   = topology().group(group_tag).tag;
    for (int layer_id : topology().layerIdsForGroup(group_tag)) {
        const auto&  group       = topology().groupForLayer(layer_id, tag);
        const size_t kv_bytes    = group.kvBlockStrideBytes();
        const size_t scale_bytes = group.kvScaleStrideBytes();
        RTP_LLM_CHECK_WITH_INFO(scale_bytes <= std::numeric_limits<size_t>::max() - kv_bytes,
                                "CacheModuleConfig tag=%s layer=%d stride overflow",
                                tag.c_str(),
                                layer_id);
        const size_t layer_bytes = kv_bytes + scale_bytes;
        RTP_LLM_CHECK_WITH_INFO(layer_bytes <= std::numeric_limits<size_t>::max() - total,
                                "CacheModuleConfig tag=%s block size overflow",
                                tag.c_str());
        total += layer_bytes;
    }
    return total;
}

size_t CacheModuleConfig::totalGroupBlockSizeBytes() const {
    size_t total = 0;
    for (const auto& group : topology().groups()) {
        const size_t group_bytes = blockSizeBytesForGroup(group.tag);
        RTP_LLM_CHECK_WITH_INFO(group_bytes <= std::numeric_limits<size_t>::max() - total,
                                "CacheModuleConfig total block size overflow");
        total += group_bytes;
    }
    return total;
}

std::string CacheModuleConfig::debugString(size_t indent) const {
    const std::string indent_str = std::string(indent, ' ');
    const std::string indent1    = indent_str + "  ";

    std::ostringstream os;
    os << indent_str << "CacheModuleConfig{\n";

#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

    os << indent1 << "# Module Configuration:\n";
    OUTPUT_FIELD_EXPR("dtype", static_cast<int>(dtype));
    OUTPUT_FIELD(layer_num);
    OUTPUT_FIELD_EXPR("use_mla", (use_mla ? "true" : "false"));
    os << "\n";

    os << indent1 << "# Block Sizing Information:\n";
    OUTPUT_FIELD_EXPR("total_group_block_size_bytes", totalGroupBlockSizeBytes());
    os << "\n";

    const auto&                   topology_groups = topology().groups();
    std::vector<uint32_t>         group_block_nums;
    std::vector<std::vector<int>> layers_by_group;
    layers_by_group.reserve(topology_groups.size());
    for (const auto& group : topology_groups) {
        layers_by_group.push_back(layerIdsForGroup(group.tag));
        group_block_nums.push_back(group.block_num);
    }

    os << indent1 << "# Attention Configuration:\n";
    OUTPUT_FIELD(reuse_cache_tail_blocks);
    OUTPUT_FIELD_EXPR("full_group_num",
                      std::count_if(topology_groups.begin(), topology_groups.end(), [](const GroupBase& group) {
                          return group.policy.group_type == CacheGroupType::FULL;
                      }));
    OUTPUT_FIELD_EXPR("linear_group_num",
                      std::count_if(topology_groups.begin(), topology_groups.end(), [](const GroupBase& group) {
                          return group.policy.group_type == CacheGroupType::LINEAR;
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
    OUTPUT_FIELD_EXPR("group_policies.size()", topology_groups.size());
    os << indent1 << "group_types=[";
    for (size_t i = 0; i < topology_groups.size(); ++i) {
        os << static_cast<int>(topology_groups[i].policy.group_type);
        if (i + 1 < topology_groups.size()) {
            os << ",";
        }
    }
    os << "]\n";
    OUTPUT_FIELD_EXPR("group_tags.size()", topology_groups.size());
    os << indent1 << "group_tags=[";
    for (size_t i = 0; i < topology_groups.size(); ++i) {
        os << topology_groups[i].tag;
        if (i + 1 < topology_groups.size()) {
            os << ",";
        }
    }
    os << "]\n";
    OUTPUT_FIELD_EXPR("layer_to_group_tags.size()", topology().layers().size());
    os << indent1 << "layer_to_group_tags=[";
    bool first_layer = true;
    for (const auto& layer : topology().layers()) {
        if (!first_layer) {
            os << ", ";
        }
        first_layer = false;
        os << "[";
        for (size_t i = 0; i < layer.group_tags.size(); ++i) {
            if (i != 0) {
                os << ",";
            }
            os << layer.group_tags[i];
        }
        os << "]";
    }
    os << "]\n";
    os << "\n";

#undef OUTPUT_FIELD
#undef OUTPUT_FIELD_EXPR

    os << indent_str << "}\n";
    return os.str();
}

}  // namespace rtp_llm
