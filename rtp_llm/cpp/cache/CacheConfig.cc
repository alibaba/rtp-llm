#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <new>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace rtp_llm {

namespace {

CacheGroupType groupTypeForSpec(const KVCacheSpec& spec) {
    return spec.type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
}

size_t checkedAdd(size_t total, size_t addition, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(addition <= std::numeric_limits<size_t>::max() - total,
                            "CacheConfig %s overflow: total=%zu addition=%zu",
                            name,
                            total,
                            addition);
    return total + addition;
}

size_t checkedMultiply(size_t lhs, size_t rhs, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(lhs == 0 || rhs <= std::numeric_limits<size_t>::max() / lhs,
                            "CacheConfig %s overflow: lhs=%zu rhs=%zu",
                            name,
                            lhs,
                            rhs);
    return lhs * rhs;
}

uint32_t checkedLayerCount(size_t layer_count) {
    RTP_LLM_CHECK_WITH_INFO(layer_count <= std::numeric_limits<uint32_t>::max(),
                            "CacheConfig layer count %zu exceeds uint32_t range",
                            layer_count);
    return static_cast<uint32_t>(layer_count);
}

}  // namespace

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.group_type == rhs.group_type && lhs.enable_prefix_reuse == rhs.enable_prefix_reuse
           && lhs.evict_policy == rhs.evict_policy && lhs.reservable == rhs.reservable
           && lhs.explicit_block_num == rhs.explicit_block_num && lhs.active_tail_blocks == rhs.active_tail_blocks
           && lhs.validate_tail_blocks == rhs.validate_tail_blocks && lhs.cp_mapping == rhs.cp_mapping
           && lhs.cp_slice == rhs.cp_slice;
}

CacheConfig::CacheConfig(std::vector<CacheGroup> new_groups,
                         std::vector<CacheLayer> new_layers,
                         uint32_t                main_layer_num):
    layer_num(main_layer_num), layer_all_num(checkedLayerCount(new_layers.size())) {
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0 && layer_num <= layer_all_num,
                            "CacheConfig main layer count %u must be in range [1, %u]",
                            layer_num,
                            layer_all_num);
    RTP_LLM_CHECK_WITH_INFO(!new_groups.empty(), "CacheConfig requires at least one cache group");
    RTP_LLM_CHECK_WITH_INFO(!new_layers.empty(), "CacheConfig requires at least one cache layer");
    for (size_t idx = 0; idx < new_groups.size(); ++idx) {
        auto& group = new_groups[idx];
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheConfig got null spec at group %zu", idx);
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheConfig requires tag for group %zu", idx);

        const auto expected_group_type = groupTypeForSpec(*group.spec);
        RTP_LLM_CHECK_WITH_INFO(expected_group_type != CacheGroupType::LINEAR
                                    || group.policy.group_type == CacheGroupType::LINEAR,
                                "CacheConfig group %zu tag=%s policy type %s does not match spec type %d",
                                idx,
                                group.tag.c_str(),
                                cacheGroupTypeName(group.policy.group_type),
                                static_cast<int>(group.spec->type));

        if (group.kv_block_stride_bytes == 0) {
            group.kv_block_stride_bytes = group.spec->block_size_bytes();
        }
        if (group.kv_scale_stride_bytes == 0) {
            group.kv_scale_stride_bytes = group.spec->scale_block_size_bytes();
        }
    }

    std::unordered_map<std::string, size_t>           new_tag_to_idx;
    std::unordered_map<std::string, std::vector<int>> new_tag_to_layer_ids;
    validateAndBuildIndex(new_groups, new_layers, new_tag_to_idx, new_tag_to_layer_ids);

    groups_.swap(new_groups);
    layers_.swap(new_layers);
    tag_to_idx_.swap(new_tag_to_idx);
    tag_to_layer_ids_.swap(new_tag_to_layer_ids);
}

CacheConfig& CacheConfig::operator=(CacheConfig&& other) noexcept {
    if (this != &other) {
        this->~CacheConfig();
        new (this) CacheConfig(std::move(other));
    }
    return *this;
}

void CacheConfig::validateAndBuildIndex(std::vector<CacheGroup>&                           groups,
                                        const std::vector<CacheLayer>&                     layers,
                                        std::unordered_map<std::string, size_t>&           tag_to_idx,
                                        std::unordered_map<std::string, std::vector<int>>& tag_to_layer_ids) {
    for (size_t idx = 0; idx < groups.size(); ++idx) {
        auto& group = groups[idx];
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheConfig group %zu has empty tag", idx);
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheConfig tag=%s has null spec", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            tag_to_idx.emplace(group.tag, idx).second, "CacheConfig duplicate group tag=%s", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            group.spec->seq_size_per_block > 0, "CacheConfig tag=%s has zero seq_size_per_block", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.spec->kernel_seq_size_per_block > 0
                                    && group.spec->seq_size_per_block % group.spec->kernel_seq_size_per_block == 0,
                                "CacheConfig tag=%s physical block %u must be divisible by kernel block %u",
                                group.tag.c_str(),
                                group.spec->seq_size_per_block,
                                group.spec->kernel_seq_size_per_block);
        tag_to_layer_ids.emplace(group.tag, std::vector<int>{});
    }

    for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
        std::unordered_set<std::string> seen;
        RTP_LLM_CHECK_WITH_INFO(!layers[layer_id].empty(), "CacheConfig layer=%zu has no cache group", layer_id);
        for (const auto& tag : layers[layer_id]) {
            RTP_LLM_CHECK_WITH_INFO(!tag.empty(), "CacheConfig layer=%zu has empty tag", layer_id);
            RTP_LLM_CHECK_WITH_INFO(
                tag_to_idx.count(tag) != 0, "CacheConfig layer=%zu references missing tag=%s", layer_id, tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(
                seen.insert(tag).second, "CacheConfig layer=%zu has duplicate tag=%s", layer_id, tag.c_str());
            tag_to_layer_ids.at(tag).push_back(static_cast<int>(layer_id));
        }
    }
}

const CacheGroup& CacheConfig::group(std::string_view tag) const {
    const std::string value(tag);
    const auto        it = tag_to_idx_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != tag_to_idx_.end(), "CacheConfig missing tag=%s", value.c_str());
    return groups_[it->second];
}

const CacheLayer& CacheConfig::groupsForLayer(int layer_id) const {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers_.size(),
                            "CacheConfig invalid layer_id=%d size=%zu",
                            layer_id,
                            layers_.size());
    return layers_[static_cast<size_t>(layer_id)];
}

const std::vector<int>& CacheConfig::groupLayerIds(std::string_view tag) const {
    const std::string value(tag);
    const auto        it = tag_to_layer_ids_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != tag_to_layer_ids_.end(), "CacheConfig missing tag=%s", value.c_str());
    return it->second;
}

const CacheGroup& CacheConfig::groupForLayer(int layer_id, std::string_view tag) const {
    const auto&       tags = groupsForLayer(layer_id);
    const std::string value(tag);
    RTP_LLM_CHECK_WITH_INFO(std::find(tags.begin(), tags.end(), value) != tags.end(),
                            "CacheConfig layer=%d does not own tag=%s",
                            layer_id,
                            value.c_str());
    return group(tag);
}

const CacheGroup& CacheConfig::soleGroupForLayer(int layer_id) const {
    const auto& tags = groupsForLayer(layer_id);
    RTP_LLM_CHECK_WITH_INFO(
        tags.size() == 1, "CacheConfig layer=%d requires exactly one group, got %zu", layer_id, tags.size());
    return group(tags.front());
}

bool CacheConfig::hasOneGroupPerLayer() const {
    return std::all_of(layers_.begin(), layers_.end(), [](const CacheLayer& layer) { return layer.size() == 1; });
}

size_t CacheConfig::pagedBlockSizeBytes() const {
    size_t bytes = 0;
    for (const auto& group_config : groups_) {
        if ((group_config.policy.group_type == CacheGroupType::FULL
             || group_config.policy.group_type == CacheGroupType::LINEAR)
            && group_config.policy.explicit_block_num == 0) {
            bytes = checkedAdd(bytes, blockSizeBytes(group_config.tag), "paged block bytes");
        }
    }
    return bytes;
}

size_t CacheConfig::swaBlockSizeBytes() const {
    size_t bytes = 0;
    for (const auto& group_config : groups_) {
        if (group_config.policy.group_type == CacheGroupType::SWA && group_config.policy.explicit_block_num == 0) {
            bytes = checkedAdd(bytes, blockSizeBytes(group_config.tag), "SWA block bytes");
        }
    }
    return bytes;
}

size_t CacheConfig::explicitlySizedPoolReserveBytes() const {
    size_t bytes = 0;
    for (const auto& group_config : groups_) {
        const auto reserve = checkedMultiply(static_cast<size_t>(group_config.policy.explicit_block_num),
                                             blockSizeBytes(group_config.tag),
                                             "explicit pool reserve bytes");
        bytes              = checkedAdd(bytes, reserve, "explicit pool reserve bytes");
    }
    return bytes;
}

size_t CacheConfig::totalGroupBlockSizeBytes() const {
    size_t bytes = 0;
    for (const auto& group_config : groups_) {
        bytes = checkedAdd(bytes, blockSizeBytes(group_config.tag), "total group block bytes");
    }
    return bytes;
}

void CacheConfig::finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
    RTP_LLM_CHECK_WITH_INFO(global_block_num > 0, "finalizeBlockNums requires positive global_block_num");
    // TODO: use RuntimeConfig when group-level block sizing needs runtime parallelism context.
    (void)runtime_config;
    block_num = global_block_num;
    for (auto& sub_cfg : mtp_sub_configs) {
        RTP_LLM_CHECK_WITH_INFO(sub_cfg != nullptr, "CacheConfig mtp_sub_config must not be null");
        sub_cfg->finalizeBlockNums(global_block_num, runtime_config);
    }

    const auto step = static_cast<uint32_t>(std::max(1, linear_step));
    for (auto& group_config : groups_) {
        const auto explicit_independent_blocks = group_config.policy.explicit_block_num;
        uint32_t   rule_blocks                 = global_block_num;
        if (explicit_independent_blocks > 0) {
            rule_blocks = explicit_independent_blocks;
        } else if (group_config.policy.group_type == CacheGroupType::SWA) {
            rule_blocks = global_block_num / step + (global_block_num % step != 0 ? 1u : 0u);
        }
        group_config.block_num = rule_blocks;
    }

    // CUDA graph and MTP modules share the physical cache pools.  Validate the
    // finalized geometry here so graph construction is not the first place that
    // discovers an incompatible sub-configuration.
    for (size_t module_idx = 0; module_idx < mtp_sub_configs.size(); ++module_idx) {
        const auto& sub_cfg = mtp_sub_configs[module_idx];
        RTP_LLM_CHECK_WITH_INFO(sub_cfg != nullptr, "CacheConfig MTP sub-config %zu must not be null", module_idx);
        RTP_LLM_CHECK_WITH_INFO(sub_cfg->groups_.size() == groups_.size(),
                                "CacheConfig MTP sub-config %zu group count mismatch: main=%zu sub=%zu",
                                module_idx,
                                groups_.size(),
                                sub_cfg->groups_.size());
        for (const auto& group_config : groups_) {
            const auto& sub_group = sub_cfg->group(group_config.tag);
            RTP_LLM_CHECK_WITH_INFO(sub_group.spec != nullptr,
                                    "CacheConfig MTP sub-config %zu tag=%s has null spec",
                                    module_idx,
                                    group_config.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(
                sub_group.seqSizePerBlock() == group_config.seqSizePerBlock()
                    && sub_group.kernelSeqSizePerBlock() == group_config.kernelSeqSizePerBlock(),
                "CacheConfig MTP sub-config %zu tag=%s geometry mismatch: main=(%zu,%zu) sub=(%zu,%zu)",
                module_idx,
                group_config.tag.c_str(),
                group_config.seqSizePerBlock(),
                group_config.kernelSeqSizePerBlock(),
                sub_group.seqSizePerBlock(),
                sub_group.kernelSeqSizePerBlock());
            RTP_LLM_CHECK_WITH_INFO(sub_group.block_num == group_config.block_num,
                                    "CacheConfig MTP sub-config %zu tag=%s block count mismatch: main=%u sub=%u",
                                    module_idx,
                                    group_config.tag.c_str(),
                                    group_config.block_num,
                                    sub_group.block_num);
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

    os << indent1 << "# Model Configuration:\n";
    OUTPUT_FIELD_EXPR("dtype", static_cast<int>(dtype));
    OUTPUT_FIELD(layer_num);
    OUTPUT_FIELD(layer_all_num);
    OUTPUT_FIELD_EXPR("use_mla", (use_mla ? "true" : "false"));
    os << "\n";

    os << indent1 << "# Block Configuration:\n";
    OUTPUT_FIELD(block_num);
    OUTPUT_FIELD(seq_size_per_block);
    os << "\n";

    os << indent1 << "# Block Sizing Information:\n";
    OUTPUT_FIELD_EXPR("paged_block_size_bytes", pagedBlockSizeBytes());
    OUTPUT_FIELD_EXPR("swa_block_size_bytes", swaBlockSizeBytes());
    OUTPUT_FIELD_EXPR("explicitly_sized_pool_reserve_bytes", explicitlySizedPoolReserveBytes());
    OUTPUT_FIELD_EXPR("total_group_block_size_bytes", totalGroupBlockSizeBytes());
    os << "\n";

    // Debug rendering walks the tagged group records directly; the printed order
    // is local storage order and carries no identity.
    const auto&                   topology_groups = groups();
    std::vector<CacheGroupPolicy> group_policies;
    std::vector<uint32_t>         group_block_nums;
    std::vector<std::string>      group_tags;
    std::vector<std::vector<int>> layers_by_group;
    group_policies.reserve(topology_groups.size());
    group_tags.reserve(topology_groups.size());
    layers_by_group.reserve(topology_groups.size());
    for (const auto& group : topology_groups) {
        group_policies.push_back(group.policy);
        group_tags.push_back(group.tag);
        layers_by_group.push_back(groupLayerIds(group.tag));
        group_block_nums.push_back(group.block_num);
    }

    std::vector<std::vector<std::string>> layer_to_group_tags;
    layer_to_group_tags.reserve(layers().size());
    for (const auto& layer : layers()) {
        layer_to_group_tags.push_back(layer);
    }

    os << indent1 << "# Attention Configuration:\n";
    OUTPUT_FIELD(linear_step);
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
    OUTPUT_FIELD_EXPR("layer_to_group_tags.size()", layer_to_group_tags.size());
    os << indent1 << "layer_to_group_tags=[";
    for (size_t layer_index = 0; layer_index < layer_to_group_tags.size(); ++layer_index) {
        if (layer_index > 0) {
            os << ",";
        }
        os << "[";
        for (size_t tag_index = 0; tag_index < layer_to_group_tags[layer_index].size(); ++tag_index) {
            if (tag_index > 0) {
                os << ",";
            }
            os << layer_to_group_tags[layer_index][tag_index];
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
