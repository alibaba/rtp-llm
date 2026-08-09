#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

CacheConfig::CacheConfig(uint32_t               main_layer_num,
                         uint32_t               total_layer_num,
                         bool                   mla,
                         bool                   sparse,
                         bool                   hybrid_attention,
                         size_t                 block_seq_size,
                         std::vector<GroupBase> groups,
                         std::vector<LayerBase> layers):
    configured_sparse_(sparse),
    layer_num(main_layer_num),
    layer_all_num(total_layer_num),
    use_mla(mla),
    enable_hybrid_attention(hybrid_attention),
    seq_size_per_block(block_seq_size) {
    setTopology(std::move(groups), std::move(layers));
}

namespace {

CacheGroupType groupTypeForSpec(const KVCacheSpec& spec) {
    return spec.type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
}

std::vector<GroupBase> copyGroups(const CacheTopology& topology) {
    const auto groups = topology.groups();
    return {groups.begin(), groups.end()};
}

bool checkedAdd(size_t lhs, size_t rhs, size_t* result) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool checkedMul(size_t lhs, size_t rhs, size_t* result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

}  // namespace

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.group_type == rhs.group_type && lhs.enable_prefix_reuse == rhs.enable_prefix_reuse
           && lhs.evict_policy == rhs.evict_policy && lhs.reservable == rhs.reservable
           && lhs.explicit_block_num == rhs.explicit_block_num && lhs.active_tail_blocks == rhs.active_tail_blocks
           && lhs.validate_tail_blocks == rhs.validate_tail_blocks && lhs.cp_mapping == rhs.cp_mapping
           && lhs.cp_slice == rhs.cp_slice;
}

std::optional<std::string> CacheConfig::kernelAddressedFullGroupTag() const {
    std::optional<std::string> attention_selected;
    std::optional<std::string> opaque_selected;
    std::optional<std::string> any_selected;
    for (const auto& group : topology().groups()) {
        const auto& spec = group.spec;
        const bool  kernel_addressed =
            group.policy.group_type == CacheGroupType::FULL && spec
            && (spec->type == KVCacheSpecType::MultiHeadAttention
                || spec->type == KVCacheSpecType::MultiHeadLatentAttention || spec->type == KVCacheSpecType::OpaqueKV);
        if (!kernel_addressed) {
            continue;
        }
        if (!any_selected.has_value()) {
            any_selected = group.tag;
        } else {
            RTP_LLM_CHECK_WITH_INFO(kernelSeqSizePerBlockForGroup(group.tag)
                                        == kernelSeqSizePerBlockForGroup(*any_selected),
                                    "kernel-addressed cache groups must use one kernel block size");
        }
        // Attention groups define the dtype and kernel geometry exposed to the
        // model, so prefer them over opaque regions regardless of topology order.
        auto& selected = spec->type == KVCacheSpecType::OpaqueKV ? opaque_selected : attention_selected;
        if (!selected.has_value()) {
            selected = group.tag;
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(spec->memoryLayoutDType() == specForGroup(*selected)->memoryLayoutDType(),
                                "kernel-addressed cache groups must use one dtype, tag=%s dtype=%d vs tag=%s dtype=%d",
                                group.tag.c_str(),
                                static_cast<int>(spec->memoryLayoutDType()),
                                selected->c_str(),
                                static_cast<int>(specForGroup(*selected)->memoryLayoutDType()));
    }
    return attention_selected.has_value() ? attention_selected : opaque_selected;
}

uint32_t CacheConfig::totalLayerNum() const {
    return static_cast<uint32_t>(topology().layers().size());
}

uint32_t CacheConfig::blockNum() const {
    RTP_LLM_CHECK_WITH_INFO(finalized_global_block_num_.has_value(),
                            "CacheConfig::blockNum requires a finalized global block count");
    return *finalized_global_block_num_;
}

uint32_t CacheConfig::blockNumForGroup(std::string_view tag) const {
    const auto explicit_block_num = policyForGroup(tag).explicit_block_num;
    if (finalized_global_block_num_ == 1) {
        return 1;
    }
    if (explicit_block_num > 0) {
        return explicit_block_num;
    }
    RTP_LLM_CHECK_WITH_INFO(finalized_global_block_num_.has_value(),
                            "CacheConfig::blockNumForGroup requires finalized config for dynamic group tag=%s",
                            std::string(tag).c_str());
    if (policyForGroup(tag).group_type == CacheGroupType::SWA) {
        const auto step = static_cast<uint32_t>(std::max(1, linear_step));
        return *finalized_global_block_num_ / step + (*finalized_global_block_num_ % step != 0 ? 1u : 0u);
    }
    return *finalized_global_block_num_;
}

size_t CacheConfig::groupLayerNum() const {
    size_t result = 0;
    for (const auto& group : topology().groups()) {
        result = std::max(result, group.layer_ids.size());
    }
    return result;
}

size_t CacheConfig::explicitReserveBytesForGroup(std::string_view tag) const {
    const auto& group = topology().group(tag);
    if (group.policy.explicit_block_num == 0) {
        return 0;
    }
    size_t stride     = 0;
    size_t slot_bytes = 0;
    size_t reserve    = 0;
    RTP_LLM_CHECK_WITH_INFO(checkedAdd(group.kv_block_stride_bytes, group.kv_scale_stride_bytes, &stride)
                                && checkedMul(group.layer_ids.size(), stride, &slot_bytes),
                            "kv cache explicit slot bytes overflow for group tag=%s",
                            group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(checkedMul(static_cast<size_t>(group.policy.explicit_block_num), slot_bytes, &reserve),
                            "kv cache explicit bytes overflow for group tag=%s block_num=%u slot_bytes=%zu",
                            group.tag.c_str(),
                            group.policy.explicit_block_num,
                            slot_bytes);
    return reserve;
}

size_t CacheConfig::explicitlySizedPoolReserveBytes() const {
    size_t result = 0;
    for (const auto& group : topology().groups()) {
        size_t next = 0;
        RTP_LLM_CHECK_WITH_INFO(checkedAdd(result, explicitReserveBytesForGroup(group.tag), &next),
                                "kv cache explicit bytes overflow for group tag=%s",
                                group.tag.c_str());
        result = next;
    }
    return result;
}

bool CacheConfig::usesTypedCacheRegions() const {
    return std::any_of(topology().groups().begin(), topology().groups().end(), [](const GroupBase& group) {
        return group.spec->type == KVCacheSpecType::OpaqueKV || group.spec->type == KVCacheSpecType::OpaqueState;
    });
}

bool CacheConfig::usesOpaqueKVCacheStore() const {
    return usesTypedCacheRegions();
}

bool CacheConfig::usesActiveOpaqueKVCacheStore() const {
    return std::any_of(topology().groups().begin(), topology().groups().end(), [](const GroupBase& group) {
        return !group.layer_ids.empty()
               && (group.spec->type == KVCacheSpecType::OpaqueKV || group.spec->type == KVCacheSpecType::OpaqueState);
    });
}

rtp_llm::DataType CacheConfig::cacheDType() const {
    const auto primary_tag = kernelAddressedFullGroupTag();
    if (primary_tag.has_value()) {
        return specForGroup(*primary_tag)->memoryLayoutDType();
    }
    RTP_LLM_CHECK_WITH_INFO(!topology().groups().empty(), "CacheConfig::cacheDType requires a cache group");
    return topology().groups().front().spec->memoryLayoutDType();
}

void CacheConfig::setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers) {
    RTP_LLM_CHECK_WITH_INFO(!new_groups.empty(), "CacheConfig::setTopology requires at least one cache group");
    RTP_LLM_CHECK_WITH_INFO(!new_layers.empty(), "CacheConfig::setTopology requires at least one cache layer");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0,
                            "CacheConfig::setTopology requires positive global seq_size_per_block");
    RTP_LLM_CHECK_WITH_INFO(layer_num == 0 || new_layers.size() >= static_cast<size_t>(layer_num),
                            "CacheConfig::setTopology layer count %zu is smaller than main layer count %u",
                            new_layers.size(),
                            layer_num);

    std::optional<size_t> full_kernel_seq_size;
    for (size_t gid = 0; gid < new_groups.size(); ++gid) {
        auto& group = new_groups[gid];
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheConfig::setTopology got null spec at group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheConfig::setTopology requires tag for group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(group.spec->tag == group.tag,
                                "CacheConfig::setTopology tag=%s does not match spec tag=%s",
                                group.tag.c_str(),
                                group.spec->tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.policy.explicit_block_num == 0 || group.policy.explicit_block_num >= 2,
                                "CacheConfig::setTopology group %zu tag=%s explicit_block_num must be 0 or >= 2, "
                                "got %u",
                                gid,
                                group.tag.c_str(),
                                group.policy.explicit_block_num);

        const auto expected_group_type = groupTypeForSpec(*group.spec);
        RTP_LLM_CHECK_WITH_INFO(expected_group_type != CacheGroupType::LINEAR
                                    || group.policy.group_type == CacheGroupType::LINEAR,
                                "CacheConfig::setTopology group %zu tag=%s policy type %s does not match spec type %d",
                                gid,
                                group.tag.c_str(),
                                cacheGroupTypeName(group.policy.group_type),
                                static_cast<int>(group.spec->type));

        group.spec                   = group.spec->clone();
        const auto physical_seq_size = group.spec->seq_size_per_block;
        const auto kernel_seq_size   = group.spec->kernel_seq_size_per_block;
        RTP_LLM_CHECK_WITH_INFO(physical_seq_size % seq_size_per_block == 0,
                                "invalid physical span for group %zu tag=%s: physical=%zu global=%zu",
                                gid,
                                group.tag.c_str(),
                                physical_seq_size,
                                seq_size_per_block);
        RTP_LLM_CHECK_WITH_INFO(kernel_seq_size > 0 && physical_seq_size >= kernel_seq_size
                                    && physical_seq_size % kernel_seq_size == 0,
                                "invalid block subdivision for group %zu tag=%s: physical=%zu kernel=%zu",
                                gid,
                                group.tag.c_str(),
                                physical_seq_size,
                                kernel_seq_size);
        if (group.policy.group_type == CacheGroupType::FULL) {
            if (!full_kernel_seq_size.has_value()) {
                full_kernel_seq_size = kernel_seq_size;
            } else {
                RTP_LLM_CHECK_WITH_INFO(*full_kernel_seq_size == kernel_seq_size,
                                        "FULL cache groups must use one kernel block size: expected=%zu "
                                        "group=%zu tag=%s actual=%zu",
                                        *full_kernel_seq_size,
                                        gid,
                                        group.tag.c_str(),
                                        kernel_seq_size);
            }
        }
        const auto expected_kv_stride    = group.spec->block_size_bytes();
        const auto expected_scale_stride = group.spec->scale_block_size_bytes();
        if (group.kv_block_stride_bytes == 0) {
            group.kv_block_stride_bytes = expected_kv_stride;
        } else {
            RTP_LLM_CHECK_WITH_INFO(group.kv_block_stride_bytes >= expected_kv_stride,
                                    "cache group %zu tag=%s kv stride %zu is smaller than spec bytes %zu",
                                    gid,
                                    group.tag.c_str(),
                                    group.kv_block_stride_bytes,
                                    expected_kv_stride);
        }
        if (group.kv_scale_stride_bytes == 0) {
            group.kv_scale_stride_bytes = expected_scale_stride;
        } else {
            RTP_LLM_CHECK_WITH_INFO(group.kv_scale_stride_bytes >= expected_scale_stride,
                                    "cache group %zu tag=%s scale stride %zu is smaller than spec bytes %zu",
                                    gid,
                                    group.tag.c_str(),
                                    group.kv_scale_stride_bytes,
                                    expected_scale_stride);
        }
        if (group.spec->type == KVCacheSpecType::MultiHeadAttention) {
            RTP_LLM_CHECK_WITH_INFO(group.kv_block_stride_bytes == expected_kv_stride
                                        && group.kv_scale_stride_bytes == expected_scale_stride,
                                    "MHA cache group %zu tag=%s does not support padded rows: "
                                    "kv=%zu/%zu scale=%zu/%zu",
                                    gid,
                                    group.tag.c_str(),
                                    group.kv_block_stride_bytes,
                                    expected_kv_stride,
                                    group.kv_scale_stride_bytes,
                                    expected_scale_stride);
        }
    }

    cache_topology = CacheTopology::create(std::move(new_groups), std::move(new_layers));
    finalized_global_block_num_.reset();
}

void CacheConfig::publishSentinelOnlyBlockNum() {
    RTP_LLM_CHECK_WITH_INFO(
        linear_step >= 1, "sentinel-only cache config requires linear_step>=1, got %d", linear_step);
    RTP_LLM_CHECK_WITH_INFO(groupNums() > 0, "sentinel-only cache config requires initialized topology");
    finalized_global_block_num_ = 1;
    for (auto& sub_cfg : mtp_sub_configs) {
        if (sub_cfg != nullptr) {
            sub_cfg->publishSentinelOnlyBlockNum();
        }
    }
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

void CacheConfig::finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
    // TODO: use RuntimeConfig when group-level block sizing needs runtime parallelism context.
    (void)runtime_config;
    RTP_LLM_CHECK_WITH_INFO(
        linear_step >= 1, "CacheConfig::finalizeBlockNums requires linear_step>=1, got %d", linear_step);
    RTP_LLM_CHECK_WITH_INFO(global_block_num >= 2,
                            "CacheConfig::finalizeBlockNums requires at least 2 total slots; "
                            "sentinel-only configs are internal, got %u",
                            global_block_num);
    for (auto& sub_cfg : mtp_sub_configs) {
        if (sub_cfg != nullptr) {
            sub_cfg->linear_step = linear_step;
            sub_cfg->finalizeBlockNums(global_block_num, runtime_config);
        }
    }

    RTP_LLM_CHECK_WITH_INFO(groupNums() > 0, "CacheConfig::finalizeBlockNums requires initialized topology");
    finalized_global_block_num_ = global_block_num;
}

std::string CacheConfig::debugString(size_t indent) const {
    const std::string indent_str = std::string(indent, ' ');
    const std::string indent1    = indent_str + "  ";

    std::ostringstream os;
    os << indent_str << "CacheConfig{\n";

#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

    os << indent1 << "# Model Configuration:\n";
    OUTPUT_FIELD_EXPR("dtype", static_cast<int>(cacheDType()));
    OUTPUT_FIELD(layer_num);
    OUTPUT_FIELD(layer_all_num);
    OUTPUT_FIELD_EXPR("use_mla", (use_mla ? "true" : "false"));
    OUTPUT_FIELD_EXPR("is_sparse", (isSparse() ? "true" : "false"));
    OUTPUT_FIELD_EXPR("enable_hybrid_attention", (enable_hybrid_attention ? "true" : "false"));
    os << "\n";

    os << indent1 << "# Block Configuration:\n";
    OUTPUT_FIELD_EXPR(
        "block_num",
        (finalized_global_block_num_.has_value() ? std::to_string(blockNum()) : std::string("<unfinalized>")));
    OUTPUT_FIELD(seq_size_per_block);
    os << "\n";

    const auto& topology_groups = topology().groups();

    os << indent1 << "# Attention Configuration:\n";
    OUTPUT_FIELD(linear_step);
    OUTPUT_FIELD_EXPR("group_layer_num", groupLayerNum());
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
        os << "{" << group.tag << ":";
        if (finalized_global_block_num_.has_value()) {
            os << blockNumForGroup(group.tag);
        } else if (group.policy.explicit_block_num > 0) {
            os << group.policy.explicit_block_num;
        } else {
            os << "<unfinalized>";
        }
        os << "}";
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
        os << indent1 << "  seq_size_per_block=" << seqSizePerBlockForGroup(group.tag) << "\n";
        os << indent1 << "  kernel_seq_size_per_block=" << kernelSeqSizePerBlockForGroup(group.tag) << "\n";
        os << indent1 << "  kernel_blocks_per_kv_block=" << kernelBlocksPerKvBlockForGroup(group.tag) << "\n";
        os << indent1 << "  kv_block_stride_bytes=" << group.kv_block_stride_bytes << "\n";
        os << indent1 << "  kv_scale_stride_bytes=" << group.kv_scale_stride_bytes << "\n";
        os << indent1 << "  group_block_size_bytes=" << blockSizeBytesForGroup(group.tag) << "\n";
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
