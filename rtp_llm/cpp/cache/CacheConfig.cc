#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

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
       << ", explicit_block_num=" << policy.explicit_block_num << ", active_tail_blocks=" << policy.active_tail_blocks
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
        os << ", seq_size_per_block=" << target_config.seqSizePerBlockForGroup(group.tag)
           << ", kernel_seq_size_per_block=" << target_config.kernelSeqSizePerBlockForGroup(group.tag)
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
            && target_group.spec->layoutFingerprint() == source_group.spec->layoutFingerprint()
            && target_config.seqSizePerBlockForGroup(target_group.tag)
                   == propose_config.seqSizePerBlockForGroup(source_group.tag)
            && target_config.kernelSeqSizePerBlockForGroup(target_group.tag)
                   == propose_config.kernelSeqSizePerBlockForGroup(source_group.tag)
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
                     propose_config.seqSizePerBlockForGroup(source_group.tag),
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
    if (finalized_global_block_num_.has_value()) {
        return *finalized_global_block_num_;
    }
    const GroupBase* canonical = nullptr;
    if (use_independent_block_pools) {
        for (const auto& group : topology().groups()) {
            const bool global_paged_group = group.policy.explicit_block_num == 0
                                            && (group.policy.group_type == CacheGroupType::FULL
                                                || group.policy.group_type == CacheGroupType::LINEAR);
            if (!global_paged_group) {
                continue;
            }
            if (canonical == nullptr) {
                canonical = &group;
            } else {
                RTP_LLM_CHECK_WITH_INFO(canonical->block_num == group.block_num,
                                        "global paged cache groups have inconsistent block counts: %s=%u %s=%u",
                                        canonical->tag.c_str(),
                                        canonical->block_num,
                                        group.tag.c_str(),
                                        group.block_num);
            }
        }
    } else {
        for (const auto& group : topology().groups()) {
            if (canonical == nullptr) {
                canonical = &group;
            } else {
                RTP_LLM_CHECK_WITH_INFO(canonical->block_num == group.block_num,
                                        "shared-pool cache groups have inconsistent block counts: %s=%u %s=%u",
                                        canonical->tag.c_str(),
                                        canonical->block_num,
                                        group.tag.c_str(),
                                        group.block_num);
            }
        }
    }
    if (canonical != nullptr) {
        return canonical->block_num;
    }
    RTP_LLM_FAIL("CacheConfig::blockNum requires a finalized global block count or a global FULL/LINEAR group");
}

uint32_t CacheConfig::blockNumForGroup(std::string_view tag) const {
    const auto explicit_block_num = policyForGroup(tag).explicit_block_num;
    if (finalized_global_block_num_ == 1) {
        return 1;
    }
    if (explicit_block_num > 0) {
        return explicit_block_num;
    }
    if (!finalized_global_block_num_.has_value()) {
        return group(tag).block_num;
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

size_t CacheConfig::layerBlockStrideBytes(int layer_id) const {
    size_t result = 0;
    for (const auto& group_ref : groupsForLayer(layer_id)) {
        const auto& group = group_ref.get();
        result            = std::max(result, group.kv_block_stride_bytes + group.kv_scale_stride_bytes);
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
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block == propose_config.seq_size_per_block,
                            "CacheConfig::mergeMTPModule global seq_size_per_block mismatch: main=%zu propose=%zu",
                            seq_size_per_block,
                            propose_config.seq_size_per_block);

    auto sub_cfg           = std::make_shared<CacheConfig>(propose_config);
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

    const auto target_group_num  = target_groups.size();
    const auto default_alias_tag = resolveDefaultMTPGroupAlias(*this, propose_config);

    std::vector<GroupBase> sub_groups;
    std::vector<LayerBase> sub_layers(static_cast<size_t>(mtp_layer_num));
    sub_groups.reserve(target_group_num);

    for (auto& target_group : target_groups) {
        const auto& tag                = target_group.tag;
        const bool  has_exact_group    = propose_config.topology().containsTag(tag);
        const bool  uses_default_alias = !has_exact_group && default_alias_tag.has_value() && tag == *default_alias_tag;
        const bool  has_propose_group  = has_exact_group || uses_default_alias;
        const auto& source_group       = has_exact_group    ? propose_config.group(tag) :
                                         uses_default_alias ? propose_config.group("default") :
                                                              target_group;

        if (has_propose_group) {
            RTP_LLM_CHECK_WITH_INFO(target_group.spec->layoutFingerprint() == source_group.spec->layoutFingerprint()
                                        && samePolicy(target_group.policy, source_group.policy)
                                        && seqSizePerBlockForGroup(target_group.tag)
                                               == propose_config.seqSizePerBlockForGroup(source_group.tag)
                                        && kernelSeqSizePerBlockForGroup(target_group.tag)
                                               == propose_config.kernelSeqSizePerBlockForGroup(source_group.tag)
                                        && target_group.kv_block_stride_bytes == source_group.kv_block_stride_bytes
                                        && target_group.kv_scale_stride_bytes == source_group.kv_scale_stride_bytes,
                                    "CacheConfig::mergeMTPModule incompatible group tag=%s",
                                    tag.c_str());
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

            const size_t main_group_layer_num = static_cast<size_t>(std::count_if(
                target_group.layer_ids.begin(), target_group.layer_ids.end(), [main_layer_num](int layer_id) {
                    return layer_id >= 0 && static_cast<uint32_t>(layer_id) < main_layer_num;
                }));
            const size_t expected_existing_layers =
                main_group_layer_num + static_cast<size_t>(module_index) * mtp_layer_num;
            RTP_LLM_CHECK_WITH_INFO(target_group.layer_ids.size() == expected_existing_layers,
                                    "CacheConfig::mergeMTPModule source_tag=%s target_tag=%s "
                                    "physical group alignment mismatch: "
                                    "existing_layers=%zu expected=%zu module=%d main_layer_num=%u module_layers=%u",
                                    source_group.tag.c_str(),
                                    tag.c_str(),
                                    target_group.layer_ids.size(),
                                    expected_existing_layers,
                                    module_index,
                                    main_layer_num,
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
                             propose_config.seqSizePerBlockForGroup(source_group.tag),
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

    sub_cfg->setTopology(std::move(sub_groups), std::move(sub_layers));
    layer_all_num = static_cast<uint32_t>(total_layers);
    setTopology(std::move(target_groups), std::move(target_layers));
    return sub_cfg;
}

void CacheConfig::finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
    // TODO: use RuntimeConfig when group-level block sizing needs runtime parallelism context.
    (void)runtime_config;
    if (global_block_num > 0) {
        for (auto& sub_cfg : mtp_sub_configs) {
            if (sub_cfg != nullptr) {
                sub_cfg->finalizeBlockNums(global_block_num, runtime_config);
            }
        }
    }

    if (!use_independent_block_pools || groupNums() == 0) {
        if (groupNums() > 0) {
            auto groups = copyGroups(topology());
            for (auto& group : groups) {
                group.block_num = global_block_num;
            }
            setTopology(std::move(groups), topology().layers());
        }
        return;
    }

    if (global_block_num > 0) {
        finalized_global_block_num_ = global_block_num;
    }

    const auto step   = static_cast<uint32_t>(std::max(1, linear_step));
    auto       groups = copyGroups(topology());
    for (auto& group : groups) {
        const auto explicit_independent_blocks = group.policy.explicit_block_num;
        uint32_t   rule_blocks                 = global_block_num;
        if (explicit_independent_blocks > 0) {
            rule_blocks = explicit_independent_blocks;
        } else if (group.policy.group_type == CacheGroupType::SWA) {
            rule_blocks = global_block_num / step + (global_block_num % step != 0 ? 1u : 0u);
        }
        group.block_num = rule_blocks;
    }
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
    OUTPUT_FIELD_EXPR("dtype", static_cast<int>(cacheDType()));
    OUTPUT_FIELD(layer_num);
    OUTPUT_FIELD(layer_all_num);
    OUTPUT_FIELD_EXPR("use_mla", (use_mla ? "true" : "false"));
    OUTPUT_FIELD_EXPR("is_sparse", (is_sparse ? "true" : "false"));
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
