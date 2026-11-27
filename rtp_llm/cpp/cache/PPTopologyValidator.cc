#include "rtp_llm/cpp/cache/PPTopologyValidator.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

std::string joinTags(const std::vector<StageCacheGroupSnapshot>& groups) {
    std::ostringstream oss;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << groups[i].tag;
    }
    return oss.str();
}

PPValidationResult fail(std::string error) {
    PPValidationResult result;
    result.ok    = false;
    result.error = std::move(error);
    return result;
}

/** Fills per-tag construction inputs from the completed canonical table. */
void deriveConstructionInputs(PPValidationResult& result) {
    uint32_t paged_min = std::numeric_limits<uint32_t>::max();
    uint32_t any_max   = 0;
    result.agreed.block_num_overrides.clear();
    result.agreed.block_num_overrides.reserve(result.canonical_groups.size());
    for (const auto& entry : result.canonical_groups) {
        result.agreed.block_num_overrides.emplace(entry.tag, entry.logical_block_num);
        any_max = std::max(any_max, entry.logical_block_num);
        const bool follows_global_budget =
            entry.explicit_block_num == 0
            && (entry.type == CacheGroupType::FULL || entry.type == CacheGroupType::LINEAR);
        if (follows_global_budget) {
            paged_min = std::min(paged_min, entry.logical_block_num);
        }
    }
    if (paged_min == std::numeric_limits<uint32_t>::max()) {
        paged_min = any_max;
    }
    RTP_LLM_CHECK_WITH_INFO(paged_min > 0, "PP canonical table yielded a non-positive top-level block count");
    result.agreed.paged_block_num = paged_min;
}

CanonicalGroupEntry makeCanonicalGroupEntry(const StageCacheGroupSnapshot& group) {
    CanonicalGroupEntry entry;
    entry.tag                       = group.tag;
    entry.type                      = group.type;
    entry.seq_size_per_block        = group.seq_size_per_block;
    entry.kernel_seq_size_per_block = group.kernel_seq_size_per_block;
    entry.cache_key_token_stride    = group.cache_key_token_stride;
    entry.logical_block_num         = group.has_local_backing ? group.block_num : 0;
    entry.explicit_block_num        = group.explicit_block_num;
    entry.policy_fingerprint        = group.policy_fingerprint;
    return entry;
}

std::string
validateAndMergeCanonicalGroup(CanonicalGroupEntry& entry, const StageCacheGroupSnapshot& group, size_t stage_index) {
    if (group.type != entry.type) {
        return "stage " + std::to_string(stage_index) + " group [" + group.tag
               + "] type differs from the canonical entry";
    }
    if (group.seq_size_per_block != entry.seq_size_per_block) {
        return "stage " + std::to_string(stage_index) + " group [" + group.tag + "] seq_size_per_block "
               + std::to_string(group.seq_size_per_block) + " != canonical " + std::to_string(entry.seq_size_per_block);
    }
    if (group.kernel_seq_size_per_block != entry.kernel_seq_size_per_block) {
        return "stage " + std::to_string(stage_index) + " group [" + group.tag + "] kernel_seq_size_per_block "
               + std::to_string(group.kernel_seq_size_per_block) + " != canonical "
               + std::to_string(entry.kernel_seq_size_per_block);
    }
    if (group.cache_key_token_stride != entry.cache_key_token_stride) {
        return "stage " + std::to_string(stage_index) + " group [" + group.tag + "] cache_key_token_stride "
               + std::to_string(group.cache_key_token_stride) + " != canonical "
               + std::to_string(entry.cache_key_token_stride);
    }
    if (group.explicit_block_num != entry.explicit_block_num) {
        return "stage " + std::to_string(stage_index) + " group [" + group.tag + "] explicit_block_num "
               + std::to_string(group.explicit_block_num) + " != canonical " + std::to_string(entry.explicit_block_num);
    }
    if (group.policy_fingerprint != entry.policy_fingerprint) {
        return "stage " + std::to_string(stage_index) + " group [" + group.tag + "] policy [" + group.policy_fingerprint
               + "] != canonical [" + entry.policy_fingerprint + "]";
    }
    if (group.has_local_backing) {
        entry.logical_block_num =
            entry.logical_block_num == 0 ? group.block_num : std::min(entry.logical_block_num, group.block_num);
    }
    return {};
}

std::string validateTierSizing(const StageCacheSnapshot& stage, const BlockTreeCacheSizing& expected) {
    const auto& sizing = stage.block_tree;
    if (!std::isfinite(sizing.host_swa_block_ratio) || sizing.host_swa_block_ratio <= 0.0) {
        return "HOST SWA block ratio must be finite and positive";
    }
    if ((sizing.host_budget_bytes > 0) != (expected.host_budget_bytes > 0)
        || (sizing.disk_budget_bytes > 0) != (expected.disk_budget_bytes > 0)) {
        return "HOST/DISK cache enablement differs across PP stages";
    }
    if (sizing.host_budget_bytes == 0 && sizing.disk_budget_bytes == 0) {
        return {};
    }
    std::unordered_set<CacheGroupType> remaining_types;
    for (const auto& group : stage.groups) {
        if (group.block_tree_group_member && group.has_local_backing) {
            remaining_types.insert(group.type);
        }
    }
    for (const auto& group_set : sizing.group_sets) {
        const auto member = std::find_if(stage.groups.begin(), stage.groups.end(), [&](const auto& group) {
            return group.tag == group_set.first_tag;
        });
        if (member == stage.groups.end() || !member->block_tree_group_member || !member->has_local_backing
            || group_set.pool_stride == 0
            || remaining_types.erase(member->type) != 1) {
            return "invalid or duplicate BlockTreeCache group-set sizing";
        }
    }
    return remaining_types.empty() ? "" : "missing BlockTreeCache group-set sizing";
}

void mergeMinBlockNums(TierBlockNumOverrides& counts, const TierBlockNumOverrides& candidates) {
    for (const auto& [type, count] : candidates) {
        auto [it, inserted] = counts.emplace(type, count);
        if (!inserted) {
            it->second = std::min(it->second, count);
        }
    }
}

}  // namespace

StageCacheSnapshot StageCacheSnapshot::captureLocalStage(const CacheConfig&          config,
                                                         uint32_t                    local_block_num,
                                                         const BlockTreeCacheSizing& block_tree_sizing) {
    /** Tier weights use the first reusable tag of each type in local layer order. */
    std::unordered_set<std::string>                 local_tags;
    std::unordered_map<CacheGroupType, std::string> first_local_tag;
    for (const auto& layer : config.topology().layers()) {
        for (const auto& tag : layer.group_tags) {
            if (!local_tags.insert(tag).second) {
                continue;
            }
            const auto& policy = config.group(tag).policy;
            if (cacheGroupParticipatesInBlockTree(policy)) {
                first_local_tag.try_emplace(policy.group_type, tag);
            }
        }
    }

    StageCacheSnapshot snapshot;
    snapshot.groups.reserve(config.topology().groups().size());
    for (const auto& group : config.topology().groups()) {
        StageCacheGroupSnapshot captured;
        captured.tag                       = group.tag;
        captured.type                      = group.policy.group_type;
        captured.has_local_backing         = local_tags.count(group.tag) != 0;
        captured.block_tree_group_member   = cacheGroupParticipatesInBlockTree(group.policy);
        captured.seq_size_per_block        = group.seqSizePerBlock();
        captured.kernel_seq_size_per_block = group.kernelSeqSizePerBlock();
        captured.cache_key_token_stride    = group.cacheKeyTokenStride();
        captured.block_num                 = captured.has_local_backing ? config.blockNumForGroup(group, local_block_num) : 0;
        captured.explicit_block_num        = group.policy.explicit_block_num;
        captured.policy_fingerprint        = group.policy.policyFingerprint();
        snapshot.groups.push_back(std::move(captured));
    }
    snapshot.block_tree = block_tree_sizing;
    for (auto& group_set : snapshot.block_tree.group_sets) {
        group_set.first_tag = first_local_tag.at(config.group(group_set.first_tag).policy.group_type);
    }
    return snapshot;
}

namespace {

constexpr char kFieldSep = '|';
constexpr char kTagSep   = '\x1f';
constexpr char kNumSep   = ',';

std::vector<std::string> splitFields(const std::string& s, char sep) {
    std::vector<std::string> parts;
    std::stringstream        ss(s);
    std::string              item;
    while (std::getline(ss, item, sep)) {
        parts.push_back(item);
    }
    return parts;
}

template<typename Item, typename Getter>
std::string joinGroupNums(const std::vector<Item>& groups, Getter getter) {
    std::ostringstream oss;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (i > 0) {
            oss << kNumSep;
        }
        oss << static_cast<unsigned long long>(getter(groups[i]));
    }
    return oss.str();
}

template<typename Item, typename Getter>
std::string joinGroupStrings(const std::vector<Item>& groups, Getter getter, const char* field_name) {
    std::ostringstream oss;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (i > 0) {
            oss << kTagSep;
        }
        const auto& value = getter(groups[i]);
        RTP_LLM_CHECK_WITH_INFO(value.find_first_of("\x1f|,") == std::string::npos,
                                "PP snapshot %s contains a wire-format delimiter: %s",
                                field_name,
                                value.c_str());
        oss << value;
    }
    return oss.str();
}

template<typename T>
std::vector<T> parseNums(const std::string& field, size_t expected_size) {
    std::vector<T> values;
    if (field.empty()) {
        RTP_LLM_CHECK_WITH_INFO(expected_size == 0, "PP snapshot is missing numeric entries");
        return values;
    }
    std::stringstream ss(field);
    std::string       item;
    while (std::getline(ss, item, kNumSep)) {
        values.push_back(static_cast<T>(std::stoull(item)));
    }
    RTP_LLM_CHECK_WITH_INFO(values.size() == expected_size,
                            "PP snapshot field has %zu entries, expected %zu",
                            values.size(),
                            expected_size);
    return values;
}

}  // namespace

std::string StageCacheSnapshot::serialize() const {
    RTP_LLM_CHECK_WITH_INFO(std::isfinite(block_tree.host_swa_block_ratio) && block_tree.host_swa_block_ratio > 0.0,
                            "PP snapshot HOST SWA block ratio must be finite and positive");
    std::ostringstream oss;
    oss << joinGroupStrings(
        groups, [](const auto& group) -> const std::string& { return group.tag; }, "tag")
        << kFieldSep << joinGroupNums(groups, [](const auto& group) { return group.type; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.block_tree_group_member; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.seq_size_per_block; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.kernel_seq_size_per_block; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.cache_key_token_stride; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.block_num; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.explicit_block_num; }) << kFieldSep
        << joinGroupStrings(
               groups,
               [](const auto& group) -> const std::string& { return group.policy_fingerprint; },
               "policy fingerprint")
        << kFieldSep
        << joinGroupStrings(
               block_tree.group_sets,
               [](const auto& group_set) -> const std::string& { return group_set.first_tag; },
               "group-set first tag")
        << kFieldSep
        << joinGroupNums(block_tree.group_sets, [](const auto& group_set) { return group_set.pool_stride; })
        << kFieldSep << joinGroupNums(groups, [](const auto& group) { return group.has_local_backing; }) << kFieldSep
        << block_tree.host_budget_bytes << kFieldSep << block_tree.disk_budget_bytes << kFieldSep
        << std::setprecision(std::numeric_limits<double>::max_digits10) << block_tree.host_swa_block_ratio;
    return oss.str();
}

StageCacheSnapshot StageCacheSnapshot::deserialize(const std::string& payload) {
    const auto fields = splitFields(payload, kFieldSep);
    RTP_LLM_CHECK_WITH_INFO(fields.size() == 15, "PP snapshot payload is malformed (field count)");
    auto group_tags = splitFields(fields[0], kTagSep);
    if (fields[0].empty()) {
        group_tags.clear();
    }
    const auto group_types                 = parseNums<CacheGroupType>(fields[1], group_tags.size());
    const auto block_tree_group_membership = parseNums<uint8_t>(fields[2], group_tags.size());
    const auto seq_size_per_block          = parseNums<size_t>(fields[3], group_tags.size());
    const auto kernel_seq_size_per_block   = parseNums<size_t>(fields[4], group_tags.size());
    const auto cache_key_token_strides     = parseNums<size_t>(fields[5], group_tags.size());
    const auto block_nums                  = parseNums<uint32_t>(fields[6], group_tags.size());
    const auto explicit_block_nums         = parseNums<uint32_t>(fields[7], group_tags.size());
    const auto local_backing_flags         = parseNums<uint8_t>(fields[11], group_tags.size());
    auto       policy_fingerprints         = splitFields(fields[8], kTagSep);
    if (fields[8].empty()) {
        policy_fingerprints.clear();
    }
    RTP_LLM_CHECK_WITH_INFO(policy_fingerprints.size() == group_tags.size(),
                            "PP snapshot policy fingerprint field has %zu entries, expected %zu",
                            policy_fingerprints.size(),
                            group_tags.size());
    RTP_LLM_CHECK_WITH_INFO(std::all_of(block_tree_group_membership.begin(),
                                        block_tree_group_membership.end(),
                                        [](uint8_t value) { return value == 0 || value == 1; }),
                            "PP snapshot block-tree membership field must contain only 0 or 1");

    StageCacheSnapshot snapshot;
    snapshot.groups.reserve(group_tags.size());
    for (size_t i = 0; i < group_tags.size(); ++i) {
        StageCacheGroupSnapshot group;
        group.tag                       = std::move(group_tags[i]);
        group.type                      = group_types[i];
        group.block_tree_group_member   = block_tree_group_membership[i] != 0;
        group.has_local_backing         = local_backing_flags[i] != 0;
        group.seq_size_per_block        = seq_size_per_block[i];
        group.kernel_seq_size_per_block = kernel_seq_size_per_block[i];
        group.cache_key_token_stride    = cache_key_token_strides[i];
        group.block_num                 = block_nums[i];
        group.explicit_block_num        = explicit_block_nums[i];
        group.policy_fingerprint        = std::move(policy_fingerprints[i]);
        snapshot.groups.push_back(std::move(group));
    }
    auto       first_tags = splitFields(fields[9], kTagSep);
    const auto strides    = parseNums<size_t>(fields[10], first_tags.size());
    for (size_t i = 0; i < first_tags.size(); ++i) {
        snapshot.block_tree.group_sets.push_back({std::move(first_tags[i]), strides[i]});
    }
    snapshot.block_tree.host_budget_bytes = std::stoull(fields[12]);
    snapshot.block_tree.disk_budget_bytes = std::stoull(fields[13]);
    size_t ratio_chars                       = 0;
    snapshot.block_tree.host_swa_block_ratio = std::stod(fields[14], &ratio_chars);
    RTP_LLM_CHECK_WITH_INFO(ratio_chars == fields[14].size() && std::isfinite(snapshot.block_tree.host_swa_block_ratio)
                                && snapshot.block_tree.host_swa_block_ratio > 0.0,
                            "PP snapshot HOST SWA block ratio must be finite and positive");
    return snapshot;
}

PPValidationResult validatePPTopology(const std::vector<StageCacheSnapshot>& stages) {
    PPValidationResult result;

    if (stages.empty()) {
        result.ok = true;
        return result;
    }

    std::unordered_map<std::string, size_t> canonical_index;
    for (size_t s = 0; s < stages.size(); ++s) {
        const auto& stage = stages[s];
        if (s == 0 && stage.groups.empty()) {
            return fail("stage 0 cache snapshot has no cache groups");
        }

        std::unordered_set<std::string> unique_tags;
        for (const auto& group : stage.groups) {
            if (group.tag.empty() || !unique_tags.emplace(group.tag).second) {
                return fail("stage " + std::to_string(s) + " has an empty or duplicate cache group tag [" + group.tag
                            + "]");
            }
            if (group.has_local_backing && group.block_num < 2) {
                return fail("stage " + std::to_string(s) + " group [" + group.tag
                            + "] has invalid local block capacity " + std::to_string(group.block_num));
            }
            if (s == 0) {
                canonical_index.emplace(group.tag, result.canonical_groups.size());
                result.canonical_groups.push_back(makeCanonicalGroupEntry(group));
                continue;
            }

            const auto it = canonical_index.find(group.tag);
            if (it == canonical_index.end()) {
                return fail("stage " + std::to_string(s) + " declares logical cache group [" + group.tag
                            + "] that is absent from stage 0 [" + joinTags(stages.front().groups) + "]");
            }
            auto& entry = result.canonical_groups[it->second];
            if (const auto error = validateAndMergeCanonicalGroup(entry, group, s); !error.empty()) {
                return fail(error);
            }
        }

        if (unique_tags.size() != canonical_index.size()) {
            return fail("stage " + std::to_string(s) + " logical cache groups [" + joinTags(stage.groups)
                        + "] do not match stage 0 [" + joinTags(stages.front().groups) + "]");
        }
        if (const auto error = validateTierSizing(stage, stages.front().block_tree); !error.empty()) {
            return fail("stage " + std::to_string(s) + ": " + error);
        }
    }

    for (const auto& group : result.canonical_groups) {
        if (group.logical_block_num == 0) {
            return fail("cache group [" + group.tag + "] has no PP stage with local backing");
        }
    }
    deriveConstructionInputs(result);
    std::unordered_map<CacheGroupType, std::pair<unsigned __int128, size_t>> pool_strides;
    for (size_t s = 0; s < stages.size(); ++s) {
        const auto& sizing = stages[s].block_tree;
        if (sizing.host_budget_bytes == 0 && sizing.disk_budget_bytes == 0) {
            continue;
        }
        try {
            std::vector<GroupSetPoolSizing> pools;
            for (const auto& group_set : sizing.group_sets) {
                const auto& group = result.canonical_groups[canonical_index.at(group_set.first_tag)];
                pools.push_back({group.type, group.logical_block_num, group_set.pool_stride});
                auto& [stride_sum, owner_count] = pool_strides[group.type];
                stride_sum += group_set.pool_stride;
                ++owner_count;
            }
            const auto counts = computeHostDiskBlockCounts(pools, sizing);
            if (!counts) {
                return fail("HOST budget cannot hold a usable block per group set on stage " + std::to_string(s));
            }
            mergeMinBlockNums(result.agreed.block_tree.host, counts->host);
            mergeMinBlockNums(result.agreed.block_tree.disk, counts->disk);
        } catch (const std::exception& e) {
            return fail("BlockTreeCache capacity on stage " + std::to_string(s) + ": " + e.what());
        }
    }
    for (const auto& [type, strides] : pool_strides) {
        const auto& [stride_sum, owner_count] = strides;
        result.agreed.eviction_block_bytes[type] = static_cast<size_t>((stride_sum + owner_count - 1) / owner_count);
    }
    result.ok = true;
    return result;
}

std::vector<std::string> collectPPStagePayloads(const std::string& payload, int64_t pp_size) {
    const auto local_size = torch::tensor({static_cast<int64_t>(payload.size())}, torch::kInt64);
    auto       sizes      = torch::empty({pp_size}, torch::kInt64);
    execAllGather({{sizes}, ParallelMode::PP, {local_size}, false});

    // The existing tensor all-gather stages CPU buffers through NCCL and copies them back before returning.
    const auto* size_data = sizes.data_ptr<int64_t>();
    const auto  max_size  = *std::max_element(size_data, size_data + pp_size);
    auto        send      = torch::zeros({max_size}, torch::kUInt8);
    std::memcpy(send.data_ptr(), payload.data(), payload.size());
    auto received = torch::empty({pp_size, max_size}, torch::kUInt8);
    execAllGather({{received}, ParallelMode::PP, {send}, false});

    const auto*              data = received.data_ptr<uint8_t>();
    std::vector<std::string> stages;
    stages.reserve(static_cast<size_t>(pp_size));
    for (int64_t s = 0; s < pp_size; ++s) {
        stages.emplace_back(reinterpret_cast<const char*>(data + s * max_size), size_data[s]);
    }
    return stages;
}

std::vector<StageCacheSnapshot> PPSnapshotCollector::collect() {
    const auto                      payloads = collectPPStagePayloads(local_.serialize(), pp_size_);
    std::vector<StageCacheSnapshot> stages;
    stages.reserve(payloads.size());
    for (size_t s = 0; s < payloads.size(); ++s) {
        try {
            stages.push_back(StageCacheSnapshot::deserialize(payloads[s]));
        } catch (const std::exception& e) {
            RTP_LLM_FAIL("PP snapshot exchange: stage %zu payload rejected: %s", s, e.what());
        }
    }
    return stages;
}

NegotiatedCapacity PPCacheCapacityNegotiator::negotiate(const CacheConfig&          topology,
                                                        uint32_t                    local_block_num,
                                                        const BlockTreeCacheSizing& block_tree_sizing,
                                                        const RuntimeConfig&        /*runtime_config*/) {
    auto snapshot = StageCacheSnapshot::captureLocalStage(topology, local_block_num, block_tree_sizing);
    PPSnapshotCollector collector(std::move(snapshot), pp_size_);
    auto                validation = validatePPTopology(collector.collect());
    if (!validation.ok) {
        RTP_LLM_FAIL("PP cache topology validation failed: %s", validation.error.c_str());
    }
    RTP_LLM_LOG_INFO("PP cache negotiation: local block_num %u -> agreed paged %u over %zu canonical groups",
                     local_block_num,
                     validation.agreed.paged_block_num,
                     validation.canonical_groups.size());
    return std::move(validation.agreed);
}

}  // namespace rtp_llm
