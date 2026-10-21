#include "rtp_llm/cpp/cache/PPTopologyValidator.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

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

using GroupSetMemberCountsByType = std::unordered_map<CacheGroupType, size_t>;

CanonicalGroupEntry makeCanonicalGroupEntry(const StageCacheGroupSnapshot& group) {
    CanonicalGroupEntry entry;
    entry.tag                       = group.tag;
    entry.type                      = group.type;
    entry.seq_size_per_block        = group.seq_size_per_block;
    entry.kernel_seq_size_per_block = group.kernel_seq_size_per_block;
    entry.cache_key_token_stride    = group.cache_key_token_stride;
    entry.logical_block_num         = group.block_num;
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
    entry.logical_block_num = std::min(entry.logical_block_num, group.block_num);
    return {};
}

std::string validateGroupSetStructure(size_t                            stage_index,
                                      const GroupSetMemberCountsByType& canonical,
                                      const GroupSetMemberCountsByType& local) {
    for (const auto& [type, local_count] : local) {
        const auto expected = canonical.find(type);
        // Tags are unique, later stages are stage-0 subsets, and shared tags
        // already agree on type and reuse policy. Equal non-zero counts
        // therefore imply an identical group set.
        const size_t expected_count = expected == canonical.end() ? 0 : expected->second;
        // TODO(pp-partial-group-set): once the sender can project each
        // CopyItem (including group_blocks) per target PP rank, allow a
        // non-empty proper subset. A wholly absent group set is skipped receiver-side.
        if (local_count != expected_count) {
            return "stage " + std::to_string(stage_index) + " has " + std::to_string(local_count) + " of "
                   + std::to_string(expected_count) + " group-set members for type "
                   + std::to_string(static_cast<int>(type)) + "; partial group-set overlap is not supported yet";
        }
    }
    return {};
}

}  // namespace

StageCacheSnapshot StageCacheSnapshot::captureLocalStage(const CacheConfig& config,
                                                         size_t             local_block_tree_combined_stride_bytes) {
    StageCacheSnapshot snapshot;
    snapshot.groups.reserve(config.topology().groups().size());
    for (const auto& group : config.topology().groups()) {
        StageCacheGroupSnapshot captured;
        captured.tag                       = group.tag;
        captured.type                      = group.policy.group_type;
        captured.block_tree_group_member   = cacheGroupParticipatesInBlockTree(group.policy);
        captured.seq_size_per_block        = group.seqSizePerBlock();
        captured.kernel_seq_size_per_block = group.kernelSeqSizePerBlock();
        captured.cache_key_token_stride    = group.cacheKeyTokenStride();
        captured.block_num                 = group.block_num;
        captured.explicit_block_num        = group.policy.explicit_block_num;
        captured.policy_fingerprint        = group.policy.policyFingerprint();
        snapshot.groups.push_back(std::move(captured));
    }
    snapshot.block_tree_combined_stride_bytes = local_block_tree_combined_stride_bytes;
    return snapshot;
}

namespace {

/** Wire format: "v1|tags|types|block_tree_member|seq|kseq|key_stride|blocks|explicit|combined_stride|fingerprints". */
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

template<typename Getter>
std::string joinGroupNums(const std::vector<StageCacheGroupSnapshot>& groups, Getter getter) {
    std::ostringstream oss;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (i > 0) {
            oss << kNumSep;
        }
        oss << static_cast<unsigned long long>(getter(groups[i]));
    }
    return oss.str();
}

template<typename Getter>
std::string
joinGroupStrings(const std::vector<StageCacheGroupSnapshot>& groups, Getter getter, const char* field_name) {
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
    std::ostringstream oss;
    oss << "v1" << kFieldSep
        << joinGroupStrings(
               groups, [](const auto& group) -> const std::string& { return group.tag; }, "tag")
        << kFieldSep << joinGroupNums(groups, [](const auto& group) { return group.type; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.block_tree_group_member; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.seq_size_per_block; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.kernel_seq_size_per_block; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.cache_key_token_stride; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.block_num; }) << kFieldSep
        << joinGroupNums(groups, [](const auto& group) { return group.explicit_block_num; }) << kFieldSep
        << block_tree_combined_stride_bytes << kFieldSep
        << joinGroupStrings(
               groups,
               [](const auto& group) -> const std::string& { return group.policy_fingerprint; },
               "policy fingerprint");
    return oss.str();
}

StageCacheSnapshot StageCacheSnapshot::deserialize(const std::string& payload) {
    const auto fields = splitFields(payload, kFieldSep);
    RTP_LLM_CHECK_WITH_INFO(fields.size() == 11 && fields[0] == "v1",
                            "PP snapshot payload is malformed (version/field count)");
    auto group_tags = splitFields(fields[1], kTagSep);
    if (fields[1].empty()) {
        group_tags.clear();
    }
    const auto group_types                 = parseNums<CacheGroupType>(fields[2], group_tags.size());
    const auto block_tree_group_membership = parseNums<uint8_t>(fields[3], group_tags.size());
    const auto seq_size_per_block          = parseNums<size_t>(fields[4], group_tags.size());
    const auto kernel_seq_size_per_block   = parseNums<size_t>(fields[5], group_tags.size());
    const auto cache_key_token_strides     = parseNums<size_t>(fields[6], group_tags.size());
    const auto block_nums                  = parseNums<uint32_t>(fields[7], group_tags.size());
    const auto explicit_block_nums         = parseNums<uint32_t>(fields[8], group_tags.size());
    auto       policy_fingerprints         = splitFields(fields[10], kTagSep);
    if (fields[10].empty()) {
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
        group.seq_size_per_block        = seq_size_per_block[i];
        group.kernel_seq_size_per_block = kernel_seq_size_per_block[i];
        group.cache_key_token_stride    = cache_key_token_strides[i];
        group.block_num                 = block_nums[i];
        group.explicit_block_num        = explicit_block_nums[i];
        group.policy_fingerprint        = std::move(policy_fingerprints[i]);
        snapshot.groups.push_back(std::move(group));
    }
    snapshot.block_tree_combined_stride_bytes = std::stoull(fields[9]);
    return snapshot;
}

PPValidationResult validatePPTopology(const std::vector<StageCacheSnapshot>& stages) {
    PPValidationResult result;

    if (stages.empty()) {
        result.ok = true;
        return result;
    }

    std::unordered_map<std::string, size_t> canonical_index;
    GroupSetMemberCountsByType              canonical_group_set_member_counts;
    for (size_t s = 0; s < stages.size(); ++s) {
        const auto& stage = stages[s];
        if (s == 0 && stage.groups.empty()) {
            return fail("stage 0 cache snapshot has no cache groups");
        }

        result.agreed.block_tree_combined_stride_bytes =
            std::max(result.agreed.block_tree_combined_stride_bytes, stage.block_tree_combined_stride_bytes);

        std::unordered_set<std::string> unique_tags;
        GroupSetMemberCountsByType      local_group_set_member_counts;
        for (const auto& group : stage.groups) {
            if (group.tag.empty() || !unique_tags.emplace(group.tag).second) {
                return fail("stage " + std::to_string(s) + " has an empty or duplicate cache group tag [" + group.tag
                            + "]");
            }
            if (group.block_num == 0) {
                return fail("stage " + std::to_string(s) + " group [" + group.tag
                            + "] has 0 KV blocks; the group could not allocate a single block");
            }
            if (group.block_tree_group_member) {
                ++local_group_set_member_counts[group.type];
            }

            if (s == 0) {
                canonical_index.emplace(group.tag, result.canonical_groups.size());
                result.canonical_groups.push_back(makeCanonicalGroupEntry(group));
                continue;
            }

            const auto it = canonical_index.find(group.tag);
            if (it == canonical_index.end()) {
                /** Stage 0 issues every block id, so later stages can only own its tags. */
                return fail("stage " + std::to_string(s) + " owns cache group [" + group.tag
                            + "] that is absent from stage 0 [" + joinTags(stages[0].groups)
                            + "]; the leading PP stage must own every cache group (bookkeeping-only "
                              "allocation is not supported)");
            }
            auto& entry = result.canonical_groups[it->second];
            if (const auto error = validateAndMergeCanonicalGroup(entry, group, s); !error.empty()) {
                return fail(error);
            }
        }

        if (s == 0) {
            canonical_group_set_member_counts = std::move(local_group_set_member_counts);
        } else if (const auto error =
                       validateGroupSetStructure(s, canonical_group_set_member_counts, local_group_set_member_counts);
                   !error.empty()) {
            return fail(error);
        }
    }

    result.ok = true;
    deriveConstructionInputs(result);
    return result;
}

PPValidationResult initPPCacheGeometry(StageSnapshotCollector& collector) {
    return validatePPTopology(collector.collect());
}

std::vector<StageCacheSnapshot> PPSnapshotCollector::collect() {
    const auto payloads = execPPSnapshotExchange(local_.serialize());
    RTP_LLM_CHECK_WITH_INFO(!payloads.empty(), "PP snapshot exchange returned no stages");
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

NegotiatedCapacity PPCacheCapacityNegotiator::negotiate(const CacheConfig&   topology,
                                                        uint32_t             local_block_num,
                                                        size_t               local_block_tree_combined_stride_bytes,
                                                        const RuntimeConfig& runtime_config) {
    /**
     * Size a throwaway copy, exchange stage-aligned snapshots, and reduce all
     * owner stages to the same per-tag minima without mutating the input.
     */
    CacheConfig sized = topology;
    sized.mtp_sub_configs.clear();
    sized.finalizeBlockNums(local_block_num, runtime_config);
    auto                snapshot = StageCacheSnapshot::captureLocalStage(sized, local_block_tree_combined_stride_bytes);
    PPSnapshotCollector collector(std::move(snapshot));
    auto                validation = initPPCacheGeometry(collector);
    if (!validation.ok) {
        RTP_LLM_FAIL("PP cache topology validation failed: %s", validation.error.c_str());
    }
    RTP_LLM_LOG_INFO("PP cache negotiation: local block_num %u -> agreed paged %u, lower-tier stride %zu over %zu "
                     "canonical groups",
                     local_block_num,
                     validation.agreed.paged_block_num,
                     validation.agreed.block_tree_combined_stride_bytes,
                     validation.canonical_groups.size());
    return std::move(validation.agreed);
}

}  // namespace rtp_llm
