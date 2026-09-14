#include "rtp_llm/cpp/cache/PPTopologyValidator.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

std::string joinTags(const std::vector<std::string>& tags) {
    std::ostringstream oss;
    for (size_t i = 0; i < tags.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << tags[i];
    }
    return oss.str();
}

PPValidationResult fail(std::string error) {
    PPValidationResult result;
    result.ok    = false;
    result.error = std::move(error);
    return result;
}

// Fills per-tag construction inputs from the completed canonical table.
void deriveConstructionInputs(PPValidationResult& result) {
    uint32_t paged_min = std::numeric_limits<uint32_t>::max();
    uint32_t any_max   = 0;
    result.agreed.block_num_overrides.clear();
    result.agreed.block_num_overrides.reserve(result.canonical_groups.size());
    for (const auto& entry : result.canonical_groups) {
        result.agreed.block_num_overrides.emplace(entry.tag, entry.logical_block_num);
        any_max = std::max(any_max, entry.logical_block_num);
        // Explicit and SWA pools are decoupled from the paged yardstick.
        const bool follows_global_budget =
            entry.explicit_block_num == 0
            && (entry.type == CacheGroupType::FULL || entry.type == CacheGroupType::LINEAR);
        if (follows_global_budget) {
            paged_min = std::min(paged_min, entry.logical_block_num);
        }
    }
    if (paged_min == std::numeric_limits<uint32_t>::max()) {
        // All pools explicitly sized, so nothing follows the budget: take the
        // largest so the top-level value never understates an actual pool.
        paged_min = any_max;
    }
    RTP_LLM_CHECK_WITH_INFO(paged_min > 0, "PP canonical table yielded a non-positive top-level block count");
    result.agreed.paged_block_num = paged_min;
}

}  // namespace

bool StageCacheSnapshot::internallyConsistent() const {
    const size_t n = group_tags.size();
    return group_types.size() == n && seq_size_per_block.size() == n && kernel_seq_size_per_block.size() == n
           && block_nums.size() == n && explicit_block_nums.size() == n && policy_fingerprints.size() == n;
}

std::string cacheGroupPolicyFingerprint(const CacheGroupPolicy& policy) {
    // Fixed-field digest of every field samePolicy() compares; wire-safe.
    std::ostringstream oss;
    oss << "t" << static_cast<int>(policy.group_type) << ":r" << (policy.enable_prefix_reuse ? 1 : 0) << ":e"
        << static_cast<int>(policy.evict_policy) << ":v" << (policy.reservable ? 1 : 0) << ":x"
        << policy.explicit_block_num << ":a" << policy.active_tail_blocks << ":w"
        << (policy.validate_tail_blocks ? 1 : 0) << ":m" << static_cast<int>(policy.cp_mapping) << ":s"
        << static_cast<int>(policy.cp_slice);
    return oss.str();
}

StageCacheSnapshot StageCacheSnapshot::fromTopologyAndCapacity(const CacheConfig&   topology,
                                                               uint32_t             local_capacity,
                                                               const RuntimeConfig& runtime_config) {
    // Derive per-group counts on a throwaway copy through the same rule the
    // final composition uses, so a stage reports exactly what it would build.
    CacheConfig sized = topology;
    sized.finalizeBlockNums(local_capacity, runtime_config);

    StageCacheSnapshot snapshot;
    // Sort by tag so identical topologies serialize identically everywhere.
    std::vector<const CacheGroup*> sorted_groups;
    sorted_groups.reserve(sized.groups().size());
    for (const auto& group : sized.groups()) {
        sorted_groups.push_back(&group);
    }
    std::sort(sorted_groups.begin(), sorted_groups.end(), [](const CacheGroup* a, const CacheGroup* b) {
        return a->tag < b->tag;
    });
    for (const auto* group : sorted_groups) {
        snapshot.group_tags.push_back(group->tag);
        snapshot.group_types.push_back(group->policy.group_type);
        snapshot.seq_size_per_block.push_back(static_cast<uint32_t>(group->seqSizePerBlock()));
        snapshot.kernel_seq_size_per_block.push_back(static_cast<uint32_t>(group->kernelSeqSizePerBlock()));
        snapshot.block_nums.push_back(group->block_num);
        snapshot.explicit_block_nums.push_back(group->policy.explicit_block_num);
        snapshot.policy_fingerprints.push_back(cacheGroupPolicyFingerprint(group->policy));
    }
    return snapshot;
}

namespace {

// Bumped only on an incompatible field-set change; all stages run one binary.
constexpr char kWireVersion[] = "v1";
// Wire layout: version, then tags, types, seq, kseq, blocks, explicit,
// fingerprints. Tags and fingerprints unit-separated, numerics by ','.
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

template<typename T>
std::string joinNums(const std::vector<T>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << kNumSep;
        }
        oss << static_cast<unsigned long long>(values[i]);
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
    oss << kWireVersion << kFieldSep;
    for (size_t i = 0; i < group_tags.size(); ++i) {
        if (i > 0) {
            oss << kTagSep;
        }
        RTP_LLM_CHECK_WITH_INFO(group_tags[i].find_first_of("\x1f|,") == std::string::npos,
                                "PP snapshot tag contains a wire-format delimiter: %s",
                                group_tags[i].c_str());
        oss << group_tags[i];
    }
    oss << kFieldSep << joinNums(group_types) << kFieldSep << joinNums(seq_size_per_block) << kFieldSep
        << joinNums(kernel_seq_size_per_block) << kFieldSep << joinNums(block_nums) << kFieldSep
        << joinNums(explicit_block_nums) << kFieldSep;
    for (size_t i = 0; i < policy_fingerprints.size(); ++i) {
        if (i > 0) {
            oss << kTagSep;
        }
        RTP_LLM_CHECK_WITH_INFO(policy_fingerprints[i].find_first_of("\x1f|,") == std::string::npos,
                                "PP snapshot policy fingerprint contains a wire-format delimiter: %s",
                                policy_fingerprints[i].c_str());
        oss << policy_fingerprints[i];
    }
    return oss.str();
}

StageCacheSnapshot StageCacheSnapshot::deserialize(const std::string& payload) {
    const auto fields = splitFields(payload, kFieldSep);
    RTP_LLM_CHECK_WITH_INFO(fields.size() == 8 && fields[0] == kWireVersion,
                            "PP snapshot payload malformed: got %zu fields with version [%s], expected 8 fields "
                            "and version [%s]",
                            fields.size(),
                            fields.empty() ? "<none>" : fields[0].c_str(),
                            kWireVersion);
    StageCacheSnapshot snapshot;
    snapshot.group_tags                = splitFields(fields[1], kTagSep);
    snapshot.group_types               = parseNums<CacheGroupType>(fields[2], snapshot.group_tags.size());
    snapshot.seq_size_per_block        = parseNums<size_t>(fields[3], snapshot.group_tags.size());
    snapshot.kernel_seq_size_per_block = parseNums<size_t>(fields[4], snapshot.group_tags.size());
    snapshot.block_nums                = parseNums<uint32_t>(fields[5], snapshot.group_tags.size());
    snapshot.explicit_block_nums       = parseNums<uint32_t>(fields[6], snapshot.group_tags.size());
    snapshot.policy_fingerprints       = splitFields(fields[7], kTagSep);
    RTP_LLM_CHECK_WITH_INFO(snapshot.internallyConsistent(), "PP snapshot payload failed consistency check");
    return snapshot;
}

PPValidationResult validatePPTopology(const std::vector<StageCacheSnapshot>& stages, double capacity_skew_threshold) {
    PPValidationResult result;

    // pp_size=1 (or nothing reported): degenerates to today's behavior.
    if (stages.size() <= 1) {
        result.ok = true;
        if (stages.size() == 1) {
            if (!stages[0].internallyConsistent()) {
                return fail("stage 0 cache snapshot is internally inconsistent");
            }
            for (size_t g = 0; g < stages[0].group_tags.size(); ++g) {
                CanonicalGroupEntry entry;
                entry.tag                       = stages[0].group_tags[g];
                entry.type                      = stages[0].group_types[g];
                entry.seq_size_per_block        = stages[0].seq_size_per_block[g];
                entry.kernel_seq_size_per_block = stages[0].kernel_seq_size_per_block[g];
                entry.logical_block_num         = stages[0].block_nums[g];
                entry.explicit_block_num        = stages[0].explicit_block_nums[g];
                entry.policy_fingerprint        = stages[0].policy_fingerprints[g];
                result.canonical_groups.push_back(std::move(entry));
            }
            deriveConstructionInputs(result);
        }
        return result;
    }

    for (size_t s = 0; s < stages.size(); ++s) {
        if (!stages[s].internallyConsistent()) {
            return fail("stage " + std::to_string(s) + " cache snapshot is internally inconsistent");
        }
        // Tags come off the wire, so reject duplicates here rather than let a
        // repeated tag masquerade as a second owner and skew the minimum.
        std::unordered_set<std::string> seen;
        for (const auto& tag : stages[s].group_tags) {
            if (!seen.insert(tag).second) {
                return fail("stage " + std::to_string(s) + " snapshot lists cache group [" + tag + "] twice");
            }
        }
        // Invariant: a hybrid stage (any LINEAR group) must keep at least one FULL group.
        const bool has_linear = std::any_of(stages[s].group_types.begin(),
                                            stages[s].group_types.end(),
                                            [](CacheGroupType t) { return t == CacheGroupType::LINEAR; });
        const bool has_full   = std::any_of(stages[s].group_types.begin(),
                                          stages[s].group_types.end(),
                                          [](CacheGroupType t) { return t == CacheGroupType::FULL; });
        if (has_linear && !has_full) {
            return fail("stage " + std::to_string(s)
                        + " has LINEAR cache groups but no FULL group; every hybrid PP stage must own at least one "
                          "full attention layer");
        }
        // Sliding-window pools use step-derived capacities whose cross-stage
        // reconciliation is not implemented yet.
        const bool has_swa = std::any_of(stages[s].group_types.begin(),
                                         stages[s].group_types.end(),
                                         [](CacheGroupType t) { return t == CacheGroupType::SWA; });
        if (has_swa) {
            return fail("stage " + std::to_string(s)
                        + " holds an SWA cache group; sliding-window pools do not support pipeline parallelism yet");
        }
    }

    // Allocation authority: the leading stage issues every block id, so it must
    // own a pool for every tag in the union.
    const auto&                           ref = stages[0];
    const std::unordered_set<std::string> stage0_tags(ref.group_tags.begin(), ref.group_tags.end());
    for (size_t s = 1; s < stages.size(); ++s) {
        for (const auto& tag : stages[s].group_tags) {
            if (stage0_tags.find(tag) == stage0_tags.end()) {
                return fail("stage " + std::to_string(s) + " owns cache group [" + tag
                            + "] which stage 0 does not; the leading stage issues every block id, so it must own a "
                              "pool for each tag (stage 0 group set: ["
                            + joinTags(ref.group_tags) + "])");
            }
        }
    }

    // Any group anywhere with zero blocks cannot serve a request.
    for (size_t s = 0; s < stages.size(); ++s) {
        for (size_t g = 0; g < stages[s].group_tags.size(); ++g) {
            if (stages[s].block_nums[g] == 0) {
                return fail("stage " + std::to_string(s) + " group [" + stages[s].group_tags[g]
                            + "] has 0 KV blocks; the group could not allocate a single block");
            }
        }
    }

    // Canonical table: cross-stage UNION by tag; owners must agree on type,
    // geometry and policy, and capacity is the min over owners only.
    std::unordered_map<std::string, size_t> canonical_index;
    // Tag-keyed, not positional: canonical_groups is sorted before publication.
    std::unordered_map<std::string, uint32_t> max_blocks_by_tag;
    for (size_t s = 0; s < stages.size(); ++s) {
        for (size_t g = 0; g < stages[s].group_tags.size(); ++g) {
            const auto& tag = stages[s].group_tags[g];
            const auto  it  = canonical_index.find(tag);
            if (it == canonical_index.end()) {
                canonical_index.emplace(tag, result.canonical_groups.size());
                CanonicalGroupEntry entry;
                entry.tag                       = tag;
                entry.type                      = stages[s].group_types[g];
                entry.seq_size_per_block        = stages[s].seq_size_per_block[g];
                entry.kernel_seq_size_per_block = stages[s].kernel_seq_size_per_block[g];
                entry.logical_block_num         = stages[s].block_nums[g];
                entry.explicit_block_num        = stages[s].explicit_block_nums[g];
                entry.policy_fingerprint        = stages[s].policy_fingerprints[g];
                result.canonical_groups.push_back(std::move(entry));
                max_blocks_by_tag.emplace(tag, stages[s].block_nums[g]);
                continue;
            }
            auto& entry = result.canonical_groups[it->second];
            if (stages[s].group_types[g] != entry.type) {
                return fail("stage " + std::to_string(s) + " group [" + tag
                            + "] type differs from the canonical entry");
            }
            if (stages[s].seq_size_per_block[g] != entry.seq_size_per_block) {
                return fail("stage " + std::to_string(s) + " group [" + tag + "] seq_size_per_block "
                            + std::to_string(stages[s].seq_size_per_block[g]) + " != canonical "
                            + std::to_string(entry.seq_size_per_block));
            }
            if (stages[s].kernel_seq_size_per_block[g] != entry.kernel_seq_size_per_block) {
                return fail("stage " + std::to_string(s) + " group [" + tag + "] kernel_seq_size_per_block "
                            + std::to_string(stages[s].kernel_seq_size_per_block[g]) + " != canonical "
                            + std::to_string(entry.kernel_seq_size_per_block));
            }
            // Explicit pool sizing comes from deployment-wide config; same-tag
            // owners diverging means the stages were launched inconsistently.
            if (stages[s].explicit_block_nums[g] != entry.explicit_block_num) {
                return fail("stage " + std::to_string(s) + " group [" + tag + "] explicit_block_num "
                            + std::to_string(stages[s].explicit_block_nums[g]) + " != canonical "
                            + std::to_string(entry.explicit_block_num));
            }
            // Full policy reconciliation: eviction/reuse/placement/tail knobs
            // must agree across owners of the same pool.
            if (stages[s].policy_fingerprints[g] != entry.policy_fingerprint) {
                return fail("stage " + std::to_string(s) + " group [" + tag + "] policy ["
                            + stages[s].policy_fingerprints[g] + "] != canonical [" + entry.policy_fingerprint + "]");
            }
            entry.logical_block_num = std::min(entry.logical_block_num, stages[s].block_nums[g]);
            auto& max_blocks        = max_blocks_by_tag[tag];
            max_blocks              = std::max(max_blocks, stages[s].block_nums[g]);
        }
    }

    // Deployment guard, not correctness (per-tag minima keep pools exact): a
    // large spread means stranded VRAM, usually an unintended partition.
    for (const auto& entry : result.canonical_groups) {
        const auto max_blocks = max_blocks_by_tag[entry.tag];
        if (static_cast<double>(max_blocks) / static_cast<double>(entry.logical_block_num) > capacity_skew_threshold) {
            std::ostringstream oss;
            oss << "group [" << entry.tag << "] KV capacity skew too large: max/min = " << max_blocks << "/"
                << entry.logical_block_num << " > threshold " << capacity_skew_threshold
                << "; the richer stage would strand the difference, adjust the layer partition to balance per-stage "
                   "KV capacity";
            return fail(oss.str());
        }
    }

    // Publish the canonical order as tag-lexicographic, keeping the invariant
    // independent of stage arrival order.
    std::sort(result.canonical_groups.begin(), result.canonical_groups.end(), [](const auto& a, const auto& b) {
        return a.tag < b.tag;
    });

    deriveConstructionInputs(result);
    result.ok = true;
    return result;
}

PPValidationResult initPPCacheGeometry(StageSnapshotCollector& collector, double capacity_skew_threshold) {
    return validatePPTopology(collector.collect(), capacity_skew_threshold);
}

std::vector<StageCacheSnapshot> PPSnapshotCollector::collect() {
    // All-gather over the PP process group; payloads come back in group-rank
    // order, which equals pp_rank order, so vector index == stage index.
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

void validatePPComposedBlockNums(const CacheConfig& composed, const NegotiatedCapacity& agreed) {
    for (const auto& group : composed.groups()) {
        const auto it = agreed.block_num_overrides.find(group.tag);
        RTP_LLM_CHECK_WITH_INFO(it != agreed.block_num_overrides.end(),
                                "local group [%s] is missing from the PP canonical group table",
                                group.tag.c_str());
        // Equality holds by construction; a mismatch means the table lost a tag
        // or local capacity moved after the negotiation.
        RTP_LLM_CHECK_WITH_INFO(group.block_num == it->second,
                                "composed group [%s] block_num %u != cross-stage agreed %u",
                                group.tag.c_str(),
                                group.block_num,
                                it->second);
    }
    for (const auto& sub_config : composed.mtp_sub_configs) {
        if (sub_config != nullptr) {
            validatePPComposedBlockNums(*sub_config, agreed);
        }
    }
}

NegotiatedCapacity PPCacheCapacityNegotiator::negotiate(const CacheConfig&   topology,
                                                        uint32_t             local_block_num,
                                                        const RuntimeConfig& runtime_config) {
    // Startup barrier: every stage reports its geometry and measured capacity,
    // then all reduce the collected snapshots to the same per-tag minima.
    PPSnapshotCollector collector(
        StageCacheSnapshot::fromTopologyAndCapacity(topology, local_block_num, runtime_config));
    const auto validation = initPPCacheGeometry(collector, capacity_skew_threshold_);
    if (!validation.ok) {
        RTP_LLM_FAIL("PP cache topology validation failed: %s", validation.error.c_str());
    }
    RTP_LLM_LOG_INFO("PP cache negotiation: local block_num %u -> agreed paged %u over %zu canonical groups",
                     local_block_num,
                     validation.agreed.paged_block_num,
                     validation.canonical_groups.size());
    return validation.agreed;
}

void PPCacheCapacityNegotiator::validateComposed(const CacheConfig& composed, const NegotiatedCapacity& agreed) {
    validatePPComposedBlockNums(composed, agreed);
}

}  // namespace rtp_llm
