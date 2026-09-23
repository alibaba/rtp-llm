#include "rtp_llm/cpp/cache/PPTopologyValidator.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_map>

#include "rtp_llm/cpp/cache/CacheConfig.h"
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
    std::ostringstream oss;
    oss << "t" << static_cast<int>(policy.group_type) << ":r" << (policy.enable_prefix_reuse ? 1 : 0) << ":e"
        << static_cast<int>(policy.evict_policy) << ":v" << (policy.reservable ? 1 : 0) << ":x"
        << policy.explicit_block_num << ":c" << (policy.charge_to_paged_budget ? 1 : 0) << ":p"
        << static_cast<int>(policy.memory_placement) << ":a" << policy.active_tail_blocks << ":w"
        << (policy.validate_tail_blocks ? 1 : 0) << ":m" << static_cast<int>(policy.cp_mapping) << ":s"
        << static_cast<int>(policy.cp_slice);
    return oss.str();
}

StageCacheSnapshot StageCacheSnapshot::fromConfig(const CacheConfig& config) {
    StageCacheSnapshot snapshot;
    for (const auto& group : config.topology().groups()) {
        snapshot.group_tags.push_back(group.tag);
        snapshot.group_types.push_back(group.policy.group_type);
        snapshot.seq_size_per_block.push_back(group.seq_size_per_block);
        snapshot.kernel_seq_size_per_block.push_back(group.kernel_seq_size_per_block);
        snapshot.block_nums.push_back(group.block_num);
        snapshot.explicit_block_nums.push_back(group.policy.explicit_block_num);
        snapshot.policy_fingerprints.push_back(cacheGroupPolicyFingerprint(group.policy));
    }
    return snapshot;
}

namespace {

// Wire format: "v1|tags|types|seq|kseq|blocks|explicit|fingerprints".
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
    oss << "v1" << kFieldSep;
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
    RTP_LLM_CHECK_WITH_INFO(fields.size() == 8 && fields[0] == "v1",
                            "PP snapshot payload is malformed (version/field count)");
    StageCacheSnapshot snapshot;
    snapshot.group_tags = splitFields(fields[1], kTagSep);
    // splitFields on an empty tag field yields zero entries (no empty tag).
    if (fields[1].empty()) {
        snapshot.group_tags.clear();
    }
    snapshot.group_types               = parseNums<CacheGroupType>(fields[2], snapshot.group_tags.size());
    snapshot.seq_size_per_block        = parseNums<size_t>(fields[3], snapshot.group_tags.size());
    snapshot.kernel_seq_size_per_block = parseNums<size_t>(fields[4], snapshot.group_tags.size());
    snapshot.block_nums                = parseNums<uint32_t>(fields[5], snapshot.group_tags.size());
    snapshot.explicit_block_nums       = parseNums<uint32_t>(fields[6], snapshot.group_tags.size());
    snapshot.policy_fingerprints       = splitFields(fields[7], kTagSep);
    if (fields[7].empty()) {
        snapshot.policy_fingerprints.clear();
    }
    RTP_LLM_CHECK_WITH_INFO(snapshot.internallyConsistent(), "PP snapshot payload failed consistency check");
    return snapshot;
}

PPValidationResult validatePPTopology(const std::vector<StageCacheSnapshot>& stages, double capacity_skew_threshold) {
    PPValidationResult result;

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
        const bool has_swa = std::any_of(stages[s].group_types.begin(),
                                         stages[s].group_types.end(),
                                         [](CacheGroupType t) { return t == CacheGroupType::SWA; });
        if (has_swa) {
            return fail("stage " + std::to_string(s)
                        + " holds an SWA cache group; sliding-window pools do not support pipeline parallelism yet");
        }
    }

    const auto& ref            = stages[0];
    const bool  tag_sets_equal = std::all_of(
        stages.begin(), stages.end(), [&](const StageCacheSnapshot& s) { return s.group_tags == ref.group_tags; });

    if (tag_sets_equal) {
        // Identical tag sets: strict equality check.
        for (size_t s = 1; s < stages.size(); ++s) {
            const auto& cur = stages[s];

            if (cur.group_types != ref.group_types) {
                return fail("stage " + std::to_string(s) + " group type sequence differs from stage 0");
            }
            for (size_t g = 0; g < ref.group_tags.size(); ++g) {
                if (cur.seq_size_per_block[g] != ref.seq_size_per_block[g]) {
                    return fail("stage " + std::to_string(s) + " group [" + ref.group_tags[g] + "] seq_size_per_block "
                                + std::to_string(cur.seq_size_per_block[g]) + " != stage 0 "
                                + std::to_string(ref.seq_size_per_block[g]));
                }
                if (cur.kernel_seq_size_per_block[g] != ref.kernel_seq_size_per_block[g]) {
                    return fail("stage " + std::to_string(s) + " group [" + ref.group_tags[g]
                                + "] kernel_seq_size_per_block " + std::to_string(cur.kernel_seq_size_per_block[g])
                                + " != stage 0 " + std::to_string(ref.kernel_seq_size_per_block[g]));
                }
            }
        }
    } else {
        /* Pairing path: stages may legitimately hold different tag subsets,
           so groups match by tag name against stage 0. Superset gate: the
           leading stage issues every block id from its own physical pools,
           so it must own every group that appears anywhere. */
        for (size_t s = 1; s < stages.size(); ++s) {
            const auto& cur = stages[s];
            for (const auto& tag : cur.group_tags) {
                if (std::find(ref.group_tags.begin(), ref.group_tags.end(), tag) == ref.group_tags.end()) {
                    return fail("stage " + std::to_string(s) + " owns cache group [" + tag
                                + "] that is absent from stage 0 [" + joinTags(ref.group_tags)
                                + "]; the leading PP stage must own every cache group (bookkeeping-only "
                                  "allocation is not supported)");
                }
            }
            for (size_t g = 0; g < cur.group_tags.size(); ++g) {
                const auto ref_it = std::find(ref.group_tags.begin(), ref.group_tags.end(), cur.group_tags[g]);
                RTP_LLM_CHECK_WITH_INFO(ref_it != ref.group_tags.end(),
                                        "unreachable: stage %zu tag [%s] passed the superset gate",
                                        s,
                                        cur.group_tags[g].c_str());
                const auto rg = static_cast<size_t>(ref_it - ref.group_tags.begin());
                if (cur.group_types[g] != ref.group_types[rg]) {
                    return fail("stage " + std::to_string(s) + " group [" + cur.group_tags[g]
                                + "] type differs from stage 0");
                }
                if (cur.seq_size_per_block[g] != ref.seq_size_per_block[rg]) {
                    return fail("stage " + std::to_string(s) + " group [" + cur.group_tags[g] + "] seq_size_per_block "
                                + std::to_string(cur.seq_size_per_block[g]) + " != stage 0 "
                                + std::to_string(ref.seq_size_per_block[rg]));
                }
                if (cur.kernel_seq_size_per_block[g] != ref.kernel_seq_size_per_block[rg]) {
                    return fail("stage " + std::to_string(s) + " group [" + cur.group_tags[g]
                                + "] kernel_seq_size_per_block " + std::to_string(cur.kernel_seq_size_per_block[g])
                                + " != stage 0 " + std::to_string(ref.kernel_seq_size_per_block[rg]));
                }
            }
        }
    }

    for (size_t s = 0; s < stages.size(); ++s) {
        for (size_t g = 0; g < stages[s].group_tags.size(); ++g) {
            if (stages[s].block_nums[g] == 0) {
                return fail("stage " + std::to_string(s) + " group [" + stages[s].group_tags[g]
                            + "] has 0 KV blocks; the group could not allocate a single block");
            }
        }
    }

    // Canonical table: stage-0 order first, then first-seen; same-tag owners must agree on geometry.
    std::unordered_map<std::string, size_t> canonical_index;
    std::vector<uint32_t>                   canonical_max_blocks;
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
                canonical_max_blocks.push_back(stages[s].block_nums[g]);
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
            if (stages[s].explicit_block_nums[g] != entry.explicit_block_num) {
                return fail("stage " + std::to_string(s) + " group [" + tag + "] explicit_block_num "
                            + std::to_string(stages[s].explicit_block_nums[g]) + " != canonical "
                            + std::to_string(entry.explicit_block_num));
            }
            if (stages[s].policy_fingerprints[g] != entry.policy_fingerprint) {
                return fail("stage " + std::to_string(s) + " group [" + tag + "] policy ["
                            + stages[s].policy_fingerprints[g] + "] != canonical [" + entry.policy_fingerprint + "]");
            }
            entry.logical_block_num          = std::min(entry.logical_block_num, stages[s].block_nums[g]);
            canonical_max_blocks[it->second] = std::max(canonical_max_blocks[it->second], stages[s].block_nums[g]);
        }
    }

    // Reject excessive capacity imbalance even though logical counts are capped at the owner minimum.
    for (size_t c = 0; c < result.canonical_groups.size(); ++c) {
        const auto& entry = result.canonical_groups[c];
        if (static_cast<double>(canonical_max_blocks[c]) / static_cast<double>(entry.logical_block_num)
            > capacity_skew_threshold) {
            std::ostringstream oss;
            oss << "group [" << entry.tag << "] KV capacity skew too large: max/min = " << canonical_max_blocks[c]
                << "/" << entry.logical_block_num << " > threshold " << capacity_skew_threshold
                << "; adjust the layer partition to balance per-stage KV capacity";
            return fail(oss.str());
        }
    }

    result.ok = true;
    deriveConstructionInputs(result);
    return result;
}

PPValidationResult initPPCacheGeometry(StageSnapshotCollector& collector, double capacity_skew_threshold) {
    return validatePPTopology(collector.collect(), capacity_skew_threshold);
}

std::vector<StageCacheSnapshot> PPSnapshotCollector::collect() {
    // Payloads return in group-rank (== pp_rank) order, so vector index == stage index.
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
    for (const auto& group : composed.topology().groups()) {
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

PPValidationResult PPCacheCapacityNegotiator::negotiate(const CacheConfig&   topology,
                                                        uint32_t             local_block_num,
                                                        const RuntimeConfig& runtime_config) {
    // Startup barrier: every stage reports its geometry and stage-aligned
    // capacity, then all reduce the collected snapshots to the same per-tag
    // minima. The snapshot derives counts on a throwaway copy through the same
    // finalize rule the composition uses.
    CacheConfig sized = topology;
    sized.mtp_sub_configs.clear();
    sized.finalizeBlockNums(local_block_num, runtime_config);
    PPSnapshotCollector collector(StageCacheSnapshot::fromConfig(sized));
    auto                validation = initPPCacheGeometry(collector, capacity_skew_threshold_);
    if (!validation.ok) {
        RTP_LLM_FAIL("PP cache topology validation failed: %s", validation.error.c_str());
    }
    RTP_LLM_LOG_INFO("PP cache negotiation: local block_num %u -> agreed paged %u over %zu canonical groups",
                     local_block_num,
                     validation.agreed.paged_block_num,
                     validation.canonical_groups.size());
    return validation;
}

void PPCacheCapacityNegotiator::validateComposed(const CacheConfig& composed, const NegotiatedCapacity& agreed) {
    validatePPComposedBlockNums(composed, agreed);
}

void applyPPCanonicalIndices(CacheConfig& config, const PPValidationResult& validation) {
    RTP_LLM_CHECK_WITH_INFO(validation.ok, "applyPPCanonicalIndices requires a successful PP validation");
    const size_t group_num = static_cast<size_t>(config.groupNums());
    if (group_num == 0) {
        return;
    }

    std::unordered_map<std::string, size_t> canonical_index;
    canonical_index.reserve(validation.canonical_groups.size());
    for (size_t c = 0; c < validation.canonical_groups.size(); ++c) {
        canonical_index.emplace(validation.canonical_groups[c].tag, c);
    }

    auto groups = config.topology().groups();
    for (auto& group : groups) {
        const auto it = canonical_index.find(group.tag);
        RTP_LLM_CHECK_WITH_INFO(it != canonical_index.end(),
                                "local group [%s] is missing from the PP canonical group table",
                                group.tag.c_str());
        group.canonical_idx = it->second;
    }
    config.setTopology(std::move(groups), config.topology().layers());
    for (auto& sub_config : config.mtp_sub_configs) {
        if (sub_config != nullptr) {
            applyPPCanonicalIndices(*sub_config, validation);
        }
    }
}

}  // namespace rtp_llm
