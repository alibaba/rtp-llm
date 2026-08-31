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

}  // namespace

bool StageCacheSnapshot::internallyConsistent() const {
    const size_t n = group_tags.size();
    return group_types.size() == n && seq_size_per_block.size() == n && kernel_seq_size_per_block.size() == n
           && block_nums.size() == n && explicit_block_nums.size() == n && policy_fingerprints.size() == n;
}

std::string cacheGroupPolicyFingerprint(const CacheGroupPolicy& policy) {
    // Fixed-field digest covering every field CacheConfig::samePolicy()
    // compares; uses ':' separators only, so it is wire-safe.
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

// Wire format: "v1|tags|types|seq|kseq|blocks|explicit|fingerprints"; tags
// and fingerprints joined with \x1f, numeric fields joined with ','.
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
        }
        return result;
    }

    for (size_t s = 0; s < stages.size(); ++s) {
        if (!stages[s].internallyConsistent()) {
            return fail("stage " + std::to_string(s) + " cache snapshot is internally inconsistent");
        }
        // Invariant: a hybrid stage (any LINEAR group) must keep at least one
        // FULL group.
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
        // v1 scope: sliding-window pools use step-derived capacities whose
        // cross-stage reconciliation is not implemented yet.
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
        // Elevated path: identical tag sets allow the strict equality check
        // (the original safety net for stage-scoped isomorphic topologies).
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
        // Pairing path: stage-scoped topologies may legitimately hold
        // different tag subsets. Match groups by tag name against stage 0
        // instead of requiring equal tag lists.
        //
        // Superset gate: the leading stage issues every block id from its own
        // physical pools, so it must own every group that appears anywhere.
        // Bookkeeping-only (layerless) pools are not supported (see the PP
        // logical bookkeeping design for the future unlock path).
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

    // Any group anywhere with zero blocks cannot serve a request.
    for (size_t s = 0; s < stages.size(); ++s) {
        for (size_t g = 0; g < stages[s].group_tags.size(); ++g) {
            if (stages[s].block_nums[g] == 0) {
                return fail("stage " + std::to_string(s) + " group [" + stages[s].group_tags[g]
                            + "] has 0 KV blocks; the group could not allocate a single block");
            }
        }
    }

    // Canonical group table: cross-stage union ordered stage-0-first (stage-0
    // order, then first-seen order on later stages). The leading allocator
    // issues block ids for every entry, so same-tag owners must agree on
    // type and geometry even when none of them is stage 0.
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
            entry.logical_block_num          = std::min(entry.logical_block_num, stages[s].block_nums[g]);
            canonical_max_blocks[it->second] = std::max(canonical_max_blocks[it->second], stages[s].block_nums[g]);
        }
    }

    // Capacity skew guard over the whole canonical table (not just stage-0
    // tags): an oversized owner would let the leading allocator issue ids
    // beyond a smaller owner's pool.
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

    // Per-stage logical counts live on the canonical entries themselves;
    // consumers look entries up by tag (gid is stage-private).
    result.ok = true;
    return result;
}

PPValidationResult initPPCacheGeometry(StageSnapshotCollector& collector, double capacity_skew_threshold) {
    return validatePPTopology(collector.collect(), capacity_skew_threshold);
}

std::vector<StageCacheSnapshot> PPSnapshotCollector::collect() {
    // All-gather over the PP process group; payloads come back in group-rank
    // order, which equals pp_rank order (lane members are ascending world
    // ranks), so vector index == stage index.
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

void applyPPLogicalBlockNums(CacheConfig& config, const PPValidationResult& validation) {
    RTP_LLM_CHECK_WITH_INFO(validation.ok, "applyPPLogicalBlockNums requires a successful PP validation");
    const size_t group_num = static_cast<size_t>(config.groupNums());
    if (group_num == 0) {
        return;
    }

    // Look every local group up in the canonical table by tag: entries carry
    // the cross-stage min over all owners, so groups owned only by later
    // stages are capped too. Strides are untouched (geometry was already
    // validated identical for same-tag owners).
    std::unordered_map<std::string, uint32_t> logical_blocks;
    logical_blocks.reserve(validation.canonical_groups.size());
    // The top-level block_num is the capacity yardstick of the paged pools
    // that follow the global budget (independent-pool semantics): explicitly
    // sized pools are decoupled from it and must not drag it down.
    uint32_t paged_min = std::numeric_limits<uint32_t>::max();
    for (const auto& entry : validation.canonical_groups) {
        logical_blocks.emplace(entry.tag, entry.logical_block_num);
        const bool follows_global_budget =
            entry.explicit_block_num == 0
            && (entry.type == CacheGroupType::FULL || entry.type == CacheGroupType::LINEAR);
        if (follows_global_budget) {
            paged_min = std::min(paged_min, entry.logical_block_num);
        }
    }

    std::vector<uint32_t> block_nums;
    std::vector<size_t>   kv_strides;
    std::vector<size_t>   scale_strides;
    block_nums.reserve(group_num);
    kv_strides.reserve(group_num);
    scale_strides.reserve(group_num);
    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto& group = config.topology().groupById(gid);
        const auto  it    = logical_blocks.find(group.tag);
        RTP_LLM_CHECK_WITH_INFO(it != logical_blocks.end(),
                                "local group [%s] is missing from the PP canonical group table",
                                group.tag.c_str());
        // The canonical min includes this stage's own snapshot value, so the
        // local count can never fall below it; a violation means the local
        // block count was lowered after the startup snapshot exchange. Fail
        // fast instead of silently building a pool smaller than the id space
        // the leading stage may issue (runtime out-of-range writes).
        RTP_LLM_CHECK_WITH_INFO(group.block_num >= it->second,
                                "local group [%s] block_num %u is below the PP canonical logical min %u; "
                                "local capacity changed after the startup snapshot exchange",
                                group.tag.c_str(),
                                group.block_num,
                                it->second);
        // After the check, min(local, canonical) == canonical: pools are
        // sized exactly to the cross-stage agreed logical capacity, leaving
        // the richer stage's surplus VRAM unallocated.
        block_nums.push_back(it->second);
        kv_strides.push_back(group.kv_block_stride_bytes);
        scale_strides.push_back(group.kv_scale_stride_bytes);
    }

    config.setGroupBlockLayout(block_nums, kv_strides, scale_strides);
    // Keep the top-level (log/scheduler-facing) block count consistent with
    // the capped paged pools; only ever decreases.
    if (paged_min != std::numeric_limits<uint32_t>::max() && config.block_num > static_cast<int>(paged_min)) {
        RTP_LLM_LOG_INFO("PP logical capacity caps local block_num %d to %u (paged-pool canonical min)",
                         config.block_num,
                         paged_min);
        config.block_num = static_cast<int>(paged_min);
    }
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
}

}  // namespace rtp_llm
