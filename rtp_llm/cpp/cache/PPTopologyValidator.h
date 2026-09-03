#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

class CacheConfig;

// Per-group cache geometry of one PP stage, exchanged at startup; tags are the cross-stage identity, gid is
// stage-private.
struct StageCacheSnapshot {
    std::vector<std::string>    group_tags;
    std::vector<CacheGroupType> group_types;
    std::vector<size_t>         seq_size_per_block;
    std::vector<size_t>         kernel_seq_size_per_block;
    std::vector<uint32_t>       block_nums;
    std::vector<uint32_t>       explicit_block_nums;  // 0 = follows the paged budget
    std::vector<std::string>    policy_fingerprints;

    // True when all per-group vectors have the same length as group_tags.
    bool internallyConsistent() const;

    static StageCacheSnapshot fromConfig(const CacheConfig& config);

    std::string               serialize() const;
    static StageCacheSnapshot deserialize(const std::string& payload);
};

// One row of the canonical group table: cross-stage union of cache groups in stage-0 order.
struct CanonicalGroupEntry {
    std::string    tag;
    CacheGroupType type                      = CacheGroupType::FULL;
    size_t         seq_size_per_block        = 0;
    size_t         kernel_seq_size_per_block = 0;
    uint32_t       logical_block_num         = 0;  // min across owner stages
    uint32_t       explicit_block_num        = 0;  // must agree across owners; 0 = budget-following
    std::string    policy_fingerprint;             // must agree across owners
};

// Digest covering every field samePolicy() compares.
std::string cacheGroupPolicyFingerprint(const struct CacheGroupPolicy& policy);

struct PPValidationResult {
    bool        ok = false;
    std::string error;
    // Cross-stage union of cache groups; valid only when ok.
    std::vector<CanonicalGroupEntry> canonical_groups;
};

/* Validates cache geometry across PP stages and builds the canonical group
   table. Groups pair by tag; identical tag lists elevate to strict equality.
   Every stage's groups must be a subset of stage 0's (the leading stage
   issues all block ids); same-tag owners must agree on type and geometry;
   logical_block_num is the min over owners; capacity skew is bounded by
   capacity_skew_threshold. stages.size() <= 1 is trivially ok. */
PPValidationResult validatePPTopology(const std::vector<StageCacheSnapshot>& stages,
                                      double                                 capacity_skew_threshold = 1.5);

class StageSnapshotCollector {
public:
    virtual ~StageSnapshotCollector()                 = default;
    virtual std::vector<StageCacheSnapshot> collect() = 0;
};

class LocalStageSnapshotCollector: public StageSnapshotCollector {
public:
    explicit LocalStageSnapshotCollector(StageCacheSnapshot snapshot): snapshot_(std::move(snapshot)) {}

    std::vector<StageCacheSnapshot> collect() override {
        return {snapshot_};
    }

private:
    StageCacheSnapshot snapshot_;
};

// Exchanges serialized snapshots over the PP process group; collective, all stages must reach it.
class PPSnapshotCollector: public StageSnapshotCollector {
public:
    explicit PPSnapshotCollector(StageCacheSnapshot local_snapshot): local_(std::move(local_snapshot)) {}

    std::vector<StageCacheSnapshot> collect() override;

private:
    StageCacheSnapshot local_;
};

// Collect + validate; on failure the caller must abort startup.
PPValidationResult initPPCacheGeometry(StageSnapshotCollector& collector, double capacity_skew_threshold = 1.5);

/* Sizes every local group to its canonical entry's cross-stage min, paired
   by tag; a local count below the min aborts startup. Top-level block_num
   follows the budget-following paged pools only. Must run after
   finalizeBlockNums and before KVCacheManager::init(). */
void applyPPLogicalBlockNums(CacheConfig& config, const PPValidationResult& validation);

// Fills each local group's canonical_idx by tag pairing; must run before KVCacheManager::init().
void applyPPCanonicalIndices(CacheConfig& config, const PPValidationResult& validation);

}  // namespace rtp_llm
