#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

class CacheConfig;

// Cache geometry snapshot of one PP stage, exchanged at startup after the
// KV pools are built. All fields are per-group and indexed by gid; the
// ordered tag list is the semantic identity of the group layout (gid is a
// stage-private topology index).
struct StageCacheSnapshot {
    std::vector<std::string>    group_tags;                 // ordered tag list
    std::vector<CacheGroupType> group_types;                // type sequence
    std::vector<size_t>         seq_size_per_block;         // per group
    std::vector<size_t>         kernel_seq_size_per_block;  // per group
    std::vector<uint32_t>       block_nums;                 // actually allocated blocks
    std::vector<uint32_t>       explicit_block_nums;        // 0 = follows the paged budget
    std::vector<std::string>    policy_fingerprints;        // CacheGroupPolicy digest per group

    // True when all per-group vectors have the same length as group_tags.
    bool internallyConsistent() const;

    // Builds the snapshot from a fully initialized CacheConfig.
    static StageCacheSnapshot fromConfig(const CacheConfig& config);

    // Self-describing text wire format; deserialize rejects malformed payloads.
    std::string               serialize() const;
    static StageCacheSnapshot deserialize(const std::string& payload);
};

// One row of the canonical group table: the cross-stage union of cache
// groups, ordered stage-0-first. Under the stage-0-superset rule (see
// validatePPTopology) the union equals stage 0's own group list, so the
// leading allocator physically owns every entry and issues all block ids
// from its own pools.
struct CanonicalGroupEntry {
    std::string    tag;
    CacheGroupType type                      = CacheGroupType::FULL;
    size_t         seq_size_per_block        = 0;
    size_t         kernel_seq_size_per_block = 0;
    uint32_t       logical_block_num         = 0;  // min across owner stages
    uint32_t       explicit_block_num        = 0;  // must agree across owners; 0 = budget-following
    std::string    policy_fingerprint;             // must agree across owners
};

// Text digest of a CacheGroupPolicy covering every field samePolicy()
// compares. Delimiter-free by construction (wire-safe).
std::string cacheGroupPolicyFingerprint(const struct CacheGroupPolicy& policy);

struct PPValidationResult {
    bool        ok = false;
    std::string error;  // human-readable reject reason
    // Whole-model union of cache groups; valid only when ok. Single source
    // of truth for cross-stage group identity, ordering (canonical_idx) and
    // logical capacity (per-entry min over owners).
    std::vector<CanonicalGroupEntry> canonical_groups;
};

// Validates cache geometry across PP stages and computes the canonical
// group table. Pairing-by-tag is the base rule; when all stages report the
// same tag list the strict equality check is elevated automatically.
// Rules:
//  - stages.size() <= 1: trivially ok (pp_size=1 behavior)
//  - internally inconsistent snapshot: fail
//  - hybrid stage (any LINEAR group) without a FULL group: fail
//  - any stage owning a group absent from stage 0: fail (stage-0-superset
//    rule: the leading stage must physically own every cache group, since
//    it issues all block ids from its own pools)
//  - shared tags with different type / seq size / kernel seq size: fail
//  - any group with 0 blocks: fail (group could not allocate)
//  - capacity skew max/min > capacity_skew_threshold on any tag: fail
//  - canonical_groups: union over all stages (stage-0 order first);
//    same-tag owners must agree on type and geometry; logical_block_num is
//    the min over owners
PPValidationResult validatePPTopology(const std::vector<StageCacheSnapshot>& stages,
                                      double                                 capacity_skew_threshold = 1.5);

// Collects cache snapshots from all PP stages.
class StageSnapshotCollector {
public:
    virtual ~StageSnapshotCollector()                 = default;
    virtual std::vector<StageCacheSnapshot> collect() = 0;
};

// Local single-stage collector (pp_size=1): returns this stage's own snapshot.
class LocalStageSnapshotCollector: public StageSnapshotCollector {
public:
    explicit LocalStageSnapshotCollector(StageCacheSnapshot snapshot): snapshot_(std::move(snapshot)) {}

    std::vector<StageCacheSnapshot> collect() override {
        return {snapshot_};
    }

private:
    StageCacheSnapshot snapshot_;
};

// Cross-stage collector: exchanges this stage's serialized snapshot with
// every PP stage over the PP process group (execPPSnapshotExchange) and
// deserializes the results in stage (pp_rank) order. Startup-only: all
// stages must reach the call.
class PPSnapshotCollector: public StageSnapshotCollector {
public:
    explicit PPSnapshotCollector(StageCacheSnapshot local_snapshot): local_(std::move(local_snapshot)) {}

    std::vector<StageCacheSnapshot> collect() override;

private:
    StageCacheSnapshot local_;
};

// One-stop startup entry: collect snapshots from all stages, then validate.
// On failure the caller must abort startup with result.error (fail fast).
PPValidationResult initPPCacheGeometry(StageSnapshotCollector& collector, double capacity_skew_threshold = 1.5);

// Applies the validated logical (min) block counts to this stage's
// CacheConfig: every local group is paired with its canonical entry by tag
// and sized to the entry's cross-stage min. A local count below the
// canonical min violates the startup invariant (the min includes this
// stage's own snapshot) and aborts startup instead of silently building an
// undersized pool. The top-level
// block_num is capped at the min over the budget-following paged pools
// only; explicitly sized pools are decoupled from it. Must run after
// finalizeBlockNums and before KVCacheManager::init(): pools are physically
// sized by the capped counts, so a richer stage simply allocates a smaller
// pool and leaves the surplus VRAM free. No-op for configs without groups.
void applyPPLogicalBlockNums(CacheConfig& config, const PPValidationResult& validation);

// Fills each local group's canonical_idx from the canonical group table by
// tag pairing. Must run before KVCacheManager::init(): plan columns are
// addressed by canonical index and downstream column selection resolves
// local groups through canonical_idx. Every local tag must appear in the
// canonical table.
void applyPPCanonicalIndices(CacheConfig& config, const PPValidationResult& validation);

}  // namespace rtp_llm
