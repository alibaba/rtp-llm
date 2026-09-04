#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheCapacityNegotiator.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

// Cache geometry of one PP stage, exchanged at startup before any pool exists.
// Per-group fields are indexed by position in the tag-sorted group_tags list.
// Carries facts only (geometry plus what this stage can afford), never a
// stage-global capacity scalar: the agreement is per tag.
struct StageCacheSnapshot {
    std::vector<std::string>    group_tags;  // tag-sorted
    std::vector<CacheGroupType> group_types;
    std::vector<size_t>         seq_size_per_block;
    std::vector<size_t>         kernel_seq_size_per_block;
    std::vector<uint32_t>       block_nums;           // what this stage can afford
    std::vector<uint32_t>       explicit_block_nums;  // 0 = follows the paged budget
    std::vector<std::string>    policy_fingerprints;

    bool internallyConsistent() const;

    // Derives the per-group counts on a throwaway copy through the same
    // finalizeBlockNums rule the final composition uses, so a stage reports
    // exactly what it would build.
    static StageCacheSnapshot
    fromTopologyAndCapacity(const CacheConfig& topology, uint32_t local_capacity, const RuntimeConfig& runtime_config);

    std::string               serialize() const;
    static StageCacheSnapshot deserialize(const std::string& payload);
};

// One row of the canonical table: the cross-stage UNION of cache groups in
// tag-sorted order. A stage may own a proper subset, so logical_block_num is
// the minimum over that tag's OWNERS only (a single-owner tag keeps its own).
struct CanonicalGroupEntry {
    std::string    tag;
    CacheGroupType type                      = CacheGroupType::FULL;
    size_t         seq_size_per_block        = 0;
    size_t         kernel_seq_size_per_block = 0;
    uint32_t       logical_block_num         = 0;  // min across owner stages
    uint32_t       explicit_block_num        = 0;  // must agree; 0 = budget-following
    std::string    policy_fingerprint;             // must agree across owners
};

// Text digest of a CacheGroupPolicy covering every field samePolicy() compares.
std::string cacheGroupPolicyFingerprint(const struct CacheGroupPolicy& policy);

struct PPValidationResult {
    bool                             ok = false;
    std::string                      error;  // human-readable reject reason
    std::vector<CanonicalGroupEntry> canonical_groups;
    NegotiatedCapacity               agreed;  // construction inputs, valid when ok
};

// Validates cache geometry across PP stages and derives the canonical table
// plus the per-tag construction inputs. Rejects: internally inconsistent
// snapshots; stage 0 not a superset of the union (ALLOCATION AUTHORITY: the
// leading stage produces the block table shipped downstream, so it must own a
// pool for every tag it issues ids for); a hybrid stage without a FULL group;
// any SWA group; shared tags disagreeing on type, geometry, explicit sizing or
// policy fingerprint; any group with 0 blocks; capacity skew above threshold.
PPValidationResult validatePPTopology(const std::vector<StageCacheSnapshot>& stages,
                                      double                                 capacity_skew_threshold = 1.5);

class StageSnapshotCollector {
public:
    virtual ~StageSnapshotCollector()                 = default;
    virtual std::vector<StageCacheSnapshot> collect() = 0;
};

// pp_size=1: returns this stage's own snapshot.
class LocalStageSnapshotCollector: public StageSnapshotCollector {
public:
    explicit LocalStageSnapshotCollector(StageCacheSnapshot snapshot): snapshot_(std::move(snapshot)) {}

    std::vector<StageCacheSnapshot> collect() override {
        return {snapshot_};
    }

private:
    StageCacheSnapshot snapshot_;
};

// All-gathers the serialized snapshot over the PP process group and
// deserializes in pp_rank order. Startup-only: all stages must reach the call.
class PPSnapshotCollector: public StageSnapshotCollector {
public:
    explicit PPSnapshotCollector(StageCacheSnapshot local_snapshot): local_(std::move(local_snapshot)) {}

    std::vector<StageCacheSnapshot> collect() override;

private:
    StageCacheSnapshot local_;
};

PPValidationResult initPPCacheGeometry(StageSnapshotCollector& collector, double capacity_skew_threshold = 1.5);

// Fuse over the composed config: every local group must carry exactly its
// agreed count. A mismatch means a tag fell back to the derivation rule, or
// local capacity moved after the negotiation.
void validatePPComposedBlockNums(const CacheConfig& composed, const NegotiatedCapacity& agreed);

// PP implementation of CacheCapacityNegotiator: exchange snapshots, validate,
// reduce the canonical table to the agreed inputs, then run the fuse above.
// Both hooks abort startup on failure.
class PPCacheCapacityNegotiator: public CacheCapacityNegotiator {
public:
    explicit PPCacheCapacityNegotiator(double capacity_skew_threshold = 1.5):
        capacity_skew_threshold_(capacity_skew_threshold) {}

    NegotiatedCapacity
    negotiate(const CacheConfig& topology, uint32_t local_block_num, const RuntimeConfig& runtime_config) override;

    void validateComposed(const CacheConfig& composed, const NegotiatedCapacity& agreed) override;

private:
    double capacity_skew_threshold_;
};

}  // namespace rtp_llm
