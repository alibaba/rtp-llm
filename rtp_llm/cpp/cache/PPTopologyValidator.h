#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheCapacityNegotiator.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

/** Per-group cache geometry of one PP stage; tag is the cross-stage identity, gid is stage-private. */
struct StageCacheGroupSnapshot {
    std::string    tag;
    CacheGroupType type                      = CacheGroupType::FULL;
    bool           block_tree_group_member   = false;
    size_t         seq_size_per_block        = 0;
    size_t         kernel_seq_size_per_block = 0;
    size_t         cache_key_token_stride    = 0;
    uint32_t       block_num                 = 0;
    uint32_t       explicit_block_num        = 0;  // 0 = follows the paged budget
    std::string    policy_fingerprint;
};

/** Per-stage cache geometry exchanged at startup. */
struct StageCacheSnapshot {
    std::vector<StageCacheGroupSnapshot> groups;
    // Sum of the aligned HOST/DISK strides of this stage's BlockTree group sets.
    size_t block_tree_combined_stride_bytes{0};

    static StageCacheSnapshot captureLocalStage(const CacheConfig& config,
                                                size_t             local_block_tree_combined_stride_bytes);

    std::string               serialize() const;
    static StageCacheSnapshot deserialize(const std::string& payload);
};

/** One row of the agreed group table: cross-stage union of cache groups in stage-0 order. */
struct CanonicalGroupEntry {
    std::string    tag;
    CacheGroupType type                      = CacheGroupType::FULL;
    size_t         seq_size_per_block        = 0;
    size_t         kernel_seq_size_per_block = 0;
    size_t         cache_key_token_stride    = 0;
    uint32_t       logical_block_num         = 0;  // min across owner stages
    uint32_t       explicit_block_num        = 0;  // must agree across owners; 0 = budget-following
    std::string    policy_fingerprint;             // must agree across owners
};

struct PPValidationResult {
    bool        ok = false;
    std::string error;
    // Cross-stage union of cache groups; valid only when ok.
    std::vector<CanonicalGroupEntry> canonical_groups;
    NegotiatedCapacity               agreed;  // construction inputs, valid when ok
};

/**
 * Validates cache geometry across PP stages and builds the agreed group
 * table. Groups pair by tag;
 * every stage's groups must be a subset of stage 0's because the leading
 * stage issues all block ids. Same-tag owners must agree on policy and token
 * geometry; capacity is the owner-stage minimum. A reusable group set must
 * be fully present or absent on each stage; partial overlap is rejected until
 * per-rank request projection is implemented.
 */
PPValidationResult validatePPTopology(const std::vector<StageCacheSnapshot>& stages);

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

/** Exchanges serialized snapshots within one (dp_rank, tp_rank) PP lane; all stages in that lane must participate. */
class PPSnapshotCollector: public StageSnapshotCollector {
public:
    explicit PPSnapshotCollector(StageCacheSnapshot local_snapshot): local_(std::move(local_snapshot)) {}

    std::vector<StageCacheSnapshot> collect() override;

private:
    StageCacheSnapshot local_;
};

/**
 * Collect + validate; PPSnapshotCollector performs a collective exchange before validation.
 * On failure the caller must abort startup.
 */
PPValidationResult initPPCacheGeometry(StageSnapshotCollector& collector);

/**
 * PP implementation of CacheCapacityNegotiator: exchange snapshots, validate,
 * and reduce the canonical table to the agreed inputs. It aborts startup on
 * failure.
 */
class PPCacheCapacityNegotiator: public CacheCapacityNegotiator {
public:
    NegotiatedCapacity negotiate(const CacheConfig&   topology,
                                 uint32_t             local_block_num,
                                 size_t               local_block_tree_combined_stride_bytes,
                                 const RuntimeConfig& runtime_config) override;
};

}  // namespace rtp_llm
