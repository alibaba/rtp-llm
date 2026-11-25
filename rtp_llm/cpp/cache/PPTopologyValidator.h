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
    bool           has_local_backing         = true;
    size_t         seq_size_per_block        = 0;
    size_t         kernel_seq_size_per_block = 0;
    size_t         cache_key_token_stride    = 0;
    uint32_t       block_num                 = 0;
    uint32_t       explicit_block_num        = 0;  // 0 = follows the paged budget
    std::string    policy_fingerprint;
    bool           is_draft = false;
};

/** Per-stage cache geometry exchanged at startup. */
struct StageCacheSnapshot {
    std::vector<StageCacheGroupSnapshot> groups;
    BlockTreeCacheSizing                 block_tree;

    static StageCacheSnapshot captureLocalStage(const CacheConfig&          config,
                                                uint32_t                    local_block_num,
                                                const BlockTreeCacheSizing& block_tree_sizing);

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
 * table. All stages declare the same logical tags and agree on their policy and
 * token geometry. The leading stage assigns IDs using the minimum capacity
 * among stages with local backing for each tag. Draft (MTP) groups carry
 * allocation-only rows on non-final stages; their physical backing lives on
 * the last PP stage.
 */
PPValidationResult validatePPTopology(const std::vector<StageCacheSnapshot>& stages);

/** Exchanges serialized snapshots within one (dp_rank, tp_rank) PP lane; all stages in that lane must participate. */
class PPSnapshotCollector {
public:
    PPSnapshotCollector(StageCacheSnapshot local_snapshot, int64_t pp_size):
        local_(std::move(local_snapshot)), pp_size_(pp_size) {}

    std::vector<StageCacheSnapshot> collect();

private:
    StageCacheSnapshot local_;
    int64_t            pp_size_;
};

std::vector<std::string> collectPPStagePayloads(const std::string& payload, int64_t pp_size);

/**
 * PP implementation of CacheCapacityNegotiator: exchange snapshots, validate,
 * and reduce the canonical table to the agreed inputs. It aborts startup on
 * failure.
 */
class PPCacheCapacityNegotiator: public CacheCapacityNegotiator {
public:
    explicit PPCacheCapacityNegotiator(int64_t pp_size): pp_size_(pp_size) {}

    NegotiatedCapacity negotiate(const CacheConfig&          topology,
                                 uint32_t                    local_block_num,
                                 const BlockTreeCacheSizing& block_tree_sizing,
                                 const RuntimeConfig&        runtime_config) override;

private:
    int64_t pp_size_;
};

}  // namespace rtp_llm
