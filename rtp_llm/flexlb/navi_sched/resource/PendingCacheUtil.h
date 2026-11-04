#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedCommon.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentPendingCacheToMasterData.h"
#include "autil/FlatHashmap.h"

#include <cstdint>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace navi_sched {

struct PendingCacheReferencedRequest {
    const NaviSchedBlockHashes *blockHashes = nullptr;
    int64_t totalTokenCount = 0;
    int64_t blockSize = 0;
};

class PendingCacheUtil {
public:
    static bool makeReservation(
        int64_t batchId, const std::vector<PendingCacheReferencedRequest> &requests,
        const NaviSchedCachePolicy &policy,
        NaviSchedPendingCacheReservation &reservation);
    static bool validReservation(
        const NaviSchedPendingCacheReservation &reservation);
    static bool equalReservation(
        const NaviSchedPendingCacheReservation &left,
        const NaviSchedPendingCacheReservation &right);
    static uint64_t reservationChecksum(
        const NaviSchedPendingCacheReservation &reservation);
    static uint64_t reservationBlockChecksum(int64_t batchId, uint64_t blockHash,
                                            bool tailReady);
};

struct PendingCacheStateStats {
    size_t batchCount = 0;
    size_t uniqueBlockCount = 0;
    size_t uniqueTailReadyBlockCount = 0;
    uint64_t checksum = 0;
};

// Locking belongs to the Agent/Master resource wrappers so the same state
// transition code is used on both sides of the synchronization protocol.
class PendingCacheState {
public:
    bool upsert(const NaviSchedPendingCacheReservation &reservation,
                bool replaceExisting, bool &changed,
                const PendingCacheState *other = nullptr,
                PendingCacheStateStats *overlap = nullptr);
    bool erase(int64_t batchId, bool &changed,
               const PendingCacheState *other = nullptr,
               PendingCacheStateStats *overlap = nullptr);
    bool replace(
        const std::vector<NaviSchedPendingCacheReservation> &reservations);
    void clear();

    bool containsFullReady(uint64_t blockHash) const;
    bool containsTailReady(uint64_t blockHash) const;
    bool containsFullReadyExcludingSchedule(uint64_t blockHash,
                                            int64_t scheduleBatchId) const;
    bool containsTailReadyExcludingSchedule(uint64_t blockHash,
                                            int64_t scheduleBatchId) const;
    bool getReservation(int64_t batchId,
                        NaviSchedPendingCacheReservation &reservation) const;
    int64_t getSourceBatchId(uint64_t blockHash, bool tailReady) const;
    PendingCacheStateStats getStats() const;
    size_t countBatchOverlap(const PendingCacheState &other) const;
    size_t countFullReadyOverlap(const PendingCacheState &other) const;
    size_t countTailReadyOverlap(const PendingCacheState &other) const;
    bool sameReservations(const PendingCacheState &other) const;
    std::vector<NaviSchedPendingCacheReservation> makeSnapshot() const;

private:
    friend class MasterPendingCacheR;
    // Master reservations are produced in canonical order by construction.
    bool upsertGenerated(
        const NaviSchedPendingCacheReservation &reservation,
        bool replaceExisting, bool &changed,
        const PendingCacheState *other,
        PendingCacheStateStats *overlap);
    bool removeGeneratedBlocks(int64_t batchId,
                               const std::vector<uint64_t> &fullReadyBlockHashes,
                               const std::vector<uint64_t> &tailReadyBlockHashes,
                               const PendingCacheState *other,
                               PendingCacheStateStats *overlap);
    void add(const NaviSchedPendingCacheReservation &reservation,
             const PendingCacheState *other, PendingCacheStateStats *overlap);
    void remove(
        std::unordered_map<int64_t, NaviSchedPendingCacheReservation>::iterator it,
        const PendingCacheState *other, PendingCacheStateStats *overlap);
    void eraseMetadata(
        std::unordered_map<int64_t, NaviSchedPendingCacheReservation>::iterator it,
        const PendingCacheState *other, PendingCacheStateStats *overlap);

private:
    // Most hashes have one owner; allocate a tree only for shared hashes.
    struct BlockOwners {
        BlockOwners() = default;
        BlockOwners(const BlockOwners &other);
        BlockOwners &operator=(const BlockOwners &other);
        BlockOwners(BlockOwners &&) noexcept = default;
        BlockOwners &operator=(BlockOwners &&) noexcept = default;

        void add(int64_t batchId);
        void remove(int64_t batchId);
        size_t size() const;
        int64_t first() const;
        bool contains(int64_t batchId) const;

        int64_t singleBatchId = 0;
        std::unique_ptr<std::set<int64_t>> sharedBatchIds;
    };

    bool hasOwnerOutsideSchedule(const BlockOwners &owners,
                                 int64_t scheduleBatchId) const;

    std::unordered_map<int64_t, NaviSchedPendingCacheReservation> _reservations;
    struct ReducedReservation {
        size_t fullReadyBlockCount = 0;
    };
    // Retain the original vectors; live owners supply the view during export.
    std::unordered_map<int64_t, ReducedReservation> _reducedReservations;
    std::unordered_map<int64_t, uint64_t> _reservationChecksums;
    // Agent-local batches can share one Master schedule batch.
    std::unordered_map<int64_t, std::unordered_set<int64_t>> _scheduleBatchIds;
    autil::flat_hash_map<uint64_t, BlockOwners> _fullReadyOwners;
    autil::flat_hash_map<uint64_t, BlockOwners> _tailReadyOwners;
    uint64_t _checksum = 0;
};

} // namespace navi_sched
