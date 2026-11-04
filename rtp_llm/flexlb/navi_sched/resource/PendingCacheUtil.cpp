#include "aios/navi_sched/resource/PendingCacheUtil.h"

#include <algorithm>
#include <unordered_set>

#include "absl/container/flat_hash_set.h"

namespace navi_sched {
namespace {

uint64_t mix(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

}

bool PendingCacheUtil::makeReservation(
    int64_t batchId, const std::vector<PendingCacheReferencedRequest> &requests,
    const NaviSchedCachePolicy &policy,
    NaviSchedPendingCacheReservation &reservation)
{
    if (batchId <= 0 || requests.empty() || !policy.valid()) {
        return false;
    }
    absl::flat_hash_set<uint64_t> fullReady;
    absl::flat_hash_set<uint64_t> tailReady;
    for (const auto &request : requests) {
        if (!request.blockHashes || request.blockSize != policy.blockSize ||
            request.totalTokenCount < 0 ||
            request.blockHashes->size() != static_cast<size_t>(
                request.totalTokenCount / request.blockSize)) {
            return false;
        }
        const auto &blockHashes = *request.blockHashes;
        const auto blockCount = blockHashes.size();
        const bool blockAligned =
            request.totalTokenCount % request.blockSize == 0;
        for (size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
            const auto blockHash = blockHashes[blockIndex];
            fullReady.emplace(blockHash);
            if (!policy.hybridTailRequired) {
                continue;
            }
            const bool periodicTail =
                (blockIndex + 1) % policy.linearStep == 0;
            const bool finalBlock = blockIndex + 1 == blockCount;
            const bool alignedPreviousBlock =
                blockAligned && blockCount >= 2 && blockIndex + 2 == blockCount;
            if (periodicTail || finalBlock || alignedPreviousBlock) {
                tailReady.emplace(blockHash);
            }
        }
    }
    if (fullReady.empty()) {
        return false;
    }
    reservation.batchId = batchId;
    reservation.scheduleBatchId = batchId;
    reservation.fullReadyBlockHashes.assign(fullReady.begin(), fullReady.end());
    reservation.tailReadyBlockHashes.assign(tailReady.begin(), tailReady.end());
    std::sort(reservation.fullReadyBlockHashes.begin(),
              reservation.fullReadyBlockHashes.end());
    std::sort(reservation.tailReadyBlockHashes.begin(),
              reservation.tailReadyBlockHashes.end());
    return true;
}

bool PendingCacheUtil::validReservation(
    const NaviSchedPendingCacheReservation &reservation)
{
    if (reservation.batchId <= 0 || reservation.scheduleBatchId <= 0 ||
        reservation.fullReadyBlockHashes.empty() ||
        !std::is_sorted(reservation.fullReadyBlockHashes.begin(),
                        reservation.fullReadyBlockHashes.end()) ||
        !std::is_sorted(reservation.tailReadyBlockHashes.begin(),
                        reservation.tailReadyBlockHashes.end()) ||
        std::adjacent_find(reservation.fullReadyBlockHashes.begin(),
                           reservation.fullReadyBlockHashes.end()) !=
            reservation.fullReadyBlockHashes.end() ||
        std::adjacent_find(reservation.tailReadyBlockHashes.begin(),
                           reservation.tailReadyBlockHashes.end()) !=
            reservation.tailReadyBlockHashes.end()) {
        return false;
    }
    return std::includes(reservation.fullReadyBlockHashes.begin(),
                         reservation.fullReadyBlockHashes.end(),
                         reservation.tailReadyBlockHashes.begin(),
                         reservation.tailReadyBlockHashes.end());
}

bool PendingCacheUtil::equalReservation(
    const NaviSchedPendingCacheReservation &left,
    const NaviSchedPendingCacheReservation &right)
{
    return left.batchId == right.batchId &&
           left.scheduleBatchId == right.scheduleBatchId &&
           left.fullReadyBlockHashes == right.fullReadyBlockHashes &&
           left.tailReadyBlockHashes == right.tailReadyBlockHashes;
}

uint64_t PendingCacheUtil::reservationChecksum(
    const NaviSchedPendingCacheReservation &reservation)
{
    const auto batchHash = mix(static_cast<uint64_t>(reservation.batchId));
    const auto scheduleBatchHash =
        mix(static_cast<uint64_t>(reservation.scheduleBatchId));
    uint64_t checksum =
        mix(batchHash ^ scheduleBatchHash ^ 0x6f4f3b2a1908d7c6ULL);
    for (auto blockHash : reservation.fullReadyBlockHashes) {
        checksum ^= mix(batchHash ^ blockHash ^ 0x46a532f9d8127b01ULL);
    }
    for (auto blockHash : reservation.tailReadyBlockHashes) {
        checksum ^= mix(batchHash ^ blockHash ^ 0xb837e14c65fa209dULL);
    }
    return checksum;
}

uint64_t PendingCacheUtil::reservationBlockChecksum(int64_t batchId,
                                                    uint64_t blockHash,
                                                    bool tailReady)
{
    return mix(mix(static_cast<uint64_t>(batchId)) ^ blockHash ^
               (tailReady ? 0xb837e14c65fa209dULL : 0x46a532f9d8127b01ULL));
}

PendingCacheState::BlockOwners::BlockOwners(const BlockOwners &other)
    : singleBatchId(other.singleBatchId)
    , sharedBatchIds(other.sharedBatchIds
                         ? std::make_unique<std::set<int64_t>>(*other.sharedBatchIds)
                         : nullptr)
{
}

PendingCacheState::BlockOwners &
PendingCacheState::BlockOwners::operator=(const BlockOwners &other)
{
    if (this != &other) {
        BlockOwners copy(other);
        *this = std::move(copy);
    }
    return *this;
}

void PendingCacheState::BlockOwners::add(int64_t batchId)
{
    if (sharedBatchIds) {
        sharedBatchIds->insert(batchId);
    } else if (singleBatchId == 0) {
        singleBatchId = batchId;
    } else if (singleBatchId != batchId) {
        sharedBatchIds = std::make_unique<std::set<int64_t>>(
            std::initializer_list<int64_t>{singleBatchId, batchId});
        singleBatchId = 0;
    }
}

void PendingCacheState::BlockOwners::remove(int64_t batchId)
{
    if (sharedBatchIds) {
        sharedBatchIds->erase(batchId);
        if (sharedBatchIds->size() == 1) {
            singleBatchId = *sharedBatchIds->begin();
            sharedBatchIds.reset();
        }
    } else if (singleBatchId == batchId) {
        singleBatchId = 0;
    }
}

size_t PendingCacheState::BlockOwners::size() const
{
    return sharedBatchIds ? sharedBatchIds->size() : (singleBatchId != 0);
}

int64_t PendingCacheState::BlockOwners::first() const
{
    return sharedBatchIds ? *sharedBatchIds->begin() : singleBatchId;
}

bool PendingCacheState::BlockOwners::contains(int64_t batchId) const
{
    return sharedBatchIds ? sharedBatchIds->count(batchId) != 0
                          : singleBatchId == batchId;
}

bool PendingCacheState::upsert(
    const NaviSchedPendingCacheReservation &reservation, bool replaceExisting,
    bool &changed, const PendingCacheState *other,
    PendingCacheStateStats *overlap)
{
    changed = false;
    if (!PendingCacheUtil::validReservation(reservation)) {
        return false;
    }
    return upsertGenerated(reservation, replaceExisting, changed, other, overlap);
}

bool PendingCacheState::upsertGenerated(
    const NaviSchedPendingCacheReservation &reservation, bool replaceExisting,
    bool &changed, const PendingCacheState *other,
    PendingCacheStateStats *overlap)
{
    changed = false;
    if ((other == nullptr) != (overlap == nullptr) || other == this) {
        return false;
    }
    auto it = _reservations.find(reservation.batchId);
    if (_reservations.end() != it) {
        if (_reducedReservations.count(reservation.batchId)) {
            NaviSchedPendingCacheReservation current;
            getReservation(reservation.batchId, current);
            it->second = std::move(current);
            _reducedReservations.erase(reservation.batchId);
        }
        if (PendingCacheUtil::equalReservation(it->second, reservation)) {
            return true;
        }
        if (!replaceExisting) {
            return false;
        }
        if (it->second.scheduleBatchId == reservation.scheduleBatchId) {
            const auto updateOwners = [&](const auto &oldHashes, const auto &newHashes,
                                          auto &owners, bool tailReady) {
                size_t oldIndex = 0;
                size_t newIndex = 0;
                while (oldIndex < oldHashes.size() || newIndex < newHashes.size()) {
                    if (oldIndex < oldHashes.size() && newIndex < newHashes.size() &&
                        oldHashes[oldIndex] == newHashes[newIndex]) {
                        ++oldIndex;
                        ++newIndex;
                        continue;
                    }
                    const bool removing = oldIndex < oldHashes.size() &&
                        (newIndex == newHashes.size() ||
                         oldHashes[oldIndex] < newHashes[newIndex]);
                    const auto hash = removing ? oldHashes[oldIndex++] : newHashes[newIndex++];
                    const bool inOther = other && (tailReady
                        ? other->containsTailReady(hash) : other->containsFullReady(hash));
                    if (removing) {
                        auto ref = owners.find(hash);
                        ref->second.remove(reservation.batchId);
                        if (ref->second.size() == 0) {
                            if (inOther) {
                                --(tailReady ? overlap->uniqueTailReadyBlockCount
                                             : overlap->uniqueBlockCount);
                            }
                            owners.erase(ref);
                        }
                    } else {
                        auto &ref = owners[hash];
                        ref.add(reservation.batchId);
                        if (ref.size() == 1 && inOther) {
                            ++(tailReady ? overlap->uniqueTailReadyBlockCount
                                         : overlap->uniqueBlockCount);
                        }
                    }
                }
            };
            updateOwners(it->second.fullReadyBlockHashes, reservation.fullReadyBlockHashes,
                         _fullReadyOwners, false);
            updateOwners(it->second.tailReadyBlockHashes, reservation.tailReadyBlockHashes,
                         _tailReadyOwners, true);
            auto &checksum = _reservationChecksums.at(reservation.batchId);
            const auto replacementChecksum = PendingCacheUtil::reservationChecksum(reservation);
            _checksum ^= checksum ^ replacementChecksum;
            checksum = replacementChecksum;
            it->second = reservation;
            changed = true;
            return true;
        }
        remove(it, other, overlap);
    }
    add(reservation, other, overlap);
    changed = true;
    return true;
}

bool PendingCacheState::removeGeneratedBlocks(
    int64_t batchId, const std::vector<uint64_t> &fullReadyBlockHashes,
    const std::vector<uint64_t> &tailReadyBlockHashes,
    const PendingCacheState *other, PendingCacheStateStats *overlap)
{
    if ((other == nullptr) != (overlap == nullptr) || other == this) {
        return false;
    }
    auto it = _reservations.find(batchId);
    if (it == _reservations.end()) {
        return false;
    }
    auto [reducedIt, inserted] = _reducedReservations.try_emplace(batchId);
    auto &reduced = reducedIt->second;
    if (inserted) {
        reduced.fullReadyBlockCount = it->second.fullReadyBlockHashes.size();
    }
    auto &batchChecksum = _reservationChecksums.at(batchId);
    auto removeHashes = [&](const auto &hashes, auto &owners, bool tailReady) {
        for (auto hash : hashes) {
            auto owner = owners.find(hash);
            owner->second.remove(batchId);
            if (owner->second.size() == 0) {
                if (other && (tailReady ? other->containsTailReady(hash)
                                       : other->containsFullReady(hash))) {
                    --(tailReady ? overlap->uniqueTailReadyBlockCount
                                 : overlap->uniqueBlockCount);
                }
                owners.erase(owner);
            }
            const auto checksum = PendingCacheUtil::reservationBlockChecksum(batchId, hash, tailReady);
            _checksum ^= checksum;
            batchChecksum ^= checksum;
        }
    };
    removeHashes(fullReadyBlockHashes, _fullReadyOwners, false);
    removeHashes(tailReadyBlockHashes, _tailReadyOwners, true);
    reduced.fullReadyBlockCount -= fullReadyBlockHashes.size();
    if (reduced.fullReadyBlockCount == 0) {
        eraseMetadata(it, other, overlap);
    }
    return true;
}

bool PendingCacheState::erase(int64_t batchId, bool &changed,
                              const PendingCacheState *other,
                              PendingCacheStateStats *overlap)
{
    changed = false;
    if ((other == nullptr) != (overlap == nullptr) || other == this) {
        return false;
    }
    if (batchId <= 0) {
        return false;
    }
    auto it = _reservations.find(batchId);
    if (_reservations.end() == it) {
        return true;
    }
    remove(it, other, overlap);
    changed = true;
    return true;
}

bool PendingCacheState::replace(
    const std::vector<NaviSchedPendingCacheReservation> &reservations)
{
    PendingCacheState replacement;
    for (const auto &reservation : reservations) {
        bool changed = false;
        if (!replacement.upsert(reservation, false, changed) || !changed) {
            return false;
        }
    }
    *this = std::move(replacement);
    return true;
}

void PendingCacheState::clear()
{
    _reservations.clear();
    _reducedReservations.clear();
    _reservationChecksums.clear();
    _scheduleBatchIds.clear();
    _fullReadyOwners.clear();
    _tailReadyOwners.clear();
    _checksum = 0;
}

bool PendingCacheState::containsFullReady(uint64_t blockHash) const
{
    return _fullReadyOwners.end() != _fullReadyOwners.find(blockHash);
}

bool PendingCacheState::containsTailReady(uint64_t blockHash) const
{
    return _tailReadyOwners.end() != _tailReadyOwners.find(blockHash);
}

bool PendingCacheState::containsFullReadyExcludingSchedule(
    uint64_t blockHash, int64_t scheduleBatchId) const
{
    auto ref = _fullReadyOwners.find(blockHash);
    if (ref == _fullReadyOwners.end()) {
        return false;
    }
    return hasOwnerOutsideSchedule(ref->second, scheduleBatchId);
}

bool PendingCacheState::containsTailReadyExcludingSchedule(
    uint64_t blockHash, int64_t scheduleBatchId) const
{
    auto ref = _tailReadyOwners.find(blockHash);
    if (ref == _tailReadyOwners.end()) {
        return false;
    }
    return hasOwnerOutsideSchedule(ref->second, scheduleBatchId);
}

bool PendingCacheState::hasOwnerOutsideSchedule(
    const BlockOwners &owners, int64_t scheduleBatchId) const
{
    if (!owners.sharedBatchIds) {
        return _reservations.at(owners.singleBatchId).scheduleBatchId != scheduleBatchId;
    }
    const auto batches = _scheduleBatchIds.find(scheduleBatchId);
    if (batches == _scheduleBatchIds.end() ||
        owners.size() > batches->second.size()) {
        return true;
    }
    for (auto batchId : *owners.sharedBatchIds) {
        if (_reservations.at(batchId).scheduleBatchId != scheduleBatchId) {
            return true;
        }
    }
    return false;
}

bool PendingCacheState::getReservation(
    int64_t batchId, NaviSchedPendingCacheReservation &reservation) const
{
    auto it = _reservations.find(batchId);
    if (_reservations.end() == it) {
        return false;
    }
    reservation = it->second;
    if (_reducedReservations.count(batchId)) {
        auto removeRetired = [batchId](auto &hashes, const auto &owners) {
            hashes.erase(std::remove_if(hashes.begin(), hashes.end(), [&](uint64_t hash) {
                auto it = owners.find(hash);
                return it == owners.end() || !it->second.contains(batchId);
            }), hashes.end());
        };
        removeRetired(reservation.fullReadyBlockHashes, _fullReadyOwners);
        removeRetired(reservation.tailReadyBlockHashes, _tailReadyOwners);
    }
    return true;
}

int64_t PendingCacheState::getSourceBatchId(uint64_t blockHash,
                                            bool tailReady) const
{
    const auto &owners = tailReady ? _tailReadyOwners : _fullReadyOwners;
    const auto it = owners.find(blockHash);
    return it == owners.end() ? 0 : it->second.first();
}

PendingCacheStateStats PendingCacheState::getStats() const
{
    return {_reservations.size(), _fullReadyOwners.size(),
            _tailReadyOwners.size(), _checksum};
}

size_t PendingCacheState::countBatchOverlap(const PendingCacheState &other) const
{
    const auto *smaller = &_reservations;
    const auto *larger = &other._reservations;
    if (smaller->size() > larger->size()) {
        std::swap(smaller, larger);
    }
    size_t count = 0;
    for (const auto &entry : *smaller) {
        count += larger->count(entry.first);
    }
    return count;
}

size_t PendingCacheState::countFullReadyOverlap(
    const PendingCacheState &other) const
{
    const auto *smaller = &_fullReadyOwners;
    const auto *larger = &other._fullReadyOwners;
    if (smaller->size() > larger->size()) {
        std::swap(smaller, larger);
    }
    size_t count = 0;
    for (const auto &entry : *smaller) {
        count += larger->count(entry.first);
    }
    return count;
}

size_t PendingCacheState::countTailReadyOverlap(
    const PendingCacheState &other) const
{
    const auto *smaller = &_tailReadyOwners;
    const auto *larger = &other._tailReadyOwners;
    if (smaller->size() > larger->size()) {
        std::swap(smaller, larger);
    }
    size_t count = 0;
    for (const auto &entry : *smaller) {
        count += larger->count(entry.first);
    }
    return count;
}

bool PendingCacheState::sameReservations(const PendingCacheState &other) const
{
    if (_reservations.size() != other._reservations.size()) {
        return false;
    }
    for (const auto &[batchId, reservation] : _reservations) {
        const auto it = other._reservations.find(batchId);
        if (it == other._reservations.end()) {
            return false;
        }
        if (_reducedReservations.count(batchId) || other._reducedReservations.count(batchId)) {
            NaviSchedPendingCacheReservation left, right;
            getReservation(batchId, left);
            other.getReservation(batchId, right);
            if (!PendingCacheUtil::equalReservation(left, right)) return false;
        } else if (!PendingCacheUtil::equalReservation(reservation, it->second)) {
            return false;
        }
    }
    return true;
}

std::vector<NaviSchedPendingCacheReservation>
PendingCacheState::makeSnapshot() const
{
    std::vector<NaviSchedPendingCacheReservation> reservations;
    reservations.reserve(_reservations.size());
    for (const auto &entry : _reservations) {
        NaviSchedPendingCacheReservation reservation;
        getReservation(entry.first, reservation);
        reservations.emplace_back(std::move(reservation));
    }
    std::sort(reservations.begin(), reservations.end(),
              [](const auto &left, const auto &right) {
                  return left.batchId < right.batchId;
              });
    return reservations;
}

void PendingCacheState::add(
    const NaviSchedPendingCacheReservation &reservation,
    const PendingCacheState *other, PendingCacheStateStats *overlap)
{
    _reservations.emplace(reservation.batchId, reservation);
    if (other && other->_reservations.count(reservation.batchId)) {
        ++overlap->batchCount;
    }
    _scheduleBatchIds[reservation.scheduleBatchId].insert(reservation.batchId);
    if (_fullReadyOwners.empty()) {
        _fullReadyOwners.reserve(reservation.fullReadyBlockHashes.size());
    }
    if (_tailReadyOwners.empty()) {
        _tailReadyOwners.reserve(reservation.tailReadyBlockHashes.size());
    }
    for (auto blockHash : reservation.fullReadyBlockHashes) {
        auto &owners = _fullReadyOwners[blockHash];
        owners.add(reservation.batchId);
        if (owners.size() == 1 && other &&
            other->containsFullReady(blockHash)) {
            ++overlap->uniqueBlockCount;
        }
    }
    for (auto blockHash : reservation.tailReadyBlockHashes) {
        auto &owners = _tailReadyOwners[blockHash];
        owners.add(reservation.batchId);
        if (owners.size() == 1 && other &&
            other->containsTailReady(blockHash)) {
            ++overlap->uniqueTailReadyBlockCount;
        }
    }
    const auto checksum = PendingCacheUtil::reservationChecksum(reservation);
    _reservationChecksums.emplace(reservation.batchId, checksum);
    _checksum ^= checksum;
}

void PendingCacheState::remove(
    std::unordered_map<int64_t, NaviSchedPendingCacheReservation>::iterator it,
    const PendingCacheState *other, PendingCacheStateStats *overlap)
{
    const bool reduced = _reducedReservations.count(it->first) != 0;
    for (auto blockHash : it->second.fullReadyBlockHashes) {
        auto ref = _fullReadyOwners.find(blockHash);
        if (reduced && (ref == _fullReadyOwners.end() || !ref->second.contains(it->first))) {
            continue;
        }
        ref->second.remove(it->first);
        if (ref->second.size() == 0) {
            if (other && other->containsFullReady(blockHash)) {
                --overlap->uniqueBlockCount;
            }
            _fullReadyOwners.erase(ref);
        }
    }
    for (auto blockHash : it->second.tailReadyBlockHashes) {
        auto ref = _tailReadyOwners.find(blockHash);
        if (reduced && (ref == _tailReadyOwners.end() || !ref->second.contains(it->first))) {
            continue;
        }
        ref->second.remove(it->first);
        if (ref->second.size() == 0) {
            if (other && other->containsTailReady(blockHash)) {
                --overlap->uniqueTailReadyBlockCount;
            }
            _tailReadyOwners.erase(ref);
        }
    }
    eraseMetadata(it, other, overlap);
}

void PendingCacheState::eraseMetadata(
    std::unordered_map<int64_t, NaviSchedPendingCacheReservation>::iterator it,
    const PendingCacheState *other, PendingCacheStateStats *overlap)
{
    if (other && other->_reservations.count(it->first)) {
        --overlap->batchCount;
    }
    auto batches = _scheduleBatchIds.find(it->second.scheduleBatchId);
    batches->second.erase(it->first);
    if (batches->second.empty()) {
        _scheduleBatchIds.erase(batches);
    }
    auto checksum = _reservationChecksums.find(it->first);
    _checksum ^= checksum->second;
    _reservationChecksums.erase(checksum);
    _reducedReservations.erase(it->first);
    _reservations.erase(it);
}

}
