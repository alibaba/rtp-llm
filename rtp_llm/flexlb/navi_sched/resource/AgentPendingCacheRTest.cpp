#include "rtp_llm/flexlb/navi_sched/resource/AgentPendingCacheR.h"

#include "unittest/unittest.h"

#include <atomic>
#include <thread>

namespace navi_sched {
namespace {

NaviSchedPendingCacheReservation makeReservation(
    int64_t batchId, int64_t scheduleBatchId,
    std::initializer_list<uint64_t> fullReady,
    std::initializer_list<uint64_t> tailReady = {})
{
    NaviSchedPendingCacheReservation reservation;
    reservation.batchId = batchId;
    reservation.scheduleBatchId = scheduleBatchId;
    reservation.fullReadyBlockHashes.assign(fullReady);
    reservation.tailReadyBlockHashes.assign(tailReady);
    return reservation;
}

class AgentPendingCacheRTest : public TESTBASE {};

TEST_F(AgentPendingCacheRTest, testReplaceAndRetirePreserveSharedHashes)
{
    AgentPendingCacheR cache;
    AgentPendingCacheToMasterData update;
    bool changed = false;
    ASSERT_TRUE(cache.reserve(makeReservation(7, 1, {11, 22}, {22}), update, changed));
    ASSERT_TRUE(changed);
    const auto other = makeReservation(8, 1, {11, 33}, {33});
    ASSERT_TRUE(cache.reserve(other, update, changed));
    ASSERT_TRUE(changed);
    const auto replacement = makeReservation(7, 1, {22, 44}, {44});
    update = {};
    ASSERT_TRUE(cache.replace(replacement, update, changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(3, update.revision);
    EXPECT_EQ(2, update.batchCount);
    EXPECT_EQ(4, update.uniqueBlockCount);
    EXPECT_EQ(2, update.uniqueTailReadyBlockCount);
    update = {};
    ASSERT_TRUE(cache.replace(replacement, update, changed));
    EXPECT_FALSE(changed);
    EXPECT_TRUE(update.addedReservations.empty());
    EXPECT_EQ(3, cache.getState().revision);

    ASSERT_TRUE(cache.retire(7, update, changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(4, update.revision);
    EXPECT_EQ(1, update.batchCount);
    EXPECT_EQ(2, update.uniqueBlockCount);
    EXPECT_EQ(1, update.uniqueTailReadyBlockCount);
    EXPECT_EQ(PendingCacheUtil::reservationChecksum(other), update.checksum);
    ASSERT_TRUE(cache.retire(7, update, changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(4, cache.getState().revision);
    cache.clear();
    const auto state = cache.getState();
    EXPECT_EQ(0, state.revision);
    EXPECT_EQ(0, state.batchCount);
    EXPECT_EQ(0, state.uniqueBlockCount);
    EXPECT_EQ(0, state.uniqueTailReadyBlockCount);
    EXPECT_EQ(0, state.checksum);
    EXPECT_TRUE(cache.makeSnapshot().addedReservations.empty());
}

TEST_F(AgentPendingCacheRTest, testRetainIgnoresMissingAndRetiredBatches)
{
    AgentPendingCacheR cache;
    AgentPendingCacheToMasterData update;
    bool changed = true;
    const auto reservation = makeReservation(7, 70, {11, 22}, {22});
    ASSERT_TRUE(cache.retain(reservation, update, changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(0, cache.getState().revision);
    ASSERT_TRUE(cache.reserve(reservation, update, changed));
    ASSERT_TRUE(cache.retire(7, update, changed));
    update = {};
    ASSERT_TRUE(cache.retain(reservation, update, changed));
    EXPECT_FALSE(changed);
    EXPECT_TRUE(update.addedReservations.empty());
    EXPECT_TRUE(update.removedBatchIds.empty());
    EXPECT_EQ(2, cache.getState().revision);
    EXPECT_EQ(0, cache.getState().batchCount);
    EXPECT_EQ(0, cache.getState().uniqueBlockCount);
    EXPECT_EQ(0, cache.getState().checksum);
}

TEST_F(AgentPendingCacheRTest, testRetainPreservesSharedOwnersAndNeverAddsHashes)
{
    AgentPendingCacheR cache;
    AgentPendingCacheToMasterData update;
    bool changed = false;
    ASSERT_TRUE(cache.reserve(makeReservation(7, 70, {11, 22}, {22}), update, changed));
    const auto other = makeReservation(8, 70, {22, 33}, {33});
    ASSERT_TRUE(cache.reserve(other, update, changed));
    update = {};
    ASSERT_TRUE(cache.retain(makeReservation(7, 70, {22, 44}, {22, 44}), update, changed));
    ASSERT_TRUE(changed);
    ASSERT_EQ(1u, update.addedReservations.size());
    EXPECT_EQ(NaviSchedBlockHashes({22}), update.addedReservations[0].fullReadyBlockHashes);
    EXPECT_EQ(NaviSchedBlockHashes({22}), update.addedReservations[0].tailReadyBlockHashes);
    EXPECT_EQ(3, update.revision);
    EXPECT_EQ(2, update.batchCount);
    EXPECT_EQ(2, update.uniqueBlockCount);
    EXPECT_EQ(2, update.uniqueTailReadyBlockCount);
    update = {};
    ASSERT_TRUE(cache.retire(7, update, changed));
    EXPECT_EQ(4, update.revision);
    EXPECT_EQ(1, update.batchCount);
    EXPECT_EQ(2, update.uniqueBlockCount);
    EXPECT_EQ(1, update.uniqueTailReadyBlockCount);
    EXPECT_EQ(PendingCacheUtil::reservationChecksum(other), update.checksum);
}

TEST_F(AgentPendingCacheRTest, testRetainRejectsScheduleMismatchAndRetiresEmptyIntersection)
{
    AgentPendingCacheR cache;
    AgentPendingCacheToMasterData update;
    bool changed = false;
    const auto reservation = makeReservation(7, 70, {11}, {11});
    ASSERT_TRUE(cache.reserve(reservation, update, changed));
    update = {};
    EXPECT_FALSE(cache.retain(makeReservation(7, 71, {11}), update, changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(1, cache.getState().revision);
    EXPECT_EQ(PendingCacheUtil::reservationChecksum(reservation), cache.getState().checksum);
    ASSERT_TRUE(cache.retain(makeReservation(7, 70, {22}, {22}), update, changed));
    ASSERT_TRUE(changed);
    EXPECT_TRUE(update.addedReservations.empty());
    EXPECT_EQ(std::vector<int64_t>({7}), update.removedBatchIds);
    EXPECT_EQ(2, update.revision);
    EXPECT_EQ(0, update.batchCount);
    EXPECT_EQ(0, update.uniqueBlockCount);
    EXPECT_EQ(0, update.uniqueTailReadyBlockCount);
    EXPECT_EQ(0, update.checksum);
}

TEST_F(AgentPendingCacheRTest, testConcurrentRetainAndRetireNeverResurrectsPending)
{
    AgentPendingCacheR cache;
    AgentPendingCacheToMasterData initial;
    bool changed = false;
    ASSERT_TRUE(cache.reserve(makeReservation(7, 70, {11, 22, 33}, {22, 33}), initial, changed));
    std::atomic<bool> begin{false};
    std::vector<std::thread> threads;
    for (size_t index = 0; index < 16; ++index) {
        threads.emplace_back([&, index]() {
            while (!begin.load(std::memory_order_acquire)) std::this_thread::yield();
            AgentPendingCacheToMasterData update;
            bool localChanged = false;
            if (index == 0) {
                EXPECT_TRUE(cache.retire(7, update, localChanged));
            } else {
                EXPECT_TRUE(cache.retain(
                    makeReservation(7, 70, {22, 33}, {33}), update, localChanged));
            }
        });
    }
    begin.store(true, std::memory_order_release);
    for (auto &thread : threads) thread.join();
    const auto state = cache.getState();
    EXPECT_TRUE(state.revision == 2 || state.revision == 3);
    EXPECT_EQ(0, state.batchCount);
    EXPECT_EQ(0, state.uniqueBlockCount);
    EXPECT_EQ(0, state.uniqueTailReadyBlockCount);
    EXPECT_EQ(0, state.checksum);
    EXPECT_TRUE(cache.makeSnapshot().addedReservations.empty());
}

TEST_F(AgentPendingCacheRTest, testSnapshotAndDeltasExposeExactState)
{
    AgentPendingCacheR cache;
    const auto second = makeReservation(20, 8, {22, 33}, {33});
    const auto first = makeReservation(10, 8, {11, 22}, {22});
    AgentPendingCacheToMasterData update;
    bool changed = false;
    ASSERT_TRUE(cache.reserve(second, update, changed));
    ASSERT_TRUE(changed);
    EXPECT_EQ(1, update.revision);
    EXPECT_EQ(1, update.batchCount);
    EXPECT_EQ(2, update.uniqueBlockCount);
    EXPECT_EQ(1, update.uniqueTailReadyBlockCount);

    update = {};
    ASSERT_TRUE(cache.reserve(first, update, changed));
    ASSERT_TRUE(changed);
    EXPECT_EQ(2, update.revision);
    EXPECT_EQ(2, update.batchCount);
    EXPECT_EQ(3, update.uniqueBlockCount);
    EXPECT_EQ(2, update.uniqueTailReadyBlockCount);
    EXPECT_EQ(PendingCacheUtil::reservationChecksum(first) ^
                  PendingCacheUtil::reservationChecksum(second),
              update.checksum);

    const auto snapshot = cache.makeSnapshot();
    ASSERT_TRUE(snapshot.fullSnapshot);
    ASSERT_EQ(2, snapshot.addedReservations.size());
    EXPECT_EQ(10, snapshot.addedReservations[0].batchId);
    EXPECT_EQ(20, snapshot.addedReservations[1].batchId);
    EXPECT_EQ(update.revision, snapshot.revision);
    EXPECT_EQ(update.batchCount, snapshot.batchCount);
    EXPECT_EQ(update.uniqueBlockCount, snapshot.uniqueBlockCount);
    EXPECT_EQ(update.uniqueTailReadyBlockCount,
              snapshot.uniqueTailReadyBlockCount);
    EXPECT_EQ(update.checksum, snapshot.checksum);

    AgentPendingCacheToMasterData duplicate;
    ASSERT_TRUE(cache.reserve(first, duplicate, changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(2, cache.getState().revision);
    EXPECT_TRUE(duplicate.addedReservations.empty());

    AgentPendingCacheToMasterData retired;
    ASSERT_TRUE(cache.retire(10, retired, changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(3, retired.revision);
    EXPECT_EQ(1, retired.batchCount);
    EXPECT_EQ(2, retired.uniqueBlockCount);
    EXPECT_EQ(1, retired.uniqueTailReadyBlockCount);
}

TEST_F(AgentPendingCacheRTest,
       testPendingAwareMatchUnionsPhysicalAndPendingWithoutDoubleCounting)
{
    AgentPendingCacheR cache;
    AgentPendingCacheToMasterData update;
    bool changed = false;
    ASSERT_TRUE(cache.reserve(makeReservation(10, 8, {11, 22, 33}),
                              update, changed));
    ASSERT_TRUE(changed);

    const std::unordered_set<uint64_t> physical{11};
    const std::unordered_set<uint64_t> empty;
    EXPECT_EQ(3, cache.getMatchedBlockCount(
                     {11, 22, 33, 44}, physical, empty, empty, false));
    EXPECT_EQ(2, cache.getMatchedBlockCount(
                     {11, 22, 33, 44}, physical, empty, empty, false, 2));

    AgentPendingCacheToMasterData retired;
    ASSERT_TRUE(cache.retire(10, retired, changed));
    ASSERT_TRUE(changed);
    EXPECT_EQ(1, cache.getMatchedBlockCount(
                     {11, 22, 33, 44}, physical, empty, empty, false));
}

TEST_F(AgentPendingCacheRTest, testPendingAwareHybridMatchRollsBackToReadyTail)
{
    AgentPendingCacheR cache;
    AgentPendingCacheToMasterData update;
    bool changed = false;
    ASSERT_TRUE(cache.reserve(makeReservation(10, 8, {11, 22, 33}, {22}),
                              update, changed));

    const std::unordered_set<uint64_t> empty;
    const std::unordered_set<uint64_t> physicalFull{11};
    const std::unordered_set<uint64_t> physicalTail{11};
    EXPECT_EQ(2, cache.getMatchedBlockCount(
                     {11, 22, 33, 44}, empty, physicalFull, physicalTail, true));

    ASSERT_TRUE(cache.replace(makeReservation(10, 8, {11, 22, 33}),
                              update, changed));
    ASSERT_TRUE(changed);
    EXPECT_EQ(1, cache.getMatchedBlockCount(
                     {11, 22, 33, 44}, empty, physicalFull, physicalTail, true));
}

} // namespace
} // namespace navi_sched
