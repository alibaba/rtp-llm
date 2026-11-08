#include "aios/navi_sched/collector/PPCacheCollectorState.h"
#include "unittest/unittest.h"
#include <algorithm>
#include <random>
#include <stdexcept>

using namespace navi_sched;
class PPCacheCollectorStateTest : public TESTBASE {};

namespace {
PPCacheShardUpdate update(uint32_t shard, uint64_t sequence, uint64_t generation = 1)
{
    PPCacheShardUpdate value;
    value.shardId = shard;
    value.sequence = sequence;
    value.streamGeneration = generation;
    return value;
}
void ready(PPCacheCollectorState &state, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        ASSERT_TRUE(state.connect(i, 1).needsFullSnapshot);
        auto snapshot = update(i, 1);
        snapshot.fullSnapshot = true;
        ASSERT_TRUE(state.apply(snapshot).accepted);
    }
}
PPCacheShardUpdate terminal(uint32_t shard, uint64_t sequence,
                            PPShardRequestStatus status, uint64_t attempt = 1,
                            uint64_t generation = 1)
{
    auto value = update(shard, sequence, generation);
    value.terminals.push_back({{10, attempt}, status});
    return value;
}
}

TEST_F(PPCacheCollectorStateTest, testAllShardsAreRequiredAndOneEvictionRetractsReady)
{
    PPCacheCollectorState state(4);
    ready(state, 4);
    for (uint32_t i = 0; i < 4; ++i) {
        auto value = update(i, 2);
        value.addedFull = {42, 42};
        value.addedTail = i == 3 ? std::vector<uint64_t>{} : std::vector<uint64_t>{42};
        const auto changes = state.apply(value);
        ASSERT_EQ(i == 3, state.fullReady(42));
        ASSERT_FALSE(state.tailReady(42));
        ASSERT_EQ(i == 3 ? 1 : 0, changes.addedFull.size());
    }
    auto removed = update(2, 3);
    removed.removedFull = {42, 42};
    const auto changes = state.apply(removed);
    ASSERT_EQ(std::vector<uint64_t>({42}), changes.removedFull);
    ASSERT_FALSE(state.fullReady(42));
}

TEST_F(PPCacheCollectorStateTest, testFailureAndCancellationFinishMergeInAnyOrder)
{
    PPCacheCollectorState state(4);
    ready(state, 4);
    ASSERT_TRUE(state.beginRequest({10, 1}).accepted);
    ASSERT_TRUE(state.apply(terminal(2, 2, PPShardRequestStatus::FAILED))
                    .completedRequests.empty());
    ASSERT_TRUE(state.apply(terminal(0, 2, PPShardRequestStatus::SUCCEEDED))
                    .completedRequests.empty());
    ASSERT_TRUE(state.apply(terminal(3, 2, PPShardRequestStatus::CANCELLED))
                    .completedRequests.empty());
    auto duplicate = terminal(2, 3, PPShardRequestStatus::SUCCEEDED);
    ASSERT_TRUE(state.apply(duplicate).completedRequests.empty());
    const auto completed = state.apply(terminal(1, 2, PPShardRequestStatus::SUCCEEDED));
    ASSERT_EQ(1, completed.completedRequests.size());
    ASSERT_FALSE(completed.completedRequests[0].succeeded);
    ASSERT_EQ(PPShardRequestStatus::FAILED,
              completed.completedRequests[0].shardStatuses[2]);
    ASSERT_EQ(PPShardRequestStatus::CANCELLED,
              completed.completedRequests[0].shardStatuses[3]);
    ASSERT_EQ(0, state.pendingRequestCount());
    ASSERT_TRUE(state.apply(terminal(1, 3, PPShardRequestStatus::SUCCEEDED))
                    .completedRequests.empty());
}

TEST_F(PPCacheCollectorStateTest, testSuccessfulRequestsCompleteExactlyOnce)
{
    PPCacheCollectorState state(2);
    ready(state, 2);
    state.beginRequest({10, 1});
    state.apply(terminal(0, 2, PPShardRequestStatus::SUCCEEDED));
    auto complete = state.apply(terminal(1, 2, PPShardRequestStatus::SUCCEEDED));
    ASSERT_EQ(1, complete.completedRequests.size());
    ASSERT_TRUE(complete.completedRequests[0].succeeded);
    ASSERT_TRUE(state.apply(terminal(1, 2, PPShardRequestStatus::SUCCEEDED))
                    .completedRequests.empty());
}

TEST_F(PPCacheCollectorStateTest, testDisconnectFinishesMissingTerminalAndRemovesCache)
{
    PPCacheCollectorState state(2);
    ready(state, 2);
    for (uint32_t i = 0; i < 2; ++i) {
        auto value = update(i, 2);
        value.addedFull = {9};
        state.apply(value);
    }
    state.beginRequest({10, 1});
    state.apply(terminal(0, 3, PPShardRequestStatus::SUCCEEDED));
    auto lost = state.disconnect(1, 1);
    ASSERT_EQ(1, lost.completedRequests.size());
    ASSERT_FALSE(lost.completedRequests[0].succeeded);
    ASSERT_EQ(PPShardRequestStatus::DISCONNECTED,
              lost.completedRequests[0].shardStatuses[1]);
    ASSERT_EQ(std::vector<uint64_t>({9}), lost.removedFull);
    ASSERT_FALSE(state.fullReady(9));
    ASSERT_EQ(0, state.pendingRequestCount());
}

TEST_F(PPCacheCollectorStateTest, testOldStreamAndOldAttemptCannotCompleteNewRequest)
{
    PPCacheCollectorState state(2);
    ready(state, 2);
    state.beginRequest({10, 1});
    state.apply(terminal(0, 2, PPShardRequestStatus::SUCCEEDED));
    ASSERT_EQ(1, state.connect(1, 2).completedRequests.size());
    auto snapshot = update(1, 1, 2);
    snapshot.fullSnapshot = true;
    state.apply(snapshot);
    state.beginRequest({10, 2});
    ASSERT_FALSE(
        state.apply(terminal(1, 2, PPShardRequestStatus::SUCCEEDED, 1, 1)).accepted);
    ASSERT_FALSE(state.disconnect(1, 1).accepted);
    state.apply(terminal(0, 3, PPShardRequestStatus::SUCCEEDED, 1));
    ASSERT_EQ(1, state.pendingRequestCount());
    state.apply(terminal(0, 4, PPShardRequestStatus::SUCCEEDED, 2));
    auto complete = state.apply(terminal(1, 2, PPShardRequestStatus::SUCCEEDED, 2, 2));
    ASSERT_EQ(1, complete.completedRequests.size());
    ASSERT_EQ(2, complete.completedRequests[0].request.attemptId);
}

TEST_F(PPCacheCollectorStateTest, testSequenceGapWithdrawsCacheAndRequestsSnapshot)
{
    PPCacheCollectorState state(2);
    ready(state, 2);
    for (uint32_t i = 0; i < 2; ++i) {
        auto value = update(i, 2);
        value.addedFull = {5};
        state.apply(value);
    }
    state.beginRequest({10, 1});
    auto gap = state.apply(terminal(1, 4, PPShardRequestStatus::SUCCEEDED));
    ASSERT_TRUE(gap.needsFullSnapshot);
    ASSERT_FALSE(gap.accepted);
    ASSERT_FALSE(state.fullReady(5));
    auto complete = state.apply(terminal(0, 3, PPShardRequestStatus::SUCCEEDED));
    ASSERT_EQ(1, complete.completedRequests.size());
    ASSERT_FALSE(complete.completedRequests[0].succeeded);
    auto snapshot = update(1, 9);
    snapshot.fullSnapshot = true;
    snapshot.addedFull = {5};
    ASSERT_TRUE(state.apply(snapshot).accepted);
    ASSERT_TRUE(state.fullReady(5));
    ASSERT_FALSE(state.apply(terminal(1, 8, PPShardRequestStatus::FAILED)).accepted);
}

TEST_F(PPCacheCollectorStateTest, testIdenticalSnapshotHasNoRedundantCacheChanges)
{
    PPCacheCollectorState state(1);
    ready(state, 1);
    auto snapshot = update(0, 2);
    snapshot.fullSnapshot = true;
    snapshot.addedFull = {1, 2};
    snapshot.addedTail = {2};
    state.apply(snapshot);
    snapshot.sequence = 3;
    auto changes = state.apply(snapshot);
    ASSERT_TRUE(changes.addedFull.empty());
    ASSERT_TRUE(changes.removedFull.empty());
    ASSERT_TRUE(changes.addedTail.empty());
    ASSERT_TRUE(changes.removedTail.empty());
}

TEST_F(PPCacheCollectorStateTest, testUnavailableCollectorsDoNotLeaveRequestWaiting)
{
    PPCacheCollectorState state(2);
    auto result = state.beginRequest({10, 1});
    ASSERT_TRUE(result.accepted);
    ASSERT_EQ(1, result.completedRequests.size());
    ASSERT_FALSE(result.completedRequests[0].succeeded);
    ASSERT_EQ(0, state.pendingRequestCount());
}

TEST_F(PPCacheCollectorStateTest, testRandomIncrementalChangesMatchFullIntersection)
{
    PPCacheCollectorState state(4);
    ready(state, 4);
    std::vector<std::unordered_set<uint64_t>> full(4), tail(4);
    std::vector<uint64_t> sequence(4, 1);
    std::mt19937 random(17);
    for (size_t step = 0; step < 5000; ++step) {
        const auto shard = random() % 4;
        const auto hash = random() % 32;
        const bool isTail = random() % 2;
        const bool adding = random() % 2;
        auto value = update(shard, ++sequence[shard]);
        auto &set = isTail ? tail[shard] : full[shard];
        if (adding) {
            set.insert(hash);
            (isTail ? value.addedTail : value.addedFull).push_back(hash);
        } else {
            set.erase(hash);
            (isTail ? value.removedTail : value.removedFull).push_back(hash);
        }
        state.apply(value);
        for (uint64_t key = 0; key < 32; ++key) {
            bool allFull = true, allTail = true;
            for (size_t node = 0; node < 4; ++node) {
                allFull &= full[node].count(key) != 0;
                allTail &= tail[node].count(key) != 0;
            }
            ASSERT_EQ(allFull, state.fullReady(key));
            ASSERT_EQ(allTail, state.tailReady(key));
        }
    }
}

TEST_F(PPCacheCollectorStateTest, testPeerReadinessRequiresEverySyncedShard)
{
    PPCacheCollectorState state(2);
    for (uint32_t shard = 0; shard < 2; ++shard) {
        state.connect(shard, 1);
        PPCacheShardUpdate snapshot;
        snapshot.shardId = shard;
        snapshot.streamGeneration = 1;
        snapshot.sequence = 1;
        snapshot.fullSnapshot = true;
        snapshot.peerCacheReadAvailable = 1;
        snapshot.addedFull = {8};
        auto changes = state.apply(snapshot);
        EXPECT_EQ(shard == 1 ? 1 : -1, changes.peerCacheReadAvailable);
    }
    PPCacheShardUpdate update;
    update.streamGeneration = 1;
    update.sequence = 2;
    update.peerCacheReadAvailable = 0;
    EXPECT_EQ(0, state.apply(update).peerCacheReadAvailable);
    EXPECT_TRUE(state.fullReady(8));
    update.sequence = 3;
    update.peerCacheReadAvailable = -1;
    EXPECT_EQ(-1, state.apply(update).peerCacheReadAvailable);
    update.sequence = 4;
    update.peerCacheReadAvailable = 1;
    EXPECT_EQ(1, state.apply(update).peerCacheReadAvailable);
    EXPECT_EQ(0, state.disconnect(1, 1).peerCacheReadAvailable);
    state.connect(1, 2);
    PPCacheShardUpdate snapshot;
    snapshot.shardId = 1;
    snapshot.streamGeneration = 2;
    snapshot.sequence = 1;
    snapshot.fullSnapshot = true;
    EXPECT_EQ(-1, state.apply(snapshot).peerCacheReadAvailable);
    snapshot.fullSnapshot = false;
    snapshot.sequence = 2;
    snapshot.peerCacheReadAvailable = 1;
    EXPECT_EQ(1, state.apply(snapshot).peerCacheReadAvailable);
    snapshot.sequence = 4;
    const auto gap = state.apply(snapshot);
    EXPECT_TRUE(gap.needsFullSnapshot);
    EXPECT_EQ(0, gap.peerCacheReadAvailable);
    snapshot.streamGeneration = 1;
    snapshot.sequence = 5;
    EXPECT_EQ(-1, state.apply(snapshot).peerCacheReadAvailable);
}
