#include "rtp_llm/flexlb/navi_sched/resource/RequestLatencyTracker.h"
#include "rtp_llm/flexlb/navi_sched/data/NaviSchedLatencyPrediction.h"

#include "unittest/unittest.h"

#include <atomic>
#include <cmath>
#include <limits>
#include <thread>

namespace navi_sched {
namespace {

class RequestLatencyTrackerTest : public TESTBASE {};

TEST_F(RequestLatencyTrackerTest, testSnapshotValidationRejectsEveryInvalidInvariant)
{
    RequestLatencyTracker::Snapshot snapshot;
    EXPECT_TRUE(snapshot.valid());

    snapshot.revision = 0;
    EXPECT_FALSE(snapshot.valid());
    snapshot.revision = 1;
    snapshot.requestCount = -1;
    EXPECT_FALSE(snapshot.valid());
    snapshot.requestCount = 1;
    snapshot.totalPredictedLatencyMs = -1.0;
    EXPECT_FALSE(snapshot.valid());
    snapshot.totalPredictedLatencyMs = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(snapshot.valid());
    snapshot.totalPredictedLatencyMs = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(snapshot.valid());
    snapshot.requestCount = 0;
    snapshot.totalPredictedLatencyMs = 1.0;
    EXPECT_FALSE(snapshot.valid());
    snapshot.requestCount = 1;
    snapshot.totalPredictedLatencyMs = 0.0;
    EXPECT_TRUE(snapshot.valid());
}

TEST_F(RequestLatencyTrackerTest, testAddAndRemoveUseStoredLatency)
{
    RequestLatencyTracker tracker;
    ASSERT_TRUE(tracker.add(1, 10.5));
    ASSERT_TRUE(tracker.add(2, 20.25));
    auto snapshot = tracker.snapshot();
    EXPECT_EQ(2, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(30.75, snapshot.totalPredictedLatencyMs);

    EXPECT_FALSE(tracker.add(1, 999.0));
    double storedLatencyMs = 0.0;
    ASSERT_TRUE(tracker.get(1, storedLatencyMs));
    EXPECT_DOUBLE_EQ(10.5, storedLatencyMs);
    ASSERT_TRUE(tracker.remove(1, &storedLatencyMs));
    EXPECT_DOUBLE_EQ(10.5, storedLatencyMs);
    snapshot = tracker.snapshot();
    EXPECT_EQ(1, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(20.25, snapshot.totalPredictedLatencyMs);
    EXPECT_EQ(1, snapshot.duplicateAddCount);

    const auto revision = snapshot.revision;
    EXPECT_FALSE(tracker.remove(1));
    snapshot = tracker.snapshot();
    EXPECT_EQ(revision, snapshot.revision);
    EXPECT_EQ(1, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(20.25, snapshot.totalPredictedLatencyMs);
    EXPECT_EQ(1, snapshot.missingRemoveCount);
}

TEST_F(RequestLatencyTrackerTest, testRejectsInvalidLatencyWithoutMutation)
{
    RequestLatencyTracker tracker;
    EXPECT_FALSE(tracker.add(0, 1.0));
    EXPECT_FALSE(tracker.add(1, -1.0));
    EXPECT_FALSE(tracker.add(2, std::numeric_limits<double>::infinity()));
    EXPECT_FALSE(tracker.add(3, std::numeric_limits<double>::quiet_NaN()));
    ASSERT_TRUE(tracker.add(4, std::numeric_limits<double>::max()));
    EXPECT_FALSE(tracker.add(5, std::numeric_limits<double>::max()));
    const auto snapshot = tracker.snapshot();
    EXPECT_TRUE(snapshot.valid());
    EXPECT_EQ(1, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(std::numeric_limits<double>::max(),
                     snapshot.totalPredictedLatencyMs);
    EXPECT_EQ(5, snapshot.invalidAddCount);
}

TEST_F(RequestLatencyTrackerTest, testConcurrentSnapshotsAreConsistent)
{
    ConcurrentRequestLatencyTracker tracker;
    constexpr size_t THREAD_COUNT = 8;
    constexpr size_t REQUESTS_PER_THREAD = 500;
    std::vector<std::thread> threads;
    std::atomic<bool> failed{false};
    for (size_t threadIndex = 0; threadIndex < THREAD_COUNT; ++threadIndex) {
        threads.emplace_back([&, threadIndex]() {
            for (size_t requestIndex = 0; requestIndex < REQUESTS_PER_THREAD;
                 ++requestIndex) {
                RequestLatencyTracker::Snapshot snapshot;
                const auto requestId =
                    threadIndex * REQUESTS_PER_THREAD + requestIndex + 1;
                if (!tracker.add(requestId, 2.5, snapshot) || !snapshot.valid()) {
                    failed.store(true, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    EXPECT_FALSE(failed.load(std::memory_order_relaxed));
    auto snapshot = tracker.snapshot();
    EXPECT_EQ(THREAD_COUNT * REQUESTS_PER_THREAD, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(THREAD_COUNT * REQUESTS_PER_THREAD * 2.5,
                     snapshot.totalPredictedLatencyMs);

    std::vector<uint64_t> requestIds;
    requestIds.reserve(THREAD_COUNT * REQUESTS_PER_THREAD);
    for (size_t index = 0; index < THREAD_COUNT * REQUESTS_PER_THREAD; ++index) {
        requestIds.emplace_back(index + 1);
    }
    std::vector<uint64_t> missing;
    EXPECT_EQ(requestIds.size(), tracker.remove(requestIds, snapshot, &missing));
    EXPECT_TRUE(missing.empty());
    EXPECT_EQ(0, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(0.0, snapshot.totalPredictedLatencyMs);
    EXPECT_TRUE(snapshot.valid());
}

TEST_F(RequestLatencyTrackerTest, testConcurrentDuplicateHasSingleOwner)
{
    ConcurrentRequestLatencyTracker tracker;
    constexpr size_t THREAD_COUNT = 16;
    std::vector<std::thread> threads;
    std::atomic<size_t> addedCount{0};
    for (size_t index = 0; index < THREAD_COUNT; ++index) {
        threads.emplace_back([&]() {
            RequestLatencyTracker::Snapshot snapshot;
            if (tracker.add(101, 7.5, snapshot)) {
                addedCount.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }

    auto snapshot = tracker.snapshot();
    EXPECT_EQ(1, addedCount.load(std::memory_order_relaxed));
    EXPECT_EQ(1, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(7.5, snapshot.totalPredictedLatencyMs);
    EXPECT_EQ(THREAD_COUNT - 1, snapshot.duplicateAddCount);
    const auto ownedRevision = snapshot.revision;

    ASSERT_TRUE(tracker.remove(101, snapshot));
    EXPECT_EQ(ownedRevision + 1, snapshot.revision);
    EXPECT_EQ(0, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(0.0, snapshot.totalPredictedLatencyMs);
    EXPECT_FALSE(tracker.remove(101, snapshot));
    EXPECT_EQ(ownedRevision + 1, snapshot.revision);
    EXPECT_EQ(1, snapshot.missingRemoveCount);
}

TEST_F(RequestLatencyTrackerTest, testBatchRemoveReportsMissingRequests)
{
    ConcurrentRequestLatencyTracker tracker;
    RequestLatencyTracker::Snapshot snapshot;
    ASSERT_TRUE(tracker.add(10, 1.0, snapshot));
    ASSERT_TRUE(tracker.add(20, 2.0, snapshot));
    std::vector<uint64_t> missing;
    EXPECT_EQ(2, tracker.remove({10, 30, 20}, snapshot, &missing));
    ASSERT_EQ(1, missing.size());
    EXPECT_EQ(30, missing.front());
    EXPECT_EQ(0, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(0.0, snapshot.totalPredictedLatencyMs);
    EXPECT_EQ(1, snapshot.missingRemoveCount);
}

TEST_F(RequestLatencyTrackerTest, testBestEffortRemoveDoesNotReportAlreadyReleasedOwners)
{
    ConcurrentRequestLatencyTracker tracker;
    RequestLatencyTracker::Snapshot snapshot;
    ASSERT_TRUE(tracker.add(10, 1.0, snapshot));
    ASSERT_TRUE(tracker.add(20, 2.0, snapshot));
    ASSERT_TRUE(tracker.remove(10, snapshot));
    const auto revision = snapshot.revision;

    EXPECT_EQ(1, tracker.removePresent({10, 20, 30}, snapshot));
    EXPECT_EQ(revision + 1, snapshot.revision);
    EXPECT_EQ(0, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(0.0, snapshot.totalPredictedLatencyMs);
    EXPECT_EQ(0, snapshot.missingRemoveCount);
}

TEST_F(RequestLatencyTrackerTest, testClearReleasesAllOwnersExactlyOnce)
{
    RequestLatencyTracker tracker;
    const auto initial = tracker.snapshot();
    tracker.clear();
    EXPECT_EQ(initial.revision, tracker.snapshot().revision);

    ASSERT_TRUE(tracker.add(10, 1.5));
    ASSERT_TRUE(tracker.add(20, 2.5));
    EXPECT_FALSE(tracker.add(10, 9.0));
    EXPECT_FALSE(tracker.remove(30));
    const auto owned = tracker.snapshot();
    ASSERT_EQ(2, owned.requestCount);
    ASSERT_DOUBLE_EQ(4.0, owned.totalPredictedLatencyMs);

    tracker.clear();
    const auto cleared = tracker.snapshot();
    EXPECT_EQ(owned.revision + 1, cleared.revision);
    EXPECT_EQ(0, cleared.requestCount);
    EXPECT_DOUBLE_EQ(0.0, cleared.totalPredictedLatencyMs);
    EXPECT_EQ(owned.invalidAddCount, cleared.invalidAddCount);
    EXPECT_EQ(owned.duplicateAddCount, cleared.duplicateAddCount);
    EXPECT_EQ(owned.missingRemoveCount, cleared.missingRemoveCount);

    tracker.clear();
    const auto duplicateClear = tracker.snapshot();
    EXPECT_EQ(cleared.revision, duplicateClear.revision);
    EXPECT_EQ(cleared.duplicateAddCount, duplicateClear.duplicateAddCount);
    EXPECT_EQ(cleared.missingRemoveCount, duplicateClear.missingRemoveCount);
}

TEST_F(RequestLatencyTrackerTest, testRecomputeUpdatesWaitingButKeepsRunningAtomic)
{
    ConcurrentRequestLatencyTracker tracker;
    auto firstPrediction = std::make_shared<NaviSchedLatencyPrediction>();
    firstPrediction->features.appendRequest(10, 20);
    firstPrediction->fixedLatencyMs = 3.0;
    firstPrediction->recomputable = true;
    auto secondPrediction = std::make_shared<NaviSchedLatencyPrediction>();
    secondPrediction->features.appendRequest(20, 40);
    secondPrediction->fixedLatencyMs = 5.0;
    secondPrediction->recomputable = true;

    RequestLatencyTracker::Snapshot snapshot;
    ASSERT_TRUE(tracker.add(1, 100.0, firstPrediction, snapshot));
    ASSERT_TRUE(tracker.add(2, 200.0, secondPrediction, snapshot));
    ASSERT_TRUE(tracker.add(3, 300.0, snapshot));
    ASSERT_TRUE(tracker.markRunning({2}, snapshot));
    const auto revisionBefore = snapshot.revision;

    size_t recomputedRequestCount = 0;
    ASSERT_TRUE(tracker.recomputeWaiting(
        [](const NaviSchedLatencyFeatures &) { return 40.0; },
        snapshot, recomputedRequestCount));
    EXPECT_EQ(1, recomputedRequestCount);
    EXPECT_EQ(revisionBefore + 1, snapshot.revision);
    EXPECT_EQ(3, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(43.0 + 200.0 + 300.0,
                     snapshot.totalPredictedLatencyMs);
    EXPECT_EQ(1, snapshot.recomputeCount);
    EXPECT_EQ(1, snapshot.recomputedRequestCount);
    EXPECT_EQ(0, snapshot.recomputeFailureCount);

    ASSERT_TRUE(tracker.remove(1, snapshot));
    EXPECT_EQ(2, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(500.0, snapshot.totalPredictedLatencyMs);
    ASSERT_TRUE(tracker.remove(2, snapshot));
    ASSERT_TRUE(tracker.remove(3, snapshot));
    EXPECT_EQ(0, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(0.0, snapshot.totalPredictedLatencyMs);
}

TEST_F(RequestLatencyTrackerTest, testRecomputeFailureDoesNotPartiallyCommit)
{
    ConcurrentRequestLatencyTracker tracker;
    auto prediction = std::make_shared<NaviSchedLatencyPrediction>();
    prediction->features.appendRequest(0, 10);
    prediction->recomputable = true;
    RequestLatencyTracker::Snapshot snapshot;
    ASSERT_TRUE(tracker.add(1, 10.0, prediction, snapshot));
    ASSERT_TRUE(tracker.add(2, 20.0, prediction, snapshot));
    const auto before = snapshot;

    size_t invocationCount = 0;
    size_t recomputedRequestCount = 0;
    EXPECT_FALSE(tracker.recomputeWaiting(
        [&invocationCount](const NaviSchedLatencyFeatures &) {
            ++invocationCount;
            return invocationCount == 1
                       ? 99.0
                       : std::numeric_limits<double>::quiet_NaN();
        },
        snapshot, recomputedRequestCount));
    EXPECT_EQ(0, recomputedRequestCount);
    EXPECT_EQ(before.revision, snapshot.revision);
    EXPECT_EQ(before.requestCount, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(before.totalPredictedLatencyMs,
                     snapshot.totalPredictedLatencyMs);
    EXPECT_EQ(before.recomputeFailureCount + 1,
              snapshot.recomputeFailureCount);

    double latencyMs = 0.0;
    RequestLatencyTracker plain;
    ASSERT_TRUE(plain.add(1, 10.0, prediction));
    ASSERT_TRUE(plain.add(2, 20.0, prediction));
    size_t plainRecomputed = 0;
    EXPECT_FALSE(plain.recomputeWaiting(
        [&invocationCount](const NaviSchedLatencyFeatures &) {
            return ++invocationCount > 3
                       ? std::numeric_limits<double>::infinity()
                       : 100.0;
        },
        plainRecomputed));
    ASSERT_TRUE(plain.get(1, latencyMs));
    EXPECT_DOUBLE_EQ(10.0, latencyMs);
    ASSERT_TRUE(plain.get(2, latencyMs));
    EXPECT_DOUBLE_EQ(20.0, latencyMs);
}

TEST_F(RequestLatencyTrackerTest, testRecomputeReevaluatesUnifiedFloorBranch)
{
    RequestLatencyTracker tracker;
    auto prediction = std::make_shared<NaviSchedLatencyPrediction>();
    prediction->features.appendRequest(0, 100);
    prediction->floorFeatures.appendRequest(0, 10);
    prediction->fixedLatencyMs = -50.0;
    prediction->hasFloor = true;
    prediction->recomputable = true;
    ASSERT_TRUE(tracker.add(1, 30.0, prediction));

    size_t recomputedRequestCount = 0;
    size_t predictionIndex = 0;
    ASSERT_TRUE(tracker.recomputeWaiting(
        [&predictionIndex](const NaviSchedLatencyFeatures &) {
            return predictionIndex++ == 0 ? 100.0 : 10.0;
        },
        recomputedRequestCount));
    EXPECT_EQ(1, recomputedRequestCount);
    double latencyMs = 0.0;
    ASSERT_TRUE(tracker.get(1, latencyMs));
    EXPECT_DOUBLE_EQ(50.0, latencyMs);

    predictionIndex = 0;
    ASSERT_TRUE(tracker.recomputeWaiting(
        [&predictionIndex](const NaviSchedLatencyFeatures &) {
            return predictionIndex++ == 0 ? 20.0 : 40.0;
        },
        recomputedRequestCount));
    ASSERT_TRUE(tracker.get(1, latencyMs));
    EXPECT_DOUBLE_EQ(40.0, latencyMs);
}

TEST_F(RequestLatencyTrackerTest, testFinishedRequestIsRemovedBeforeRecompute)
{
    RequestLatencyTracker tracker;
    auto prediction = std::make_shared<NaviSchedLatencyPrediction>();
    prediction->features.appendRequest(0, 10);
    prediction->recomputable = true;
    ASSERT_TRUE(tracker.add(1, 10.0, prediction));
    ASSERT_TRUE(tracker.add(2, 20.0, prediction));
    ASSERT_TRUE(tracker.remove(1));

    size_t invocationCount = 0;
    size_t recomputedRequestCount = 0;
    ASSERT_TRUE(tracker.recomputeWaiting(
        [&invocationCount](const NaviSchedLatencyFeatures &) {
            ++invocationCount;
            return 30.0;
        },
        recomputedRequestCount));
    EXPECT_EQ(1, invocationCount);
    EXPECT_EQ(1, recomputedRequestCount);
    const auto snapshot = tracker.snapshot();
    EXPECT_EQ(1, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(30.0, snapshot.totalPredictedLatencyMs);
    EXPECT_FALSE(tracker.contains(1));
    EXPECT_TRUE(tracker.contains(2));
}

TEST_F(RequestLatencyTrackerTest, testMarkRunningValidatesWholeTransitionBeforeMutation)
{
    RequestLatencyTracker tracker;
    ASSERT_TRUE(tracker.add(1, 10.0));
    ASSERT_TRUE(tracker.add(2, 20.0));
    EXPECT_FALSE(tracker.markRunning({1, 3}));
    ASSERT_TRUE(tracker.markRunning({1, 2}));
    EXPECT_FALSE(tracker.markRunning({1}));
    const auto snapshot = tracker.snapshot();
    EXPECT_EQ(2, snapshot.invalidStateTransitionCount);
    EXPECT_EQ(2, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(30.0, snapshot.totalPredictedLatencyMs);
}

TEST_F(RequestLatencyTrackerTest, testRequestCountIsCappedAtFiveThousand)
{
    RequestLatencyTracker tracker;
    for (size_t index = 0; index < RequestLatencyTracker::MAX_REQUEST_COUNT; ++index) {
        ASSERT_TRUE(tracker.add(index + 1, 1.0));
    }
    EXPECT_FALSE(tracker.add(RequestLatencyTracker::MAX_REQUEST_COUNT + 1, 1.0));
    const auto snapshot = tracker.snapshot();
    EXPECT_EQ(RequestLatencyTracker::MAX_REQUEST_COUNT, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(RequestLatencyTracker::MAX_REQUEST_COUNT,
                     snapshot.totalPredictedLatencyMs);
    EXPECT_EQ(1, snapshot.invalidAddCount);
}

} // namespace
} // namespace navi_sched
