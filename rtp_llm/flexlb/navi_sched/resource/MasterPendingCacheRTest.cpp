#include "rtp_llm/flexlb/navi_sched/resource/MasterPendingCacheR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterBatchReservation.h"
#include "rtp_llm/flexlb/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "rtp_llm/flexlb/navi_sched/data/BlockHashUtil.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/ScheduleContext.h"
#include "autil/legacy/jsonizable.h"

#include "unittest/unittest.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <malloc.h>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <time.h>

namespace navi_sched {

class MasterPendingCacheRTest : public TESTBASE {};

bool reserveFlat(MasterPendingCacheR &cache, uint64_t agentIdHash,
                 int64_t batchId, int64_t scheduleTimeUs,
                 const NaviSchedBlockHashes &blockHashes)
{
    MasterPendingCacheR::ReferencedRequest request{
        &blockHashes, static_cast<int64_t>(blockHashes.size()), 1};
    return cache.reserve(agentIdHash, batchId, scheduleTimeUs, {request},
                         NaviSchedCachePolicy{1, 1, false});
}

AgentPendingCacheToMasterData makePendingSnapshot(
    uint64_t revision,
    const std::vector<NaviSchedPendingCacheReservation> &reservations)
{
    AgentPendingCacheToMasterData update;
    update.fullSnapshot = true;
    update.revision = revision;
    update.addedReservations = reservations;
    PendingCacheState state;
    if (!state.replace(reservations)) {
        return {};
    }
    const auto stats = state.getStats();
    update.batchCount = stats.batchCount;
    update.uniqueBlockCount = stats.uniqueBlockCount;
    update.uniqueTailReadyBlockCount = stats.uniqueTailReadyBlockCount;
    update.checksum = stats.checksum;
    return update;
}

TEST_F(MasterPendingCacheRTest, testMasterRemovalAndAgentUpdatesPreserveIndependentOwners)
{
    for (const bool agentFirst : {false, true}) {
        SCOPED_TRACE(agentFirst);
        MasterPendingCacheR cache;
        MasterBatchReservation batch;
        const auto request = [](uint64_t id, NaviSchedBlockHashes blocks) {
            const auto count = blocks.size();
            return MasterBatchReservation::Request{
                id, static_cast<int64_t>(count), 1, 0,
                std::make_shared<const NaviSchedBlockHashes>(std::move(blocks)), 1.0};
        };
        const NaviSchedCachePolicy policy{1, 3, true};
        ASSERT_TRUE(batch.init(101, 1, 4, {request(1, {11, 22, 33}),
                                         request(2, {11, 44, 33}),
                                         request(3, {11, 22, 33}), request(4, {99})}, &policy));
        ASSERT_TRUE(cache.updateScheduled(42, batch.getPendingCacheReservation()));
        NaviSchedPendingCacheReservation independent{102, 102, {11, 77}, {11, 77}};
        ASSERT_TRUE(cache.updateScheduled(42, independent));
        NaviSchedPendingCacheReservation agent{201, 101, {11, 22, 33}, {22, 33}};
        MasterBatchReservation::Transition transition;
        ASSERT_TRUE(batch.markFailed(1, transition));
        EXPECT_FALSE(transition.cacheChanged);
        EXPECT_TRUE(transition.removedFullReadyBlockHashes.empty());
        EXPECT_TRUE(transition.removedTailReadyBlockHashes.empty());
        if (agentFirst) {
            ASSERT_TRUE(cache.applyAgentUpdate(42, makePendingSnapshot(7, {agent})).applied);
        }
        const auto scans = batch._pendingRebuildVisitedNodes;
        ASSERT_TRUE(batch.markFailed(3, transition));
        EXPECT_EQ(scans, batch._pendingRebuildVisitedNodes);
        EXPECT_EQ((std::vector<uint64_t>{22}), transition.removedFullReadyBlockHashes);
        EXPECT_EQ((std::vector<uint64_t>{22}), transition.removedTailReadyBlockHashes);
        ASSERT_TRUE(cache.removeScheduledBlocks(42, 101,
            transition.removedFullReadyBlockHashes, transition.removedTailReadyBlockHashes));
        if (!agentFirst) {
            EXPECT_EQ(1, cache.getMatchedBlockCount(42, {11, 22, 33}, {}, {}, {}, false));
            ASSERT_TRUE(cache.applyAgentUpdate(42, makePendingSnapshot(7, {agent})).applied);
        }
        EXPECT_EQ(3, cache.getMatchedBlockCount(42, {11, 22, 33}, {}, {}, {}, false));
        auto stats = cache.getStats(42);
        EXPECT_EQ(6u, stats.uniqueBlockCount);
        EXPECT_EQ(6u, stats.uniqueTailReadyBlockCount);
        EXPECT_EQ(2u, stats.scheduledBatchCount);
        EXPECT_EQ(1u, stats.agentBatchCount);
        EXPECT_EQ(7u, stats.agentRevision);
        const auto snapshot = cache.captureReplaySnapshot(42);
        PendingCacheState oracle;
        ASSERT_TRUE(oracle.replace({batch.getPendingCacheReservation(), independent}));
        EXPECT_EQ(oracle.getStats().checksum, snapshot.scheduledStats.checksum);
        EXPECT_EQ(oracle.getStats().uniqueBlockCount, snapshot.scheduledStats.uniqueBlockCount);
        EXPECT_EQ(oracle.countFullReadyOverlap(cache.getNodeCache(42)->agent),
                  cache.getNodeCache(42)->overlap.uniqueBlockCount);
        MasterPendingCacheR restored;
        ASSERT_TRUE(restored.restoreReplayState(42, snapshot.toJson()));
        EXPECT_EQ(6u, restored.getStats(42).uniqueBlockCount);
        ASSERT_TRUE(batch.markFailed(3, transition));
        EXPECT_FALSE(transition.changed);
        EXPECT_FALSE(transition.cacheChanged);
        for (auto id : {2, 4}) {
            ASSERT_TRUE(batch.markFailed(id, transition));
            ASSERT_TRUE(cache.removeScheduledBlocks(42, 101,
                transition.removedFullReadyBlockHashes, transition.removedTailReadyBlockHashes));
        }
        EXPECT_TRUE(batch.isResolved());
        stats = cache.getStats(42);
        EXPECT_EQ(4u, stats.uniqueBlockCount);
        EXPECT_EQ(1u, stats.scheduledBatchCount);
        EXPECT_EQ(1u, stats.agentBatchCount);
        EXPECT_EQ(3, cache.getMatchedBlockCount(42, {11, 22, 33}, {}, {}, {}, false));
        ASSERT_TRUE(cache.applyAgentUpdate(42, makePendingSnapshot(8, {})).applied);
        EXPECT_EQ(2u, cache.getStats(42).uniqueBlockCount);
        ASSERT_TRUE(cache.retire(42, 102));
        EXPECT_EQ(0u, cache.getStats(42).uniqueBlockCount);
        EXPECT_EQ(0u, cache.getStats(42).uniqueTailReadyBlockCount);
        EXPECT_EQ(0u, cache.getStats(42).batchCount);
        EXPECT_EQ(8u, cache.getStats(42).agentRevision);
        EXPECT_EQ(0u, cache.getNodeCache(42)->scheduled.getStats().checksum);
        EXPECT_TRUE(cache.getNodeCache(42)->scheduled._reducedReservations.empty());
        EXPECT_TRUE(cache.getNodeCache(42)->scheduled._reservationChecksums.empty());
    }
}

TEST_F(MasterPendingCacheRTest, testGeneratedRemovalMatchesIndependentOracle)
{
    for (uint32_t seed = 0; seed < 24; ++seed) {
        SCOPED_TRACE(seed);
        std::mt19937 random(seed);
        const NaviSchedCachePolicy policy{16, 1 + seed % 7, bool(seed % 2)};
        MasterBatchReservation batch;
        MasterPendingCacheR cache;
        std::vector<MasterBatchReservation::Request> requests;
        std::vector<bool> active(48, true);
        for (size_t index = 0; index < active.size(); ++index) {
            auto hashes = std::make_shared<NaviSchedBlockHashes>();
            const auto count = 1 + random() % 16;
            for (size_t block = 0; block < count; ++block) {
                hashes->push_back(random() % 29);
            }
            requests.push_back({index + 1, int64_t(count * 16 + index % 2),
                                16, 0, hashes, 1.0});
        }
        ASSERT_TRUE(batch.init(101, 1, 48, requests, &policy));
        ASSERT_TRUE(cache.updateScheduled(42, batch.getPendingCacheReservation()));
        const NaviSchedPendingCacheReservation independent{102, 102, {0, 5, 32}, {5}};
        ASSERT_TRUE(cache.updateScheduled(42, independent));
        std::vector<size_t> order;
        for (size_t index = 0; index < active.size(); ++index) {
            order.push_back(index);
        }
        std::shuffle(order.begin(), order.end(), random);
        for (size_t step = 0; step < order.size(); ++step) {
            SCOPED_TRACE(step);
            const bool withAgent = step % 3 != 0;
            const NaviSchedPendingCacheReservation agent{201, 101, {0, 5, 17, 31}, {5, 17}};
            ASSERT_TRUE(cache.applyAgentUpdate(42, makePendingSnapshot(
                step + 1, withAgent ? std::vector{agent}
                                    : std::vector<NaviSchedPendingCacheReservation>{})).applied);
            const auto index = order[step];
            MasterBatchReservation::Transition transition;
            if (index % 2 == 0) {
                ASSERT_TRUE(batch.markFailed(index + 1, transition));
            } else {
                NaviSchedLocalBatchInfo local;
                local.scheduleBatchId = 101;
                local.localBatchId = 301 + index;
                local.requestIds = {index + 1};
                local.requestCount = 1;
                ASSERT_TRUE(batch.markAdmitted(local, transition));
                ASSERT_TRUE(batch.markCompleted(local, true, transition));
            }
            active[index] = false;
            if (transition.cacheChanged) {
                ASSERT_TRUE(cache.removeScheduledBlocks(42, 101,
                    transition.removedFullReadyBlockHashes,
                    transition.removedTailReadyBlockHashes));
            }
            std::set<uint64_t> full;
            std::set<uint64_t> tail;
            for (size_t requestIndex = 0; requestIndex < active.size(); ++requestIndex) {
                if (!active[requestIndex]) {
                    continue;
                }
                const auto &request = requests[requestIndex];
                const auto &hashes = *request.blockHashes;
                full.insert(hashes.begin(), hashes.end());
                if (policy.hybridTailRequired) {
                    for (size_t block = 0; block < hashes.size(); ++block) {
                        if ((block + 1) % policy.linearStep == 0 ||
                            block + 1 == hashes.size() ||
                            (request.totalTokenCount % 16 == 0 && block + 2 == hashes.size())) {
                            tail.insert(hashes[block]);
                        }
                    }
                }
            }
            PendingCacheState oracle;
            auto expectedScheduled = std::vector{independent};
            if (!full.empty()) {
                expectedScheduled.push_back({101, 101, {full.begin(), full.end()},
                                                       {tail.begin(), tail.end()}});
            }
            ASSERT_TRUE(oracle.replace(expectedScheduled));
            const auto snapshot = cache.captureReplaySnapshot(42);
            PendingCacheState actual;
            ASSERT_TRUE(actual.replace(snapshot.scheduled));
            EXPECT_TRUE(oracle.sameReservations(actual));
            EXPECT_EQ(oracle.getStats().checksum, snapshot.scheduledStats.checksum);
            auto expectedFull = full;
            auto expectedTail = tail;
            expectedFull.insert(independent.fullReadyBlockHashes.begin(), independent.fullReadyBlockHashes.end());
            expectedTail.insert(independent.tailReadyBlockHashes.begin(), independent.tailReadyBlockHashes.end());
            if (withAgent) {
                expectedFull.insert(agent.fullReadyBlockHashes.begin(), agent.fullReadyBlockHashes.end());
                expectedTail.insert(agent.tailReadyBlockHashes.begin(), agent.tailReadyBlockHashes.end());
            }
            const auto stats = cache.getStats(42);
            EXPECT_EQ(expectedFull.size(), stats.uniqueBlockCount);
            EXPECT_EQ(expectedTail.size(), stats.uniqueTailReadyBlockCount);
            EXPECT_EQ(expectedScheduled.size(), stats.scheduledBatchCount);
            EXPECT_EQ(step + 1, stats.agentRevision);
            if (step % 11 == 0) {
                MasterPendingCacheR restored;
                ASSERT_TRUE(restored.restoreReplayState(42, snapshot.toJson()));
                EXPECT_EQ(expectedFull.size(), restored.getStats(42).uniqueBlockCount);
            }
        }
        EXPECT_TRUE(batch.isResolved());
        ASSERT_TRUE(cache.retire(42, 102));
        ASSERT_TRUE(cache.applyAgentUpdate(42, makePendingSnapshot(49, {})).applied);
        EXPECT_EQ(0u, cache.getStats(42).uniqueBlockCount);
        EXPECT_TRUE(cache.getNodeCache(42)->scheduled._reducedReservations.empty());
        EXPECT_TRUE(cache.getNodeCache(42)->scheduled._reservationChecksums.empty());
    }
}

TEST_F(MasterPendingCacheRTest, testGeneratedRemovalConcurrentWithAgentUpdates)
{
    MasterPendingCacheR cache;
    MasterBatchReservation batch;
    const NaviSchedCachePolicy policy{1, 1, false};
    std::vector<MasterBatchReservation::Request> requests;
    for (uint64_t id = 1; id <= 256; ++id) {
        requests.push_back({id, 2, 1, 0,
            std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{7, 1000 + id}), 1.0});
    }
    ASSERT_TRUE(batch.init(101, 1, 256, std::move(requests), &policy));
    ASSERT_TRUE(cache.updateScheduled(42, batch.getPendingCacheReservation()));
    const NaviSchedPendingCacheReservation agent{201, 101, {7, 999}, {}};
    std::atomic<bool> start{false};
    std::atomic<size_t> failures{0};
    std::thread updater([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (uint64_t revision = 1; revision <= 512; ++revision) {
            const auto update = makePendingSnapshot(revision, revision % 2
                ? std::vector{agent} : std::vector<NaviSchedPendingCacheReservation>{});
            if (!cache.applyAgentUpdate(42, update).applied) {
                ++failures;
            }
            std::this_thread::yield();
        }
    });
    start.store(true, std::memory_order_release);
    for (uint64_t id = 1; id <= 256; ++id) {
        MasterBatchReservation::Transition transition;
        EXPECT_TRUE(batch.markFailed(id, transition));
        EXPECT_TRUE(cache.removeScheduledBlocks(42, 101,
            transition.removedFullReadyBlockHashes, transition.removedTailReadyBlockHashes));
        std::this_thread::yield();
    }
    updater.join();
    EXPECT_EQ(0u, failures.load());
    EXPECT_TRUE(batch.isResolved());
    const auto stats = cache.getStats(42);
    EXPECT_EQ(0u, stats.scheduledBatchCount);
    EXPECT_EQ(0u, stats.agentBatchCount);
    EXPECT_EQ(0u, stats.uniqueBlockCount);
    EXPECT_EQ(0u, stats.uniqueTailReadyBlockCount);
    EXPECT_EQ(512u, stats.agentRevision);
    const auto snapshot = cache.captureReplaySnapshot(42);
    EXPECT_EQ(0u, snapshot.scheduledStats.checksum);
    EXPECT_EQ(0u, snapshot.agentStats.checksum);
    EXPECT_EQ(0u, cache.getNodeCache(42)->overlap.uniqueBlockCount);
}

TEST_F(MasterPendingCacheRTest, testReplayRestoresBothOwnersAndScheduleExclusion)
{
    MasterPendingCacheR original;
    ASSERT_TRUE(reserveFlat(original, 42, 101, 1000, {1, 2, 3}));
    ASSERT_TRUE(reserveFlat(original, 42, 102, 1000, {1, 2, 4}));
    NaviSchedPendingCacheReservation agent;
    agent.batchId = 201;
    agent.scheduleBatchId = 101;
    agent.fullReadyBlockHashes = {1, 2, 3};
    const auto applied = original.applyAgentUpdate(42, makePendingSnapshot(7, {agent}));
    ASSERT_TRUE(applied.applied);
    NaviSchedScheduleReplay replay;
    replay.schedulerInfo["pending_state"] = autil::legacy::ToJson(original.makeReplayState(42));
    NaviSchedScheduleReplay jsonReplay;
    NaviSchedScheduleReplayJson::fromJsonString(NaviSchedScheduleReplayJson::toJsonString(replay), jsonReplay);
    autil::DataBuffer buffer;
    jsonReplay.serialize(buffer);
    NaviSchedScheduleReplay binaryReplay;
    binaryReplay.deserialize(buffer);
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJson(state, binaryReplay.schedulerInfo.at("pending_state"));
    MasterPendingCacheR restored;
    ASSERT_TRUE(restored.restoreReplayState(42, state));
    const auto stats = restored.getStats(42);
    EXPECT_EQ(2u, stats.scheduledBatchCount);
    EXPECT_EQ(1u, stats.agentBatchCount);
    EXPECT_EQ(4u, stats.uniqueBlockCount);
    EXPECT_EQ(7u, stats.agentRevision);
    EXPECT_EQ(original.getStats(42).agentChecksum, stats.agentChecksum);
    EXPECT_EQ(3, restored.getMatchedBlockCount(42, {1, 2, 3}, {}, {}, {}, false));
    EXPECT_EQ(2, restored.getMatchedBlockCountExcludingSchedule(42, 101, {1, 2, 3}, {}, {}, {}, false));
    ASSERT_TRUE(restored.retire(42, 101));
    EXPECT_EQ(3, restored.getMatchedBlockCount(42, {1, 2, 3}, {}, {}, {}, false));
    EXPECT_EQ(2, restored.getMatchedBlockCountExcludingSchedule(42, 101, {1, 2, 3}, {}, {}, {}, false));
    ASSERT_TRUE(restored.applyAgentUpdate(42, makePendingSnapshot(8, {})).applied);
    EXPECT_EQ(2, restored.getMatchedBlockCount(42, {1, 2, 3}, {}, {}, {}, false));
    ASSERT_TRUE(restored.retire(42, 102));
    EXPECT_EQ(0u, restored.getStats(42).uniqueBlockCount);
    EXPECT_EQ(0u, restored.getStats(42).batchCount);
    state["agent_checksum"] = autil::legacy::ToJson(uint64_t(1));
    EXPECT_FALSE(restored.restoreReplayState(42, state));
    EXPECT_EQ(0u, restored.getStats(42).batchCount);
}

TEST_F(MasterPendingCacheRTest, testCapturedReplaySurvivesRetirementAndNodeRemoval)
{
    MasterPendingCacheR cache;
    ASSERT_TRUE(reserveFlat(cache, 42, 101, 1000, {1, 2, 3}));
    NaviSchedPendingCacheReservation agent;
    agent.batchId = 201;
    agent.scheduleBatchId = 101;
    agent.fullReadyBlockHashes = {1, 2, 3};
    ASSERT_TRUE(cache.applyAgentUpdate(42, makePendingSnapshot(7, {agent})).applied);
    const auto captured = cache.captureReplaySnapshot(42);
    ASSERT_TRUE(captured.present);
    ASSERT_TRUE(cache.retire(42, 101));
    ASSERT_TRUE(cache.applyAgentUpdate(42, makePendingSnapshot(8, {})).applied);
    cache.remove(42);

    MasterPendingCacheR restored;
    ASSERT_TRUE(restored.restoreReplayState(42, captured.toJson()));
    const auto stats = restored.getStats(42);
    EXPECT_EQ(1u, stats.scheduledBatchCount);
    EXPECT_EQ(1u, stats.agentBatchCount);
    EXPECT_EQ(3u, stats.uniqueBlockCount);
    EXPECT_EQ(7u, stats.agentRevision);
    EXPECT_EQ(captured.agentStats.checksum, stats.agentChecksum);
    EXPECT_TRUE(cache.captureReplaySnapshot(42).toJson().empty());
}

TEST_F(MasterPendingCacheRTest, testRepeatedStatsQueriesDoNotRescanPendingHashes)
{
    auto cpuTimeNs = []() {
        timespec value{};
        EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value));
        return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
    };
    auto measure = [&](size_t agentBatchCount, size_t scheduledBatchCount) {
        MasterPendingCacheR cache;
        std::vector<NaviSchedPendingCacheReservation> reservations;
        for (size_t index = 0; index < agentBatchCount; ++index) {
            NaviSchedPendingCacheReservation reservation;
            reservation.batchId = index + 1;
            reservation.scheduleBatchId = index + 1;
            for (uint64_t block = 0; block < 400; ++block) {
                reservation.fullReadyBlockHashes.push_back(index * 400 + block + 1);
            }
            reservations.push_back(std::move(reservation));
        }
        EXPECT_TRUE(cache.applyAgentUpdate(1, makePendingSnapshot(1, reservations)).applied);
        for (size_t index = 0; index < scheduledBatchCount; ++index) {
            EXPECT_TRUE(reserveFlat(cache, 1, 100000 + index, 1,
                                    reservations[index].fullReadyBlockHashes));
        }
        const auto expectedBatches = agentBatchCount + scheduledBatchCount;
        EXPECT_EQ(expectedBatches, cache.getStats(1).batchCount);
        const auto begin = cpuTimeNs();
        for (size_t query = 0; query < 300; ++query) {
            const auto stats = cache.getStats(1);
            EXPECT_EQ(expectedBatches, stats.batchCount);
            EXPECT_EQ(agentBatchCount * 400, stats.uniqueBlockCount);
        }
        const auto elapsed = cpuTimeNs() - begin;
        EXPECT_TRUE(cache.retire(1, 100000));
        EXPECT_EQ(expectedBatches - 1, cache.getStats(1).batchCount);
        EXPECT_TRUE(reserveFlat(cache, 1, 200000, 1, {99999999}));
        EXPECT_EQ(expectedBatches, cache.getStats(1).batchCount);
        EXPECT_EQ(agentBatchCount * 400 + 1, cache.getStats(1).uniqueBlockCount);
        return elapsed;
    };
    const auto smallNs = measure(8, 2);
    const auto largeNs = measure(2450, 100);
    RecordProperty("small_repeated_stats_cpu_ns", std::to_string(smallNs));
    RecordProperty("large_repeated_stats_cpu_ns", std::to_string(largeNs));
    EXPECT_LT(largeNs, smallNs * 16 + 50000000);
}

TEST_F(MasterPendingCacheRTest, testStatsAfterSmallMutationsDoNotRescanHistoricalHashes)
{
    auto cpuTimeNs = []() {
        timespec value{};
        EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value));
        return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
    };
    auto measure = [&](size_t batchCount, size_t scheduledCount, bool agentMutation) {
        MasterPendingCacheR cache;
        std::vector<NaviSchedPendingCacheReservation> reservations;
        for (size_t index = 0; index < batchCount; ++index) {
            NaviSchedPendingCacheReservation reservation;
            reservation.batchId = index + 1;
            reservation.scheduleBatchId = index + 1;
            for (uint64_t block = 0; block < 400; ++block) {
                reservation.fullReadyBlockHashes.push_back(index * 400 + block + 1);
            }
            reservations.push_back(std::move(reservation));
        }
        auto initial = makePendingSnapshot(1, reservations);
        EXPECT_TRUE(cache.applyAgentUpdate(1, initial).applied);
        PendingCacheState agent;
        EXPECT_TRUE(agent.replace(reservations));
        for (size_t index = 0; index < scheduledCount; ++index) {
            EXPECT_TRUE(reserveFlat(cache, 1, 100000 + index, 1,
                                    reservations[index].fullReadyBlockHashes));
        }
        NaviSchedPendingCacheReservation added;
        added.batchId = 900000;
        added.scheduleBatchId = 900000;
        added.fullReadyBlockHashes = {99999999};
        const auto begin = cpuTimeNs();
        for (size_t step = 0; step < 300; ++step) {
            const bool adding = step % 2 == 0;
            if (agentMutation) {
                bool changed = false;
                EXPECT_TRUE(adding ? agent.upsert(added, true, changed)
                                   : agent.erase(added.batchId, changed));
                EXPECT_TRUE(changed);
                AgentPendingCacheToMasterData update;
                update.revision = step + 2;
                if (adding) {
                    update.addedReservations.push_back(added);
                } else {
                    update.removedBatchIds.push_back(added.batchId);
                }
                const auto stats = agent.getStats();
                update.batchCount = stats.batchCount;
                update.uniqueBlockCount = stats.uniqueBlockCount;
                update.uniqueTailReadyBlockCount = stats.uniqueTailReadyBlockCount;
                update.checksum = stats.checksum;
                EXPECT_TRUE(cache.applyAgentUpdate(1, update).applied);
            } else {
                EXPECT_TRUE(adding ? reserveFlat(cache, 1, added.batchId, 1,
                                                 added.fullReadyBlockHashes)
                                   : cache.retire(1, added.batchId));
            }
            const auto stats = cache.getStats(1);
            EXPECT_EQ(batchCount + scheduledCount + size_t(adding), stats.batchCount);
            EXPECT_EQ(batchCount * 400 + size_t(adding), stats.uniqueBlockCount);
            EXPECT_EQ(0, stats.uniqueTailReadyBlockCount);
        }
        return cpuTimeNs() - begin;
    };
    for (bool agentMutation : {false, true}) {
        const auto smallNs = measure(8, 2, agentMutation);
        const auto largeNs = measure(2450, 100, agentMutation);
        const std::string name = agentMutation ? "agent_delta" : "scheduled_delta";
        RecordProperty(name + "_small_cpu_ns", std::to_string(smallNs));
        RecordProperty(name + "_large_cpu_ns", std::to_string(largeNs));
        EXPECT_LT(largeNs, smallNs * 16 + 50000000) << name;
    }
}

TEST_F(MasterPendingCacheRTest, testIncrementalStatsMatchIndependentUnion)
{
    using Reservations = std::map<int64_t, NaviSchedPendingCacheReservation>;
    for (uint32_t seed : {7, 41, 103, 997}) {
        std::mt19937 random(seed);
        MasterPendingCacheR cache;
        Reservations scheduled;
        Reservations agent;
        uint64_t revision = 0;
        bool snapshotReady = false;
        auto snapshot = [&]() {
            std::vector<NaviSchedPendingCacheReservation> result;
            for (const auto &entry : agent) {
                result.push_back(entry.second);
            }
            return makePendingSnapshot(revision, result);
        };
        auto verify = [&]() {
            // This oracle rebuilds unions from raw reservations, without the
            // production reference counts or incremental overlap state.
            std::set<int64_t> batches;
            std::set<uint64_t> full;
            std::set<uint64_t> tail;
            for (const auto *source : {&scheduled, &agent}) {
                for (const auto &entry : *source) {
                    batches.insert(entry.first);
                    full.insert(entry.second.fullReadyBlockHashes.begin(),
                                entry.second.fullReadyBlockHashes.end());
                    tail.insert(entry.second.tailReadyBlockHashes.begin(),
                                entry.second.tailReadyBlockHashes.end());
                }
            }
            const auto actual = cache.getStats(1);
            EXPECT_EQ(batches.size(), actual.batchCount);
            EXPECT_EQ(full.size(), actual.uniqueBlockCount);
            EXPECT_EQ(tail.size(), actual.uniqueTailReadyBlockCount);
            EXPECT_EQ(scheduled.size(), actual.scheduledBatchCount);
            EXPECT_EQ(agent.size(), actual.agentBatchCount);
            EXPECT_EQ(revision, actual.agentRevision);
            EXPECT_EQ(snapshotReady, actual.agentSnapshotReady);
            EXPECT_EQ(0, actual.agentRevisionGapCount);
            EXPECT_EQ(0, actual.agentStateMismatchCount);
        };
        ASSERT_TRUE(cache.applyAgentUpdate(1, snapshot()).applied);
        snapshotReady = true;
        for (size_t step = 0; step < 800; ++step) {
            SCOPED_TRACE("seed=" + std::to_string(seed) +
                         " step=" + std::to_string(step));
            const int64_t batchId = random() % 24 + 1;
            NaviSchedPendingCacheReservation reservation;
            reservation.batchId = batchId;
            reservation.scheduleBatchId = random() % 6 + 1;
            // A small hash domain forces shared references across batches,
            // shared batch IDs across states and differing full/tail overlap.
            for (uint64_t hash = 0; hash < 16; ++hash) {
                if (hash == 0 || random() % 3 == 0) {
                    reservation.fullReadyBlockHashes.push_back(hash);
                    if (random() % 2 == 0) {
                        reservation.tailReadyBlockHashes.push_back(hash);
                    }
                }
            }
            const auto operation = random() % 8;
            if (operation <= 1) {
                reservation.scheduleBatchId = batchId;
                ASSERT_TRUE(cache.updateScheduled(1, reservation));
                scheduled[batchId] = reservation;
                ASSERT_TRUE(cache.updateScheduled(1, reservation));
            } else if (operation == 2) {
                const bool existed = scheduled.erase(batchId) != 0;
                EXPECT_EQ(existed, cache.retire(1, batchId));
            } else if (operation <= 4) {
                agent[batchId] = reservation;
                ++revision;
                auto update = snapshot();
                update.fullSnapshot = false;
                update.addedReservations = {reservation};
                ASSERT_TRUE(cache.applyAgentUpdate(1, update).applied);
            } else if (operation == 5) {
                agent.erase(batchId);
                ++revision;
                auto update = snapshot();
                update.fullSnapshot = false;
                update.addedReservations.clear();
                update.removedBatchIds = {batchId};
                ASSERT_TRUE(cache.applyAgentUpdate(1, update).applied);
            } else if (operation == 6) {
                agent[batchId] = reservation;
                ++revision;
                ASSERT_TRUE(cache.applyAgentUpdate(1, snapshot()).applied);
                ASSERT_TRUE(cache.applyAgentUpdate(1, snapshot()).applied);
            } else {
                cache.clear(1);
                scheduled.clear();
                agent.clear();
                revision = 0;
                snapshotReady = false;
                verify();
                ASSERT_TRUE(cache.applyAgentUpdate(1, snapshot()).applied);
                snapshotReady = true;
            }
            verify();
        }
    }
}

TEST_F(MasterPendingCacheRTest, testStatsOverlapResetsAfterRejectedAgentUpdates)
{
    for (size_t failure = 0; failure < 6; ++failure) {
        SCOPED_TRACE("failure=" + std::to_string(failure));
        MasterPendingCacheR cache;
        NaviSchedPendingCacheReservation first{1, 1, {0, 11, 22}, {11}};
        NaviSchedPendingCacheReservation second{2, 1, {11, 22, 33}, {22}};
        ASSERT_TRUE(cache.updateScheduled(1, first));
        ASSERT_TRUE(cache.updateScheduled(1, {3, 3, {11, 44}, {11}}));
        const auto initial = makePendingSnapshot(10, {first, second});
        ASSERT_TRUE(cache.applyAgentUpdate(1, initial).applied);
        EXPECT_EQ(3, cache.getStats(1).batchCount);
        EXPECT_EQ(5, cache.getStats(1).uniqueBlockCount);
        EXPECT_EQ(2, cache.getStats(1).uniqueTailReadyBlockCount);

        auto bad = initial;
        bad.revision = 11;
        if (failure == 0) {
            bad.batchCount = -1;
        } else if (failure == 1) {
            bad.addedReservations.push_back(first);
        } else if (failure == 2) {
            bad.checksum ^= 1;
        } else if (failure == 3) {
            bad.fullSnapshot = false;
            bad.addedReservations = {second};
            bad.revision = 12;
        } else if (failure == 4) {
            bad.fullSnapshot = false;
            second.fullReadyBlockHashes = {11, 11};
            bad.addedReservations = {second};
        } else {
            bad.revision = 10;
            bad = makePendingSnapshot(10, {second});
        }
        const auto rejected = cache.applyAgentUpdate(1, bad);
        ASSERT_FALSE(rejected.applied);
        EXPECT_TRUE(rejected.stateMismatch || rejected.revisionGap);
        EXPECT_FALSE(rejected.stats.agentSnapshotReady);
        EXPECT_EQ(0, rejected.stats.agentBatchCount);
        EXPECT_EQ(2, rejected.stats.batchCount);
        EXPECT_EQ(4, rejected.stats.uniqueBlockCount);
        EXPECT_EQ(1, rejected.stats.uniqueTailReadyBlockCount);

        ASSERT_TRUE(cache.applyAgentUpdate(
            1, makePendingSnapshot(20, {first})).applied);
        EXPECT_EQ(2, cache.getStats(1).batchCount);
        EXPECT_EQ(4, cache.getStats(1).uniqueBlockCount);
        EXPECT_EQ(1, cache.getStats(1).uniqueTailReadyBlockCount);
        ASSERT_TRUE(cache.retire(1, 1));
        EXPECT_EQ(2, cache.getStats(1).batchCount);
        EXPECT_EQ(4, cache.getStats(1).uniqueBlockCount);
        EXPECT_EQ(1, cache.getStats(1).uniqueTailReadyBlockCount);
        ASSERT_TRUE(cache.retire(1, 3));
        EXPECT_EQ(1, cache.getStats(1).batchCount);
        EXPECT_EQ(3, cache.getStats(1).uniqueBlockCount);
        cache.clear(1);
        EXPECT_EQ(0, cache.getStats(1).batchCount);
        EXPECT_EQ(0, cache.getStats(1).uniqueBlockCount);
        EXPECT_EQ(0, cache.getStats(1).uniqueTailReadyBlockCount);
    }
}

TEST_F(MasterPendingCacheRTest, testStatsReadersObserveConsistentMutations)
{
    MasterPendingCacheR cache;
    NaviSchedPendingCacheReservation reservation;
    reservation.batchId = 1;
    reservation.scheduleBatchId = 1;
    reservation.fullReadyBlockHashes = {11};
    ASSERT_TRUE(cache.applyAgentUpdate(1, makePendingSnapshot(1, {reservation})).applied);
    std::atomic<bool> start{false};
    std::vector<std::thread> readers;
    for (size_t index = 0; index < 4; ++index) {
        readers.emplace_back([&]() {
            while (!start.load()) {
                std::this_thread::yield();
            }
            for (size_t query = 0; query < 1000; ++query) {
                const auto stats = cache.getStats(1);
                EXPECT_TRUE(stats.batchCount == 1 || stats.batchCount == 2);
                EXPECT_EQ(stats.batchCount, stats.uniqueBlockCount);
                EXPECT_EQ(stats.batchCount - 1, stats.scheduledBatchCount);
                EXPECT_EQ(1, stats.agentBatchCount);
                EXPECT_EQ(1, stats.agentRevision);
                EXPECT_TRUE(stats.agentSnapshotReady);
            }
        });
    }
    start.store(true);
    for (size_t update = 0; update < 1000; ++update) {
        EXPECT_TRUE(reserveFlat(cache, 1, 100000, 1, {22}));
        EXPECT_TRUE(cache.retire(1, 100000));
    }
    for (auto &reader : readers) {
        reader.join();
    }
    EXPECT_EQ(1, cache.getStats(1).batchCount);
    cache.clear(1);
    const auto cleared = cache.getStats(1);
    EXPECT_EQ(0, cleared.batchCount);
    EXPECT_EQ(0, cleared.uniqueBlockCount);
    EXPECT_EQ(0, cleared.agentRevision);
    EXPECT_FALSE(cleared.agentSnapshotReady);
}

TEST_F(MasterPendingCacheRTest, testIgnoresStaleAgentFullSnapshotWithoutDataLoss)
{
    MasterPendingCacheR cache;
    NaviSchedPendingCacheReservation current;
    current.batchId = 7;
    current.scheduleBatchId = 7;
    current.fullReadyBlockHashes = {11, 22};
    auto currentSnapshot = makePendingSnapshot(5, {current});
    auto result = cache.applyAgentUpdate(1, currentSnapshot);
    ASSERT_TRUE(result.applied);

    NaviSchedPendingCacheReservation stale;
    stale.batchId = 8;
    stale.scheduleBatchId = 8;
    stale.fullReadyBlockHashes = {33};
    auto staleSnapshot = makePendingSnapshot(4, {stale});
    staleSnapshot.batchCount = -1;
    staleSnapshot.removedBatchIds = {7};
    result = cache.applyAgentUpdate(1, staleSnapshot);
    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.staleSnapshotIgnored);
    EXPECT_EQ("stale_full_snapshot_ignored", result.reason);
    EXPECT_EQ(5, result.stats.agentRevision);
    EXPECT_TRUE(result.stats.agentSnapshotReady);
    EXPECT_EQ(1, result.stats.agentBatchCount);
    EXPECT_EQ(PendingCacheUtil::reservationChecksum(current),
              result.stats.agentChecksum);

    const MasterPendingCacheR::PhysicalBlockSet empty;
    EXPECT_EQ(2, cache.getMatchedBlockCount(1, {11, 22}, empty, empty,
                                            empty, false));
    EXPECT_EQ(0, cache.getMatchedBlockCount(1, {33}, empty, empty, empty,
                                            false));
}

TEST_F(MasterPendingCacheRTest, testRejectsDifferentStateAtSameAgentRevision)
{
    MasterPendingCacheR cache;
    NaviSchedPendingCacheReservation current;
    current.batchId = 7;
    current.scheduleBatchId = 7;
    current.fullReadyBlockHashes = {11, 22};
    const auto currentSnapshot = makePendingSnapshot(5, {current});
    ASSERT_TRUE(cache.applyAgentUpdate(1, currentSnapshot).applied);

    auto duplicate = cache.applyAgentUpdate(1, currentSnapshot);
    EXPECT_TRUE(duplicate.applied);
    EXPECT_FALSE(duplicate.staleSnapshotIgnored);
    EXPECT_TRUE(duplicate.reason.empty());
    EXPECT_TRUE(duplicate.stats.agentSnapshotReady);

    NaviSchedPendingCacheReservation conflicting;
    conflicting.batchId = 8;
    conflicting.scheduleBatchId = 8;
    conflicting.fullReadyBlockHashes = {33};
    const auto conflictingSnapshot = makePendingSnapshot(5, {conflicting});
    const auto mismatch = cache.applyAgentUpdate(1, conflictingSnapshot);
    EXPECT_FALSE(mismatch.applied);
    EXPECT_TRUE(mismatch.stateMismatch);
    EXPECT_EQ("same_revision_state_mismatch", mismatch.reason);
    EXPECT_FALSE(mismatch.stats.agentSnapshotReady);
    EXPECT_EQ(0, mismatch.stats.agentBatchCount);
    EXPECT_EQ(1, mismatch.stats.agentStateMismatchCount);
}

TEST_F(MasterPendingCacheRTest, testReservationEqualityMatchesSortedSnapshotOracle)
{
    PendingCacheState left;
    PendingCacheState right;
    auto oracle = [](const PendingCacheState &a, const PendingCacheState &b) {
        const auto first = a.makeSnapshot();
        const auto second = b.makeSnapshot();
        return first.size() == second.size() &&
               std::equal(first.begin(), first.end(), second.begin(),
                          PendingCacheUtil::equalReservation);
    };
    std::mt19937 generator(7931);
    for (size_t step = 0; step < 512; ++step) {
        const int64_t batchId = 1 + generator() % 40;
        NaviSchedPendingCacheReservation reservation{
            batchId, static_cast<int64_t>(1 + generator() % 7), {11, 22, 33}, {22}};
        bool changed = false;
        if (generator() % 3 == 0) {
            ASSERT_TRUE(left.erase(batchId, changed));
        } else {
            ASSERT_TRUE(left.upsert(reservation, true, changed));
        }
        if (step % 4 == 0) {
            auto reversed = left.makeSnapshot();
            std::reverse(reversed.begin(), reversed.end());
            ASSERT_TRUE(right.replace(reversed));
        }
        EXPECT_EQ(oracle(left, right), left.sameReservations(right)) << step;
        EXPECT_EQ(oracle(right, left), right.sameReservations(left)) << step;
        EXPECT_TRUE(left.sameReservations(left));
    }

    const NaviSchedPendingCacheReservation original{1, 2, {11, 22, 33}, {22}};
    ASSERT_TRUE(left.replace({original}));
    for (size_t field = 0; field < 4; ++field) {
        auto changed = original;
        if (field == 0) changed.batchId = 3;
        if (field == 1) changed.scheduleBatchId = 3;
        if (field == 2) changed.fullReadyBlockHashes.back() = 44;
        if (field == 3) changed.tailReadyBlockHashes = {11};
        ASSERT_TRUE(right.replace({changed}));
        // Equality must inspect definitions even if checksum metadata collides.
        right._checksum = left._checksum;
        EXPECT_FALSE(left.sameReservations(right)) << field;
    }
    ASSERT_TRUE(right.replace({original}));
    EXPECT_TRUE(left.sameReservations(right));
}

TEST_F(MasterPendingCacheRTest, testReservationEqualityAvoidsSnapshotCopyAndSort)
{
    std::vector<NaviSchedPendingCacheReservation> reservations;
    for (int64_t batch = 1; batch <= 2048; ++batch) {
        NaviSchedPendingCacheReservation reservation;
        reservation.batchId = batch;
        reservation.scheduleBatchId = batch / 4 + 1;
        for (uint64_t block = 1; block <= 128; ++block) {
            reservation.fullReadyBlockHashes.push_back(batch * 128 + block);
        }
        reservation.tailReadyBlockHashes = {reservation.fullReadyBlockHashes.back()};
        reservations.push_back(std::move(reservation));
    }
    PendingCacheState left;
    PendingCacheState right;
    ASSERT_TRUE(left.replace(reservations));
    std::reverse(reservations.begin(), reservations.end());
    ASSERT_TRUE(right.replace(reservations));
    auto measure = [&](bool indexed) {
        std::vector<int64_t> samples;
        for (size_t repeat = 0; repeat < 9; ++repeat) {
            timespec begin{}, end{};
            EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &begin));
            for (size_t iteration = 0; iteration < 8; ++iteration) {
                if (indexed) {
                    EXPECT_TRUE(left.sameReservations(right));
                } else {
                    const auto first = left.makeSnapshot();
                    const auto second = right.makeSnapshot();
                    EXPECT_EQ(first.size(), second.size());
                    EXPECT_TRUE(std::equal(first.begin(), first.end(), second.begin(),
                                           PendingCacheUtil::equalReservation));
                }
            }
            EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end));
            if (repeat > 0) {
                samples.push_back((end.tv_sec - begin.tv_sec) * int64_t{1000000000} +
                                  end.tv_nsec - begin.tv_nsec);
            }
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    };
    const auto copiedNs = measure(false);
    const auto indexedNs = measure(true);
    RecordProperty("snapshot_equality_median_8_calls_ns", std::to_string(copiedNs));
    RecordProperty("indexed_equality_median_8_calls_ns", std::to_string(indexedNs));
    EXPECT_LT(indexedNs, copiedNs / 3);
}

TEST_F(MasterPendingCacheRTest, testKeepsSharedBlocksUntilLastBatchRetires)
{
    MasterPendingCacheR cache;
    ASSERT_TRUE(reserveFlat(cache, 1, 1, 100, {11, 22}));
    ASSERT_TRUE(reserveFlat(cache, 1, 2, 200, {22, 33}));
    EXPECT_FALSE(reserveFlat(cache, 1, 2, 300, {44}));
    EXPECT_EQ(2, cache.getStats(1).batchCount);
    EXPECT_EQ(3, cache.getStats(1).uniqueBlockCount);

    const MasterPendingCacheR::PhysicalBlockSet empty;
    EXPECT_EQ(3, cache.getMatchedBlockCount(1, {11, 22, 33}, empty, empty,
                                            empty, false));
    ASSERT_TRUE(cache.retire(1, 1));
    EXPECT_EQ(0, cache.getMatchedBlockCount(1, {11}, empty, empty, empty,
                                            false));
    EXPECT_EQ(2, cache.getMatchedBlockCount(1, {22, 33}, empty, empty, empty,
                                            false));
    ASSERT_TRUE(cache.retire(1, 2));
    EXPECT_EQ(0, cache.getStats(1).batchCount);
    EXPECT_EQ(0, cache.getStats(1).uniqueBlockCount);
}

TEST_F(MasterPendingCacheRTest, testIsolatesAgents)
{
    MasterPendingCacheR cache;
    ASSERT_TRUE(reserveFlat(cache, 1, 1, 100, {11}));
    ASSERT_TRUE(reserveFlat(cache, 2, 1, 100, {22}));
    const MasterPendingCacheR::PhysicalBlockSet empty;
    EXPECT_EQ(1, cache.getMatchedBlockCount(1, {11}, empty, empty, empty,
                                            false));
    EXPECT_EQ(0, cache.getMatchedBlockCount(2, {11}, empty, empty, empty,
                                            false));
    EXPECT_EQ(1, cache.getMatchedBlockCount(2, {22}, empty, empty, empty,
                                            false));
    cache.clear(1);
    EXPECT_EQ(0, cache.getStats(1).batchCount);
    EXPECT_EQ(1, cache.getStats(2).batchCount);
}

TEST_F(MasterPendingCacheRTest, testRealCacheTakesOverBeforePendingBatchRetires)
{
    MasterPendingCacheR cache;
    ASSERT_TRUE(reserveFlat(cache, 1, 7, 100, {11, 22}));
    MasterPendingCacheR::PhysicalBlockSet cachedBlockHashes;
    const MasterPendingCacheR::PhysicalBlockSet empty;
    EXPECT_EQ(2, cache.getMatchedBlockCount(1, {11, 22}, cachedBlockHashes,
                                            empty, empty, false));

    cachedBlockHashes.insert({11, 22});
    ASSERT_TRUE(cache.retire(1, 7));
    EXPECT_EQ(2, cache.getMatchedBlockCount(1, {11, 22}, cachedBlockHashes,
                                            empty, empty, false));
}

TEST_F(MasterPendingCacheRTest, testMatchBreakdownSeparatesPendingOverlay)
{
    MasterPendingCacheR cache;
    ASSERT_TRUE(reserveFlat(cache, 1, 7, 100, {11, 22}));
    const MasterPendingCacheR::PhysicalBlockSet empty;
    const auto breakdown = cache.getMatchBreakdown(
        1, {11, 22}, empty, empty, empty, false);
    EXPECT_EQ(0, breakdown.staticMatchedBlockCount);
    EXPECT_EQ(2, breakdown.matchedBlockCount);
    EXPECT_EQ(7, breakdown.firstPendingBlockSourceBatchId);
    EXPECT_EQ(0, breakdown.pendingTailSourceBatchId);
}

TEST_F(MasterPendingCacheRTest, testOnlyExplicitLifecycleClearsReservations)
{
    MasterPendingCacheR cache;
    ASSERT_TRUE(reserveFlat(cache, 1, 1, 1000, {11}));
    ASSERT_TRUE(reserveFlat(cache, 1, 2, 120LL * 1000 * 1000, {22}));
    EXPECT_EQ(2, cache.getStats(1).batchCount);
    EXPECT_EQ(2, cache.getStats(1).uniqueBlockCount);

    cache.remove(1);
    EXPECT_EQ(0, cache.getStats(1).batchCount);
    EXPECT_EQ(0, cache.getStats(1).uniqueBlockCount);
    EXPECT_FALSE(cache.retire(1, 2));
}

TEST_F(MasterPendingCacheRTest, testPendingBlocksCompleteHybridPrefixAndTail)
{
    MasterPendingCacheR cache;
    const NaviSchedBlockHashes produced = {22, 33, 44, 55, 66};
    MasterPendingCacheR::ReferencedRequest request{&produced, 81, 16};
    ASSERT_TRUE(cache.reserve(1, 1, 100, {request},
                              NaviSchedCachePolicy{16, 4, true}));
    const MasterPendingCacheR::PhysicalBlockSet cachedBlockHashes = {11};
    const MasterPendingCacheR::PhysicalBlockSet fullReadyBlockHashes = {11};
    const MasterPendingCacheR::PhysicalBlockSet tailReadyBlockHashes = {11};

    EXPECT_EQ(1, cache.getMatchedBlockCount(
                     1, {11, 22, 33}, cachedBlockHashes, fullReadyBlockHashes,
                     tailReadyBlockHashes, true));
    EXPECT_EQ(5, cache.getMatchedBlockCount(
                     1, {11, 22, 33, 44, 55}, cachedBlockHashes,
                     fullReadyBlockHashes, tailReadyBlockHashes, true));
    EXPECT_EQ(2, cache.getStats(1).uniqueTailReadyBlockCount);
    ASSERT_TRUE(cache.retire(1, 1));
    EXPECT_EQ(1, cache.getMatchedBlockCount(
                     1, {11, 22, 33}, cachedBlockHashes, fullReadyBlockHashes,
                     tailReadyBlockHashes, true));
}

TEST_F(MasterPendingCacheRTest, testMatchBreakdownTracksPendingTailSource)
{
    MasterPendingCacheR cache;
    const NaviSchedBlockHashes produced = {22, 33, 44};
    MasterPendingCacheR::ReferencedRequest request{&produced, 49, 16};
    ASSERT_TRUE(cache.reserve(1, 7, 100, {request},
                              NaviSchedCachePolicy{16, 16, true}));
    const MasterPendingCacheR::PhysicalBlockSet empty;
    const auto breakdown = cache.getMatchBreakdown(
        1, {22, 33, 44}, empty, empty, empty, true);
    EXPECT_EQ(0, breakdown.staticMatchedBlockCount);
    EXPECT_EQ(3, breakdown.matchedBlockCount);
    EXPECT_EQ(7, breakdown.firstPendingBlockSourceBatchId);
    EXPECT_EQ(7, breakdown.pendingTailSourceBatchId);
}

TEST_F(MasterPendingCacheRTest, testAlignedHybridRequestKeepsPreviousAndFinalTail)
{
    MasterPendingCacheR cache;
    const NaviSchedBlockHashes produced = {22, 33, 44};
    MasterPendingCacheR::ReferencedRequest request{&produced, 48, 16};
    ASSERT_TRUE(cache.reserve(1, 1, 100, {request},
                              NaviSchedCachePolicy{16, 128, true}));
    const MasterPendingCacheR::PhysicalBlockSet ready = {11};

    EXPECT_EQ(1, cache.getMatchedBlockCount(1, {11, 22}, ready, ready, ready,
                                            true));
    EXPECT_EQ(3, cache.getMatchedBlockCount(1, {11, 22, 33}, ready, ready, ready,
                                            true));
    EXPECT_EQ(4, cache.getMatchedBlockCount(1, {11, 22, 33, 44}, ready, ready,
                                            ready, true));
    EXPECT_EQ(2, cache.getStats(1).uniqueTailReadyBlockCount);
}

TEST_F(MasterPendingCacheRTest, testMatchesRtpHybridTailMaterializationAcrossLengths)
{
    constexpr int64_t blockSize = 8;
    const MasterPendingCacheR::PhysicalBlockSet empty;
    int64_t batchId = 1;
    for (int64_t linearStep : {1, 2, 3, 7}) {
        for (int64_t tokenCount = blockSize; tokenCount < 20 * blockSize;
             ++tokenCount) {
            NaviSchedBlockHashes hashes;
            const auto blockCount = static_cast<size_t>(tokenCount / blockSize);
            for (size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
                hashes.emplace_back(100000 * linearStep +
                                    1000 * tokenCount + blockIndex + 1);
            }
            MasterPendingCacheR cache;
            MasterPendingCacheR::ReferencedRequest request{
                &hashes, tokenCount, blockSize};
            ASSERT_TRUE(cache.reserve(
                1, batchId++, 100, {request},
                NaviSchedCachePolicy{blockSize, linearStep, true}));

            const bool aligned = tokenCount % blockSize == 0;
            for (size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
                const bool periodic = (blockIndex + 1) % linearStep == 0;
                const bool finalFullBlock = blockIndex + 1 == blockCount;
                const bool alignedPreviousBlock =
                    aligned && blockCount >= 2 && blockIndex + 2 == blockCount;
                const bool tailReady =
                    periodic || finalFullBlock || alignedPreviousBlock;

                EXPECT_EQ(1,
                          cache.getMatchedBlockCount(
                              1, {hashes[blockIndex]}, empty, empty, empty,
                              false))
                    << "linearStep=" << linearStep
                    << " tokenCount=" << tokenCount
                    << " blockIndex=" << blockIndex;
                EXPECT_EQ(tailReady ? 1 : 0,
                          cache.getMatchedBlockCount(
                              1, {hashes[blockIndex]}, empty, empty, empty,
                              true))
                    << "linearStep=" << linearStep
                    << " tokenCount=" << tokenCount
                    << " blockIndex=" << blockIndex;
            }
        }
    }
}

TEST_F(MasterPendingCacheRTest, testRejectsPolicyAndBlockSizeMismatch)
{
    MasterPendingCacheR cache;
    const NaviSchedBlockHashes produced = {11, 22};
    MasterPendingCacheR::ReferencedRequest request{&produced, 32, 16};

    EXPECT_FALSE(cache.reserve(1, 1, 100, {request}, {}));
    EXPECT_FALSE(cache.reserve(1, 1, 100, {request},
                               NaviSchedCachePolicy{32, 1, false}));
    EXPECT_EQ(0, cache.getStats(1).batchCount);
}

TEST_F(MasterPendingCacheRTest, testBatchMatchingUsesOnePolicyAndReportsMismatch)
{
    MasterPendingCacheR cache;
    ASSERT_TRUE(reserveFlat(cache, 1, 1, 100, {11, 22, 33}));
    std::vector<NaviSchedScheduleItem> requests(3);
    requests[0].request.blockSize = 1;
    requests[0].request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    requests[1].request.blockSize = 2;
    requests[1].request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    requests[2].request.blockSize = 1;
    requests[2].request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{33, 44});
    int64_t matchedBlockCounts[] = {-1, -1, -1};
    const MasterPendingCacheR::PhysicalBlockSet empty;

    EXPECT_EQ(1, cache.getMatchedBlockCounts(
                     1, requests, 1, empty, empty, empty, false,
                     matchedBlockCounts));
    EXPECT_EQ(2, matchedBlockCounts[0]);
    EXPECT_EQ(0, matchedBlockCounts[1]);
    EXPECT_EQ(1, matchedBlockCounts[2]);
}

TEST_F(MasterPendingCacheRTest, testBatchMatchingTimingPreservesResultsAndResetsState)
{
    MasterPendingCacheR cache;
    ASSERT_TRUE(reserveFlat(cache, 1, 1, 100, {11, 22, 33}));
    std::vector<NaviSchedScheduleItem> requests(2);
    requests[0].request.blockSize = 1;
    requests[0].request.promptTokenCount = 3;
    requests[0].request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    requests[1].request = requests[0].request;
    requests[1].request.blockSize = 2;
    const MasterPendingCacheR::PhysicalBlockSet empty;
    const MasterPendingCacheR::PhysicalBlockSet physical = {11};
    int64_t plain[] = {-1, -1};
    int64_t timed[] = {-1, -1};
    MasterPendingCacheR::BatchMatchTiming timing;
    EXPECT_EQ(1, cache.getMatchedBlockCounts(
                     1, requests, 1, physical, empty, empty, false, plain));
    EXPECT_EQ(1, cache.getMatchedBlockCounts(
                     1, requests, 1, physical, empty, empty, false, timed, &timing));
    EXPECT_EQ(2, plain[0]);
    EXPECT_EQ(0, plain[1]);
    EXPECT_EQ(plain[0], timed[0]);
    EXPECT_EQ(plain[1], timed[1]);
    EXPECT_GE(timing.nodeLookupUs, 0);
    EXPECT_GE(timing.lockWaitUs, 0);
    EXPECT_GE(timing.matchUs, 0);
    EXPECT_EQ(1u, timing.scheduledBatches);
    EXPECT_EQ(3u, timing.scheduledBlocks);
    EXPECT_EQ(0u, timing.agentBatches);
    EXPECT_EQ(0u, timing.agentBlocks);

    EXPECT_EQ(1, cache.getMatchedBlockCounts(
                     2, requests, 1, physical, empty, empty, false, timed, &timing));
    EXPECT_EQ(1, timed[0]);
    EXPECT_EQ(0, timed[1]);
    EXPECT_EQ(0, timing.lockWaitUs);
    EXPECT_EQ(0u, timing.scheduledBatches);
    EXPECT_EQ(0u, timing.scheduledBlocks);
    EXPECT_EQ(0u, timing.agentBatches);
    EXPECT_EQ(0u, timing.agentBlocks);

    timing.nodeLookupUs = timing.lockWaitUs = timing.matchUs = 123;
    timing.scheduledBatches = timing.scheduledBlocks = 123;
    EXPECT_EQ(2, cache.getMatchedBlockCounts(
                     1, requests, 1, physical, empty, empty, false, nullptr, &timing));
    EXPECT_EQ(0, timing.nodeLookupUs);
    EXPECT_EQ(0, timing.lockWaitUs);
    EXPECT_EQ(0, timing.matchUs);
    EXPECT_EQ(0u, timing.scheduledBatches);
    EXPECT_EQ(0u, timing.scheduledBlocks);
}

TEST_F(MasterPendingCacheRTest, testBatchMatchingWorkCountsIncludeMissAndHybridTail)
{
    MasterPendingCacheR cache;
    const MasterPendingCacheR::PhysicalBlockSet full = {11, 22, 33};
    const MasterPendingCacheR::PhysicalBlockSet tail = {22};
    const MasterPendingCacheR::PhysicalBlockSet empty;
    std::vector<NaviSchedScheduleItem> requests(3);
    for (auto &item : requests) {
        item.request.blockSize = 1;
        item.request.promptTokenCount = 5;
    }
    requests[0].request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22, 33, 44});
    requests[1].request.blockHashes = requests[0].request.blockHashes;
    requests[2].request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22, 55, 66});
    int64_t matched[3] = {-1, -1, -1};
    MasterPendingCacheR::BatchMatchTiming timing;
    EXPECT_EQ(0u, cache.getMatchedBlockCounts(
        1, requests, 1, empty, full, tail, true, matched, &timing));
    EXPECT_EQ(2, matched[0]);
    EXPECT_EQ(2, matched[1]);
    EXPECT_EQ(2, matched[2]);
    EXPECT_EQ(3u, timing.prefixComparisonCount);
    EXPECT_EQ(5u, timing.reusedFullPrefixBlockCount);
    EXPECT_EQ(6u, timing.fullPrefixProbeCount);
    EXPECT_EQ(5u, timing.tailProbeCount);

    NaviSchedScheduleContext context;
    context._requests = requests;
    context.prepareCacheMatchPrefixes();
    EXPECT_EQ(std::vector<size_t>({0, 4, 2}), context._cacheMatchCommonPrefixes);
    EXPECT_EQ(0u, cache.getMatchedBlockCounts(
        1, requests, 1, empty, full, tail, true, matched, &timing,
        &context._cacheMatchCommonPrefixes));
    EXPECT_EQ(2, matched[0]);
    EXPECT_EQ(2, matched[1]);
    EXPECT_EQ(2, matched[2]);
    EXPECT_EQ(0u, timing.prefixComparisonCount);
    EXPECT_EQ(5u, timing.reusedFullPrefixBlockCount);
    EXPECT_EQ(6u, timing.fullPrefixProbeCount);
    EXPECT_EQ(5u, timing.tailProbeCount);

    requests.clear();
    EXPECT_EQ(0u, cache.getMatchedBlockCounts(
        1, requests, 1, empty, full, tail, true, matched, &timing));
    EXPECT_EQ(0u, timing.prefixComparisonCount);
    EXPECT_EQ(0u, timing.reusedFullPrefixBlockCount);
    EXPECT_EQ(0u, timing.fullPrefixProbeCount);
    EXPECT_EQ(0u, timing.tailProbeCount);
}

TEST_F(MasterPendingCacheRTest, testBatchMatchingPrefixReuseMatchesScalarAcrossUpdates)
{
    MasterPendingCacheR cache;
    std::mt19937 random(9182);
    const MasterPendingCacheR::PhysicalBlockSet cached = {1, 2, 3, 4, 5};
    const MasterPendingCacheR::PhysicalBlockSet full = {1, 2, 3, 4, 5, 6};
    const MasterPendingCacheR::PhysicalBlockSet tail = {2, 5};
    std::vector<NaviSchedScheduleItem> requests;
    auto append = [&](NaviSchedBlockHashes hashes, int64_t tokens, int64_t blockSize) {
        NaviSchedScheduleItem item;
        item.request.blockHashes = std::make_shared<const NaviSchedBlockHashes>(std::move(hashes));
        item.request.promptTokenCount = tokens;
        item.request.blockSize = blockSize;
        requests.push_back(std::move(item));
    };
    append({1, 2, 3, 4}, 5, 1); // Tail fallback must not become the next full-prefix bound.
    append({1, 2, 3, 4, 5}, 6, 1);
    requests.push_back(requests.back());
    append({1, 9, 3, 4}, 5, 1);
    append({1, 2, 3, 4, 5, 6}, 6, 1);
    append({}, 10, 1);
    requests.emplace_back();
    for (size_t i = 0; i < 300; ++i) {
        NaviSchedBlockHashes hashes;
        for (size_t j = 0, size = random() % 10; j < size; ++j) {
            hashes.push_back(random() % 4 ? j + 1 : 20 + random() % 4);
        }
        append(std::move(hashes), random() % 12, random() % 5 ? 1 : 2);
    }
    std::vector<int64_t> results(requests.size());
    const NaviSchedBlockHashes empty;
    NaviSchedScheduleContext context;
    context._requests = requests;
    context.prepareCacheMatchPrefixes();
    for (size_t update = 0; update < 4; ++update) {
        SCOPED_TRACE(update);
        if (update == 1) {
            ASSERT_TRUE(cache.updateScheduled(1, {1, 1, {1, 2, 3, 7, 8}, {3, 8}}));
            ASSERT_TRUE(cache.applyAgentUpdate(1, makePendingSnapshot(
                1, {{2, 2, {1, 2, 3, 4, 5, 6, 7}, {6, 7}}})).applied);
        } else if (update == 2) {
            ASSERT_TRUE(cache.retire(1, 1));
        } else if (update == 3) {
            ASSERT_TRUE(cache.applyAgentUpdate(1, makePendingSnapshot(2, {})).applied);
        }
        for (const bool hybrid : {false, true}) {
            for (const int64_t blockSize : {-1, 0, 1, 2}) {
                uint64_t mismatches = 0;
                for (const auto &item : requests) {
                    mismatches += blockSize <= 0 || item.request.blockSize != blockSize;
                }
                EXPECT_EQ(mismatches, cache.getMatchedBlockCounts(
                    1, requests, blockSize, cached, full, tail, hybrid, results.data()));
                for (size_t i = 0; i < requests.size(); ++i) {
                    const auto &request = requests[i].request;
                    const auto expected = blockSize <= 0 || request.blockSize != blockSize ? 0
                        : cache.getMatchedBlockCount(
                            1, request.blockHashes ? *request.blockHashes : empty,
                            cached, full, tail, hybrid,
                            BlockHashUtil::getMatchableBlockCount(
                                request.promptTokenCount, request.blockSize, request.blockHashes));
                    EXPECT_EQ(expected, results[i]) << "request=" << i
                        << " blockSize=" << blockSize << " hybrid=" << hybrid;
                }
                std::vector<int64_t> reused(requests.size());
                EXPECT_EQ(mismatches, cache.getMatchedBlockCounts(
                    1, requests, blockSize, cached, full, tail, hybrid, reused.data(),
                    nullptr, &context._cacheMatchCommonPrefixes));
                EXPECT_EQ(results, reused);
            }
        }
    }
}

TEST_F(MasterPendingCacheRTest, testBatchMatchingPrefixReuseKeepsReadSnapshot)
{
    MasterPendingCacheR cache;
    NaviSchedBlockHashes blocks;
    for (uint64_t i = 1; i <= 128; ++i) blocks.push_back(i);
    ASSERT_TRUE(cache.updateScheduled(1, {1, 1, blocks, blocks}));
    std::vector<NaviSchedScheduleItem> requests(64);
    for (auto &item : requests) {
        item.request.blockHashes = std::make_shared<const NaviSchedBlockHashes>(blocks);
        item.request.blockSize = 1;
        item.request.promptTokenCount = 129;
    }
    std::atomic<bool> start{false};
    std::atomic<size_t> updates{0};
    NaviSchedScheduleContext context;
    context._requests = requests;
    context.prepareCacheMatchPrefixes();
    std::thread writer([&] {
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        for (size_t i = 0; i < 1000; ++i) {
            const auto length = i % 2 ? 128 : 32;
            const NaviSchedBlockHashes subset(blocks.begin(), blocks.begin() + length);
            EXPECT_TRUE(cache.updateScheduled(1, {1, 1, subset, subset}));
            updates.fetch_add(1, std::memory_order_relaxed);
        }
    });
    start.store(true, std::memory_order_release);
    std::vector<int64_t> counts(requests.size());
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_EQ(0, cache.getMatchedBlockCounts(
            1, requests, 1, {}, {}, {}, true, counts.data(), nullptr,
            &context._cacheMatchCommonPrefixes));
        EXPECT_TRUE(counts[0] == 32 || counts[0] == 128);
        for (auto count : counts) EXPECT_EQ(counts[0], count);
    }
    writer.join();
    EXPECT_EQ(1000u, updates.load());
}

TEST_F(MasterPendingCacheRTest, testBatchMatchingPrefixReuseOnIncidentRequests)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation fixture;
    ASSERT_TRUE(fixture.restoreReplayState(state));
    std::vector<NaviSchedScheduleItem> requests;
    MasterPendingCacheR::PhysicalBlockSet all;
    size_t totalBlocks = 0;
    for (const auto &entry : fixture._requests) {
        NaviSchedScheduleItem item;
        item.request.blockHashes = entry.request.blockHashes;
        item.request.blockSize = entry.request.blockSize;
        item.request.promptTokenCount = entry.request.totalTokenCount;
        totalBlocks += item.request.blockHashes->size();
        all.insert(item.request.blockHashes->begin(), item.request.blockHashes->end());
        requests.push_back(std::move(item));
    }
    ASSERT_EQ(512u, requests.size());
    ASSERT_EQ(110169u, totalBlocks);
    MasterPendingCacheR cache;
    ASSERT_TRUE(cache.updateScheduled(1, fixture.getPendingCacheReservation()));
    std::vector<int64_t> counts(requests.size());
    for (const bool hybrid : {false, true}) {
        for (const bool includePhysical : {false, true}) {
            const MasterPendingCacheR::PhysicalBlockSet empty;
            const auto &physical = includePhysical ? all : empty;
            EXPECT_EQ(0, cache.getMatchedBlockCounts(
                1, requests, 16, physical, physical, physical, hybrid, counts.data()));
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto &request = requests[i].request;
                EXPECT_EQ(cache.getMatchedBlockCount(
                    1, *request.blockHashes, physical, physical, physical, hybrid,
                    BlockHashUtil::getMatchableBlockCount(
                        request.promptTokenCount, request.blockSize, request.blockHashes)), counts[i]);
            }
            auto cpuTimeNs = []() {
                timespec value{};
                EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value));
                return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
            };
            uint64_t scalarChecksum = 0, batchChecksum = 0;
            std::vector<int64_t> scalarTimes, batchTimes;
            for (size_t trial = 0; trial < 3; ++trial) {
                auto begin = cpuTimeNs();
                for (size_t repeat = 0; repeat < 100; ++repeat) {
                    for (const auto &item : requests) {
                        const auto &request = item.request;
                        scalarChecksum += cache.getMatchedBlockCount(
                            1, *request.blockHashes, physical, physical, physical, hybrid,
                            BlockHashUtil::getMatchableBlockCount(
                                request.promptTokenCount, request.blockSize, request.blockHashes));
                    }
                }
                scalarTimes.push_back(cpuTimeNs() - begin);
                begin = cpuTimeNs();
                for (size_t repeat = 0; repeat < 100; ++repeat) {
                    EXPECT_EQ(0, cache.getMatchedBlockCounts(
                        1, requests, 16, physical, physical, physical, hybrid, counts.data()));
                    for (auto count : counts) batchChecksum += count;
                }
                batchTimes.push_back(cpuTimeNs() - begin);
            }
            EXPECT_EQ(scalarChecksum, batchChecksum);
            std::sort(scalarTimes.begin(), scalarTimes.end());
            std::sort(batchTimes.begin(), batchTimes.end());
            const auto name = std::string(hybrid ? "hybrid" : "full") +
                (includePhysical ? "_all_physical" : "_fixture_pending");
            RecordProperty(name + "_scalar_cpu_ns", std::to_string(scalarTimes[1]));
            RecordProperty(name + "_batch_cpu_ns", std::to_string(batchTimes[1]));
        }
    }
}

TEST_F(MasterPendingCacheRTest, testBatchMatchingReusesConfirmedPrefixCpu)
{
    constexpr size_t REQUESTS = 64;
    constexpr size_t BLOCKS = 256;
    constexpr size_t REPEATS = 2000;
    MasterPendingCacheR cache;
    NaviSchedBlockHashes blocks;
    for (size_t i = 0; i < BLOCKS; ++i) {
        blocks.push_back(i + 1);
    }
    ASSERT_TRUE(cache.updateScheduled(1, {1, 1, blocks, blocks}));
    const auto shared = std::make_shared<const NaviSchedBlockHashes>(blocks);
    const MasterPendingCacheR::PhysicalBlockSet empty;
    auto cpuTimeNs = []() {
        timespec value{};
        EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value));
        return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
    };
    for (const bool sharedPointer : {false, true}) {
        SCOPED_TRACE(sharedPointer);
        std::vector<NaviSchedScheduleItem> requests(REQUESTS);
        for (auto &item : requests) {
            item.request.blockSize = 1;
            item.request.promptTokenCount = BLOCKS + 1;
            item.request.blockHashes = sharedPointer ? shared
                : std::make_shared<const NaviSchedBlockHashes>(blocks);
        }
        std::vector<int64_t> counts(REQUESTS);
        std::vector<int64_t> scalarTimes;
        std::vector<int64_t> batchTimes;
        uint64_t checksum = 0;
        for (size_t trial = 0; trial < 3; ++trial) {
            auto begin = cpuTimeNs();
            for (size_t repeat = 0; repeat < REPEATS; ++repeat) {
                for (const auto &item : requests) {
                    checksum += cache.getMatchedBlockCount(
                        1, *item.request.blockHashes, empty, empty, empty, true, BLOCKS);
                }
            }
            scalarTimes.push_back(cpuTimeNs() - begin);
            begin = cpuTimeNs();
            for (size_t repeat = 0; repeat < REPEATS; ++repeat) {
                checksum += cache.getMatchedBlockCounts(
                    1, requests, 1, empty, empty, empty, true, counts.data());
                for (auto count : counts) {
                    checksum += count;
                }
            }
            batchTimes.push_back(cpuTimeNs() - begin);
        }
        EXPECT_EQ(uint64_t(6) * REPEATS * REQUESTS * BLOCKS, checksum);
        for (auto count : counts) {
            EXPECT_EQ(BLOCKS, count);
        }
        std::sort(scalarTimes.begin(), scalarTimes.end());
        std::sort(batchTimes.begin(), batchTimes.end());
        RecordProperty(sharedPointer ? "shared_scalar_cpu_ns" : "distinct_scalar_cpu_ns",
                       std::to_string(scalarTimes[1]));
        RecordProperty(sharedPointer ? "shared_batch_cpu_ns" : "distinct_batch_cpu_ns",
                       std::to_string(batchTimes[1]));
        EXPECT_LT(batchTimes[1], scalarTimes[1] * 0.65);
    }
}

TEST_F(MasterPendingCacheRTest, testBatchMatchingKeepsOneBlockForAlignedPrefill)
{
    MasterPendingCacheR cache;
    const MasterPendingCacheR::PhysicalBlockSet ready = {11, 22, 33};
    std::vector<NaviSchedScheduleItem> requests(3);
    for (auto &item : requests) {
        item.request.blockSize = 16;
        item.request.blockHashes = std::make_shared<const NaviSchedBlockHashes>(
            NaviSchedBlockHashes{11, 22, 33});
    }
    requests[0].request.promptTokenCount = 48;
    requests[1].request.promptTokenCount = 49;
    requests[2].request.promptTokenCount = 64;
    int64_t matchedBlockCounts[] = {-1, -1, -1};

    EXPECT_EQ(0, cache.getMatchedBlockCounts(
                     1, requests, 16, ready, ready, ready, false,
                     matchedBlockCounts));
    EXPECT_EQ(2, matchedBlockCounts[0]);
    EXPECT_EQ(3, matchedBlockCounts[1]);
    EXPECT_EQ(3, matchedBlockCounts[2]);
}

TEST_F(MasterPendingCacheRTest, testExcludingScheduleTracksSplitBatchesAndUpdates)
{
    PendingCacheState state;
    bool changed = false;
    NaviSchedPendingCacheReservation first{1, 41, {11, 22}, {22}};
    NaviSchedPendingCacheReservation second{2, 41, {11, 33}, {11}};
    NaviSchedPendingCacheReservation other{3, 42, {11, 44}, {44}};
    ASSERT_TRUE(state.replace({first, second, other}));
    EXPECT_TRUE(state.containsFullReadyExcludingSchedule(11, 41));
    EXPECT_FALSE(state.containsTailReadyExcludingSchedule(11, 41));
    EXPECT_FALSE(state.containsFullReadyExcludingSchedule(22, 41));
    EXPECT_FALSE(state.containsFullReadyExcludingSchedule(33, 41));
    EXPECT_TRUE(state.containsTailReadyExcludingSchedule(44, 41));
    EXPECT_TRUE(state.containsTailReadyExcludingSchedule(22, 999));
    EXPECT_FALSE(state.containsFullReadyExcludingSchedule(999, 41));

    other.scheduleBatchId = 41;
    ASSERT_TRUE(state.upsert(other, true, changed));
    EXPECT_TRUE(changed);
    EXPECT_FALSE(state.containsFullReadyExcludingSchedule(11, 41));
    EXPECT_FALSE(state.containsTailReadyExcludingSchedule(44, 41));
    EXPECT_TRUE(state.containsFullReadyExcludingSchedule(11, 42));

    first.scheduleBatchId = 43;
    first.fullReadyBlockHashes = {22, 55};
    first.tailReadyBlockHashes = {55};
    ASSERT_TRUE(state.upsert(first, true, changed));
    EXPECT_TRUE(changed);
    EXPECT_FALSE(state.containsFullReadyExcludingSchedule(11, 41));
    EXPECT_TRUE(state.containsFullReadyExcludingSchedule(22, 41));
    EXPECT_FALSE(state.containsTailReadyExcludingSchedule(22, 41));
    EXPECT_TRUE(state.containsTailReadyExcludingSchedule(55, 41));
    ASSERT_TRUE(state.erase(1, changed));
    EXPECT_TRUE(changed);
    EXPECT_FALSE(state.containsFullReadyExcludingSchedule(22, 41));
    EXPECT_FALSE(state.containsTailReadyExcludingSchedule(55, 41));

    auto invalid = first;
    invalid.batchId = 0;
    EXPECT_FALSE(state.replace({first, invalid}));
    EXPECT_FALSE(state.containsFullReadyExcludingSchedule(11, 41));
    EXPECT_TRUE(state.containsFullReadyExcludingSchedule(11, 43));
    ASSERT_TRUE(state.replace({first}));
    EXPECT_TRUE(state.containsTailReadyExcludingSchedule(55, 41));
    EXPECT_FALSE(state.containsFullReadyExcludingSchedule(55, 43));
    state.clear();
    EXPECT_FALSE(state.containsFullReadyExcludingSchedule(55, 41));
    EXPECT_EQ(0, state.getStats().batchCount);
    ASSERT_TRUE(state.upsert(second, false, changed));
    EXPECT_TRUE(changed);
    EXPECT_FALSE(state.containsFullReadyExcludingSchedule(11, 41));
    EXPECT_TRUE(state.containsTailReadyExcludingSchedule(11, 43));
}

TEST_F(MasterPendingCacheRTest, testExcludingScheduleCostDoesNotScaleWithUnrelatedBatches)
{
    auto cpuTimeNs = []() {
        timespec value{};
        EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value));
        return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
    };
    auto measure = [&](size_t batchCount) {
        PendingCacheState state;
        for (size_t index = 0; index < batchCount; ++index) {
            NaviSchedPendingCacheReservation reservation;
            reservation.batchId = index + 1;
            reservation.scheduleBatchId = index == 0 ? 41 : index + 100;
            reservation.fullReadyBlockHashes = {11, 22};
            reservation.tailReadyBlockHashes = {22};
            bool changed = false;
            EXPECT_TRUE(state.upsert(reservation, false, changed));
            EXPECT_TRUE(changed);
        }
        size_t hits = 0;
        const auto begin = cpuTimeNs();
        for (size_t index = 0; index < 16384; ++index) {
            hits += state.containsFullReadyExcludingSchedule(11, 41);
            hits += state.containsTailReadyExcludingSchedule(22, 41);
        }
        const auto elapsed = cpuTimeNs() - begin;
        EXPECT_EQ(32768, hits);
        return elapsed;
    };
    const auto smallNs = measure(8);
    const auto largeNs = measure(8192);
    RecordProperty("small_query_cpu_ns", std::to_string(smallNs));
    RecordProperty("large_query_cpu_ns", std::to_string(largeNs));
    // Compare thread CPU time with ample slack for timer and cache effects.
    EXPECT_LE(largeNs, smallNs * 16 + 50000000);
}

TEST_F(MasterPendingCacheRTest, testExcludingScheduleUsesBlockOwners)
{
    auto cpuTimeNs = []() {
        timespec value{};
        EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value));
        return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
    };
    auto measure = [&](size_t batchCount) {
        PendingCacheState state;
        for (size_t index = 0; index < batchCount; ++index) {
            NaviSchedPendingCacheReservation reservation;
            reservation.batchId = index + 1;
            reservation.scheduleBatchId = 41;
            for (uint64_t block = 0; block < 128; ++block) {
                reservation.fullReadyBlockHashes.push_back(index * 128 + block + 1);
            }
            reservation.tailReadyBlockHashes = reservation.fullReadyBlockHashes;
            bool changed = false;
            EXPECT_TRUE(state.upsert(reservation, false, changed));
        }
        std::vector<int64_t> samples;
        size_t hits = 0;
        for (size_t repeat = 0; repeat < 7; ++repeat) {
            const auto begin = cpuTimeNs();
            for (size_t query = 0; query < 8192; ++query) {
                const auto hash = (query % batchCount) * 128 + query % 128 + 1;
                hits += state.containsFullReadyExcludingSchedule(hash, 41);
                hits += state.containsTailReadyExcludingSchedule(hash, 41);
            }
            if (repeat != 0) samples.push_back(cpuTimeNs() - begin);
        }
        EXPECT_EQ(0u, hits);
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    };
    const auto smallNs = measure(1);
    const auto splitNs = measure(128);
    RecordProperty("single_batch_exclusion_cpu_ns", std::to_string(smallNs));
    RecordProperty("split_batch_exclusion_cpu_ns", std::to_string(splitNs));
    EXPECT_LT(splitNs, smallNs * 8 + 1000000);
}

TEST_F(MasterPendingCacheRTest, testExcludingScheduleMatchesReservationOracle)
{
    PendingCacheState state;
    std::map<int64_t, NaviSchedPendingCacheReservation> oracle;
    std::mt19937 random(14591);
    for (size_t step = 0; step < 512; ++step) {
        const int64_t batchId = random() % 24 + 1;
        bool changed = false;
        if (random() % 4 == 0) {
            const bool erased = oracle.erase(batchId) != 0;
            ASSERT_TRUE(state.erase(batchId, changed));
            EXPECT_EQ(erased, changed);
        } else {
            NaviSchedPendingCacheReservation reservation;
            reservation.batchId = batchId;
            reservation.scheduleBatchId = random() % 5 + 1;
            for (uint64_t hash = 1; hash <= 16; ++hash) {
                if (random() % 3 == 0) {
                    reservation.fullReadyBlockHashes.push_back(hash);
                    if (random() % 2 == 0) reservation.tailReadyBlockHashes.push_back(hash);
                }
            }
            if (reservation.fullReadyBlockHashes.empty()) reservation.fullReadyBlockHashes = {1};
            ASSERT_TRUE(state.upsert(reservation, true, changed));
            oracle[batchId] = reservation;
        }
        for (int64_t schedule = 1; schedule <= 6; ++schedule) {
            for (uint64_t hash = 1; hash <= 17; ++hash) {
                bool full = false, tail = false;
                for (const auto &[id, reservation] : oracle) {
                    if (reservation.scheduleBatchId == schedule) continue;
                    full |= std::find(reservation.fullReadyBlockHashes.begin(),
                                      reservation.fullReadyBlockHashes.end(), hash) != reservation.fullReadyBlockHashes.end();
                    tail |= std::find(reservation.tailReadyBlockHashes.begin(),
                                      reservation.tailReadyBlockHashes.end(), hash) != reservation.tailReadyBlockHashes.end();
                }
                ASSERT_EQ(full, state.containsFullReadyExcludingSchedule(hash, schedule));
                ASSERT_EQ(tail, state.containsTailReadyExcludingSchedule(hash, schedule));
            }
        }
    }
}

TEST_F(MasterPendingCacheRTest, testSourceLookupPreservesMinimumAcrossTransitions)
{
    PendingCacheState state;
    bool changed = false;
    ASSERT_TRUE(state.replace({{30, 3, {11, 22}, {22}},
                               {10, 1, {11, 33}, {11}},
                               {20, 2, {11, 22}, {22}}}));
    EXPECT_EQ(10, state.getSourceBatchId(11, false));
    EXPECT_EQ(20, state.getSourceBatchId(22, false));
    EXPECT_EQ(10, state.getSourceBatchId(11, true));
    EXPECT_EQ(20, state.getSourceBatchId(22, true));
    EXPECT_EQ(0, state.getSourceBatchId(33, true));
    EXPECT_EQ(0, state.getSourceBatchId(999, false));

    ASSERT_TRUE(state.erase(10, changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(20, state.getSourceBatchId(11, false));
    EXPECT_EQ(0, state.getSourceBatchId(11, true));
    ASSERT_TRUE(state.upsert({20, 2, {44}, {44}}, true, changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(30, state.getSourceBatchId(11, false));
    EXPECT_EQ(30, state.getSourceBatchId(22, true));
    EXPECT_EQ(20, state.getSourceBatchId(44, true));
    ASSERT_TRUE(state.erase(30, changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(0, state.getSourceBatchId(11, false));
    EXPECT_EQ(0, state.getSourceBatchId(22, true));
    EXPECT_EQ(1, state.getStats().uniqueBlockCount);
    EXPECT_EQ(1, state.getStats().uniqueTailReadyBlockCount);

    ASSERT_TRUE(state.replace({{40, 4, {55}, {55}}}));
    EXPECT_EQ(0, state.getSourceBatchId(44, false));
    EXPECT_EQ(40, state.getSourceBatchId(55, true));
    state.clear();
    EXPECT_EQ(0, state.getSourceBatchId(55, true));
    EXPECT_EQ(0, state.getStats().batchCount);
}

TEST_F(MasterPendingCacheRTest, testSourceLookupMatchesIndependentOracleAndCopies)
{
    PendingCacheState state;
    std::map<int64_t, NaviSchedPendingCacheReservation> oracle;
    std::mt19937 random(73921);
    auto verify = [&](const PendingCacheState &value) {
        for (uint64_t hash = 1; hash <= 17; ++hash) {
            for (bool tail : {false, true}) {
                int64_t expected = 0;
                for (const auto &entry : oracle) {
                    const auto &hashes = tail ? entry.second.tailReadyBlockHashes
                                              : entry.second.fullReadyBlockHashes;
                    if (std::find(hashes.begin(), hashes.end(), hash) != hashes.end()) {
                        expected = entry.first;
                        break;
                    }
                }
                EXPECT_EQ(expected, value.getSourceBatchId(hash, tail));
            }
        }
        EXPECT_EQ(oracle.size(), value.getStats().batchCount);
    };
    for (size_t step = 0; step < 1024; ++step) {
        const int64_t batchId = random() % 64 + 1;
        bool changed = false;
        if (random() % 3 == 0) {
            ASSERT_TRUE(state.erase(batchId, changed));
            EXPECT_EQ(oracle.erase(batchId) != 0, changed);
        } else {
            NaviSchedPendingCacheReservation reservation{batchId, batchId % 4 + 1, {}, {}};
            for (uint64_t hash = 1; hash <= 16; ++hash) {
                if (hash == 1 || random() % 4 == 0) {
                    reservation.fullReadyBlockHashes.push_back(hash);
                    if (random() % 2 == 0) {
                        reservation.tailReadyBlockHashes.push_back(hash);
                    }
                }
            }
            ASSERT_TRUE(state.upsert(reservation, true, changed));
            oracle[batchId] = reservation;
        }
        verify(state);
        if (step % 32 == 0) {
            PendingCacheState copied(state);
            PendingCacheState assigned;
            assigned = state;
            verify(copied);
            verify(assigned);
            for (const auto &entry : oracle) {
                ASSERT_TRUE(copied.erase(entry.first, changed));
                EXPECT_TRUE(changed);
            }
            EXPECT_EQ(0, copied.getStats().uniqueBlockCount);
            EXPECT_EQ(0, copied.getStats().uniqueTailReadyBlockCount);
            verify(state);
            assigned.clear();
            verify(state);
        }
    }
}

TEST_F(MasterPendingCacheRTest, testSourceLookupCostDoesNotScaleWithPendingBatches)
{
    auto cpuTimeNs = []() {
        timespec value{};
        EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value));
        return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
    };
    auto measure = [&](size_t batchCount) {
        PendingCacheState state;
        for (size_t index = 0; index < batchCount; ++index) {
            const auto batchId = static_cast<int64_t>(index + 1);
            bool changed = false;
            EXPECT_TRUE(state.upsert({batchId, batchId, {11, 22}, {22}},
                                      false, changed));
            EXPECT_TRUE(changed);
        }
        int64_t sources = 0;
        const auto begin = cpuTimeNs();
        for (size_t index = 0; index < 4096; ++index) {
            sources += state.getSourceBatchId(11, false);
            sources += state.getSourceBatchId(22, true);
            sources += state.getSourceBatchId(999, false);
        }
        const auto elapsed = cpuTimeNs() - begin;
        EXPECT_EQ(8192, sources);
        return elapsed;
    };
    const auto smallNs = measure(8);
    const auto largeNs = measure(8192);
    RecordProperty("small_source_query_cpu_ns", std::to_string(smallNs));
    RecordProperty("large_source_query_cpu_ns", std::to_string(largeNs));
    EXPECT_LE(largeNs, smallNs * 16 + 50000000);
}

TEST_F(MasterPendingCacheRTest, testCapturedPendingSourceLookup)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/agent-316-reservations.txt");
    ASSERT_TRUE(input.is_open());
    size_t count = 0, fullCount = 0, tailCount = 0;
    uint64_t checksum = 0;
    ASSERT_TRUE(bool(input >> count >> checksum >> fullCount >> tailCount));
    ASSERT_EQ(316, count);
    ASSERT_EQ(177564, fullCount);
    std::vector<NaviSchedPendingCacheReservation> reservations(count);
    std::set<uint64_t> fullHashes, tailHashes;
    size_t fullReferences = 0;
    for (auto &reservation : reservations) {
        ASSERT_TRUE(bool(input >> reservation.batchId >> reservation.scheduleBatchId));
        for (auto *hashes : {&reservation.fullReadyBlockHashes,
                             &reservation.tailReadyBlockHashes}) {
            size_t length = 0;
            ASSERT_TRUE(bool(input >> length));
            ASSERT_LE(length, 65536);
            hashes->resize(length);
            for (auto &hash : *hashes) {
                ASSERT_TRUE(bool(input >> hash));
            }
            ASSERT_TRUE(std::is_sorted(hashes->begin(), hashes->end()));
            ASSERT_EQ(hashes->end(), std::adjacent_find(hashes->begin(), hashes->end()));
        }
        fullReferences += reservation.fullReadyBlockHashes.size();
        fullHashes.insert(reservation.fullReadyBlockHashes.begin(),
                          reservation.fullReadyBlockHashes.end());
        tailHashes.insert(reservation.tailReadyBlockHashes.begin(),
                          reservation.tailReadyBlockHashes.end());
    }
    input >> std::ws;
    ASSERT_TRUE(input.eof());
    ASSERT_EQ(215311, fullReferences);
    ASSERT_EQ(fullCount, fullHashes.size());
    ASSERT_EQ(tailCount, tailHashes.size());
    PendingCacheState state;
    ASSERT_TRUE(state.replace(reservations));
    ASSERT_EQ(checksum, state.getStats().checksum);
    ASSERT_EQ(count, state.getStats().batchCount);
    ASSERT_EQ(fullCount, state.getStats().uniqueBlockCount);
    ASSERT_EQ(tailCount, state.getStats().uniqueTailReadyBlockCount);

    std::vector<std::pair<uint64_t, bool>> queries;
    for (bool tail : {false, true}) {
        const auto &hashes = tail ? tailHashes : fullHashes;
        const auto stride = std::max(size_t{1}, hashes.size() / 128);
        size_t index = 0, selected = 0;
        for (auto hash : hashes) {
            if (index++ % stride == 0 && selected < 128) {
                queries.emplace_back(hash, tail);
                ++selected;
            }
        }
        uint64_t missing = 0;
        while (hashes.count(missing)) {
            ++missing;
        }
        queries.emplace_back(missing, tail);
    }
    ASSERT_EQ(130, queries.size());
    int64_t expectedSum = 0;
    for (const auto &[hash, tail] : queries) {
        int64_t expected = 0;
        for (const auto &reservation : reservations) {
            const auto &hashes = tail ? reservation.tailReadyBlockHashes
                                      : reservation.fullReadyBlockHashes;
            // An independent linear membership oracle checks the source semantics.
            if (std::find(hashes.begin(), hashes.end(), hash) != hashes.end() &&
                (expected == 0 || reservation.batchId < expected)) {
                expected = reservation.batchId;
            }
        }
        ASSERT_EQ(expected, state.getSourceBatchId(hash, tail));
        expectedSum += expected;
    }
    auto cpuTimeNs = []() {
        timespec value{};
        EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value));
        return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
    };
    auto measure = [&](bool referenceScan) {
        int64_t sum = 0;
        const auto begin = cpuTimeNs();
        for (size_t repeat = 0; repeat < 128; ++repeat) {
            for (const auto &[hash, tail] : queries) {
                int64_t source = 0;
                if (referenceScan) {
                    for (const auto &reservation : reservations) {
                        const auto &hashes = tail ? reservation.tailReadyBlockHashes
                                                  : reservation.fullReadyBlockHashes;
                        if (std::binary_search(hashes.begin(), hashes.end(), hash) &&
                            (source == 0 || reservation.batchId < source)) {
                            source = reservation.batchId;
                        }
                    }
                } else {
                    source = state.getSourceBatchId(hash, tail);
                }
                sum += source;
            }
        }
        const auto elapsed = cpuTimeNs() - begin;
        EXPECT_EQ(expectedSum * 128, sum);
        return elapsed;
    };
    const auto scanNs = measure(true);
    const auto indexedNs = measure(false);
    RecordProperty("captured_scan_cpu_ns", std::to_string(scanNs));
    RecordProperty("captured_source_query_cpu_ns", std::to_string(indexedNs));
    EXPECT_LE(indexedNs, scanNs / 16 + 5000000);
}

TEST_F(MasterPendingCacheRTest, testSourceLookupMutationFootprint)
{
    auto cpuTimeNs = []() {
        timespec value{};
        EXPECT_EQ(0, clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value));
        return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
    };
    auto allocatedBytes = []() {
        const auto info = mallinfo();
        return int64_t(info.uordblks) + info.hblkhd;
    };
    for (bool shared : {false, true}) {
        const std::string prefix = shared ? "shared_" : "unique_";
        PendingCacheState state;
        const auto bytesBefore = allocatedBytes();
        const auto insertBegin = cpuTimeNs();
        for (int64_t batchId = 8192; batchId > 0; --batchId) {
            NaviSchedPendingCacheReservation reservation{batchId, batchId, {}, {}};
            for (uint64_t block = 1; block <= 96; ++block) {
                const uint64_t hash = shared ? block : batchId * 96 + block;
                reservation.fullReadyBlockHashes.push_back(hash);
                if (block == 96) {
                    reservation.tailReadyBlockHashes.push_back(hash);
                }
            }
            bool changed = false;
            ASSERT_TRUE(state.upsert(reservation, false, changed));
            ASSERT_TRUE(changed);
        }
        RecordProperty(prefix + "insert_cpu_ns", std::to_string(cpuTimeNs() - insertBegin));
        RecordProperty(prefix + "live_bytes", std::to_string(allocatedBytes() - bytesBefore));
        EXPECT_EQ(shared ? 96 : 8192 * 96, state.getStats().uniqueBlockCount);
        const auto eraseBegin = cpuTimeNs();
        for (int64_t batchId = 1; batchId <= 8192; ++batchId) {
            // Ascending retirement repeatedly removes the minimum shared owner.
            EXPECT_EQ(batchId, state.getSourceBatchId(shared ? 96 : batchId * 96 + 96, true));
            bool changed = false;
            ASSERT_TRUE(state.erase(batchId, changed));
            ASSERT_TRUE(changed);
        }
        RecordProperty(prefix + "erase_cpu_ns", std::to_string(cpuTimeNs() - eraseBegin));
        RecordProperty(prefix + "retired_bytes", std::to_string(allocatedBytes() - bytesBefore));
        EXPECT_EQ(0, state.getSourceBatchId(96, true));
        EXPECT_EQ(0, state.getStats().batchCount);
        EXPECT_EQ(0, state.getStats().uniqueBlockCount);
        EXPECT_EQ(0, state.getStats().uniqueTailReadyBlockCount);
        EXPECT_EQ(0, state.getStats().checksum);
    }
}

} // namespace navi_sched
