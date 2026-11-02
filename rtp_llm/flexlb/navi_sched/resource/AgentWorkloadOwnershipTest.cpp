#include "aios/navi_sched/resource/AgentWorkloadOwnership.h"

#include <random>
#include <chrono>
#include <iostream>
#include <unordered_map>

#include "unittest/unittest.h"

namespace navi_sched {
namespace {

NaviSchedRequestStats request(uint64_t id, int64_t prompt, int64_t hit)
{
    NaviSchedRequestStats value;
    value.requestId = id;
    value.totalTokenCount = prompt;
    value.cacheHitTokenCount = hit;
    value.blockSize = 16;
    return value;
}

NaviSchedWorkloadSnapshot emptyObserved(int32_t capacity = 0)
{
    NaviSchedWorkloadSnapshot value;
    value.revision = 1;
    value.eventTimeUs = 1;
    value.prefillCapacitySplit = true;
    value.maxRunningRequestCount = capacity;
    return value;
}

TEST(AgentWorkloadOwnershipTest, PartialCancellationAndExactHandoff)
{
    AgentWorkloadOwnership ownership;
    ASSERT_TRUE(ownership.addBatch(1, {request(11, 100, 20), request(12, 200, 50)}));
    ASSERT_TRUE(ownership.addBatch(2, {request(21, 300, 100)}));
    NaviSchedWorkloadSnapshot snapshot;
    ASSERT_TRUE(ownership.combine(emptyObserved(1), snapshot));
    EXPECT_EQ(2, snapshot.batchCount);
    EXPECT_EQ(3, snapshot.requestCount);
    EXPECT_EQ(600, snapshot.totalPromptTokenCount);
    EXPECT_EQ(1, snapshot.runningCapacitySlotCount);
    EXPECT_EQ(80, snapshot.runningCapacityPrefillTokenCount);
    EXPECT_EQ(350, snapshot.waitingCapacityPrefillTokenCount);
    EXPECT_DOUBLE_EQ(62500, snapshot.waitingCapacityPrefillTokenSquareSum);
    ASSERT_TRUE(ownership.removeRequests(1, {11, 11, 999}));
    ASSERT_TRUE(ownership.combine(emptyObserved(1), snapshot));
    EXPECT_EQ(2, snapshot.requestCount);
    EXPECT_EQ(500, snapshot.totalPromptTokenCount);
    EXPECT_EQ(150, snapshot.runningCapacityPrefillTokenCount);
    EXPECT_EQ(200, snapshot.waitingCapacityPrefillTokenCount);

    auto observed = emptyObserved(1);
    observed.batchCount = observed.requestCount = observed.waitingRequestCount = 1;
    observed.remainingPrefillRequestCount = observed.runningCapacityPrefillRequestCount = 1;
    observed.runningCapacitySlotCount = 1;
    observed.remainingPrefillContextTokenCount = observed.remainingTotalContextTokenCount =
        observed.runningCapacityPrefillContextTokenCount = 50;
    observed.remainingPrefillTokenCount = observed.remainingTotalTokenCount =
        observed.runningCapacityPrefillTokenCount = 150;
    observed.remainingPrefillTokenSquareSum = observed.remainingTotalTokenSquareSum =
        observed.runningCapacityPrefillTokenSquareSum = 22500;
    observed.contextRemainingPrefillTokenProductSum = observed.contextRemainingTotalTokenProductSum =
        observed.runningCapacityContextRemainingPrefillTokenProductSum = 7500;
    observed.totalPromptTokenCount = 200;
    ASSERT_TRUE(observed.valid());
    AgentWorkloadOwnership::RetiredBatches retired;
    ASSERT_TRUE(ownership.handoff({1, 1, 999}, retired));
    EXPECT_EQ(1u, retired.size());
    ASSERT_TRUE(ownership.combine(observed, snapshot));
    EXPECT_EQ(2, snapshot.batchCount);
    EXPECT_EQ(2, snapshot.requestCount);
    EXPECT_EQ(500, snapshot.totalPromptTokenCount);
    EXPECT_EQ(150, snapshot.runningCapacityPrefillTokenCount);
    EXPECT_EQ(200, snapshot.waitingCapacityPrefillTokenCount);
    ASSERT_TRUE(ownership.removeRequests(1, {12}));
    EXPECT_EQ(1u, ownership.requestCount());
    ASSERT_TRUE(ownership.handoff({2}, retired));
    EXPECT_EQ(0u, ownership.requestCount());
    EXPECT_EQ(0u, ownership.batchCount());
    ASSERT_TRUE(ownership.combine(emptyObserved(), snapshot));
    EXPECT_EQ(0, snapshot.requestCount);
    EXPECT_EQ(0, snapshot.totalPromptTokenCount);
    EXPECT_DOUBLE_EQ(0, snapshot.remainingPrefillTokenSquareSum);
}

TEST(AgentWorkloadOwnershipTest, FullyCachedRequestUsesSlotWithoutPrefillCost)
{
    AgentWorkloadOwnership ownership;
    ASSERT_TRUE(ownership.addBatch(1, {request(11, 100, 100), request(12, 200, 20)}));
    NaviSchedWorkloadSnapshot snapshot;
    ASSERT_TRUE(ownership.combine(emptyObserved(1), snapshot));
    EXPECT_EQ(2, snapshot.requestCount);
    EXPECT_EQ(1, snapshot.remainingPrefillRequestCount);
    EXPECT_EQ(1, snapshot.runningCapacitySlotCount);
    EXPECT_EQ(0, snapshot.runningCapacityPrefillRequestCount);
    EXPECT_EQ(1, snapshot.waitingCapacityPrefillRequestCount);
    EXPECT_EQ(180, snapshot.remainingPrefillTokenCount);
    EXPECT_EQ(300, snapshot.totalPromptTokenCount);
    ASSERT_TRUE(ownership.removeRequests(1, {11, 12}));
    EXPECT_EQ(0u, ownership.batchCount());
}

TEST(AgentWorkloadOwnershipTest, LargeContributionRemovalPreservesSmallSurvivor)
{
    AgentWorkloadOwnership ownership;
    ASSERT_TRUE(ownership.addBatch(1, {request(11, 100000000, 0)}));
    ASSERT_TRUE(ownership.addBatch(2, {request(21, 1, 0)}));
    AgentWorkloadOwnership::RetiredBatches retired;
    ASSERT_TRUE(ownership.handoff({1}, retired));
    NaviSchedWorkloadSnapshot snapshot;
    ASSERT_TRUE(ownership.combine(emptyObserved(), snapshot));
    EXPECT_EQ(1, snapshot.requestCount);
    EXPECT_EQ(1, snapshot.remainingPrefillTokenCount);
    EXPECT_DOUBLE_EQ(1.0, snapshot.remainingPrefillTokenSquareSum);
    ASSERT_TRUE(ownership.removeRequests(2, {21}));
    ASSERT_TRUE(ownership.combine(emptyObserved(), snapshot));
    EXPECT_EQ(0, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(0.0, snapshot.remainingPrefillTokenSquareSum);

    ASSERT_TRUE(ownership.addBatch(3, {request(31, 100000000, 0), request(32, 1, 0)}));
    ASSERT_TRUE(ownership.combine(emptyObserved(1), snapshot));
    EXPECT_EQ(1, snapshot.waitingCapacityPrefillRequestCount);
    EXPECT_DOUBLE_EQ(1.0, snapshot.waitingCapacityPrefillTokenSquareSum);
    ASSERT_TRUE(ownership.removeRequests(3, {31}));
    ASSERT_TRUE(ownership.combine(emptyObserved(), snapshot));
    EXPECT_EQ(1, snapshot.requestCount);
    EXPECT_DOUBLE_EQ(1.0, snapshot.remainingPrefillTokenSquareSum);
}

TEST(AgentWorkloadOwnershipTest, RandomLifecycleMatchesIndependentRecalculation)
{
    AgentWorkloadOwnership ownership;
    std::unordered_map<int64_t, std::unordered_map<uint64_t, NaviSchedRequestStats>> live;
    std::mt19937 random(3181);
    uint64_t nextRequest = 1;
    int64_t nextBatch = 1;
    for (int step = 0; step < 3000; ++step) {
        if (live.empty() || random() % 3 == 0) {
            std::vector<NaviSchedRequestStats> requests;
            const auto count = 1 + random() % 16;
            auto &batch = live[nextBatch];
            for (size_t i = 0; i < count; ++i) {
                const int64_t prompt = 1 + random() % 32000;
                const auto value = request(nextRequest++, prompt, random() % (prompt + 1));
                batch.emplace(value.requestId, value);
                requests.emplace_back(value);
            }
            ASSERT_TRUE(ownership.addBatch(nextBatch++, requests));
        } else {
            const auto batch = live.begin();
            if (random() % 2) {
                const auto id = batch->second.begin()->first;
                ASSERT_TRUE(ownership.removeRequests(batch->first, {id}));
                batch->second.erase(id);
                if (batch->second.empty()) live.erase(batch);
            } else {
                AgentWorkloadOwnership::RetiredBatches retired;
                ASSERT_TRUE(ownership.handoff({batch->first}, retired));
                live.erase(batch);
            }
        }
        int64_t count = 0, prompt = 0, miss = 0, hit = 0, prefillCount = 0;
        double squares = 0, products = 0;
        for (const auto &batch : live) {
            for (const auto &entry : batch.second) {
                const auto &r = entry.second;
                ++count;
                prompt += r.totalTokenCount;
                const auto compute = r.totalTokenCount - r.cacheHitTokenCount;
                if (compute == 0) continue;
                ++prefillCount;
                hit += r.cacheHitTokenCount;
                miss += compute;
                squares += static_cast<double>(compute) * compute;
                products += static_cast<double>(r.cacheHitTokenCount) * compute;
            }
        }
        NaviSchedWorkloadSnapshot snapshot;
        ASSERT_TRUE(ownership.combine(emptyObserved(random() % 65), snapshot));
        EXPECT_EQ(live.size(), ownership.batchCount());
        EXPECT_EQ(count, snapshot.requestCount);
        EXPECT_EQ(prompt, snapshot.totalPromptTokenCount);
        EXPECT_EQ(prefillCount, snapshot.remainingPrefillRequestCount);
        EXPECT_EQ(hit, snapshot.remainingPrefillContextTokenCount);
        EXPECT_EQ(miss, snapshot.remainingPrefillTokenCount);
        EXPECT_DOUBLE_EQ(squares, snapshot.remainingPrefillTokenSquareSum);
        EXPECT_DOUBLE_EQ(products, snapshot.contextRemainingPrefillTokenProductSum);
    }
}

TEST(AgentWorkloadOwnershipTest, AcceptedRequestsPromoteWithoutDoubleCounting)
{
    AgentWorkloadOwnership ownership;
    ASSERT_TRUE(ownership.addUnbatchedRequest(11, ownership.prepareRequest(request(11, 100, 20))));
    ASSERT_TRUE(ownership.addUnbatchedRequest(12, ownership.prepareRequest(request(12, 200, 50))));
    EXPECT_FALSE(ownership.addUnbatchedRequest(11, ownership.prepareRequest(request(11, 100, 20))));
    NaviSchedWorkloadSnapshot snapshot;
    ASSERT_TRUE(ownership.combine(emptyObserved(1), snapshot));
    EXPECT_EQ(0, snapshot.batchCount);
    EXPECT_EQ(2, snapshot.requestCount);
    EXPECT_EQ(300, snapshot.totalPromptTokenCount);
    EXPECT_EQ(80, snapshot.runningCapacityPrefillTokenCount);
    EXPECT_EQ(150, snapshot.waitingCapacityPrefillTokenCount);
    EXPECT_FALSE(ownership.addBatch(1, {request(11, 101, 20)}));
    EXPECT_EQ(2u, ownership.requestCount());
    ASSERT_TRUE(ownership.addBatch(1, {request(11, 100, 20)}));
    ASSERT_TRUE(ownership.combine(emptyObserved(1), snapshot));
    EXPECT_EQ(1, snapshot.batchCount);
    EXPECT_EQ(2, snapshot.requestCount);
    EXPECT_EQ(300, snapshot.totalPromptTokenCount);
    EXPECT_DOUBLE_EQ(28900, snapshot.remainingPrefillTokenSquareSum);
    ASSERT_TRUE(ownership.removeUnbatchedRequests({11, 12, 12, 99}));
    ASSERT_TRUE(ownership.combine(emptyObserved(), snapshot));
    EXPECT_EQ(1, snapshot.requestCount);
    EXPECT_EQ(100, snapshot.totalPromptTokenCount);
    EXPECT_EQ(80, snapshot.remainingPrefillTokenCount);
    AgentWorkloadOwnership::RetiredBatches retired;
    ASSERT_TRUE(ownership.handoff({1}, retired));
    ASSERT_TRUE(ownership.combine(emptyObserved(), snapshot));
    EXPECT_EQ(0, snapshot.requestCount);
    EXPECT_EQ(0, snapshot.batchCount);
}

TEST(AgentWorkloadOwnershipTest, AdmissionAndPromotionWorkAtLargeBatchSizes)
{
    for (int count : {512, 4096, 20000}) {
        AgentWorkloadOwnership ownership;
        auto observed = emptyObserved(256);
        NaviSchedWorkloadSnapshot snapshot;
        std::vector<NaviSchedRequestStats> requests;
        requests.reserve(count);
        for (int i = 0; i < count; ++i) requests.emplace_back(request(i + 1, 3000, 0));
        const auto begin = std::chrono::steady_clock::now();
        for (const auto &value : requests) {
            ASSERT_TRUE(ownership.addUnbatchedRequest(value.requestId, ownership.prepareRequest(value)));
            ASSERT_TRUE(ownership.combine(observed, snapshot));
        }
        const auto accepted = std::chrono::steady_clock::now();
        EXPECT_EQ(count, snapshot.requestCount);
        EXPECT_EQ(0, snapshot.batchCount);
        auto prepared = ownership.prepareBatch(requests);
        const auto preparedTime = std::chrono::steady_clock::now();
        ASSERT_TRUE(ownership.addBatch(1, prepared));
        prepared.reset();
        const auto promoted = std::chrono::steady_clock::now();
        EXPECT_EQ(0u, ownership.unbatchedRequestCount());
        ASSERT_TRUE(ownership.combine(observed, snapshot));
        EXPECT_EQ(1, snapshot.batchCount);
        EXPECT_EQ(count, snapshot.requestCount);
        EXPECT_EQ(static_cast<int64_t>(count) * 3000, snapshot.remainingPrefillTokenCount);
        EXPECT_DOUBLE_EQ(static_cast<double>(count) * 9000000, snapshot.remainingPrefillTokenSquareSum);
        AgentWorkloadOwnership::RetiredBatches retired;
        const auto detachBegin = std::chrono::steady_clock::now();
        ASSERT_TRUE(ownership.handoff({1}, retired));
        const auto detached = std::chrono::steady_clock::now();
        retired.clear();
        const auto freed = std::chrono::steady_clock::now();
        auto micros = [](const auto &first, const auto &last) {
            return std::chrono::duration_cast<std::chrono::microseconds>(last - first).count();
        };
        std::cout << "ADMISSION_OWNERSHIP_BENCHMARK requests=" << count
                  << " accept_and_snapshot_us=" << micros(begin, accepted)
                  << " prepare_batch_us=" << micros(accepted, preparedTime)
                  << " promote_us=" << micros(preparedTime, promoted)
                  << " detach_us=" << micros(detachBegin, detached)
                  << " free_outside_lock_us=" << micros(detached, freed) << std::endl;
        ASSERT_TRUE(ownership.combine(observed, snapshot));
        EXPECT_EQ(0, snapshot.requestCount);
        EXPECT_EQ(0, snapshot.remainingPrefillTokenCount);
    }
}

}
}
