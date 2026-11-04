#include "rtp_llm/flexlb/navi_sched/resource/MasterBatchReservation.h"
#include "rtp_llm/flexlb/navi_sched/resource/PendingCacheUtil.h"

#include "unittest/unittest.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include "autil/legacy/jsonizable.h"

#include <chrono>
#include <fstream>
#include <set>
#include <thread>

namespace navi_sched {
namespace {

using RequestState = MasterBatchReservation::RequestState;

TEST(MasterPendingReplayTest, FullFailureSkipsPendingIndexConstruction)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation fixture;
    ASSERT_TRUE(fixture.restoreReplayState(state));
    ASSERT_EQ(512u, fixture.getRequestCount());
    std::vector<MasterBatchReservation::Request> requests;
    for (const auto &entry : fixture._requests) requests.push_back(entry.request);
    std::vector<size_t> indexes(requests.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    const NaviSchedCachePolicy policy{16, 1, false};
    for (bool indexed : {false, true}) {
        for (size_t repeat = 0; repeat < 5; ++repeat) {
            MasterBatchReservation batch;
            ASSERT_TRUE(batch.init(123, 1, 0, requests, &policy));
            if (indexed) ASSERT_TRUE(batch.buildPendingCacheReferences());
            const auto snapshot = batch;
            const auto checksum = PendingCacheUtil::reservationChecksum(
                snapshot.getPendingCacheReservation());
            MasterBatchReservation::Transition transition;
            const auto begin = std::chrono::steady_clock::now();
            ASSERT_TRUE(batch.markFailedIndexes(indexes, transition));
            const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - begin).count();
            std::cout << "FULL_FAILURE_BENCHMARK indexed=" << indexed
                      << " requests=512 repeat=" << repeat << " mark_us=" << elapsedUs
                      << std::endl;
            EXPECT_TRUE(transition.changed);
            EXPECT_TRUE(transition.fullBatchFastPath);
            EXPECT_TRUE(transition.retireCache);
            EXPECT_TRUE(transition.cacheChanged);
            EXPECT_TRUE(transition.resolved);
            EXPECT_TRUE(transition.removedFullReadyBlockHashes.empty());
            EXPECT_TRUE(transition.removedTailReadyBlockHashes.empty());
            EXPECT_FALSE(batch.hasPendingCacheReferences());
            EXPECT_EQ(0u, batch.getWaitingRequestCount());
            EXPECT_EQ(512u, batch.getFailedRequestCount());
            EXPECT_EQ(0, batch.getUnreportedInflightStats().requestCount);
            EXPECT_DOUBLE_EQ(0.0, batch.getUnreportedInflightStats().totalPredictedLatencyMs);
            EXPECT_EQ(0, batch.getWaitingFeatures().getBatchSize());
            EXPECT_TRUE(batch.getPendingCacheReservation().fullReadyBlockHashes.empty());
            EXPECT_EQ(512u, snapshot.getWaitingRequestCount());
            EXPECT_EQ(0u, snapshot.getFailedRequestCount());
            EXPECT_EQ(indexed, snapshot.hasPendingCacheReferences());
            EXPECT_EQ(checksum, PendingCacheUtil::reservationChecksum(
                                    snapshot.getPendingCacheReservation()));
            for (const auto &request : requests) {
                EXPECT_EQ(RequestState::FAILED, batch.getRequestState(request.requestId));
                EXPECT_EQ(RequestState::WAITING, snapshot.getRequestState(request.requestId));
            }
        }
    }
}

TEST(MasterPendingReplayTest, LargeRoundPendingIndexMatchesSurvivingRequests)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation fixture;
    ASSERT_TRUE(fixture.restoreReplayState(state));
    ASSERT_EQ(512u, fixture.getRequestCount());
    std::vector<MasterBatchReservation::Request> requests;
    for (uint64_t group = 0; group < 8; ++group) {
        for (const auto &entry : fixture._requests) {
            auto request = entry.request;
            request.requestId = requests.size() + 1;
            auto hashes = std::make_shared<NaviSchedBlockHashes>();
            for (auto hash : *request.blockHashes) {
                // A bijection preserves fixture sharing without sequential-key bias.
                auto value = hash + group * 1000000;
                value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
                value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
                hashes->push_back(value ^ (value >> 31));
            }
            request.blockHashes = std::move(hashes);
            requests.push_back(std::move(request));
        }
    }
    const NaviSchedCachePolicy policy{16, 1, false};
    std::vector<size_t> failed(3072);
    std::iota(failed.begin(), failed.end(), 0);
    const auto clock = [] { return std::chrono::steady_clock::now(); };
    const auto elapsed = [](auto begin) {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count();
    };
    for (size_t repeat = 0; repeat < 5; ++repeat) {
        MasterBatchReservation batch;
        auto begin = clock();
        ASSERT_TRUE(batch.init(123, 1, 0, requests, &policy));
        const auto initUs = elapsed(begin);
        const auto &pending = batch.getPendingCacheReservation();
        ASSERT_EQ(588352u, pending.fullReadyBlockHashes.size());
        PendingCacheState cache, agent;
        PendingCacheStateStats overlap;
        bool changed = false;
        begin = clock();
        ASSERT_TRUE(cache.upsertGenerated(pending, false, changed, &agent, &overlap));
        const auto publishUs = elapsed(begin);
        ASSERT_TRUE(changed);
        ASSERT_EQ(588352u, cache.getStats().uniqueBlockCount);
        ASSERT_EQ(0u, overlap.uniqueBlockCount);
        uint64_t hits = 0;
        begin = clock();
        for (const auto &request : requests) {
            for (auto hash : *request.blockHashes) hits += cache.containsFullReady(hash);
        }
        const auto matchUs = elapsed(begin);
        ASSERT_EQ(881352u, hits);
        MasterBatchReservation::Transition transition;
        begin = clock();
        ASSERT_TRUE(batch.markFailedIndexes(failed, transition));
        const auto markUs = elapsed(begin);
        ASSERT_EQ(441264u, transition.removedFullReadyBlockHashes.size());
        begin = clock();
        ASSERT_TRUE(cache.removeGeneratedBlocks(123, transition.removedFullReadyBlockHashes,
                                               transition.removedTailReadyBlockHashes,
                                               &agent, &overlap));
        const auto removeUs = elapsed(begin);
        ASSERT_EQ(147088u, cache.getStats().uniqueBlockCount);
        ASSERT_EQ(1u, cache.getStats().batchCount);
        NaviSchedPendingCacheReservation actual;
        ASSERT_TRUE(cache.getReservation(123, actual));
        EXPECT_TRUE(PendingCacheUtil::equalReservation(
            batch.getPendingCacheReservation(), actual));
        EXPECT_EQ(PendingCacheUtil::reservationChecksum(actual), cache.getStats().checksum);
        EXPECT_EQ(1024u, batch.getWaitingRequestCount());
        begin = clock();
        ASSERT_TRUE(cache.erase(123, changed, &agent, &overlap));
        const auto retireUs = elapsed(begin);
        EXPECT_TRUE(changed);
        EXPECT_EQ(0u, cache.getStats().uniqueBlockCount);
        EXPECT_EQ(0u, cache.getStats().batchCount);
        EXPECT_EQ(0u, cache.getStats().checksum);
        std::cout << "PENDING_INDEX_BENCHMARK repeat=" << repeat
                  << " requests=4096 blocks=881352 unique_blocks=588352 init_us=" << initUs
                  << " publish_us=" << publishUs << " match_us=" << matchUs
                  << " mark_failed_us=" << markUs << " remove_us=" << removeUs
                  << " retire_us=" << retireUs << std::endl;
    }
    for (size_t repeat = 0; repeat < 3; ++repeat) {
        MasterBatchReservation batch;
        ASSERT_TRUE(batch.init(123, 1, 0, requests, &policy));
        ASSERT_TRUE(batch.buildPendingCacheReferences());
        const auto elapsedNs = [](auto begin) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin).count();
        };
        int64_t copyNs = 0, markNs = 0, commitNs = 0;
        const auto transactionBegin = clock();
        for (size_t index = 0; index < requests.size(); ++index) {
            auto begin = clock();
            auto candidate = batch;
            copyNs += elapsedNs(begin);
            MasterBatchReservation::Transition transition;
            begin = clock();
            ASSERT_TRUE(candidate.markFailedIndexes({index}, transition));
            markNs += elapsedNs(begin);
            ASSERT_TRUE(transition.changed);
            EXPECT_EQ(RequestState::WAITING, batch._requests[index].state);
            EXPECT_EQ(requests.size() - index,
                      batch.getUnreportedInflightStats().requestCount);
            EXPECT_EQ(requests.size() - index - 1,
                      candidate.getUnreportedInflightStats().requestCount);
            begin = clock();
            batch = std::move(candidate);
            commitNs += elapsedNs(begin);
        }
        const auto totalUs = elapsed(transactionBegin);
        EXPECT_EQ(4096u, batch.getFailedRequestCount());
        EXPECT_EQ(0u, batch.getWaitingRequestCount());
        EXPECT_EQ(0, batch.getUnreportedInflightStats().requestCount);
        EXPECT_DOUBLE_EQ(0.0, batch.getUnreportedInflightStats().totalPredictedLatencyMs);
        EXPECT_TRUE(batch.getPendingCacheReservation().fullReadyBlockHashes.empty());
        EXPECT_TRUE(batch.getPendingCacheReservation().tailReadyBlockHashes.empty());
        EXPECT_TRUE(batch.isResolved());
        std::cout << "SINGLE_FAILURE_TRANSACTION_BENCHMARK repeat=" << repeat
                  << " requests=4096 copy_us=" << copyNs / 1000
                  << " mark_us=" << markNs / 1000 << " commit_us=" << commitNs / 1000
                  << " total_us=" << totalUs << std::endl;
    }
}

TEST(MasterPendingReplayTest, PrefixTopologyIsSharedWhileReferencesRemainIndependent)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation original;
    ASSERT_TRUE(original.restoreReplayState(state));
    ASSERT_EQ(73544u, original._pendingReferences.size());
    const auto begin = std::chrono::steady_clock::now();
    uint64_t copiedNodes = 0;
    for (size_t i = 0; i < 128; ++i) {
        auto copy = original;
        copiedNodes += copy._pendingReferences.size();
    }
    EXPECT_EQ(128u * 73544u, copiedNodes);
    std::cout << "RESERVATION_COPY_BENCHMARK copies=128 elapsed_us="
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - begin).count()
              << " ref_bytes=" << sizeof(MasterBatchReservation::BlockReferences) * original._pendingReferences.size()
              << std::endl;
    ASSERT_TRUE(original._pendingBlockHashes);
    EXPECT_EQ(73544u, original._pendingBlockHashes->size());
    auto copy = original;
    EXPECT_EQ(original._pendingBlockHashes.get(), copy._pendingBlockHashes.get());
    EXPECT_EQ(original._requestBlockIndexes.get(), copy._requestBlockIndexes.get());
    ASSERT_EQ(110169u, original._requestBlockIndexes->size());
    const auto waiting = std::find_if(original._requests.begin(), original._requests.end(),
        [](const auto &entry) { return entry.state == RequestState::WAITING; });
    ASSERT_NE(original._requests.end(), waiting);
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(copy.markFailed(waiting->request.requestId, transition));
    EXPECT_EQ(232u, original.getWaitingRequestCount());
    EXPECT_EQ(231u, copy.getWaitingRequestCount());
    EXPECT_EQ(33327u, original.getPendingCacheReservation().fullReadyBlockHashes.size());
    MasterBatchReservation restored;
    ASSERT_TRUE(restored.restoreReplayState(copy.makeReplayState()));
    EXPECT_TRUE(PendingCacheUtil::equalReservation(copy.getPendingCacheReservation(),
                                                 restored.getPendingCacheReservation()));
    EXPECT_EQ(231u, restored.getWaitingRequestCount());
    EXPECT_EQ(original._pendingBlockHashes.get(), copy._pendingBlockHashes.get());
}

TEST(MasterPendingReplayTest, ConcurrentCopiesRetireSharedPrefixesIndependently)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation original;
    ASSERT_TRUE(original.restoreReplayState(state));
    std::vector<uint64_t> waiting;
    for (const auto &entry : original._requests) {
        if (entry.state == RequestState::WAITING) waiting.push_back(entry.request.requestId);
    }
    ASSERT_EQ(232u, waiting.size());
    std::array<MasterBatchReservation, 4> copies;
    std::vector<std::thread> threads;
    for (size_t worker = 0; worker < copies.size(); ++worker) {
        copies[worker] = original;
        ASSERT_EQ(original._pendingBlockHashes.get(), copies[worker]._pendingBlockHashes.get());
    }
    for (size_t worker = 0; worker < copies.size(); ++worker) {
        threads.emplace_back([&, worker] {
            auto &copy = copies[worker];
            MasterBatchReservation::Transition transition;
            for (size_t i = 0; i < waiting.size(); ++i) {
                const auto requestId = waiting[(i + worker * 59) % waiting.size()];
                ASSERT_TRUE(copy.markFailed(requestId, transition));
                EXPECT_EQ(231u - i, copy.getWaitingRequestCount());
            }
        });
    }
    for (auto &thread : threads) thread.join();
    EXPECT_EQ(232u, original.getWaitingRequestCount());
    EXPECT_EQ(280u, original.getFailedRequestCount());
    EXPECT_EQ(33327u, original.getPendingCacheReservation().fullReadyBlockHashes.size());
    for (const auto &copy : copies) {
        EXPECT_TRUE(copy.isResolved());
        EXPECT_EQ(512u, copy.getFailedRequestCount());
        EXPECT_EQ(0, copy.getUnreportedInflightStats().requestCount);
        EXPECT_TRUE(copy.getPendingCacheReservation().fullReadyBlockHashes.empty());
        EXPECT_EQ(original._pendingBlockHashes.get(), copy._pendingBlockHashes.get());
    }
}

TEST(MasterPendingReplayTest, RequestCopiesShareUntouchedPagesOnly)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation original;
    ASSERT_TRUE(original.restoreReplayState(state));
    auto copy = original;
    ASSERT_EQ(512u, original._requests.size());
    EXPECT_EQ(&original._requests[0], &copy._requests[0]);
    EXPECT_EQ(&original._requests[511], &copy._requests[511]);
    const auto waiting = std::find_if(original._requests.begin(), original._requests.end(),
        [](const auto &entry) { return entry.state == RequestState::WAITING; });
    ASSERT_NE(original._requests.end(), waiting);
    const auto requestId = waiting->request.requestId;
    const auto index = original.findRequest(requestId);
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(copy.markFailed(requestId, transition));
    EXPECT_EQ(RequestState::WAITING, original._requests[index].state);
    EXPECT_EQ(RequestState::FAILED, copy._requests[index].state);
    EXPECT_NE(&original._requests[index], &copy._requests[index]);
    size_t shared = 0;
    for (size_t i = 0; i < original._requests.size(); ++i) {
        shared += &original._requests[i] == &copy._requests[i];
    }
    EXPECT_GE(shared, original._requests.size() - 32);
    EXPECT_EQ(232u, original.getWaitingRequestCount());
    EXPECT_EQ(231u, copy.getWaitingRequestCount());
}

TEST(MasterPendingReplayTest, ReferenceCopiesShareUntouchedPagesOnly)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation original;
    ASSERT_TRUE(original.restoreReplayState(state));
    const auto &originalNodes = original._pendingReferences;
    auto copy = original;
    const auto &copyNodes = copy._pendingReferences;
    ASSERT_EQ(73544u, originalNodes.size());
    EXPECT_EQ(&originalNodes[0], &copyNodes[0]);
    const auto waiting = std::find_if(original._requests.begin(), original._requests.end(),
        [](const auto &entry) { return entry.state == RequestState::WAITING; });
    ASSERT_NE(original._requests.end(), waiting);
    const auto leaf = (*original._requestBlockIndexes)[
        waiting->blockIndexBegin + waiting->request.blockHashes->size() - 1];
    const auto before = originalNodes[leaf].fullRefCount;
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(copy.markFailed(waiting->request.requestId, transition));
    EXPECT_EQ(before, originalNodes[leaf].fullRefCount);
    EXPECT_EQ(before - 1, copyNodes[leaf].fullRefCount);
    EXPECT_NE(&originalNodes[leaf], &copyNodes[leaf]);
    size_t shared = 0;
    for (size_t index = 0; index < originalNodes.size(); ++index) {
        shared += &originalNodes[index] == &copyNodes[index];
    }
    EXPECT_GT(shared, originalNodes.size() / 2);
    EXPECT_TRUE(PendingCacheUtil::equalReservation(original.getPendingCacheReservation(),
        [&] {
            MasterBatchReservation restored;
            EXPECT_TRUE(restored.restoreReplayState(state));
            return restored.getPendingCacheReservation();
        }()));
    std::cout << "REFERENCE_PAGE_BENCHMARK total_nodes=" << originalNodes.size()
              << " shared_nodes=" << shared << " detached_nodes="
              << originalNodes.size() - shared << std::endl;
}

TEST(MasterPendingReplayTest, SharedPagesIsolateDirectoryBoundariesAndAppendedBranches)
{
    MasterBatchReservation::SharedPages<uint64_t, 2> original;
    for (uint64_t value = 0; value < 302; ++value) original.push_back(value);
    auto branch = original;
    const std::set<size_t> changedPages{0, 63, 64, 128};
    const std::set<size_t> changedIndexes{0, 127, 128, 257};
    for (auto index : changedIndexes) branch.writable(index) = 10000 + index;
    for (size_t index = 0; index < original.size(); ++index) {
        EXPECT_EQ(index, original[index]);
        EXPECT_EQ(changedIndexes.count(index) ? 10000 + index : index, branch[index]);
        EXPECT_EQ(changedPages.count(index / 2) == 0,
                  &original[index] == &branch[index]);
    }
    auto fork = branch;
    fork.writable(129) = 50000;
    fork.writable(260) = 60000;
    EXPECT_EQ(129u, branch[129]);
    EXPECT_EQ(260u, branch[260]);
    EXPECT_EQ(50000u, fork[129]);
    EXPECT_EQ(60000u, fork[260]);
    branch.push_back(70000);
    fork.push_back(80000);
    fork.push_back(80001);
    EXPECT_EQ(302u, original.size());
    EXPECT_EQ(303u, branch.size());
    EXPECT_EQ(304u, fork.size());
    EXPECT_EQ(70000u, branch[302]);
    EXPECT_EQ(80000u, fork[302]);
    EXPECT_EQ(80001u, fork[303]);
    auto moved = std::move(branch);
    EXPECT_TRUE(branch.empty());
    branch.push_back(77);
    EXPECT_EQ(77u, branch[0]);
    EXPECT_EQ(70000u, moved[302]);
    fork.clear();
    fork.push_back(88);
    EXPECT_EQ(1u, fork.size());
    EXPECT_EQ(88u, fork[0]);
    EXPECT_EQ(129u, original[129]);
    EXPECT_EQ(10000u, moved[0]);
}

TEST(MasterPendingReplayTest, RealBatchFailureTransactionsDrainAllReferences)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.restoreReplayState(state));
    std::vector<uint64_t> waiting;
    for (const auto &entry : reservation._requests) {
        if (entry.state == RequestState::WAITING) {
            waiting.push_back(entry.request.requestId);
        }
    }
    ASSERT_EQ(232u, waiting.size());
    const auto begin = std::chrono::steady_clock::now();
    for (auto id : waiting) {
        auto candidate = reservation;
        MasterBatchReservation::Transition transition;
        ASSERT_TRUE(candidate.markFailed(std::vector<uint64_t>{id}, transition));
        ASSERT_TRUE(transition.changed);
        reservation = std::move(candidate);
    }
    const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin).count();
    EXPECT_EQ(0u, reservation.getWaitingRequestCount());
    EXPECT_EQ(512u, reservation.getFailedRequestCount());
    EXPECT_EQ(0, reservation.getUnreportedInflightStats().requestCount);
    EXPECT_TRUE(reservation.isResolved());
    EXPECT_TRUE(reservation.getPendingCacheReservation().fullReadyBlockHashes.empty());
    EXPECT_TRUE(reservation.getPendingCacheReservation().tailReadyBlockHashes.empty());
    EXPECT_TRUE(reservation.getPendingCacheReservation().fullReadyBlockHashes.empty());
    std::cout << "FAILURE_TRANSACTION_BENCHMARK calls=" << waiting.size()
              << " elapsed_us=" << elapsedUs << std::endl;
}

TEST(MasterPendingReplayTest, RealBatchInflightHandoffCopyCost)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation original;
    ASSERT_TRUE(original.restoreReplayState(state));
    ASSERT_EQ(73544u, original._pendingReferences.size());
    const auto waiting = std::find_if(original._requests.begin(), original._requests.end(),
        [&](const auto &entry) {
            if (entry.state != RequestState::WAITING) return false;
            auto candidate = original;
            bool changed = false;
            return candidate.markInflightReported(entry.request.requestId, changed) && changed;
        });
    ASSERT_NE(original._requests.end(), waiting);
    const auto requestId = waiting->request.requestId;
    int64_t copyNs = 0;
    int64_t directNs = 0;
    for (size_t repeat = 0; repeat < 128; ++repeat) {
        auto oldPath = original;
        auto newPath = original;
        bool oldChanged = false;
        bool newChanged = false;
        const auto oldBegin = std::chrono::steady_clock::now();
        {
            auto candidate = oldPath;
            ASSERT_TRUE(candidate.markInflightReported(requestId, oldChanged));
            oldPath = std::move(candidate);
        }
        copyNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - oldBegin).count();
        const auto newBegin = std::chrono::steady_clock::now();
        ASSERT_TRUE(newPath.markInflightReported(requestId, newChanged));
        directNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - newBegin).count();
        EXPECT_EQ(oldChanged, newChanged);
        EXPECT_TRUE(newChanged);
        EXPECT_EQ(oldPath.getUnreportedInflightStats().requestCount,
                  newPath.getUnreportedInflightStats().requestCount);
        EXPECT_EQ(original._pendingReferences.size(), newPath._pendingReferences.size());
    }
    std::cout << "INFLIGHT_HANDOFF_BENCHMARK calls=128 copy_ns=" << copyNs
              << " direct_ns=" << directNs << std::endl;
}

TEST(MasterPendingReplayTest, RequestIndexIsSharedWhileTransitionsRemainIndependent)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.restoreReplayState(state));
    ASSERT_EQ(512u, reservation.getRequestCount());
    const auto begin = std::chrono::steady_clock::now();
    uint64_t sum = 0;
    for (size_t repeat = 0; repeat < 256; ++repeat) {
        for (uint64_t id = 1; id <= 512; ++id) {
            sum += reservation.findRequest(id);
        }
    }
    const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin).count();
    EXPECT_EQ(256u * 512u * 511u / 2u, sum);
    EXPECT_EQ(512u, reservation.findRequest(0));
    EXPECT_EQ(512u, reservation.findRequest(513));
    std::cout << "REQUEST_LOOKUP_BENCHMARK lookups=131072 elapsed_us=" << elapsedUs
              << std::endl;
    ASSERT_TRUE(reservation._requestIndexes);
    EXPECT_EQ(512u, reservation._requestIndexes->size());
    auto copy = reservation;
    EXPECT_EQ(reservation._requestIndexes.get(), copy._requestIndexes.get());
    auto waiting = std::find_if(reservation._requests.begin(), reservation._requests.end(),
                               [](const auto &entry) { return entry.state == RequestState::WAITING; });
    ASSERT_NE(reservation._requests.end(), waiting);
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(copy.markFailed(waiting->request.requestId, transition));
    EXPECT_EQ(RequestState::WAITING, reservation.getRequestState(waiting->request.requestId));
    EXPECT_EQ(RequestState::FAILED, copy.getRequestState(waiting->request.requestId));
    EXPECT_EQ(reservation._requestIndexes.get(), copy._requestIndexes.get());
}

TEST(MasterPendingReplayTest, ExplicitCompletionDefersMaterializationAndRestoresState)
{
    MasterBatchReservation original;
    NaviSchedCachePolicy policy{16, 1, false};
    std::vector<MasterBatchReservation::Request> requests;
    for (uint64_t id = 1; id <= 64; ++id) {
        auto blocks = std::make_shared<NaviSchedBlockHashes>();
        for (uint64_t block = 0; block < 64; ++block) blocks->push_back(id * 1000 + block);
        requests.push_back({id, 1024, 16, 0, blocks, 2.0});
    }
    ASSERT_TRUE(original.init(101, 1000000, 128, std::move(requests), &policy));
    ASSERT_TRUE(original.buildPendingCacheReferences());
    NaviSchedLocalBatchInfo local;
    local.scheduleBatchId = 101;
    local.localBatchId = 201;
    local.requestCount = 32;
    local.pendingCacheTracked = true;
    for (uint64_t id = 1; id <= 32; ++id) local.requestIds.push_back(id);
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(original.markAdmitted(local, transition));
    std::vector<uint64_t> acknowledged;
    ASSERT_TRUE(original.acknowledgeDispatch(local, acknowledged));
    bool changed = false;
    ASSERT_TRUE(original.markInflightReported(1, changed));
    ASSERT_TRUE(changed);
    const auto state = original.makeReplayState();
    MasterBatchReservation restored;
    ASSERT_TRUE(restored.restoreReplayState(state));
    EXPECT_EQ(original.getWaitingRequestCount(), restored.getWaitingRequestCount());
    EXPECT_EQ(32u, restored.getDispatchedRequestCount());
    EXPECT_EQ(63, restored.getUnreportedInflightStats().requestCount);
    EXPECT_EQ(4096u, restored.getPendingCacheReservation().fullReadyBlockHashes.size());
    const auto before = restored._pendingRebuildCount;
    const auto visited = restored._pendingRebuildVisitedNodes;
    ASSERT_TRUE(restored.markCompleted(local, true, transition));
    EXPECT_EQ(0u, restored._pendingRebuildCount - before);
    EXPECT_EQ(0u, restored._pendingRebuildVisitedNodes - visited);
    EXPECT_EQ(32u, restored.getCompletedRequestCount());
    EXPECT_EQ(32u, restored.getWaitingRequestCount());
    EXPECT_EQ(2048u, restored.getPendingCacheReservation().fullReadyBlockHashes.size());
    MasterBatchReservation afterCompletion;
    ASSERT_TRUE(afterCompletion.restoreReplayState(restored.makeReplayState()));
    EXPECT_TRUE(PendingCacheUtil::equalReservation(restored.getPendingCacheReservation(),
                                                 afterCompletion.getPendingCacheReservation()));
    EXPECT_EQ(32u, afterCompletion.getCompletedRequestCount());
    for (uint64_t id = 33; id <= 64; ++id) {
        ASSERT_TRUE(afterCompletion.markFailed(id, transition));
    }
    EXPECT_TRUE(afterCompletion.isResolved());
    EXPECT_EQ(0u, afterCompletion.getPendingCacheReservation().fullReadyBlockHashes.size());
    EXPECT_EQ(0, afterCompletion.getUnreportedInflightStats().requestCount);
    auto malformed = state;
    malformed.erase("requests");
    EXPECT_FALSE(afterCompletion.restoreReplayState(malformed));
    EXPECT_TRUE(afterCompletion.isResolved());
}

TEST(MasterPendingReplayTest, ZeroBlockFailureDoesNotMaterializePendingVectors)
{
    MasterBatchReservation reservation;
    NaviSchedCachePolicy policy{16, 1, false};
    std::vector<MasterBatchReservation::Request> requests;
    for (uint64_t id = 1; id <= 64; ++id) {
        auto blocks = std::make_shared<NaviSchedBlockHashes>();
        for (uint64_t block = 0; block < 64; ++block) {
            blocks->push_back(id * 1000 + block);
        }
        requests.push_back({id, 1024, 16, 0, blocks, 2.0});
    }
    requests.push_back({65, 1, 16, 0, {}, 1.0});
    ASSERT_TRUE(reservation.init(101, 1000000, 129, std::move(requests), &policy));
    ASSERT_TRUE(reservation.buildPendingCacheReferences());
    std::vector<uint64_t> failed(63);
    std::iota(failed.begin(), failed.end(), 1);
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(failed, transition));
    ASSERT_EQ(64u, reservation.getPendingCacheReservation().fullReadyBlockHashes.size());
    const auto visited = reservation._pendingRebuildVisitedNodes;
    ASSERT_TRUE(reservation.markFailed(65, transition));
    EXPECT_EQ(0u, reservation._pendingRebuildVisitedNodes - visited);
    EXPECT_EQ(4096u, reservation._pendingBlockHashes->size());
    EXPECT_EQ(1u, reservation.getWaitingRequestCount());
    NaviSchedBlockHashes expected(64);
    std::iota(expected.begin(), expected.end(), 64000);
    EXPECT_EQ(expected, reservation.getPendingCacheReservation().fullReadyBlockHashes);
    ASSERT_TRUE(reservation.markFailed(64, transition));
    EXPECT_TRUE(reservation.isResolved());
    EXPECT_TRUE(reservation.getPendingCacheReservation().fullReadyBlockHashes.empty());
    EXPECT_TRUE(reservation.getPendingCacheReservation().fullReadyBlockHashes.empty());
}

TEST(MasterPendingReplayTest, RealBatchRetirementDoesNotScanUnchangedPrefixes)
{
    std::ifstream input("rtp_llm/flexlb/navi_sched/testdata/pending_cache/master-batch-3516.json");
    ASSERT_TRUE(input.is_open());
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    autil::legacy::json::JsonMap state;
    autil::legacy::FromJsonString(state, json);
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.restoreReplayState(state));
    ASSERT_EQ(512u, reservation.getRequestCount());
    ASSERT_EQ(280u, reservation.getFailedRequestCount());
    ASSERT_EQ(232u, reservation.getWaitingRequestCount());
    ASSERT_EQ(33327u, reservation.getPendingCacheReservation().fullReadyBlockHashes.size());
    ASSERT_EQ(73544u, reservation._pendingReferences.size());
    const auto before = reservation._pendingRebuildVisitedNodes;
    std::vector<uint64_t> waiting;
    for (const auto &entry : reservation._requests) {
        if (entry.state == RequestState::WAITING) waiting.push_back(entry.request.requestId);
    }
    MasterBatchReservation::Transition transition;
    int64_t transitionElapsedUs = 0;
    for (auto id : waiting) {
        const auto shapeCount = reservation._pendingBlockHashes->size();
        const auto visited = reservation._pendingRebuildVisitedNodes;
        const auto start = std::chrono::steady_clock::now();
        ASSERT_TRUE(reservation.markFailed(id, transition));
        transitionElapsedUs += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        EXPECT_EQ(0u, reservation._pendingRebuildVisitedNodes - visited);
        EXPECT_EQ(73544u, shapeCount);
        std::set<uint64_t> expected;
        for (const auto &entry : reservation._requests) {
            if (entry.state == RequestState::WAITING && entry.request.blockHashes) {
                expected.insert(entry.request.blockHashes->begin(), entry.request.blockHashes->end());
            }
        }
        EXPECT_EQ(NaviSchedBlockHashes(expected.begin(), expected.end()),
                  reservation.getPendingCacheReservation().fullReadyBlockHashes);
    }
    const auto visits = reservation._pendingRebuildVisitedNodes - before;
    EXPECT_LE(visits, 73544u * waiting.size());
    EXPECT_TRUE(reservation.getPendingCacheReservation().fullReadyBlockHashes.empty());
    EXPECT_TRUE(reservation.isResolved());
    EXPECT_EQ(512u, reservation.getFailedRequestCount());
    std::cerr << "real_pending_retirement requests=" << waiting.size()
              << " visited_nodes=" << visits
              << " transition_elapsed_us=" << transitionElapsedUs << std::endl;
}

TEST(PendingCacheReplacementTest, DifferentialUpdatesMatchFreshState)
{
    PendingCacheState state;
    PendingCacheState other;
    PendingCacheStateStats overlap;
    std::vector<NaviSchedPendingCacheReservation> reservations;
    bool changed = false;
    for (int64_t id = 1; id <= 4; ++id) {
        NaviSchedPendingCacheReservation reservation;
        reservation.batchId = id;
        reservation.scheduleBatchId = id;
        reservation.fullReadyBlockHashes = {1, 2, static_cast<uint64_t>(id + 10)};
        reservation.tailReadyBlockHashes = {2};
        reservations.push_back(reservation);
        ASSERT_TRUE(state.upsert(reservation, false, changed, &other, &overlap));
        ASSERT_TRUE(other.upsert(reservation, false, changed));
    }
    overlap.batchCount = 4;
    overlap.uniqueBlockCount = 6;
    overlap.uniqueTailReadyBlockCount = 1;
    for (size_t round = 0; round < 100; ++round) {
        auto &reservation = reservations[round % reservations.size()];
        reservation.fullReadyBlockHashes = {1, 2, 20 + round};
        reservation.tailReadyBlockHashes = round % 2 ? std::vector<uint64_t>{2}
                                                     : std::vector<uint64_t>{20 + round};
        if (round % 7 == 0) {
            ++reservation.scheduleBatchId;
        }
        ASSERT_TRUE(state.upsert(reservation, true, changed, &other, &overlap));
        ASSERT_TRUE(changed);
        PendingCacheState fresh;
        ASSERT_TRUE(fresh.replace(reservations));
        EXPECT_EQ(fresh.getStats().checksum, state.getStats().checksum);
        EXPECT_EQ(fresh.getStats().uniqueBlockCount, state.getStats().uniqueBlockCount);
        EXPECT_EQ(fresh.getStats().uniqueTailReadyBlockCount,
                  state.getStats().uniqueTailReadyBlockCount);
        EXPECT_EQ(fresh.countBatchOverlap(other), overlap.batchCount);
        EXPECT_EQ(fresh.countFullReadyOverlap(other), overlap.uniqueBlockCount);
        EXPECT_EQ(fresh.countTailReadyOverlap(other), overlap.uniqueTailReadyBlockCount);
    }
}

NaviSchedBlockHashesConstPtr hashes(std::initializer_list<uint64_t> values)
{
    return std::make_shared<const NaviSchedBlockHashes>(values);
}

MasterBatchReservation::Request makeRequest(uint64_t requestId, int64_t tokenCount,
                                            int64_t blockSize, int64_t hitTokens,
                                            NaviSchedBlockHashesConstPtr blockHashes,
                                            double predictedLatencyMs = 0.0)
{
    return {requestId,         tokenCount, blockSize, hitTokens, std::move(blockHashes),
            predictedLatencyMs};
}

NaviSchedLocalBatchInfo makeFullBatch(int64_t scheduleBatchId, int64_t localBatchId,
                                      int32_t requestCount)
{
    NaviSchedLocalBatchInfo batch;
    batch.scheduleBatchId = scheduleBatchId;
    batch.localBatchId = localBatchId;
    batch.requestCount = requestCount;
    batch.pendingCacheTracked = true;
    return batch;
}

NaviSchedLocalBatchInfo makePartialBatch(int64_t scheduleBatchId, int64_t localBatchId,
                                         std::initializer_list<uint64_t> requestIds)
{
    NaviSchedLocalBatchInfo batch;
    batch.scheduleBatchId = scheduleBatchId;
    batch.localBatchId = localBatchId;
    batch.requestCount = static_cast<int32_t>(requestIds.size());
    batch.pendingCacheTracked = true;
    batch.requestIds.assign(requestIds);
    return batch;
}

TEST(MasterBatchReservationRetirementTest, RepeatedCancellationPreservesHashUnion)
{
    constexpr size_t REQUESTS = 128;
    constexpr size_t BLOCKS = 256;
    MasterBatchReservation reservation;
    std::vector<MasterBatchReservation::Request> requests;
    for (size_t request = 0; request < REQUESTS; ++request) {
        auto blockHashes = std::make_shared<NaviSchedBlockHashes>();
        for (size_t block = 0; block < BLOCKS; ++block) {
            blockHashes->push_back((1 + request * BLOCKS + block) * 0x9e3779b97f4a7c15ULL);
        }
        requests.push_back(makeRequest(request + 1, BLOCKS * 16, 16, 0, blockHashes));
    }
    const NaviSchedCachePolicy policy{16, 1, false};
    ASSERT_TRUE(reservation.init(123, 1, 1, std::move(requests), &policy));
    PendingCacheState cache;
    bool changed = false;
    ASSERT_TRUE(cache.upsert(reservation.getPendingCacheReservation(), false, changed));
    int64_t elapsedUs = 0;
    for (size_t request = 0; request < REQUESTS; ++request) {
        const auto begin = std::chrono::steady_clock::now();
        MasterBatchReservation::Transition transition;
        ASSERT_TRUE(reservation.markFailed(request + 1, transition));
        if (transition.resolved) {
            ASSERT_TRUE(cache.erase(123, changed));
        } else {
            ASSERT_TRUE(cache.upsert(reservation.getPendingCacheReservation(), true, changed));
        }
        elapsedUs += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count();
        const auto &pending = reservation.getPendingCacheReservation();
        EXPECT_EQ((REQUESTS - request - 1) * BLOCKS, pending.fullReadyBlockHashes.size());
        EXPECT_TRUE(std::is_sorted(pending.fullReadyBlockHashes.begin(),
                                   pending.fullReadyBlockHashes.end()));
        EXPECT_EQ(pending.fullReadyBlockHashes.size(), cache.getStats().uniqueBlockCount);
        EXPECT_EQ(pending.fullReadyBlockHashes.empty() ? 0 :
                  PendingCacheUtil::reservationChecksum(pending), cache.getStats().checksum);
    }
    std::cout << "RETIREMENT_BENCHMARK requests=" << REQUESTS << " blocks_per_request="
              << BLOCKS << " elapsed_us=" << elapsedUs << std::endl;
    EXPECT_TRUE(reservation.isResolved());
    EXPECT_EQ(REQUESTS, reservation.getFailedRequestCount());
}

TEST(MasterBatchReservationBulkFailureTest, AllFailedRequestsClearFeatures)
{
    MasterBatchReservation reservation;
    std::vector<MasterBatchReservation::Request> requests;
    std::vector<uint64_t> ids(30);
    std::iota(ids.begin(), ids.end(), uint64_t{1});
    for (auto id : ids) {
        requests.push_back(makeRequest(id, 32, 16, 0, hashes({id, id + 100}), 1.0));
    }
    const NaviSchedCachePolicy policy{16, 1, false};
    ASSERT_TRUE(reservation.init(101, 1, 30.0, std::move(requests), &policy));
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(ids, transition));
    EXPECT_TRUE(transition.changed);
    EXPECT_TRUE(transition.resolved);
    EXPECT_EQ(30, reservation.getFailedRequestCount());
    EXPECT_EQ(0, reservation.getWaitingRequestCount());
    EXPECT_EQ(0, reservation.getUnreportedInflightStats().requestCount);
    EXPECT_TRUE(reservation.getPendingCacheReservation().fullReadyBlockHashes.empty());
}

TEST(MasterBatchReservationBulkFailureTest, PartialFailurePreservesSurvivorFeatures)
{
    MasterBatchReservation reservation;
    const NaviSchedCachePolicy policy{16, 1, false};
    ASSERT_TRUE(reservation.init(101, 1, 30.0,
        {makeRequest(1, 32, 16, 0, hashes({11, 12}), 10.0),
         makeRequest(2, 32, 16, 0, hashes({11, 13}), 10.0),
         makeRequest(3, 32, 16, 16, hashes({21, 22}), 10.0)}, &policy));
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(std::vector<uint64_t>{1, 1, 2}, transition));
    EXPECT_EQ(2, reservation.getFailedRequestCount());
    EXPECT_EQ(1, reservation.getWaitingRequestCount());
    EXPECT_EQ(RequestState::WAITING, reservation.getRequestState(3));
    EXPECT_EQ((NaviSchedBlockHashes{21, 22}),
              reservation.getPendingCacheReservation().fullReadyBlockHashes);
    EXPECT_EQ(16, reservation.getWaitingFeatures().getCacheHitTokenCount());
    EXPECT_EQ(16, reservation.getWaitingFeatures().getCacheMissTokenCount());
    EXPECT_EQ(1, reservation.getUnreportedInflightStats().requestCount);
    EXPECT_FALSE(transition.resolved);
}

TEST(MasterBatchReservationBulkFailureTest, UnknownRequestDoesNotChangeKnownRequests)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(101, 1, 10.0,
        {makeRequest(1, 32, 16, 0, hashes({11, 12}), 10.0)}, nullptr));
    MasterBatchReservation::Transition transition;
    EXPECT_FALSE(reservation.markFailed(std::vector<uint64_t>{1, 99}, transition));
    EXPECT_FALSE(transition.changed);
    EXPECT_EQ(RequestState::WAITING, reservation.getRequestState(1));
    EXPECT_EQ(1, reservation.getWaitingRequestCount());
    EXPECT_EQ(0, reservation.getFailedRequestCount());
    EXPECT_EQ(1, reservation.getUnreportedInflightStats().requestCount);
}

void expectFeatures(const NaviSchedLatencyFeatures &features, int64_t batchCount,
                    int64_t batchSize, int64_t hitTokens, int64_t missTokens,
                    double missSquareSum, double hitMissProductSum)
{
    EXPECT_EQ(batchCount, features.getBatchCount());
    EXPECT_EQ(batchSize, features.getBatchSize());
    EXPECT_EQ(hitTokens, features.getCacheHitTokenCount());
    EXPECT_EQ(missTokens, features.getCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(missSquareSum, features.getCacheMissTokenSquareSum());
    EXPECT_DOUBLE_EQ(hitMissProductSum, features.getCacheHitMissTokenProductSum());
}

void expectCapacityFeatures(const NaviSchedLatencyFeatures &features, int64_t batchCount,
                            int64_t runningBatchSize, int64_t runningHitTokens,
                            int64_t runningMissTokens, double runningMissSquareSum,
                            double runningHitMissProductSum, int64_t waitingBatchSize,
                            int64_t waitingHitTokens, int64_t waitingMissTokens,
                            double waitingMissSquareSum, double waitingHitMissProductSum,
                            int64_t precedingRequestCount = 0)
{
    EXPECT_TRUE(features.usesCapacitySplit());
    EXPECT_EQ(batchCount, features.getBatchCount());
    EXPECT_EQ(precedingRequestCount, features.getPrecedingRequestCount());
    EXPECT_EQ(runningBatchSize, features.getRunningBatchSize());
    EXPECT_EQ(runningHitTokens, features.getRunningCacheHitTokenCount());
    EXPECT_EQ(runningMissTokens, features.getRunningCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(runningMissSquareSum, features.getRunningCacheMissTokenSquareSum());
    EXPECT_DOUBLE_EQ(runningHitMissProductSum,
                     features.getRunningCacheHitMissTokenProductSum());
    EXPECT_EQ(waitingBatchSize, features.getWaitingBatchSize());
    EXPECT_EQ(waitingHitTokens, features.getWaitingCacheHitTokenCount());
    EXPECT_EQ(waitingMissTokens, features.getWaitingCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(waitingMissSquareSum, features.getWaitingCacheMissTokenSquareSum());
    EXPECT_DOUBLE_EQ(waitingHitMissProductSum,
                     features.getWaitingCacheHitMissTokenProductSum());
}

void expectHashes(const NaviSchedPendingCacheReservation &reservation,
                  std::initializer_list<uint64_t> full,
                  std::initializer_list<uint64_t> tail = {})
{
    EXPECT_EQ(std::vector<uint64_t>(full), reservation.fullReadyBlockHashes);
    EXPECT_EQ(std::vector<uint64_t>(tail), reservation.tailReadyBlockHashes);
    if (full.size() == 0) {
        EXPECT_EQ(0, reservation.batchId);
        EXPECT_EQ(0, reservation.scheduleBatchId);
    } else {
        EXPECT_GT(reservation.batchId, 0);
        EXPECT_EQ(reservation.batchId, reservation.scheduleBatchId);
    }
}

class MasterBatchReservationTest : public TESTBASE {};

TEST_F(MasterBatchReservationTest, testAcceptedWorkloadDoesNotAcknowledgeBatchReady)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(101, 1, 10.0,
        {makeRequest(1, 32, 16, 0, hashes({11, 12}), 10.0),
         makeRequest(2, 32, 16, 0, hashes({11, 12}), 10.0)}, nullptr));
    ASSERT_TRUE(reservation.markAcceptedWorkloadReported(1));
    EXPECT_TRUE(reservation.isAwaitingBatchReady(1));
    EXPECT_FALSE(reservation.isDispatchAcknowledged(1));
    EXPECT_EQ(1u, reservation.getWaitingRequestCount());
    EXPECT_EQ(1u, reservation.getDispatchedRequestCount());
    EXPECT_EQ(1, reservation.getWaitingFeatures().getBatchSize());
    EXPECT_EQ(32, reservation.getWaitingFeatures().getCacheMissTokenCount());
    ASSERT_TRUE(reservation.markAcceptedWorkloadReported(1));
    EXPECT_EQ(1, reservation.getWaitingFeatures().getBatchSize());
    MasterBatchReservation restored;
    ASSERT_TRUE(restored.restoreReplayState(reservation.makeReplayState()));
    EXPECT_TRUE(restored.isAwaitingBatchReady(1));
    EXPECT_EQ(1, restored.getWaitingFeatures().getBatchSize());
    NaviSchedLocalBatchInfo ready;
    ready.scheduleBatchId = 101;
    ready.localBatchId = 7;
    ready.requestCount = 1;
    ready.requestIds = {1};
    std::vector<uint64_t> acknowledged;
    ASSERT_TRUE(restored.acknowledgeDispatch(ready, acknowledged));
    EXPECT_EQ(std::vector<uint64_t>({1}), acknowledged);
    EXPECT_FALSE(restored.isAwaitingBatchReady(1));
    ASSERT_TRUE(restored.acknowledgeDispatch(ready, acknowledged));
    EXPECT_TRUE(acknowledged.empty());
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(1, transition));
    EXPECT_FALSE(reservation.isAwaitingBatchReady(1));
    EXPECT_EQ(1u, reservation.getFailedRequestCount());
    EXPECT_EQ(1u, reservation.getWaitingRequestCount());
}

TEST_F(MasterBatchReservationTest, testPartialFailureDoesNotScanUnchangedPrefixes)
{
    const NaviSchedCachePolicy policy{16, 1, false};
    MasterBatchReservation reservation;
    std::vector<MasterBatchReservation::Request> requests;
    for (uint64_t id = 1; id <= 64; ++id) {
        auto blocks = std::make_shared<NaviSchedBlockHashes>();
        for (uint64_t block = 0; block < 64; ++block) {
            blocks->push_back(id * 1000 + block);
        }
        requests.push_back(makeRequest(id, 1024, 16, 0, blocks));
    }
    ASSERT_TRUE(reservation.init(302, 100, 10.0, std::move(requests), &policy));
    ASSERT_TRUE(reservation.buildPendingCacheReferences());
    const auto scanned = reservation._pendingRebuildVisitedNodes;
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(1, transition));
    EXPECT_TRUE(transition.changed);
    EXPECT_EQ(scanned, reservation._pendingRebuildVisitedNodes);
    EXPECT_EQ(4032u, reservation.getPendingCacheReservation().fullReadyBlockHashes.size());
    EXPECT_EQ(63u, reservation.getWaitingRequestCount());
}

TEST_F(MasterBatchReservationTest, testFirstFailureRebuildsOnlyChangedPending)
{
    const NaviSchedCachePolicy policy{16, 3, true};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(302, 100, 10.0,
        {makeRequest(1, 48, 16, 0, hashes({11, 22, 33})),
         makeRequest(2, 49, 16, 0, hashes({11, 22, 44})),
         makeRequest(3, 32, 16, 0, hashes({55, 66}))}, &policy));
    ASSERT_FALSE(reservation.hasPendingCacheReferences());
    ASSERT_EQ(0u, reservation._pendingRebuildCount);
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(1, transition));
    EXPECT_TRUE(transition.changed);
    EXPECT_TRUE(transition.cacheChanged);
    EXPECT_TRUE(reservation.hasPendingCacheReferences());
    EXPECT_EQ(0u, reservation._pendingRebuildCount);
    expectHashes(reservation.getPendingCacheReservation(), {11, 22, 44, 55, 66}, {44, 55, 66});
    EXPECT_EQ(2u, reservation.getWaitingRequestCount());
    EXPECT_EQ(1u, reservation.getFailedRequestCount());
    ASSERT_TRUE(reservation.markFailed(1, transition));
    EXPECT_FALSE(transition.changed);
    EXPECT_EQ(1u, reservation._pendingRebuildCount);
}

TEST_F(MasterBatchReservationTest, testGeneratedPendingIsCanonicalAcrossPoliciesAndTransitions)
{
    for (const bool hybrid : {false, true}) {
        for (const int64_t step : {1, 3, 16}) {
            SCOPED_TRACE(::testing::Message() << "hybrid=" << hybrid << " step=" << step);
            const NaviSchedCachePolicy policy{16, step, hybrid};
            std::vector<MasterBatchReservation::Request> requests;
            std::vector<NaviSchedBlockHashesConstPtr> requestHashes;
            std::vector<int64_t> tokenCounts;
            std::vector<bool> active(48, true);
            for (uint64_t index = 0; index < active.size(); ++index) {
                auto blocks = std::make_shared<NaviSchedBlockHashes>();
                for (size_t block = 0; block < 1 + index % 19; ++block) {
                    blocks->push_back((index * 13 + block * 7) % 31);
                }
                const auto tokens = static_cast<int64_t>(blocks->size() * 16 + index % 2);
                requests.push_back(makeRequest(index + 1, tokens, 16, 0, blocks));
                requestHashes.push_back(blocks);
                tokenCounts.push_back(tokens);
            }
            MasterBatchReservation reservation;
            ASSERT_TRUE(reservation.init(301, 100, 10.0, std::move(requests), &policy));
            const auto verify = [&]() {
                std::set<uint64_t> full;
                std::set<uint64_t> tail;
                for (size_t index = 0; index < active.size(); ++index) {
                    if (!active[index]) {
                        continue;
                    }
                    const auto &blocks = *requestHashes[index];
                    full.insert(blocks.begin(), blocks.end());
                    if (hybrid) {
                        for (size_t block = 0; block < blocks.size(); ++block) {
                            if ((block + 1) % step == 0 || block + 1 == blocks.size() ||
                                (tokenCounts[index] % 16 == 0 && block + 2 == blocks.size())) {
                                tail.insert(blocks[block]);
                            }
                        }
                    }
                }
                const auto &pending = reservation.getPendingCacheReservation();
                EXPECT_EQ(NaviSchedBlockHashes(full.begin(), full.end()), pending.fullReadyBlockHashes);
                EXPECT_EQ(NaviSchedBlockHashes(tail.begin(), tail.end()), pending.tailReadyBlockHashes);
                if (full.empty()) {
                    EXPECT_EQ(0, pending.batchId);
                    EXPECT_EQ(0, pending.scheduleBatchId);
                } else {
                    EXPECT_EQ(301, pending.batchId);
                    EXPECT_EQ(301, pending.scheduleBatchId);
                    EXPECT_TRUE(PendingCacheUtil::validReservation(pending));
                }
            };
            verify();
            for (size_t offset = 0; offset < active.size(); ++offset) {
                const auto index = (offset * 17) % active.size();
                const auto id = index + 1;
                MasterBatchReservation::Transition transition;
                if (index % 2 == 0) {
                    ASSERT_TRUE(reservation.markFailed(id, transition));
                } else {
                    const auto batch = makePartialBatch(301, 401 + index, {id});
                    ASSERT_TRUE(reservation.markAdmitted(batch, transition));
                    verify();
                    ASSERT_TRUE(reservation.markCompleted(batch, true, transition));
                }
                active[index] = false;
                EXPECT_TRUE(transition.changed);
                verify();
                ASSERT_TRUE(reservation.markFailed(id, transition));
                EXPECT_FALSE(transition.changed);
                verify();
            }
            EXPECT_TRUE(reservation.isResolved());
        }
    }
}

TEST_F(MasterBatchReservationTest, testRejectsInvalidInitialization)
{
    const NaviSchedCachePolicy policy{2, 1, false};
    const auto valid = makeRequest(1, 4, 2, 2, hashes({11, 12}));
    MasterBatchReservation reservation;
    EXPECT_FALSE(reservation.init(0, 100, 1.0, {valid}, &policy));
    EXPECT_FALSE(reservation.init(1, 0, 1.0, {valid}, &policy));
    EXPECT_FALSE(reservation.init(1, 100, -1.0, {valid}, &policy));
    EXPECT_FALSE(reservation.init(1, 100, 1.0, {}, &policy));

    auto duplicate = valid;
    EXPECT_FALSE(reservation.init(1, 100, 1.0, {valid, duplicate}, &policy));
    EXPECT_FALSE(reservation.init(1, 100, 1.0,
                                  {makeRequest(2, 4, 2, 5, hashes({11, 12}))}, &policy));
    EXPECT_FALSE(
        reservation.init(1, 100, 1.0, {makeRequest(2, 4, 2, 2, hashes({11}))}, &policy));
    EXPECT_FALSE(
        reservation.init(1, 100, 1.0, {makeRequest(2, 4, 2, 2, hashes({}))}, &policy));
}

TEST_F(MasterBatchReservationTest, testFailedInitializationIsAtomicAndCanBeRetried)
{
    MasterBatchReservation reservation;
    const auto maxLatency = std::numeric_limits<double>::max();
    EXPECT_FALSE(reservation.init(1, 100, 1.0,
                                  {makeRequest(1, 1, 1, 0, nullptr, maxLatency),
                                   makeRequest(2, 1, 1, 0, nullptr, maxLatency)},
                                  nullptr));
    EXPECT_EQ(0, reservation.getBatchId());
    EXPECT_EQ(0, reservation.getRequestCount());
    EXPECT_EQ(0, reservation.getWaitingRequestCount());
    const auto failedInflight = reservation.getUnreportedInflightStats();
    EXPECT_EQ(0, failedInflight.requestCount);
    EXPECT_DOUBLE_EQ(0.0, failedInflight.totalPredictedLatencyMs);

    ASSERT_TRUE(
        reservation.init(2, 200, 3.0, {makeRequest(3, 1, 1, 0, nullptr, 3.0)}, nullptr));
    EXPECT_EQ(2, reservation.getBatchId());
    EXPECT_EQ(1, reservation.getRequestCount());
    EXPECT_EQ(1, reservation.getWaitingRequestCount());
    const auto committedInflight = reservation.getUnreportedInflightStats();
    EXPECT_EQ(1, committedInflight.requestCount);
    EXPECT_DOUBLE_EQ(3.0, committedInflight.totalPredictedLatencyMs);

    MasterBatchReservation capacityBoundary;
    std::vector<MasterBatchReservation::Request> requests;
    for (uint64_t id = 1; id <= RequestLatencyTracker::MAX_REQUEST_COUNT + 1; ++id) {
        requests.push_back(makeRequest(id, 1, 1, 0, nullptr, 1.0));
    }
    EXPECT_FALSE(capacityBoundary.init(3, 300, 1.0, requests, nullptr));
    EXPECT_EQ(0u, capacityBoundary.getRequestCount());
    EXPECT_EQ(0, capacityBoundary.getUnreportedInflightStats().requestCount);
    requests.pop_back();
    ASSERT_TRUE(capacityBoundary.init(4, 400, 1.0, std::move(requests), nullptr));
    EXPECT_EQ(RequestLatencyTracker::MAX_REQUEST_COUNT,
              capacityBoundary.getUnreportedInflightStats().requestCount);
    EXPECT_DOUBLE_EQ(RequestLatencyTracker::MAX_REQUEST_COUNT,
                     capacityBoundary.getUnreportedInflightStats().totalPredictedLatencyMs);
}

TEST_F(MasterBatchReservationTest,
       testRequestWithoutBlockHashesStillTracksWorkloadReservation)
{
    const NaviSchedCachePolicy policy{2, 1, false};
    MasterBatchReservation reservation;
    ASSERT_TRUE(
        reservation.init(3, 100, 7.5, {makeRequest(1, 5, 2, 0, hashes({}))}, &policy));
    EXPECT_FALSE(reservation.tracksPendingCache());
    EXPECT_FALSE(reservation.hasPendingCacheReferences());
    expectHashes(reservation.getPendingCacheReservation(), {});
    expectFeatures(reservation.getWaitingFeatures(), 1, 1, 0, 5, 25, 0);

    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(1, transition));
    EXPECT_TRUE(transition.changed);
    EXPECT_FALSE(transition.cacheChanged);
    EXPECT_TRUE(transition.resolved);
    EXPECT_EQ(RequestState::FAILED, reservation.getRequestState(1));
}

TEST_F(MasterBatchReservationTest,
       testShortRequestWithoutHashesDoesNotDisableBatchPendingCache)
{
    const NaviSchedCachePolicy policy{4, 1, false};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(
        4, 100, 7.5,
        {makeRequest(1, 3, 4, 0, nullptr), makeRequest(2, 8, 4, 0, hashes({11, 12}))},
        &policy));
    EXPECT_TRUE(reservation.tracksPendingCache());
    expectHashes(reservation.getPendingCacheReservation(), {11, 12});

    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(1, transition));
    EXPECT_TRUE(transition.changed);
    EXPECT_FALSE(transition.cacheChanged);
    expectHashes(reservation.getPendingCacheReservation(), {11, 12});
    ASSERT_TRUE(reservation.markFailed(2, transition));
    EXPECT_TRUE(transition.resolved);
    expectHashes(reservation.getPendingCacheReservation(), {});
}

TEST_F(MasterBatchReservationTest, testDefinitionEqualityIsExact)
{
    const NaviSchedCachePolicy policy{2, 1, false};
    const auto requests = std::vector<MasterBatchReservation::Request>{
        makeRequest(1, 4, 2, 2, hashes({11, 12})),
        makeRequest(2, 4, 2, 0, hashes({11, 13})),
    };
    auto makeReservation = [&](int64_t scheduleTimeUs, double predictedLatencyMs,
                               const std::vector<MasterBatchReservation::Request> &values,
                               const NaviSchedCachePolicy &cachePolicy) {
        MasterBatchReservation reservation;
        EXPECT_TRUE(reservation.init(7, scheduleTimeUs, predictedLatencyMs, values,
                                     &cachePolicy));
        return reservation;
    };

    auto original = makeReservation(100, 9.5, requests, policy);
    auto same = makeReservation(100, 9.5, requests, policy);
    EXPECT_TRUE(original.sameDefinition(same));

    auto nullHashes = std::vector<MasterBatchReservation::Request>{
        makeRequest(1, 1, 2, 0, nullptr),
    };
    auto emptyHashes = nullHashes;
    emptyHashes.front().blockHashes = hashes({});
    auto nullHashReservation = makeReservation(100, 9.5, nullHashes, policy);
    auto emptyHashReservation = makeReservation(100, 9.5, emptyHashes, policy);
    EXPECT_TRUE(nullHashReservation.sameDefinition(emptyHashReservation));

    auto changedTime = makeReservation(101, 9.5, requests, policy);
    EXPECT_FALSE(original.sameDefinition(changedTime));
    auto changedLatency = makeReservation(100, 9.6, requests, policy);
    EXPECT_FALSE(original.sameDefinition(changedLatency));
    auto changedRequests = requests;
    changedRequests[1].blockHashes = hashes({11, 14});
    auto changedHash = makeReservation(100, 9.5, changedRequests, policy);
    EXPECT_FALSE(original.sameDefinition(changedHash));
    const NaviSchedCachePolicy changedPolicy{2, 2, false};
    auto differentPolicy = makeReservation(100, 9.5, requests, changedPolicy);
    EXPECT_FALSE(original.sameDefinition(differentPolicy));
}

TEST_F(MasterBatchReservationTest, testLargeRoundSingletonAdmissionsKeepExactWaitingFeatures)
{
    constexpr size_t REQUEST_COUNT = RequestLatencyTracker::MAX_REQUEST_COUNT;
    constexpr size_t REPEAT_COUNT = 32;
    const NaviSchedCachePolicy policy{512, 1, false};
    const auto sharedHashes = hashes({11, 22, 33, 44});
    int64_t totalElapsedUs = 0;
    for (size_t repeat = 0; repeat < REPEAT_COUNT; ++repeat) {
        std::vector<MasterBatchReservation::Request> requests;
        requests.reserve(REQUEST_COUNT);
        int64_t hitTokens = 0, missTokens = 0;
        double missSquareSum = 0, hitMissProductSum = 0;
        for (size_t index = 0; index < REQUEST_COUNT; ++index) {
            const int64_t total = 2048 + index % 512;
            const int64_t hit = (index % 3) * 512;
            const auto miss = total - hit;
            requests.push_back(makeRequest(index + 1, total, 512, hit, sharedHashes));
            hitTokens += hit;
            missTokens += miss;
            missSquareSum += double(miss) * miss;
            hitMissProductSum += double(hit) * miss;
        }
        MasterBatchReservation reservation;
        ASSERT_TRUE(reservation.init(41, 100, 9.5, std::move(requests), &policy));
        const auto begin = std::chrono::steady_clock::now();
        for (size_t index = 0; index < REQUEST_COUNT; ++index) {
            const int64_t total = 2048 + index % 512;
            const int64_t hit = (index % 3) * 512;
            const auto miss = total - hit;
            MasterBatchReservation::Transition transition;
            ASSERT_TRUE(reservation.markAdmitted(
                makePartialBatch(41, index + 101, {index + 1}), transition));
            ASSERT_TRUE(transition.changed);
            ASSERT_EQ(std::vector<uint64_t>({index + 1}), transition.dispatchedRequestIds);
            hitTokens -= hit;
            missTokens -= miss;
            missSquareSum -= double(miss) * miss;
            hitMissProductSum -= double(hit) * miss;
            const auto waiting = REQUEST_COUNT - index - 1;
            ASSERT_EQ(waiting, reservation.getWaitingRequestCount());
            ASSERT_EQ(index + 1, reservation.getDispatchedRequestCount());
            expectFeatures(reservation.getWaitingFeatures(), waiting ? 1 : 0, waiting,
                           hitTokens, missTokens, missSquareSum, hitMissProductSum);
            ASSERT_FALSE(reservation.hasPendingCacheReferences());
        }
        const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count();
        totalElapsedUs += elapsedUs;
        EXPECT_DOUBLE_EQ(0.0, reservation.getPredictedLatencyMs());
        EXPECT_EQ(0u, reservation.getFailedRequestCount());
        EXPECT_EQ(0u, reservation.getCompletedRequestCount());
        expectHashes(reservation.getPendingCacheReservation(), {11, 22, 33, 44});
    }
    std::cout << "SINGLETON_ADMISSION_BENCHMARK requests=" << REQUEST_COUNT
              << " repeats=" << REPEAT_COUNT << " elapsed_us=" << totalElapsedUs
              << std::endl;
    EXPECT_LT(totalElapsedUs, 500'000);
}

TEST_F(MasterBatchReservationTest, testFullFailureCoveragePreservesCompletedRequests)
{
    const NaviSchedCachePolicy policy{2, 1, true};
    MasterBatchReservation batch;
    ASSERT_TRUE(batch.init(7, 100, 0,
        {makeRequest(1, 4, 2, 0, hashes({11, 12})),
         makeRequest(2, 4, 2, 0, hashes({11, 13})),
         makeRequest(3, 4, 2, 0, hashes({11, 14}))}, &policy));
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(batch.markAdmitted(makePartialBatch(7, 101, {2, 3}), transition));
    ASSERT_TRUE(batch.markCompleted(makePartialBatch(7, 101, {2}), true, transition));
    std::vector<uint64_t> acknowledged;
    ASSERT_TRUE(batch.acknowledgeDispatch(makePartialBatch(7, 102, {1}), acknowledged));
    ASSERT_EQ((std::vector<uint64_t>{1}), acknowledged);
    const auto snapshot = batch;
    const auto checksum = PendingCacheUtil::reservationChecksum(snapshot.getPendingCacheReservation());
    ASSERT_TRUE(batch.markFailedIndexes({2, 0, 2, 1}, transition));
    EXPECT_TRUE(transition.fullBatchFastPath);
    EXPECT_TRUE(transition.retireCache);
    EXPECT_TRUE(transition.resolved);
    EXPECT_EQ(2u, batch.getFailedRequestCount());
    EXPECT_EQ(1u, batch.getCompletedRequestCount());
    EXPECT_EQ(0u, batch.getWaitingRequestCount());
    EXPECT_EQ(0u, batch.getDispatchedRequestCount());
    EXPECT_EQ(0u, batch.getAcknowledgedWaitingRequestCount());
    EXPECT_EQ(0, batch.getUnreportedInflightStats().requestCount);
    EXPECT_EQ(RequestState::COMPLETED, batch.getRequestState(2));
    expectHashes(batch.getPendingCacheReservation(), {});
    EXPECT_FALSE(batch.hasPendingCacheReferences());
    EXPECT_EQ(1u, snapshot.getWaitingRequestCount());
    EXPECT_EQ(1u, snapshot.getDispatchedRequestCount());
    EXPECT_EQ(1u, snapshot.getCompletedRequestCount());
    EXPECT_EQ(1u, snapshot.getAcknowledgedWaitingRequestCount());
    EXPECT_EQ(checksum, PendingCacheUtil::reservationChecksum(snapshot.getPendingCacheReservation()));
    ASSERT_TRUE(batch.markFailedIndexes({0, 0}, transition));
    EXPECT_FALSE(transition.changed);
    EXPECT_FALSE(transition.retireCache);
    EXPECT_EQ(2u, batch.getFailedRequestCount());
}

TEST_F(MasterBatchReservationTest, testDuplicateFailureIndexesCannotRetireSurvivors)
{
    const NaviSchedCachePolicy policy{2, 1, true};
    MasterBatchReservation batch;
    ASSERT_TRUE(batch.init(7, 100, 0,
        {makeRequest(1, 4, 2, 0, hashes({11, 12})),
         makeRequest(2, 4, 2, 0, hashes({11, 13})),
         makeRequest(3, 4, 2, 0, hashes({11, 14}))}, &policy));
    const auto snapshot = batch;
    MasterBatchReservation::Transition transition;
    ASSERT_FALSE(batch.markFailedIndexes({0, 3}, transition));
    EXPECT_TRUE(batch.sameDefinition(snapshot));
    EXPECT_EQ(0u, batch.getFailedRequestCount());
    ASSERT_TRUE(batch.markFailedIndexes({0, 0, 0}, transition));
    EXPECT_FALSE(transition.fullBatchFastPath);
    EXPECT_FALSE(transition.retireCache);
    EXPECT_FALSE(transition.resolved);
    EXPECT_EQ(1u, batch.getFailedRequestCount());
    EXPECT_EQ(2u, batch.getWaitingRequestCount());
    EXPECT_EQ(RequestState::WAITING, batch.getRequestState(2));
    EXPECT_EQ(RequestState::WAITING, batch.getRequestState(3));
    expectHashes(batch.getPendingCacheReservation(), {11, 13, 14}, {11, 13, 14});
    EXPECT_EQ(3u, snapshot.getWaitingRequestCount());
}

TEST_F(MasterBatchReservationTest, testCompleteBatchNeverBuildsPrefixTree)
{
    const NaviSchedCachePolicy policy{2, 1, false};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(7, 100, 9.5,
                                 {makeRequest(1, 4, 2, 2, hashes({11, 12})),
                                  makeRequest(2, 4, 2, 0, hashes({11, 13}))},
                                 &policy));
    EXPECT_FALSE(reservation.hasPendingCacheReferences());
    expectFeatures(reservation.getWaitingFeatures(), 1, 2, 2, 6, 20, 4);
    expectHashes(reservation.getPendingCacheReservation(), {11, 12, 13});

    MasterBatchReservation::Transition admitted;
    ASSERT_TRUE(reservation.markAdmitted(makeFullBatch(7, 101, 2), admitted));
    EXPECT_TRUE(admitted.changed);
    EXPECT_TRUE(admitted.fullBatchFastPath);
    EXPECT_EQ((std::vector<uint64_t>{1, 2}), admitted.dispatchedRequestIds);
    EXPECT_FALSE(reservation.hasPendingCacheReferences());
    expectHashes(reservation.getPendingCacheReservation(), {11, 12, 13});
    expectFeatures(reservation.getWaitingFeatures(), 0, 0, 0, 0, 0, 0);
    EXPECT_DOUBLE_EQ(0.0, reservation.getPredictedLatencyMs());

    MasterBatchReservation::Transition completed;
    ASSERT_TRUE(reservation.markCompleted(makeFullBatch(7, 101, 2), true, completed));
    EXPECT_TRUE(completed.changed);
    EXPECT_TRUE(completed.fullBatchFastPath);
    EXPECT_TRUE(completed.cacheChanged);
    EXPECT_TRUE(completed.resolved);
    EXPECT_FALSE(reservation.hasPendingCacheReferences());
    EXPECT_EQ(RequestState::COMPLETED, reservation.getRequestState(1));
    EXPECT_EQ(RequestState::COMPLETED, reservation.getRequestState(2));
    expectHashes(reservation.getPendingCacheReservation(), {});
}

TEST_F(MasterBatchReservationTest,
       testAcknowledgeDispatchDoesNotChangeWorkloadReservation)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(
        8, 100, 9.5,
        {makeRequest(1, 4, 2, 2, nullptr), makeRequest(2, 4, 2, 0, nullptr),
         makeRequest(3, 4, 2, 0, nullptr)},
        nullptr));

    std::vector<uint64_t> requestIds;
    ASSERT_TRUE(
        reservation.acknowledgeDispatch(makePartialBatch(8, 101, {1, 3}), requestIds));
    EXPECT_EQ((std::vector<uint64_t>{1, 3}), requestIds);
    EXPECT_EQ(3, reservation.getWaitingRequestCount());
    EXPECT_EQ(2, reservation.getAcknowledgedWaitingRequestCount());
    EXPECT_TRUE(reservation.isDispatchAcknowledged(1));
    EXPECT_FALSE(reservation.isDispatchAcknowledged(2));
    EXPECT_TRUE(reservation.isDispatchAcknowledged(3));
    ASSERT_TRUE(
        reservation.acknowledgeDispatch(makePartialBatch(8, 102, {1, 3}), requestIds));
    EXPECT_TRUE(requestIds.empty());
    EXPECT_EQ(2, reservation.getAcknowledgedWaitingRequestCount());

    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(1, transition));
    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(8, 103, {3}), transition));
    EXPECT_EQ(0, reservation.getAcknowledgedWaitingRequestCount());
    EXPECT_FALSE(reservation.isDispatchAcknowledged(1));
    EXPECT_FALSE(reservation.isDispatchAcknowledged(3));
    ASSERT_TRUE(reservation.acknowledgeDispatch(makeFullBatch(8, 104, 3), requestIds));
    EXPECT_EQ((std::vector<uint64_t>{2}), requestIds);
    EXPECT_EQ(1, reservation.getWaitingRequestCount());
    EXPECT_EQ(1, reservation.getAcknowledgedWaitingRequestCount());

    EXPECT_FALSE(
        reservation.acknowledgeDispatch(makePartialBatch(8, 105, {99}), requestIds));
    EXPECT_TRUE(requestIds.empty());
    EXPECT_EQ(1, reservation.getAcknowledgedWaitingRequestCount());
}

TEST_F(MasterBatchReservationTest, testInflightOwnershipTransfersExactlyOncePerRequest)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(
        8, 100, 9.5,
        {makeRequest(1, 4, 2, 2, nullptr, 10.0), makeRequest(2, 4, 2, 0, nullptr, 20.0),
         makeRequest(3, 4, 2, 0, nullptr, 30.0)},
        nullptr));
    auto inflight = reservation.getUnreportedInflightStats();
    ASSERT_EQ(3, inflight.requestCount);
    ASSERT_DOUBLE_EQ(60.0, inflight.totalPredictedLatencyMs);

    std::vector<uint64_t> acknowledgedRequestIds;
    ASSERT_TRUE(reservation.acknowledgeDispatch(makePartialBatch(8, 101, {1, 2}),
                                                acknowledgedRequestIds));
    EXPECT_EQ((std::vector<uint64_t>{1, 2}), acknowledgedRequestIds);
    auto afterAck = reservation.getUnreportedInflightStats();
    EXPECT_EQ(inflight.revision, afterAck.revision);
    EXPECT_EQ(3, afterAck.requestCount);
    EXPECT_DOUBLE_EQ(60.0, afterAck.totalPredictedLatencyMs);

    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(8, 102, {1, 2}), transition));
    auto afterAdmit = reservation.getUnreportedInflightStats();
    EXPECT_EQ(inflight.revision, afterAdmit.revision);
    EXPECT_EQ(3, afterAdmit.requestCount);
    EXPECT_DOUBLE_EQ(60.0, afterAdmit.totalPredictedLatencyMs);

    bool changed = false;
    ASSERT_TRUE(reservation.markInflightReported(1, changed));
    EXPECT_TRUE(changed);
    inflight = reservation.getUnreportedInflightStats();
    EXPECT_EQ(2, inflight.requestCount);
    EXPECT_DOUBLE_EQ(50.0, inflight.totalPredictedLatencyMs);
    const auto firstHandoffRevision = inflight.revision;

    ASSERT_TRUE(reservation.markInflightReported(1, changed));
    EXPECT_FALSE(changed);
    inflight = reservation.getUnreportedInflightStats();
    EXPECT_EQ(firstHandoffRevision, inflight.revision);
    EXPECT_EQ(2, inflight.requestCount);
    EXPECT_DOUBLE_EQ(50.0, inflight.totalPredictedLatencyMs);

    ASSERT_TRUE(reservation.markFailed(2, transition));
    inflight = reservation.getUnreportedInflightStats();
    EXPECT_EQ(1, inflight.requestCount);
    EXPECT_DOUBLE_EQ(30.0, inflight.totalPredictedLatencyMs);
    EXPECT_EQ(RequestState::FAILED, reservation.getRequestState(2));

    std::vector<uint64_t> handedOff;
    ASSERT_TRUE(
        reservation.markInflightReported(makePartialBatch(8, 103, {1, 3}), handedOff));
    EXPECT_EQ((std::vector<uint64_t>{3}), handedOff);
    inflight = reservation.getUnreportedInflightStats();
    EXPECT_EQ(0, inflight.requestCount);
    EXPECT_DOUBLE_EQ(0.0, inflight.totalPredictedLatencyMs);

    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(8, 104, {3}), transition));
    ASSERT_TRUE(
        reservation.markCompleted(makePartialBatch(8, 104, {3}), true, transition));
    inflight = reservation.getUnreportedInflightStats();
    EXPECT_EQ(0, inflight.requestCount);
    EXPECT_DOUBLE_EQ(0.0, inflight.totalPredictedLatencyMs);
}

TEST_F(MasterBatchReservationTest, testUnreportedCopyAndReplayPreserveStoredContributions)
{
    MasterBatchReservation original;
    ASSERT_TRUE(original.init(9, 100, 9.5,
        {makeRequest(1, 4, 2, 0, nullptr, 10.0),
         makeRequest(2, 4, 2, 0, nullptr, 20.0),
         makeRequest(3, 4, 2, 0, nullptr, 30.0)}, nullptr));
    auto copy = original;
    bool changed = false;
    ASSERT_TRUE(copy.markInflightReported(1, changed));
    ASSERT_TRUE(changed);
    EXPECT_EQ(3, original.getUnreportedInflightStats().requestCount);
    EXPECT_EQ(4, original.getUnreportedInflightStats().revision);
    EXPECT_DOUBLE_EQ(60.0, original.getUnreportedInflightStats().totalPredictedLatencyMs);
    EXPECT_EQ(2, copy.getUnreportedInflightStats().requestCount);
    EXPECT_EQ(5, copy.getUnreportedInflightStats().revision);
    auto state = copy.makeReplayState();
    std::vector<autil::legacy::json::JsonMap> records;
    autil::legacy::FromJson(records, state.at("requests"));
    records[1]["unreported_latency_ms"] = autil::legacy::ToJson(123.75);
    state["requests"] = autil::legacy::ToJson(records);
    MasterBatchReservation restored;
    ASSERT_TRUE(restored.restoreReplayState(state));
    EXPECT_EQ(2, restored.getUnreportedInflightStats().requestCount);
    EXPECT_EQ(7, restored.getUnreportedInflightStats().revision);
    EXPECT_DOUBLE_EQ(153.75, restored.getUnreportedInflightStats().totalPredictedLatencyMs);
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(restored.markFailed(2, transition));
    EXPECT_EQ(1, restored.getUnreportedInflightStats().requestCount);
    EXPECT_EQ(8, restored.getUnreportedInflightStats().revision);
    EXPECT_DOUBLE_EQ(30.0, restored.getUnreportedInflightStats().totalPredictedLatencyMs);
    ASSERT_TRUE(restored.markInflightReported(3, changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(0, restored.getUnreportedInflightStats().requestCount);
    EXPECT_EQ(9, restored.getUnreportedInflightStats().revision);
    EXPECT_DOUBLE_EQ(0.0, restored.getUnreportedInflightStats().totalPredictedLatencyMs);
    ASSERT_TRUE(restored.markInflightReported(3, changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(9, restored.getUnreportedInflightStats().revision);
    EXPECT_DOUBLE_EQ(50.0, copy.getUnreportedInflightStats().totalPredictedLatencyMs);
    EXPECT_DOUBLE_EQ(60.0, original.getUnreportedInflightStats().totalPredictedLatencyMs);
    EXPECT_TRUE(restored.getUnreportedInflightStats().valid());
    EXPECT_EQ(0u, restored.getUnreportedInflightStats().missingRemoveCount);
}

TEST_F(MasterBatchReservationTest, testFailedRequestRemovesOnlyItsStoredInflightLatency)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(
        9, 100, 12.0,
        {makeRequest(1, 2, 1, 0, nullptr, 1.25), makeRequest(2, 2, 1, 0, nullptr, 100.5)},
        nullptr));

    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(1, transition));
    auto inflight = reservation.getUnreportedInflightStats();
    EXPECT_EQ(1, inflight.requestCount);
    EXPECT_DOUBLE_EQ(100.5, inflight.totalPredictedLatencyMs);
    const auto failedRevision = inflight.revision;

    ASSERT_TRUE(reservation.markFailed(1, transition));
    EXPECT_FALSE(transition.changed);
    inflight = reservation.getUnreportedInflightStats();
    EXPECT_EQ(failedRevision, inflight.revision);
    EXPECT_EQ(1, inflight.requestCount);
    EXPECT_DOUBLE_EQ(100.5, inflight.totalPredictedLatencyMs);

    bool changed = false;
    ASSERT_TRUE(reservation.markInflightReported(2, changed));
    EXPECT_TRUE(changed);
    inflight = reservation.getUnreportedInflightStats();
    EXPECT_EQ(0, inflight.requestCount);
    EXPECT_DOUBLE_EQ(0.0, inflight.totalPredictedLatencyMs);
}

TEST_F(MasterBatchReservationTest,
       testFirstFailureBuildsAllPathsThenDecrementsSharedReferences)
{
    const NaviSchedCachePolicy policy{1, 1, false};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(9, 100, 12.0,
                                 {makeRequest(1, 3, 1, 0, hashes({11, 12, 13})),
                                  makeRequest(2, 3, 1, 0, hashes({11, 12, 14})),
                                  makeRequest(3, 2, 1, 0, hashes({11, 15}))},
                                 &policy));
    EXPECT_FALSE(reservation.hasPendingCacheReferences());

    MasterBatchReservation::Transition failed;
    ASSERT_TRUE(reservation.markFailed(1, failed));
    EXPECT_TRUE(reservation.hasPendingCacheReferences());
    ASSERT_EQ(5, reservation._pendingReferences.size());
    const auto findNode = [&reservation](uint64_t hash) {
        const auto &topology = *reservation._pendingBlockHashes;
        const auto it = std::find(topology.begin(), topology.end(), hash);
        return &reservation._pendingReferences[it - topology.begin()];
    };
    EXPECT_EQ(2, findNode(11)->fullRefCount);
    EXPECT_EQ(1, findNode(12)->fullRefCount);
    EXPECT_EQ(0, findNode(13)->fullRefCount);
    EXPECT_EQ(1, findNode(14)->fullRefCount);
    EXPECT_EQ(1, findNode(15)->fullRefCount);
    expectHashes(reservation.getPendingCacheReservation(), {11, 12, 14, 15});
    EXPECT_EQ(RequestState::FAILED, reservation.getRequestState(1));

    ASSERT_TRUE(reservation.markFailed(2, failed));
    expectHashes(reservation.getPendingCacheReservation(), {11, 15});
    ASSERT_TRUE(reservation.markFailed(3, failed));
    EXPECT_TRUE(failed.resolved);
    expectHashes(reservation.getPendingCacheReservation(), {});
    EXPECT_EQ(3, reservation.getFailedRequestCount());
    EXPECT_EQ(0, reservation.getWaitingRequestCount());
}

TEST_F(MasterBatchReservationTest,
       testFailureIsIdempotentAndLateAdmissionDoesNotReactivate)
{
    const NaviSchedCachePolicy policy{1, 1, false};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(9, 100, 12.0,
                                 {makeRequest(1, 2, 1, 0, hashes({11, 12})),
                                  makeRequest(2, 2, 1, 0, hashes({11, 13}))},
                                 &policy));

    MasterBatchReservation::Transition failed;
    ASSERT_TRUE(reservation.markFailed(1, failed));
    ASSERT_TRUE(failed.changed);
    const auto nodeCount = reservation._pendingReferences.size();
    ASSERT_TRUE(reservation.markFailed(1, failed));
    EXPECT_FALSE(failed.changed);
    EXPECT_EQ(nodeCount, reservation._pendingReferences.size());
    expectHashes(reservation.getPendingCacheReservation(), {11, 13});

    MasterBatchReservation::Transition admitted;
    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(9, 101, {1}), admitted));
    EXPECT_FALSE(admitted.changed);
    EXPECT_TRUE(admitted.dispatchedRequestIds.empty());
    EXPECT_EQ(RequestState::FAILED, reservation.getRequestState(1));
    expectHashes(reservation.getPendingCacheReservation(), {11, 13});
}

TEST_F(MasterBatchReservationTest, testRepeatedHashRetiresAfterLastFullAndTailOwner)
{
    const NaviSchedCachePolicy policy{2, 2, true};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(9, 100, 12.0,
        {makeRequest(1, 7, 2, 0, hashes({90, 10, 90})),
         makeRequest(2, 7, 2, 0, hashes({40, 10, 90})),
         makeRequest(3, 7, 2, 0, hashes({90, 10, 80})),
         makeRequest(4, 3, 2, 0, hashes({7}))}, &policy));
    expectHashes(reservation.getPendingCacheReservation(), {7, 10, 40, 80, 90},
                 {7, 10, 80, 90});
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(1, transition));
    EXPECT_TRUE(transition.removedFullReadyBlockHashes.empty());
    EXPECT_TRUE(transition.removedTailReadyBlockHashes.empty());
    EXPECT_EQ(0u, reservation._pendingRebuildCount);
    ASSERT_TRUE(reservation.markFailed(2, transition));
    EXPECT_EQ((NaviSchedBlockHashes{40}), transition.removedFullReadyBlockHashes);
    EXPECT_EQ((NaviSchedBlockHashes{90}), transition.removedTailReadyBlockHashes);
    EXPECT_EQ(0u, reservation._pendingRebuildCount);
    expectHashes(reservation.getPendingCacheReservation(), {7, 10, 80, 90}, {7, 10, 80});

    auto copy = reservation;
    ASSERT_TRUE(copy.markAdmitted(makePartialBatch(9, 101, {3}), transition));
    ASSERT_TRUE(copy.markCompleted(makePartialBatch(9, 101, {3}), true, transition));
    auto removed = transition.removedFullReadyBlockHashes;
    std::sort(removed.begin(), removed.end());
    EXPECT_EQ((NaviSchedBlockHashes{10, 80, 90}), removed);
    removed = transition.removedTailReadyBlockHashes;
    std::sort(removed.begin(), removed.end());
    EXPECT_EQ((NaviSchedBlockHashes{10, 80}), removed);
    expectHashes(copy.getPendingCacheReservation(), {7}, {7});
    expectHashes(reservation.getPendingCacheReservation(), {7, 10, 80, 90}, {7, 10, 80});
    MasterBatchReservation restored;
    ASSERT_TRUE(restored.restoreReplayState(copy.makeReplayState()));
    expectHashes(restored.getPendingCacheReservation(), {7}, {7});
    ASSERT_TRUE(restored.markFailed(4, transition));
    EXPECT_TRUE(restored.isResolved());
    expectHashes(restored.getPendingCacheReservation(), {});
}

TEST_F(MasterBatchReservationTest, testBlockIndexesShareCountersAcrossParentsAndDepths)
{
    const NaviSchedCachePolicy policy{2, 2, true};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(9, 100, 12.0,
        {makeRequest(1, 7, 2, 0, hashes({90, 10, 90})),
         makeRequest(2, 7, 2, 0, hashes({40, 10, 90})),
         makeRequest(3, 7, 2, 0, hashes({90, 10, 80})),
         makeRequest(4, 3, 2, 0, hashes({7}))}, &policy));
    ASSERT_TRUE(reservation.buildPendingCacheReferences());
    ASSERT_EQ(5u, reservation._pendingReferences.size());
    EXPECT_EQ((NaviSchedBlockHashes{90, 10, 40, 80, 7}), *reservation._pendingBlockHashes);
    EXPECT_EQ((std::vector<uint32_t>{0, 1, 0, 2, 1, 0, 0, 1, 3, 4}),
              *reservation._requestBlockIndexes);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(i * 3, reservation._requests[i].blockIndexBegin);
    }
    EXPECT_EQ(4u, reservation._pendingReferences[0].fullRefCount);
    EXPECT_EQ(2u, reservation._pendingReferences[0].tailRefCount);
    EXPECT_EQ(3u, reservation._pendingReferences[1].fullRefCount);
    EXPECT_EQ(3u, reservation._pendingReferences[1].tailRefCount);
    EXPECT_EQ(1u, reservation._pendingReferences[2].fullRefCount);
    EXPECT_EQ(0u, reservation._pendingReferences[2].tailRefCount);

    auto copy = reservation;
    EXPECT_EQ(reservation._requestBlockIndexes.get(), copy._requestBlockIndexes.get());
    EXPECT_EQ(reservation._pendingBlockHashes.get(), copy._pendingBlockHashes.get());
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(copy.markFailed(1, transition));
    EXPECT_EQ(2u, copy._pendingReferences[0].fullRefCount);
    EXPECT_EQ(1u, copy._pendingReferences[0].tailRefCount);
    EXPECT_EQ(2u, copy._pendingReferences[1].fullRefCount);
    EXPECT_EQ(2u, copy._pendingReferences[1].tailRefCount);
    EXPECT_EQ(4u, reservation._pendingReferences[0].fullRefCount);
    EXPECT_EQ(2u, reservation._pendingReferences[0].tailRefCount);
    EXPECT_EQ(3u, reservation._pendingReferences[1].fullRefCount);
    EXPECT_EQ(3u, reservation._pendingReferences[1].tailRefCount);
    EXPECT_EQ(0u, copy._pendingRebuildVisitedNodes);
    EXPECT_EQ(MasterBatchReservation::INVALID_BLOCK_OFFSET, copy._requests[0].blockIndexBegin);
    EXPECT_EQ(0u, reservation._requests[0].blockIndexBegin);
}

TEST_F(MasterBatchReservationTest, testPartialAdmissionKeepsKeysAndDoesNotBuildTree)
{
    const NaviSchedCachePolicy policy{1, 1, false};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(11, 100, 15.0,
                                 {makeRequest(1, 2, 1, 1, hashes({11, 12})),
                                  makeRequest(2, 2, 1, 0, hashes({11, 13})),
                                  makeRequest(3, 2, 1, 0, hashes({11, 14}))},
                                 &policy));

    MasterBatchReservation::Transition admitted;
    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(11, 101, {1, 3}), admitted));
    EXPECT_EQ((std::vector<uint64_t>{1, 3}), admitted.dispatchedRequestIds);
    EXPECT_FALSE(reservation.hasPendingCacheReferences());
    EXPECT_EQ(1, reservation.getWaitingRequestCount());
    EXPECT_EQ(2, reservation.getDispatchedRequestCount());
    expectFeatures(reservation.getWaitingFeatures(), 1, 1, 0, 2, 4, 0);
    expectHashes(reservation.getPendingCacheReservation(), {11, 12, 13, 14});
}

TEST_F(MasterBatchReservationTest,
       testPartialCompletionLazilyBuildsTreeAndKeepsSharedKeys)
{
    const NaviSchedCachePolicy policy{1, 1, false};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(11, 100, 15.0,
                                 {makeRequest(1, 3, 1, 0, hashes({11, 12, 13})),
                                  makeRequest(2, 3, 1, 0, hashes({11, 12, 14})),
                                  makeRequest(3, 2, 1, 0, hashes({11, 15}))},
                                 &policy));
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(11, 101, {1, 2}), transition));
    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(11, 102, {3}), transition));
    EXPECT_FALSE(reservation.hasPendingCacheReferences());

    ASSERT_TRUE(
        reservation.markCompleted(makePartialBatch(11, 101, {1}), true, transition));
    EXPECT_TRUE(reservation.hasPendingCacheReferences());
    expectHashes(reservation.getPendingCacheReservation(), {11, 12, 14, 15});
    EXPECT_EQ(RequestState::COMPLETED, reservation.getRequestState(1));

    ASSERT_TRUE(
        reservation.markCompleted(makePartialBatch(11, 103, {2}), false, transition));
    expectHashes(reservation.getPendingCacheReservation(), {11, 15});
    EXPECT_EQ(RequestState::FAILED, reservation.getRequestState(2));
    ASSERT_TRUE(
        reservation.markCompleted(makePartialBatch(11, 102, {3}), true, transition));
    EXPECT_TRUE(transition.resolved);
    expectHashes(reservation.getPendingCacheReservation(), {});
}

TEST_F(MasterBatchReservationTest, testHybridTailReferencesUseLinearStep)
{
    const NaviSchedCachePolicy policy{2, 2, true};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(13, 100, 10.0,
                                 {makeRequest(1, 8, 2, 0, hashes({11, 12, 13, 14})),
                                  makeRequest(2, 7, 2, 0, hashes({11, 12, 13}))},
                                 &policy));
    expectHashes(reservation.getPendingCacheReservation(), {11, 12, 13, 14},
                 {12, 13, 14});

    MasterBatchReservation::Transition failed;
    ASSERT_TRUE(reservation.markFailed(1, failed));
    expectHashes(reservation.getPendingCacheReservation(), {11, 12, 13}, {12, 13});
    ASSERT_TRUE(reservation.markFailed(2, failed));
    expectHashes(reservation.getPendingCacheReservation(), {});
}

TEST_F(MasterBatchReservationTest, testLinearStepOneMarksEveryTail)
{
    const NaviSchedCachePolicy policy{2, 1, true};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(
        13, 100, 10.0, {makeRequest(1, 6, 2, 0, hashes({11, 12, 13}))}, &policy));
    expectHashes(reservation.getPendingCacheReservation(), {11, 12, 13}, {11, 12, 13});
    MasterBatchReservation::Transition failed;
    ASSERT_TRUE(reservation.markFailed(1, failed));
    expectHashes(reservation.getPendingCacheReservation(), {});
}

TEST_F(MasterBatchReservationTest,
       testRecomputeUpdatesOnlyWaitingRequestsAndAllFeatureTerms)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(17, 100, 8.0,
                                 {makeRequest(1, 10, 2, 2, hashes({11, 12, 13, 14, 15})),
                                  makeRequest(2, 8, 2, 0, hashes({21, 22, 23, 24})),
                                  makeRequest(3, 6, 2, 4, hashes({31, 32, 33}))},
                                 nullptr));
    MasterBatchReservation::Transition admitted;
    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(17, 101, {3}), admitted));

    ASSERT_TRUE(reservation.recomputeWaitingFeatures(
        [](const MasterBatchReservation::Request &request) {
            return request.requestId == 1 ? 4 : 1;
        }));
    // Request 1: hit=8 miss=2; request 2: hit=2 miss=6.
    expectFeatures(reservation.getWaitingFeatures(), 1, 2, 10, 8, 40, 28);
    EXPECT_TRUE(reservation.setPredictedLatencyMs(6.25));
    EXPECT_DOUBLE_EQ(6.25, reservation.getPredictedLatencyMs());
}

TEST_F(MasterBatchReservationTest,
       testFailedRecomputeDoesNotPartiallyChangeRequestCacheHits)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(17, 100, 8.0,
                                 {makeRequest(1, 4, 2, 0, hashes({11, 12})),
                                  makeRequest(2, 4, 2, 2, hashes({21, 22}))},
                                 nullptr));
    const auto before = reservation.getWaitingFeatures();

    EXPECT_FALSE(reservation.recomputeWaitingFeatures(
        [](const MasterBatchReservation::Request &request) {
            return request.requestId == 1 ? 1 : 3;
        }));

    EXPECT_EQ(0, reservation._requests[0].request.predictedCacheHitTokenCount);
    EXPECT_EQ(2, reservation._requests[1].request.predictedCacheHitTokenCount);
    expectFeatures(reservation.getWaitingFeatures(), before.getBatchCount(),
                   before.getBatchSize(), before.getCacheHitTokenCount(),
                   before.getCacheMissTokenCount(), before.getCacheMissTokenSquareSum(),
                   before.getCacheHitMissTokenProductSum());
    ASSERT_TRUE(reservation.recomputeWaitingFeatures(
        [](const MasterBatchReservation::Request &) { return 1; }));
    EXPECT_EQ(2, reservation._requests[0].request.predictedCacheHitTokenCount);
    EXPECT_EQ(2, reservation._requests[1].request.predictedCacheHitTokenCount);
    expectFeatures(reservation.getWaitingFeatures(), 1, 2, 4, 4, 8, 8);
}

TEST_F(MasterBatchReservationTest,
       testWaitingCapacityFeaturesFillAndCrossCapacityBoundary)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(
        18, 100, 8.0,
        {makeRequest(1, 10, 1, 2, nullptr), makeRequest(2, 20, 1, 4, nullptr),
         makeRequest(3, 30, 1, 6, nullptr), makeRequest(4, 40, 1, 8, nullptr)},
        nullptr));

    NaviSchedLatencyFeatures features;
    int32_t occupiedSlots = 1;
    ASSERT_TRUE(reservation.appendWaitingCapacityFeatures(features, 3, occupiedSlots));
    EXPECT_EQ(3, occupiedSlots);
    expectCapacityFeatures(features, 1, 2, 6, 24, 320.0, 80.0, 2, 14, 56, 1600.0, 400.0,
                           4);

    NaviSchedLatencyFeatures allWaiting;
    occupiedSlots = 0;
    ASSERT_TRUE(reservation.appendWaitingCapacityFeatures(allWaiting, 0, occupiedSlots));
    EXPECT_EQ(0, occupiedSlots);
    expectCapacityFeatures(allWaiting, 1, 0, 0, 0, 0.0, 0.0, 4, 20, 80, 1920.0, 480.0, 4);

    NaviSchedLatencyFeatures invalid;
    occupiedSlots = 4;
    EXPECT_FALSE(reservation.appendWaitingCapacityFeatures(invalid, 3, occupiedSlots));
    EXPECT_EQ(4, occupiedSlots);
    EXPECT_EQ(0, invalid.getBatchCount());
    EXPECT_EQ(0, invalid.getBatchSize());
}

TEST_F(MasterBatchReservationTest,
       testWaitingCapacityFeaturesPreserveSmallRemainderAfterLargeRequestLeaves)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(
        18, 100, 8.0,
        {makeRequest(1, 100000000, 1, 0, nullptr),
         makeRequest(2, 1, 1, 0, nullptr)}, nullptr));
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(1, transition));
    ASSERT_EQ(1u, reservation.getWaitingRequestCount());
    NaviSchedLatencyFeatures features;
    int32_t occupiedSlots = 1;
    ASSERT_TRUE(reservation.appendWaitingCapacityFeatures(features, 1, occupiedSlots));
    EXPECT_EQ(1, occupiedSlots);
    expectCapacityFeatures(features, 1, 0, 0, 0, 0.0, 0.0,
                           1, 0, 1, 1.0, 0.0, 1);
}

TEST_F(MasterBatchReservationTest, testFullCapacitySnapshotLargeWaitingBatch)
{
    std::vector<MasterBatchReservation::Request> requests;
    for (uint64_t id = 1; id <= 5000; ++id) {
        requests.push_back(makeRequest(id, 3072, 1, 1024, nullptr));
    }
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(18, 100, 8.0, std::move(requests), nullptr));
    const auto begin = std::chrono::steady_clock::now();
    for (size_t repeat = 0; repeat < 1000; ++repeat) {
        NaviSchedLatencyFeatures features;
        int32_t slots = 64;
        ASSERT_TRUE(reservation.appendWaitingCapacityFeatures(features, 64, slots));
        ASSERT_EQ(64, slots);
        expectCapacityFeatures(features, 1, 0, 0, 0, 0.0, 0.0,
                               5000, 5120000, 10240000,
                               20971520000.0, 10485760000.0, 5000);
    }
    std::cout << "FULL_CAPACITY_SNAPSHOT_BENCHMARK requests=5000 repeats=1000 elapsed_us="
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - begin).count()
              << std::endl;
}

TEST_F(MasterBatchReservationTest,
       testWaitingCapacityFeaturesTrackOnlyWaitingRequestsAcrossReservations)
{
    MasterBatchReservation first;
    ASSERT_TRUE(
        first.init(20, 100, 8.0,
                   {makeRequest(1, 10, 1, 2, nullptr), makeRequest(2, 20, 1, 4, nullptr),
                    makeRequest(3, 30, 1, 6, nullptr)},
                   nullptr));
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(first.markFailed(1, transition));
    ASSERT_TRUE(first.markAdmitted(makePartialBatch(20, 201, {2}), transition));

    MasterBatchReservation second;
    ASSERT_TRUE(second.init(
        21, 101, 9.0,
        {makeRequest(4, 40, 1, 8, nullptr), makeRequest(5, 50, 1, 10, nullptr)},
        nullptr));

    NaviSchedLatencyFeatures features;
    int32_t occupiedSlots = 1;
    ASSERT_TRUE(first.appendWaitingCapacityFeatures(features, 2, occupiedSlots));
    EXPECT_EQ(2, occupiedSlots);
    expectCapacityFeatures(features, 1, 1, 6, 24, 576.0, 144.0, 0, 0, 0, 0.0, 0.0, 1);
    ASSERT_TRUE(second.appendWaitingCapacityFeatures(features, 2, occupiedSlots));
    expectCapacityFeatures(features, 2, 1, 6, 24, 576.0, 144.0, 2, 18, 72, 2624.0, 656.0,
                           3);

    ASSERT_TRUE(first.markAdmitted(makePartialBatch(20, 202, {3}), transition));
    NaviSchedLatencyFeatures empty;
    occupiedSlots = 1;
    ASSERT_TRUE(first.appendWaitingCapacityFeatures(empty, 2, occupiedSlots));
    EXPECT_EQ(1, occupiedSlots);
    EXPECT_FALSE(empty.usesCapacitySplit());
    EXPECT_EQ(0, empty.getBatchCount());
}

TEST_F(MasterBatchReservationTest, testEmptyWaitingCapacitySnapshotPreservesInputs)
{
    for (auto terminal : {RequestState::DISPATCHED, RequestState::COMPLETED,
                          RequestState::FAILED}) {
        std::vector<MasterBatchReservation::Request> requests;
        std::vector<uint64_t> ids;
        for (uint64_t id = 1; id <= 128; ++id) {
            requests.push_back(makeRequest(id, 10, 1, 2, nullptr));
            ids.push_back(id);
        }
        MasterBatchReservation reservation;
        ASSERT_TRUE(reservation.init(18, 100, 8.0, std::move(requests), nullptr));
        MasterBatchReservation::Transition transition;
        const auto batch = makeFullBatch(18, 201, 128);
        if (terminal == RequestState::FAILED) {
            ASSERT_TRUE(reservation.markFailed(ids, transition));
        } else {
            ASSERT_TRUE(reservation.markAdmitted(batch, transition));
            if (terminal == RequestState::COMPLETED) {
                ASSERT_TRUE(reservation.markCompleted(batch, true, transition));
            }
        }
        ASSERT_EQ(0u, reservation.getWaitingRequestCount());
        ASSERT_EQ(128u, reservation.getRequestCount());
        for (auto id : ids) ASSERT_EQ(terminal, reservation.getRequestState(id));
        for (size_t repeat = 0; repeat < 4; ++repeat) {
            NaviSchedLatencyFeatures existing;
            existing.appendRequest(1, 2);
            int32_t slots = 2;
            EXPECT_TRUE(reservation.appendWaitingCapacityFeatures(existing, 3, slots));
            EXPECT_EQ(2, slots);
            EXPECT_FALSE(existing.usesCapacitySplit());
            expectFeatures(existing, 0, 1, 1, 2, 4, 2);

            slots = 4;
            EXPECT_FALSE(reservation.appendWaitingCapacityFeatures(existing, 3, slots));
            EXPECT_EQ(4, slots);
            expectFeatures(existing, 0, 1, 1, 2, 4, 2);

            NaviSchedLatencyFeatures empty;
            slots = 0;
            EXPECT_TRUE(reservation.appendWaitingCapacityFeatures(empty, 0, slots));
            EXPECT_EQ(0, slots);
            EXPECT_FALSE(empty.usesCapacitySplit());
            expectFeatures(empty, 0, 0, 0, 0, 0, 0);
        }
    }
}

TEST_F(MasterBatchReservationTest,
       testFailedCapacityFeatureAppendDoesNotConsumeSlotsOrChangeFeatures)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(
        18, 100, 8.0,
        {makeRequest(1, 10, 1, 2, nullptr), makeRequest(2, 20, 1, 4, nullptr)}, nullptr));
    NaviSchedLatencyFeatures incompatibleFeatures;
    incompatibleFeatures.appendRequest(1, 2);
    int32_t occupiedSlots = 0;

    EXPECT_FALSE(reservation.appendWaitingCapacityFeatures(incompatibleFeatures, 1,
                                                           occupiedSlots));

    EXPECT_EQ(0, occupiedSlots);
    EXPECT_FALSE(incompatibleFeatures.usesCapacitySplit());
    expectFeatures(incompatibleFeatures, 0, 1, 1, 2, 4, 2);
}

TEST_F(MasterBatchReservationTest, testRejectsMalformedLocalBatchAtomically)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(19, 100, 8.0,
                                 {makeRequest(1, 2, 1, 0, hashes({11, 12})),
                                  makeRequest(2, 2, 1, 0, hashes({11, 13}))},
                                 nullptr));
    MasterBatchReservation::Transition transition;
    EXPECT_FALSE(reservation.markAdmitted(makeFullBatch(18, 101, 2), transition));
    EXPECT_FALSE(reservation.markAdmitted(makeFullBatch(19, 101, 1), transition));
    EXPECT_FALSE(reservation.markAdmitted(makePartialBatch(19, 101, {1, 3}), transition));
    EXPECT_EQ(2, reservation.getWaitingRequestCount());
    EXPECT_EQ(0, reservation.getDispatchedRequestCount());
}

TEST_F(MasterBatchReservationTest, testCompletionBeforeAdmissionIsRejected)
{
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(23, 100, 8.0,
                                 {makeRequest(1, 2, 1, 0, hashes({11, 12}))}, nullptr));
    MasterBatchReservation::Transition transition;
    EXPECT_FALSE(reservation.markCompleted(makeFullBatch(23, 101, 1), true, transition));
    EXPECT_EQ(RequestState::WAITING, reservation.getRequestState(1));
    EXPECT_EQ(1, reservation.getWaitingRequestCount());
}

TEST_F(MasterBatchReservationTest,
       testDispatchedFailureRemovesOnlyItsKeysAndLateAckCannotReactivate)
{
    const NaviSchedCachePolicy policy{1, 1, false};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(25, 100, 8.0,
                                 {makeRequest(1, 3, 1, 1, hashes({11, 12, 13})),
                                  makeRequest(2, 3, 1, 0, hashes({11, 12, 14}))},
                                 &policy));

    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(25, 101, {1}), transition));
    EXPECT_EQ(RequestState::DISPATCHED, reservation.getRequestState(1));
    EXPECT_EQ(1, reservation.getDispatchedRequestCount());
    EXPECT_EQ(1, reservation.getWaitingRequestCount());
    EXPECT_FALSE(reservation.hasPendingCacheReferences());

    ASSERT_TRUE(reservation.markFailed(1, transition));
    EXPECT_TRUE(transition.changed);
    EXPECT_EQ(RequestState::FAILED, reservation.getRequestState(1));
    EXPECT_EQ(0, reservation.getDispatchedRequestCount());
    EXPECT_EQ(1, reservation.getWaitingRequestCount());
    EXPECT_TRUE(reservation.hasPendingCacheReferences());
    expectHashes(reservation.getPendingCacheReservation(), {11, 12, 14});
    expectFeatures(reservation.getWaitingFeatures(), 1, 1, 0, 3, 9, 0);

    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(25, 102, {1}), transition));
    EXPECT_FALSE(transition.changed);
    EXPECT_EQ(RequestState::FAILED, reservation.getRequestState(1));
    ASSERT_TRUE(
        reservation.markCompleted(makePartialBatch(25, 102, {1}), true, transition));
    EXPECT_FALSE(transition.changed);
    EXPECT_EQ(RequestState::FAILED, reservation.getRequestState(1));

    ASSERT_TRUE(reservation.markAdmitted(makeFullBatch(25, 103, 2), transition));
    EXPECT_EQ((std::vector<uint64_t>{2}), transition.dispatchedRequestIds);
    EXPECT_EQ(RequestState::FAILED, reservation.getRequestState(1));
    EXPECT_EQ(RequestState::DISPATCHED, reservation.getRequestState(2));
    ASSERT_TRUE(reservation.markCompleted(makeFullBatch(25, 103, 2), true, transition));
    EXPECT_TRUE(transition.resolved);
    EXPECT_EQ(RequestState::COMPLETED, reservation.getRequestState(2));
    expectHashes(reservation.getPendingCacheReservation(), {});
}

TEST_F(MasterBatchReservationTest,
       testMultiplePartialAdmissionsAndOutOfOrderCompletionsAreIdempotent)
{
    const NaviSchedCachePolicy policy{1, 1, false};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(27, 100, 10.0,
                                 {makeRequest(1, 2, 1, 0, hashes({11, 12})),
                                  makeRequest(2, 2, 1, 0, hashes({11, 13})),
                                  makeRequest(3, 2, 1, 0, hashes({11, 14})),
                                  makeRequest(4, 2, 1, 0, hashes({11, 15}))},
                                 &policy));

    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(27, 101, {1, 3}), transition));
    EXPECT_EQ((std::vector<uint64_t>{1, 3}), transition.dispatchedRequestIds);
    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(27, 101, {1, 3}), transition));
    EXPECT_FALSE(transition.changed);

    ASSERT_TRUE(reservation.markAdmitted(makePartialBatch(27, 102, {2}), transition));
    ASSERT_TRUE(reservation.markFailed(4, transition));
    EXPECT_EQ(0, reservation.getWaitingRequestCount());
    EXPECT_EQ(3, reservation.getDispatchedRequestCount());
    EXPECT_EQ(1, reservation.getFailedRequestCount());
    expectFeatures(reservation.getWaitingFeatures(), 0, 0, 0, 0, 0, 0);

    ASSERT_TRUE(
        reservation.markCompleted(makePartialBatch(27, 102, {2}), true, transition));
    ASSERT_TRUE(
        reservation.markCompleted(makePartialBatch(27, 101, {1}), true, transition));
    ASSERT_TRUE(
        reservation.markCompleted(makePartialBatch(27, 101, {1}), true, transition));
    EXPECT_FALSE(transition.changed);
    ASSERT_TRUE(
        reservation.markCompleted(makePartialBatch(27, 103, {3}), false, transition));
    EXPECT_TRUE(transition.resolved);
    EXPECT_EQ(2, reservation.getCompletedRequestCount());
    EXPECT_EQ(2, reservation.getFailedRequestCount());
    expectHashes(reservation.getPendingCacheReservation(), {});
}

TEST_F(MasterBatchReservationTest, testDuplicateFullCompletionIsNoOp)
{
    const NaviSchedCachePolicy policy{1, 1, false};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(28, 100, 10.0,
                                 {makeRequest(1, 2, 1, 0, hashes({11, 12})),
                                  makeRequest(2, 2, 1, 0, hashes({11, 13}))},
                                 &policy));
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markAdmitted(makeFullBatch(28, 101, 2), transition));
    ASSERT_TRUE(reservation.markCompleted(makeFullBatch(28, 101, 2), true, transition));
    EXPECT_TRUE(transition.changed);
    ASSERT_TRUE(reservation.markCompleted(makeFullBatch(28, 101, 2), true, transition));
    EXPECT_FALSE(transition.changed);
    EXPECT_FALSE(transition.cacheChanged);
    EXPECT_TRUE(transition.resolved);
    EXPECT_EQ(2, reservation.getCompletedRequestCount());
}

TEST_F(MasterBatchReservationTest, testRandomizedTransitionsMatchSimpleReferenceModel)
{
    const NaviSchedCachePolicy policy{1, 1, false};
    struct ReferenceRequest {
        uint64_t requestId;
        RequestState state;
        bool dispatchAcknowledged;
        int64_t hitTokens;
        NaviSchedBlockHashesConstPtr blockHashes;
    };
    std::vector<ReferenceRequest> reference;
    std::vector<MasterBatchReservation::Request> requests;
    for (uint64_t requestId = 1; requestId <= 12; ++requestId) {
        auto blockHashes = hashes({11, 20 + requestId % 3, 100 + requestId});
        const auto hitTokens = static_cast<int64_t>(requestId % 3);
        reference.push_back(
            {requestId, RequestState::WAITING, false, hitTokens, blockHashes});
        requests.emplace_back(makeRequest(requestId, 3, 1, hitTokens, blockHashes));
    }
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(31, 100, 50.0, std::move(requests), &policy));

    uint64_t randomState = 0x9e3779b97f4a7c15ULL;
    const auto nextRandom = [&randomState]() {
        randomState = randomState * 6364136223846793005ULL + 1;
        return randomState;
    };
    int64_t localBatchId = 100;
    for (size_t step = 0; step < 256; ++step) {
        const auto operation = nextRandom() % 4;
        const auto index = nextRandom() % reference.size();
        auto &expectedRequest = reference[index];
        MasterBatchReservation::Transition transition;
        if (operation == 0) {
            const bool expectedChanged =
                expectedRequest.state == RequestState::WAITING ||
                expectedRequest.state == RequestState::DISPATCHED;
            ASSERT_TRUE(reservation.markFailed(expectedRequest.requestId, transition));
            EXPECT_EQ(expectedChanged, transition.changed);
            if (expectedChanged) {
                expectedRequest.state = RequestState::FAILED;
                expectedRequest.dispatchAcknowledged = false;
            }
        } else if (operation == 1) {
            const bool expectedChanged = expectedRequest.state == RequestState::WAITING;
            ASSERT_TRUE(reservation.markAdmitted(
                makePartialBatch(31, ++localBatchId, {expectedRequest.requestId}),
                transition));
            EXPECT_EQ(expectedChanged, transition.changed);
            if (expectedChanged) {
                expectedRequest.state = RequestState::DISPATCHED;
                expectedRequest.dispatchAcknowledged = false;
            }
        } else if (operation == 2) {
            const auto previousState = expectedRequest.state;
            const auto success = reservation.markCompleted(
                makePartialBatch(31, ++localBatchId, {expectedRequest.requestId}), true,
                transition);
            if (previousState == RequestState::WAITING) {
                EXPECT_FALSE(success);
            } else {
                ASSERT_TRUE(success);
                EXPECT_EQ(previousState == RequestState::DISPATCHED, transition.changed);
                if (previousState == RequestState::DISPATCHED) {
                    expectedRequest.state = RequestState::COMPLETED;
                }
            }
        } else {
            std::vector<uint64_t> acknowledgedRequestIds;
            ASSERT_TRUE(reservation.acknowledgeDispatch(
                makePartialBatch(31, ++localBatchId, {expectedRequest.requestId}),
                acknowledgedRequestIds));
            const bool expectedChanged = expectedRequest.state == RequestState::WAITING &&
                                         !expectedRequest.dispatchAcknowledged;
            if (expectedChanged) {
                EXPECT_EQ((std::vector<uint64_t>{expectedRequest.requestId}),
                          acknowledgedRequestIds);
                expectedRequest.dispatchAcknowledged = true;
            } else {
                EXPECT_TRUE(acknowledgedRequestIds.empty());
            }
        }

        size_t waitingCount = 0;
        size_t acknowledgedWaitingCount = 0;
        size_t dispatchedCount = 0;
        size_t failedCount = 0;
        size_t completedCount = 0;
        int64_t hitTokens = 0;
        int64_t missTokens = 0;
        double missSquareSum = 0.0;
        double hitMissProductSum = 0.0;
        std::vector<uint64_t> activeHashes;
        for (const auto &entry : reference) {
            EXPECT_EQ(entry.state, reservation.getRequestState(entry.requestId));
            EXPECT_EQ(entry.dispatchAcknowledged,
                      reservation.isDispatchAcknowledged(entry.requestId));
            if (entry.state == RequestState::WAITING) {
                ++waitingCount;
                acknowledgedWaitingCount += entry.dispatchAcknowledged ? 1 : 0;
                hitTokens += entry.hitTokens;
                const auto misses =
                    static_cast<int64_t>(entry.blockHashes->size()) - entry.hitTokens;
                missTokens += misses;
                missSquareSum += static_cast<double>(misses) * misses;
                hitMissProductSum += static_cast<double>(entry.hitTokens) * misses;
            } else if (entry.state == RequestState::DISPATCHED) {
                ++dispatchedCount;
            } else if (entry.state == RequestState::FAILED) {
                ++failedCount;
            } else {
                ++completedCount;
            }
            if (entry.state == RequestState::WAITING ||
                entry.state == RequestState::DISPATCHED) {
                activeHashes.insert(activeHashes.end(), entry.blockHashes->begin(),
                                    entry.blockHashes->end());
            }
        }
        std::sort(activeHashes.begin(), activeHashes.end());
        activeHashes.erase(std::unique(activeHashes.begin(), activeHashes.end()),
                           activeHashes.end());
        EXPECT_EQ(waitingCount, reservation.getWaitingRequestCount());
        EXPECT_EQ(acknowledgedWaitingCount,
                  reservation.getAcknowledgedWaitingRequestCount());
        EXPECT_EQ(dispatchedCount, reservation.getDispatchedRequestCount());
        EXPECT_EQ(failedCount, reservation.getFailedRequestCount());
        EXPECT_EQ(completedCount, reservation.getCompletedRequestCount());
        EXPECT_EQ(waitingCount == 0 && dispatchedCount == 0, reservation.isResolved());
        expectFeatures(reservation.getWaitingFeatures(), waitingCount ? 1 : 0,
                       waitingCount, hitTokens, missTokens, missSquareSum,
                       hitMissProductSum);
        EXPECT_EQ(activeHashes,
                  reservation.getPendingCacheReservation().fullReadyBlockHashes);
        if (activeHashes.empty()) {
            EXPECT_EQ(0, reservation.getPendingCacheReservation().scheduleBatchId);
        } else {
            EXPECT_EQ(31, reservation.getPendingCacheReservation().scheduleBatchId);
        }
    }
}

TEST_F(MasterBatchReservationTest, testBatchWithoutMatchableBlocks)
{
    const NaviSchedCachePolicy policy{4, 2, true};
    MasterBatchReservation reservation;
    ASSERT_TRUE(
        reservation.init(29, 100, 2.0, {makeRequest(1, 3, 4, 0, hashes({}))}, &policy));
    EXPECT_FALSE(reservation.hasPendingCacheReferences());
    expectHashes(reservation.getPendingCacheReservation(), {});
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(reservation.markFailed(1, transition));
    EXPECT_TRUE(transition.resolved);
    EXPECT_FALSE(reservation.hasPendingCacheReferences());
}

} // namespace
} // namespace navi_sched
