#include "rtp_llm/flexlb/navi_sched/resource/MasterStateR.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/ScheduleContextInternal.h"
#include "rtp_llm/flexlb/navi_sched/test/ScheduleReplayRunner.h"
#include "rtp_llm/flexlb/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "autil/legacy/jsonizable.h"
#include "autil/TimeUtility.h"
#include "unittest/unittest.h"

#include <algorithm>
#include <atomic>
#include <thread>

namespace navi_sched {
namespace {

class MutableReplayMatchNode final : public NaviSchedNode {
public:
    NaviSchedNodeInfo getInfo() const override
    {
        NaviSchedNodeInfo info;
        info.agentId = "agent";
        info.bizName = "model";
        info.online = true;
        info.totalBlocks = 100;
        return info;
    }

    bool getScheduleSnapshot(const std::string &,
                             NaviSchedNodeSnapshot &) const override
    {
        return false;
    }

    int64_t getMatchedBlockCount(const NaviSchedBlockHashes &) const override
    {
        ++queryCount;
        return liveMatch;
    }

    void getMatchedBlockCounts(const std::vector<NaviSchedScheduleItem> &requests,
                               int64_t *counts) const override
    {
        ++queryCount;
        if (expectedRequests) {
            EXPECT_EQ(expectedRequests, &requests);
        }
        observedRequestIds.clear();
        for (size_t index = 0; index < requests.size(); ++index) {
            counts[index] = liveMatch;
            observedRequestIds.push_back(requests[index].request.requestId);
        }
    }

    int64_t liveMatch = 1;
    mutable size_t queryCount = 0;
    const std::vector<NaviSchedScheduleItem> *expectedRequests = nullptr;
    mutable std::vector<uint64_t> observedRequestIds;
};

std::vector<NaviSchedScheduleItem> makeReplayMatchRequests()
{
    std::vector<NaviSchedScheduleItem> requests(2);
    for (size_t index = 0; index < requests.size(); ++index) {
        auto &request = requests[index].request;
        request.requestId = index + 1;
        request.bizName = "model";
        request.promptTokenCount = 32;
        request.blockSize = 16;
        request.blockHashes = std::make_shared<const NaviSchedBlockHashes>(
            NaviSchedBlockHashes{11, 22});
        request.timeoutMs = 1000;
        requests[index].remainingTimeoutMs = 1000;
    }
    return requests;
}

} // namespace

class ScheduleContextReplayTest : public TESTBASE {};

TEST_F(ScheduleContextReplayTest, testCachePrefixPreparationTracksViewOrderAndResets)
{
    auto requests = makeReplayMatchRequests();
    requests.resize(5);
    requests[0].request.blockHashes = std::make_shared<const NaviSchedBlockHashes>(
        NaviSchedBlockHashes{11, 22, 33});
    requests[1].request.blockHashes = std::make_shared<const NaviSchedBlockHashes>(
        NaviSchedBlockHashes{11, 22, 44});
    requests[2].request.blockHashes.reset();
    requests[3].request.blockHashes = requests[0].request.blockHashes;
    requests[4].request.blockHashes = requests[0].request.blockHashes;
    NaviSchedScheduleContext context;
    ScheduleReplayRunner::initialize(context, 1000, std::move(requests),
                                    {std::make_shared<MutableReplayMatchNode>()});
    context.prepareCacheMatchPrefixes();
    EXPECT_EQ(std::vector<size_t>({0, 2, 0, 0, 3}), context._cacheMatchCommonPrefixes);
    ASSERT_TRUE(context.activateScheduleView({4, 1, 0}, {0}));
    EXPECT_TRUE(context._cacheMatchCommonPrefixes.empty());
    context.prepareCacheMatchPrefixes();
    EXPECT_EQ(std::vector<size_t>({0, 2, 2}), context._cacheMatchCommonPrefixes);
    context.clearScheduleView();
    EXPECT_TRUE(context._cacheMatchCommonPrefixes.empty());
    context.prepareCacheMatchPrefixes();
    EXPECT_EQ(std::vector<size_t>({0, 2, 0, 0, 3}), context._cacheMatchCommonPrefixes);
}

TEST_F(ScheduleContextReplayTest, testIdentityCostViewQueriesOriginalRequests)
{
    auto first = std::make_shared<MutableReplayMatchNode>();
    auto second = std::make_shared<MutableReplayMatchNode>();
    NaviSchedScheduleContext context;
    ScheduleReplayRunner::initialize(context, 1000, makeReplayMatchRequests(), {first, second});
    first->expectedRequests = &context._requests;
    second->expectedRequests = &context._requests;
    ASSERT_TRUE(context.activateScheduleView({0, 1}, {1, 0}));
    std::vector<int64_t> matches;
    for (size_t n = 0; n < 2; ++n) {
        ASSERT_TRUE(context.getCacheHitBlockCounts(n, matches));
        EXPECT_EQ(std::vector<int64_t>({1, 1}), matches);
    }
    EXPECT_EQ(std::vector<uint64_t>({1, 2}), first->observedRequestIds);
    EXPECT_EQ(std::vector<uint64_t>({1, 2}), second->observedRequestIds);
    context.clearScheduleView();
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(2, first->queryCount);
}

TEST_F(ScheduleContextReplayTest, testFilteredCostViewIsSharedByParallelNodeQueries)
{
    std::vector<NaviSchedNodeConstPtr> nodes;
    std::vector<std::shared_ptr<MutableReplayMatchNode>> workers;
    for (size_t i = 0; i < 10; ++i) {
        workers.push_back(std::make_shared<MutableReplayMatchNode>());
        nodes.push_back(workers.back());
    }
    NaviSchedScheduleContext context;
    ScheduleReplayRunner::initialize(context, 1000, makeReplayMatchRequests(), nodes);
    auto hashes = context._requests[1].request.blockHashes;
    const auto baselineReferences = hashes.use_count();
    EXPECT_FALSE(context.activateScheduleView({1, 1}, {0}));
    EXPECT_FALSE(context._scheduleViewActive);
    EXPECT_TRUE(context._scheduleViewRequests.empty());
    ASSERT_TRUE(context.activateScheduleView({1}, {9, 7, 5, 3, 1, 0, 2, 4, 6, 8}));
    EXPECT_EQ(baselineReferences + 1, hashes.use_count());
    EXPECT_FALSE(context._scheduleViewUsesAllRequests);
    std::vector<std::thread> threads;
    for (const auto &worker : workers) {
        worker->expectedRequests = &context._scheduleViewRequests;
    }
    for (size_t i = 0; i < workers.size(); ++i) {
        threads.emplace_back([&context, i]() {
            for (size_t repeat = 0; repeat < 100; ++repeat) {
                std::vector<int64_t> matches;
                ASSERT_TRUE(context.getCacheHitBlockCounts(i, matches));
                EXPECT_EQ(std::vector<int64_t>({1}), matches);
            }
        });
    }
    for (auto &thread : threads) thread.join();
    for (const auto &worker : workers) {
        EXPECT_EQ(100, worker->queryCount);
        EXPECT_EQ(std::vector<uint64_t>({2}), worker->observedRequestIds);
        worker->expectedRequests = &context._requests;
    }
    EXPECT_EQ(baselineReferences + 1, hashes.use_count());
    context.clearScheduleView();
    EXPECT_EQ(baselineReferences, hashes.use_count());
    EXPECT_TRUE(context._scheduleViewRequests.empty());
    // A full permutation is a filtered view too; it must keep its order.
    ASSERT_TRUE(context.activateScheduleView({1, 0}, {0}));
    workers[0]->expectedRequests = &context._scheduleViewRequests;
    std::vector<int64_t> matches;
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<uint64_t>({2, 1}), workers[0]->observedRequestIds);
    context.clearScheduleView();
    EXPECT_EQ(baselineReferences, hashes.use_count());
}

TEST_F(ScheduleContextReplayTest, testCostViewPreparationFixedWork)
{
    const size_t nodeCount = 432;
    for (size_t requestCount : {size_t{128}, size_t{512}}) {
        auto node = std::make_shared<MutableReplayMatchNode>();
        auto requests = makeReplayMatchRequests();
        requests.resize(requestCount, requests.front());
        for (size_t i = 0; i < requestCount; ++i) {
            requests[i].request.requestId = i + 1;
            requests[i].request.bizName = "omega_llm_976371_5613449_DeepSeek-VL";
        }
        NaviSchedScheduleContext context;
        ScheduleReplayRunner::initialize(context, 1000, requests,
                                         std::vector<NaviSchedNodeConstPtr>(nodeCount, node));
        std::vector<size_t> nodeIndexes(nodeCount);
        for (size_t i = 0; i < nodeCount; ++i) nodeIndexes[i] = i;
        for (bool filtered : {false, true}) {
            std::vector<size_t> requestIndexes;
            for (size_t i = 0; i < requestCount; ++i) {
                if (!filtered || i % 2) requestIndexes.push_back(i);
            }
            std::vector<int64_t> durations;
            for (size_t repeat = 0; repeat < 9; ++repeat) {
                const auto begin = autil::TimeUtility::monotonicTimeUs();
                ASSERT_TRUE(context.activateScheduleView(requestIndexes, nodeIndexes));
                std::vector<int64_t> matches;
                for (size_t n = 0; n < nodeCount; ++n) {
                    ASSERT_TRUE(context.getCacheHitBlockCounts(n, matches));
                }
                context.clearScheduleView();
                const auto elapsed = autil::TimeUtility::monotonicTimeUs() - begin;
                EXPECT_EQ(std::vector<int64_t>(requestIndexes.size(), 1), matches);
                EXPECT_EQ(requestIndexes.back() + 1, node->observedRequestIds.back());
                if (repeat > 0) durations.push_back(elapsed);
            }
            std::sort(durations.begin(), durations.end());
            const auto prefix = std::to_string(requestCount) + (filtered ? "_filtered" : "_identity");
            RecordProperty(prefix + "_median_us", std::to_string(durations[4]));
            RecordProperty(prefix + "_min_us", std::to_string(durations.front()));
            RecordProperty(prefix + "_max_us", std::to_string(durations.back()));
        }
    }
}

TEST_F(ScheduleContextReplayTest, testPendingReplayQueriesRealIndex)
{
    MasterPendingCacheR pending;
    NaviSchedBlockHashes hashes{11, 22};
    ASSERT_TRUE(pending.reserve(42, 101, 1000, {{&hashes, 32, 16}}, {16, 1, false}));
    MasterStateR::NodeState node;
    node._info = MutableReplayMatchNode().getInfo();
    node._info.agentIdHash = 42;
    node._cachePolicy = {16, 1, false};
    node._cachePolicyReady = true;
    node._pendingCacheR = &pending;
    NaviSchedScheduleReplay replay;
    replay.bizName = "model";
    replay.scheduleTimeUs = 1000;
    replay.requests = makeReplayMatchRequests();
    for (auto &item : replay.requests) item.request.promptTokenCount = 33;
    replay.nodes = {node._info};
    // Deliberately wrong recorded matches: restoration must query the real index.
    replay.nodeMatches = {{{0}}, {{0}}};
    replay.schedulerInfo["pending_state"] = autil::legacy::ToJson(
        std::vector<autil::legacy::json::JsonMap>{node.makePendingReplayState()});
    NaviSchedScheduleReplay decoded;
    NaviSchedScheduleReplayJson::fromJsonString(NaviSchedScheduleReplayJson::toJsonString(replay), decoded);
    NaviSchedScheduleContext context;
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(decoded, context));
    std::vector<int64_t> matches;
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<int64_t>({2, 2}), matches);
    EXPECT_EQ(2, context.getCacheHitBlockCount(0, 0));
    auto output = std::make_shared<NaviSchedScheduleReplay>(decoded);
    ScheduleReplayRunner::setReplay(context, output);
    context.setSchedulerInfo({});
    EXPECT_EQ(1u, output->schedulerInfo.count("pending_state"));
}

TEST_F(ScheduleContextReplayTest, testPendingReplaySharesIndexAndReadsAtomicNodeBatches)
{
    MasterPendingCacheR recorded;
    NaviSchedBlockHashes full{11, 22}, shortPrefix{11};
    const NaviSchedCachePolicy policy{16, 1, false};
    ASSERT_TRUE(recorded.reserve(42, 101, 1000, {{&full, 32, 16}}, policy));
    ASSERT_TRUE(recorded.reserve(43, 102, 1000, {{&shortPrefix, 16, 16}}, policy));
    NaviSchedScheduleReplay replay;
    replay.bizName = "model";
    replay.scheduleTimeUs = 1000;
    replay.requests = makeReplayMatchRequests();
    for (auto &item : replay.requests) item.request.promptTokenCount = 33;
    std::vector<autil::legacy::json::JsonMap> states;
    for (const uint64_t agent : {42, 43}) {
        MasterStateR::NodeState node;
        node._info = MutableReplayMatchNode().getInfo();
        node._info.agentId = std::to_string(agent);
        node._info.agentIdHash = agent;
        node._cachePolicy = policy;
        node._cachePolicyReady = true;
        node._pendingCacheR = &recorded;
        replay.nodes.push_back(node._info);
        states.push_back(node.makePendingReplayState());
    }
    replay.nodeMatches = {{{0}, {0}}, {{0}, {0}}};
    replay.schedulerInfo["pending_state"] = autil::legacy::ToJson(states);
    auto pending = std::make_shared<MasterPendingCacheR>();
    NaviSchedScheduleContext context;
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context, pending));
    EXPECT_EQ(1u, pending->getStats(42).batchCount);
    EXPECT_EQ(2u, pending->getStats(42).uniqueBlockCount);
    EXPECT_EQ(1u, pending->getStats(43).uniqueBlockCount);
    std::vector<int64_t> matches;
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<int64_t>({2, 2}), matches);
    ASSERT_TRUE(context.getCacheHitBlockCounts(1, matches));
    EXPECT_EQ(std::vector<int64_t>({1, 1}), matches);
    NaviSchedPendingCacheReservation shortReservation, fullReservation;
    ASSERT_TRUE(PendingCacheUtil::makeReservation(
        101, {{&shortPrefix, 16, 16}}, policy, shortReservation));
    ASSERT_TRUE(PendingCacheUtil::makeReservation(
        101, {{&full, 32, 16}}, policy, fullReservation));
    ASSERT_TRUE(pending->updateScheduled(42, shortReservation));
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<int64_t>({1, 1}), matches);
    EXPECT_EQ(1, context.getCacheHitBlockCount(0, 0));
    EXPECT_EQ(1u, pending->getStats(43).uniqueBlockCount);

    std::atomic<bool> start{false};
    std::atomic<size_t> invalidReads{0}, completedReads{0};
    std::vector<std::thread> readers;
    for (size_t worker = 0; worker < 4; ++worker) {
        readers.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            for (size_t iteration = 0; iteration < 1000; ++iteration) {
                std::vector<int64_t> first, second;
                if (!context.getCacheHitBlockCounts(0, first) ||
                    !context.getCacheHitBlockCounts(1, second) ||
                    (first != std::vector<int64_t>({1, 1}) &&
                     first != std::vector<int64_t>({2, 2})) ||
                    second != std::vector<int64_t>({1, 1})) {
                    invalidReads.fetch_add(1, std::memory_order_relaxed);
                }
                completedReads.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (size_t iteration = 0; iteration < 1000; ++iteration) {
        EXPECT_TRUE(pending->updateScheduled(
            42, iteration % 2 == 0 ? fullReservation : shortReservation));
    }
    for (auto &reader : readers) reader.join();
    EXPECT_EQ(4000u, completedReads.load());
    EXPECT_EQ(0u, invalidReads.load());
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<int64_t>({1, 1}), matches);
    EXPECT_EQ(1u, pending->getStats(42).uniqueBlockCount);
    EXPECT_EQ(1u, pending->getStats(43).uniqueBlockCount);
}

TEST_F(ScheduleContextReplayTest, testCapturedMatchesDoNotRequeryChangingCache)
{
    auto node = std::make_shared<MutableReplayMatchNode>();
    NaviSchedScheduleContext context;
    ScheduleReplayRunner::initialize(context, 1000, makeReplayMatchRequests(), {node});
    MasterStateR master;
    auto replay = master.makeScheduleReplay(context);
    ASSERT_EQ(1, node->queryCount);
    ASSERT_EQ(1, replay->nodeMatches[0][0].matchedBlockCount);
    ScheduleReplayRunner::setReplay(context, replay);

    node->liveMatch = 0;
    std::vector<int64_t> matches;
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<int64_t>({1, 1}), matches);
    EXPECT_EQ(1, context.getCacheHitBlockCount(1, 0));
    EXPECT_EQ(1, node->queryCount);
    EXPECT_EQ(0, context.getCacheHitBlockCount(2, 0));
    EXPECT_FALSE(context.getCacheHitBlockCounts(1, matches));
}

TEST_F(ScheduleContextReplayTest, testCapturedMatchesFollowScheduleViewIndexes)
{
    auto first = std::make_shared<MutableReplayMatchNode>();
    auto second = std::make_shared<MutableReplayMatchNode>();
    NaviSchedScheduleContext context;
    ScheduleReplayRunner::initialize(context, 1000, makeReplayMatchRequests(),
                                     {first, second});
    MasterStateR master;
    auto replay = master.makeScheduleReplay(context);
    replay->nodeMatches[0][1].matchedBlockCount = 0;
    replay->nodeMatches[1][1].matchedBlockCount = 2;
    ScheduleReplayRunner::setReplay(context, replay);
    ASSERT_TRUE(context.activateScheduleView({1, 0}, {1}));
    std::vector<int64_t> matches;
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<int64_t>({2, 0}), matches);
    EXPECT_EQ(2, context.getCacheHitBlockCount(0, 0));
    EXPECT_EQ(0, context.getCacheHitBlockCount(1, 0));
    EXPECT_EQ(1, first->queryCount);
    EXPECT_EQ(1, second->queryCount);
    context.clearScheduleView();
    ASSERT_TRUE(context.getCacheHitBlockCounts(1, matches));
    EXPECT_EQ(std::vector<int64_t>({0, 2}), matches);
}

TEST_F(ScheduleContextReplayTest, testTraceOnlyReplayKeepsLiveQueries)
{
    auto node = std::make_shared<MutableReplayMatchNode>();
    NaviSchedScheduleContext context;
    ScheduleReplayRunner::initialize(context, 1000, makeReplayMatchRequests(), {node});
    ScheduleReplayRunner::setReplay(context, std::make_shared<NaviSchedScheduleReplay>());
    node->liveMatch = 2;
    std::vector<int64_t> matches;
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<int64_t>({2, 2}), matches);
    EXPECT_EQ(2, context.getCacheHitBlockCount(0, 0));
    EXPECT_EQ(2, node->queryCount);
}

TEST_F(ScheduleContextReplayTest, testCapturedMatchesSurviveRequestCompaction)
{
    auto node = std::make_shared<MutableReplayMatchNode>();
    NaviSchedScheduleContext context;
    ScheduleReplayRunner::initialize(context, 1000, makeReplayMatchRequests(), {node});
    MasterStateR master;
    auto replay = master.makeScheduleReplay(context);
    replay->nodeMatches[0][0].matchedBlockCount = 0;
    replay->nodeMatches[1][0].matchedBlockCount = 2;
    ScheduleReplayRunner::setReplay(context, replay);
    context._requests.erase(context._requests.begin());
    std::vector<int64_t> matches;
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<int64_t>({2}), matches);
    EXPECT_EQ(2, context.getCacheHitBlockCount(0, 0));
    EXPECT_EQ(1, node->queryCount);

    ScheduleReplayRunner::setReplay(context, nullptr);
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<int64_t>({1}), matches);
    EXPECT_EQ(2, node->queryCount);
}

TEST_F(ScheduleContextReplayTest, testIncompleteCapturedMatrixKeepsLiveQueries)
{
    auto node = std::make_shared<MutableReplayMatchNode>();
    NaviSchedScheduleContext context;
    ScheduleReplayRunner::initialize(context, 1000, makeReplayMatchRequests(), {node});
    MasterStateR master;
    auto replay = master.makeScheduleReplay(context);
    replay->nodeMatches[1].clear();
    ScheduleReplayRunner::setReplay(context, replay);
    node->liveMatch = 2;
    std::vector<int64_t> matches;
    ASSERT_TRUE(context.getCacheHitBlockCounts(0, matches));
    EXPECT_EQ(std::vector<int64_t>({2, 2}), matches);
    EXPECT_EQ(2, context.getCacheHitBlockCount(0, 0));
    EXPECT_EQ(3, node->queryCount);
}

} // namespace navi_sched
