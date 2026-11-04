#include "rtp_llm/flexlb/navi_sched/resource/ClientResponseR.h"
#include "unittest/unittest.h"
#include <atomic>
#include <chrono>
#include <thread>

using namespace navi_sched;

class ClientResponseRTest : public TESTBASE {
};

namespace {

MasterResponseToClientDataPtr makeDecision(uint64_t requestId)
{
    auto decision = std::make_shared<MasterResponseToClientData>();
    decision->requestId = requestId;
    decision->target.address = "127.0.0.1:8000";
    return decision;
}

AgentResponseToClientDataPtr makeOutput(uint64_t requestId,
                                        const std::string &frame,
                                        bool terminal)
{
    auto output = std::make_shared<AgentResponseToClientData>();
    output->requestId = requestId;
    output->kind = AgentResponseKind::OUTPUT;
    output->frames = {frame};
    output->terminal = terminal;
    return output;
}

} // namespace

TEST_F(ClientResponseRTest, testStreamsResponsesUntilGraphFinishes)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(1, false));
    ASSERT_TRUE(resource.publishDecision(1, makeDecision(101)));
    auto decision = resource.waitDecision(1, 100);
    ASSERT_EQ(navi::EC_NONE, decision.errorCode);
    ASSERT_EQ(1, decision.decision->requestId);

    ASSERT_TRUE(resource.publishResponse(1, makeOutput(101, "first", false)));
    ASSERT_TRUE(resource.publishResponse(1, makeOutput(101, "last", true)));
    auto first = resource.takeNextResponse(100);
    auto last = resource.takeNextResponse(100);
    ASSERT_TRUE(first);
    ASSERT_TRUE(last);
    EXPECT_EQ(std::vector<std::string>({"first"}), first->frames);
    EXPECT_FALSE(first->terminal);
    EXPECT_EQ(std::vector<std::string>({"last"}), last->frames);
    EXPECT_TRUE(last->terminal);

    resource.finishGraph(1, navi::EC_NONE, "");
    EXPECT_EQ(0, resource.requestCount());
    EXPECT_FALSE(resource.takeNextResponse(0));
}

TEST_F(ClientResponseRTest, testConcurrentRequestsKeepTheirIds)
{
    ClientResponseR resource;
    for (uint64_t requestId : {1, 2}) {
        ASSERT_TRUE(resource.begin(requestId, false));
        ASSERT_TRUE(resource.publishDecision(requestId,
                                             makeDecision(100 + requestId)));
        ASSERT_EQ(navi::EC_NONE,
                  resource.waitDecision(requestId, 100).errorCode);
    }

    ASSERT_TRUE(resource.publishResponse(2, makeOutput(102, "two", true)));
    ASSERT_TRUE(resource.publishResponse(1, makeOutput(101, "one", true)));
    auto two = resource.takeNextResponse(100);
    auto one = resource.takeNextResponse(100);
    ASSERT_TRUE(two);
    ASSERT_TRUE(one);
    EXPECT_EQ(2, two->requestId);
    EXPECT_EQ(1, one->requestId);
    resource.finishGraph(1, navi::EC_NONE, "");
    resource.finishGraph(2, navi::EC_NONE, "");
    EXPECT_EQ(0, resource.requestCount());
}

TEST_F(ClientResponseRTest, testGraphFailureSynthesizesTerminalResponse)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(3, false));
    ASSERT_TRUE(resource.publishDecision(3, makeDecision(3)));
    ASSERT_EQ(navi::EC_NONE, resource.waitDecision(3, 100).errorCode);

    resource.finishGraph(3, navi::EC_ABORT, "agent disconnected");
    auto response = resource.takeNextResponse(100);
    ASSERT_TRUE(response);
    EXPECT_EQ(3, response->requestId);
    EXPECT_EQ(AgentResponseKind::ERROR, response->kind);
    EXPECT_TRUE(response->terminal);
    EXPECT_EQ("agent disconnected", response->errorMessage);
    EXPECT_EQ(0, resource.requestCount());
}

TEST_F(ClientResponseRTest, testSuppressedResponseModeConsumesStreamWithoutQueue)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(31, false, false));
    ASSERT_TRUE(resource.publishDecision(31, makeDecision(131)));
    ASSERT_EQ(navi::EC_NONE, resource.waitDecision(31, 100).errorCode);

    ASSERT_TRUE(resource.publishResponse(31, makeOutput(131, "token-1", false)));
    ASSERT_TRUE(resource.publishResponse(31, makeOutput(131, "token-2", false)));
    ASSERT_TRUE(resource.publishResponse(31, makeOutput(131, "token-3", true)));
    resource.finishGraph(31, navi::EC_NONE, "");

    EXPECT_EQ(0, resource.requestCount());
    EXPECT_FALSE(resource.takeNextResponse(0));
}

TEST_F(ClientResponseRTest, testSuppressedResponseModeDoesNotSynthesizeGraphError)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(32, false, false));
    ASSERT_TRUE(resource.publishDecision(32, makeDecision(32)));
    ASSERT_EQ(navi::EC_NONE, resource.waitDecision(32, 100).errorCode);

    resource.finishGraph(32, navi::EC_NONE, "");

    EXPECT_EQ(0, resource.requestCount());
    EXPECT_FALSE(resource.takeNextResponse(0));
}

TEST_F(ClientResponseRTest, testDetachedGraphCompletionReturnsDecisionAndCleansState)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(33, false, false, true));

    ASSERT_TRUE(resource.publishDecision(33, makeDecision(133)));
    EXPECT_EQ(1u, resource.requestCount());

    ASSERT_TRUE(resource.publishResponse(33, makeOutput(133, "token", false)));
    ASSERT_TRUE(resource.publishResponse(33, makeOutput(133, "terminal", true)));
    EXPECT_FALSE(resource.takeNextResponse(0));
    auto result = resource.finishGraph(33, navi::EC_NONE, "", true);

    ASSERT_EQ(navi::EC_NONE, result.errorCode);
    ASSERT_TRUE(result.decision);
    EXPECT_EQ(33u, result.decision->requestId);
    EXPECT_GT(result.decisionTimeUs, 0);
    EXPECT_EQ(1u, resource.requestCount());
    resource.finishClosureCompletion(33);
    EXPECT_EQ(0u, resource.requestCount());
}

TEST_F(ClientResponseRTest, testDetachedGraphCompletionReturnsFailureWithoutDecision)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(34, false, false, true));

    auto result = resource.finishGraph(
        34, navi::EC_STREAM_RECEIVE, "master stream failed", true);

    EXPECT_EQ(navi::EC_STREAM_RECEIVE, result.errorCode);
    EXPECT_EQ("master stream failed", result.errorMessage);
    EXPECT_EQ(1u, resource.requestCount());
    resource.finishClosureCompletion(34);
    EXPECT_EQ(0u, resource.requestCount());
}

TEST_F(ClientResponseRTest, testReplayWaitPreservesGraphErrorAfterDecision)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(36, true, false));
    ASSERT_TRUE(resource.publishDecision(36, makeDecision(136)));
    resource.finishGraph(36, navi::EC_STREAM_RECEIVE, "replay stream failed");

    const auto result = resource.waitDecision(36, 100);
    EXPECT_EQ(navi::EC_STREAM_RECEIVE, result.errorCode);
    EXPECT_EQ("replay stream failed", result.errorMessage);
    EXPECT_FALSE(result.decision);
    EXPECT_FALSE(result.replay);
    EXPECT_EQ(0u, resource.requestCount());
}

TEST_F(ClientResponseRTest, testReplayWaitRejectsCompletedGraphWithoutReplay)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(37, true, false));
    ASSERT_TRUE(resource.publishDecision(37, makeDecision(137)));
    resource.finishGraph(37, navi::EC_NONE, "");

    const auto result = resource.waitDecision(37, 100);
    EXPECT_EQ(navi::EC_NO_DATA, result.errorCode);
    EXPECT_EQ("schedule graph returned no decision", result.errorMessage);
    EXPECT_FALSE(result.decision);
    EXPECT_FALSE(result.replay);
    EXPECT_EQ(0u, resource.requestCount());
}

TEST_F(ClientResponseRTest, testReplayWaitReturnsBothOutputs)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(38, true, false));
    ASSERT_TRUE(resource.publishDecision(38, makeDecision(138)));
    auto replay = std::make_shared<MasterResponseToClientData>();
    replay->replay = std::make_shared<NaviSchedScheduleReplay>();
    ASSERT_TRUE(resource.publishReplay(38, replay));

    const auto result = resource.waitDecision(38, 100);
    ASSERT_EQ(navi::EC_NONE, result.errorCode);
    ASSERT_TRUE(result.decision);
    EXPECT_EQ(38u, result.decision->requestId);
    EXPECT_EQ(replay->replay, result.replay);
    resource.finishGraph(38, navi::EC_NONE, "");
    EXPECT_EQ(0u, resource.requestCount());
}

TEST_F(ClientResponseRTest, testStopWaitsForDetachedClosureCompletion)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(35, false, false, true));
    ASSERT_TRUE(resource.publishDecision(35, makeDecision(135)));
    ASSERT_EQ(navi::EC_NONE,
              resource.finishGraph(35, navi::EC_NONE, "", true).errorCode);

    std::atomic<bool> stopped{false};
    std::thread stopper([&]() {
        resource.finishAll();
        stopped.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_FALSE(stopped.load(std::memory_order_acquire));

    resource.finishClosureCompletion(35);
    stopper.join();
    EXPECT_TRUE(stopped.load(std::memory_order_acquire));
    EXPECT_EQ(0u, resource.requestCount());
}

TEST_F(ClientResponseRTest, testStopUnblocksDecisionWaiter)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(4, false));
    ClientResponseR::DecisionResult result;
    std::thread waiter([&]() { result = resource.waitDecision(4, 5000); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::atomic<bool> stopFinished(false);
    std::thread stopper([&]() {
        resource.finishAll();
        stopFinished.store(true);
    });
    waiter.join();

    EXPECT_EQ(navi::EC_ABORT, result.errorCode);
    EXPECT_EQ("NaviSched client stopped", result.errorMessage);
    EXPECT_FALSE(stopFinished.load());
    EXPECT_EQ(1, resource.requestCount());
    resource.finishGraph(4, navi::EC_ABORT, "aborted");
    stopper.join();

    EXPECT_TRUE(stopFinished.load());
    EXPECT_EQ(0, resource.requestCount());
    EXPECT_FALSE(resource.begin(5, false));
}

TEST_F(ClientResponseRTest, testConcurrentFinishAllWaitsForEveryGraph)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(5, false));
    ASSERT_TRUE(resource.begin(6, false));
    ClientResponseR::DecisionResult firstResult;
    ClientResponseR::DecisionResult secondResult;
    std::thread firstWaiter(
        [&]() { firstResult = resource.waitDecision(5, 5000); });
    std::thread secondWaiter(
        [&]() { secondResult = resource.waitDecision(6, 5000); });

    std::atomic<size_t> stopCount(0);
    std::thread firstStopper([&]() {
        resource.finishAll();
        stopCount.fetch_add(1);
    });
    firstWaiter.join();
    secondWaiter.join();
    ASSERT_EQ(navi::EC_ABORT, firstResult.errorCode);
    ASSERT_EQ(navi::EC_ABORT, secondResult.errorCode);

    std::thread secondStopper([&]() {
        resource.finishAll();
        stopCount.fetch_add(1);
    });
    resource.finishGraph(5, navi::EC_ABORT, "aborted");
    EXPECT_EQ(0, stopCount.load());
    resource.finishGraph(6, navi::EC_ABORT, "aborted");
    firstStopper.join();
    secondStopper.join();

    EXPECT_EQ(2, stopCount.load());
    EXPECT_EQ(0, resource.requestCount());
}

TEST_F(ClientResponseRTest, testAbortBeforeAttachIsRemembered)
{
    ClientResponseR resource;
    ASSERT_TRUE(resource.begin(7, false));
    EXPECT_TRUE(resource.abort(7));
    EXPECT_TRUE(resource.abort(7));

    resource.finishGraph(7, navi::EC_ABORT, "aborted before attach");
    auto result = resource.waitDecision(7, 100);
    EXPECT_EQ(navi::EC_ABORT, result.errorCode);
    EXPECT_EQ("aborted before attach", result.errorMessage);
    EXPECT_EQ(0, resource.requestCount());
}
