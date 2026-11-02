#include "aios/navi_sched/adapters/rtp_llm/RtpLLMAdapter.h"

#include <navi_sched/NaviSchedAgent.h>
#include "aios/navi_sched/data/AgentResponseToClientData.h"
#include "aios/navi_sched/resource/AgentBatchR.h"
#include "aios/navi_sched/resource/AgentRequestR.h"
#include "aios/navi_sched/service/NaviSchedAgentImpl.h"
#include "aios/navi_sched/service/NaviSchedRoleRuntimeAccess.h"
#include "autil/EnvUtil.h"
#include "autil/legacy/json.h"
#include "navi/engine/AsyncPipe.h"
#include "unittest/unittest.h"
#include <unistd.h>

using namespace navi_sched;

namespace {
NaviSchedBlockHashesConstPtr makeBlockHashes(std::initializer_list<uint64_t> values)
{
    return std::make_shared<const NaviSchedBlockHashes>(values);
}

NaviSchedRequestStats makeRequestStats(uint64_t requestId)
{
    NaviSchedRequestStats request;
    request.requestId = requestId;
    return request;
}

NaviSchedBatchRequest makeBatchRequest(uint64_t requestId)
{
    NaviSchedBatchRequest request;
    request.requestId = requestId;
    return request;
}

NaviSchedBatchRequest makeBatchRequest(const NaviSchedRequestStats &stats)
{
    NaviSchedBatchRequest request;
    request.requestId = stats.requestId;
    request.totalTokenCount = stats.totalTokenCount;
    return request;
}

void initializeCachePolicy(const std::shared_ptr<NaviSchedAgent> &agent)
{
    reportRtpLLMCacheStatus(agent, 1024, 256, 1, false, 100);
}

class CaptureReadyBatchHandler {
public:
    void handleReadyBatch(const NaviSchedBatchPtr &batch)
    {
        autil::ScopedLock lock(_condition);
        _batch = batch;
        _called = true;
        _condition.broadcast();
    }

    bool wait(int64_t timeoutUs)
    {
        autil::ScopedLock lock(_condition);
        if (!_called) {
            _condition.wait(timeoutUs);
        }
        return _called;
    }

    NaviSchedBatchPtr getBatch() const
    {
        autil::ScopedLock lock(_condition);
        return _batch;
    }

private:
    mutable autil::ThreadCond _condition;
    NaviSchedBatchPtr _batch;
    bool _called = false;
};

class CaptureCompletionPipe : public navi::AsyncPipe {
public:
    CaptureCompletionPipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}

    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        autil::ScopedLock lock(_condition);
        _data = data;
        return navi::EC_NONE;
    }

    navi::ErrorCode setEof() override
    {
        autil::ScopedLock lock(_condition);
        _eof = true;
        _condition.broadcast();
        return navi::EC_NONE;
    }

    bool waitForEof(int64_t timeoutUs)
    {
        autil::ScopedLock lock(_condition);
        if (!_eof) {
            _condition.wait(timeoutUs);
        }
        return _eof;
    }

    bool hasResponse() const
    {
        autil::ScopedLock lock(_condition);
        return bool(std::dynamic_pointer_cast<AgentResponseToClientData>(_data));
    }

    AgentResponseToClientDataPtr getResponse() const
    {
        autil::ScopedLock lock(_condition);
        return std::dynamic_pointer_cast<AgentResponseToClientData>(_data);
    }

private:
    mutable autil::ThreadCond _condition;
    navi::DataPtr _data;
    bool _eof = false;
};

NaviSchedForwardRequest makeForwardRequest(uint64_t requestId, int64_t batchId,
                                           const std::string &payload)
{
    NaviSchedForwardRequest request;
    request.request.requestId = requestId;
    request.request.promptTokenCount = requestId * 10;
    request.request.forwardPayload = payload;
    request.decision.batchId = batchId;
    request.decision.expectedBatchSize = 2;
    request.decision.scheduleTimeUs = 1000 + requestId;
    request.decision.scheduleLatencyUs = 100 + requestId;
    request.decision.predictedQueueLatencyMs = 1.25;
    request.decision.predictedExecutionLatencyMs = 2.5;
    return request;
}

}

class RtpLLMAdapterTest : public TESTBASE {};

TEST_F(RtpLLMAdapterTest, testLatencyFitBatchSizeHasOneDefault)
{
    ASSERT_EQ(4, NaviSchedConfig().latencyFitBatchSize);
    ASSERT_EQ(0, NaviSchedConfig().latencyFitInterval);
    ASSERT_DOUBLE_EQ(0.05, NaviSchedConfig().latencyLearningRate);
    ASSERT_DOUBLE_EQ(1.0, NaviSchedConfig().latencyOnlineLearningRateScale);
    ASSERT_DOUBLE_EQ(4.0, NaviSchedConfig().latencyCurrentSampleWeight);
    ASSERT_DOUBLE_EQ(1500.0, NaviSchedConfig().latencyNonlinearScale);
    ASSERT_DOUBLE_EQ(0.02, NaviSchedConfig().queueLearningRate);
    ASSERT_EQ(100, NaviSchedConfig().latencyColdStartFitEpochs);
    ASSERT_EQ(10000, NaviSchedConfig().latencyPersistIntervalMs);
    ASSERT_EQ(64, NaviSchedConfig().latencyHistoryBucketCapacity);
    ASSERT_EQ(10, NaviSchedConfig().latencyWarmupSampleCount);
    ASSERT_FALSE(NaviSchedConfig().latencyValidationEnabled);
    ASSERT_DOUBLE_EQ(0.0, NaviSchedConfig().residualLearningRate);
    ASSERT_DOUBLE_EQ(100.0, NaviSchedConfig().residualUpdateErrorFloorMs);
    ASSERT_DOUBLE_EQ(0.1, NaviSchedConfig().residualUpdateErrorRatio);
    ASSERT_EQ(120000, NaviSchedConfig().residualIdleDecayStartMs);
    ASSERT_EQ(300000, NaviSchedConfig().residualIdleDecayEndMs);
}

TEST_F(RtpLLMAdapterTest, testRequestAndCacheState)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_adapter_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    config.latencyFitBatchSize = 1;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    agentConfig.runtime.threadNum = 1;
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));
    ASSERT_TRUE(agent->init());
    reportRtpLLMCacheStatus(agent, 1024, 256, 16, false, 100);

    reportRtpLLMCacheStateChanged(agent, makeBlockHashes({42}), makeBlockHashes({42}),
                                  makeBlockHashes({}), makeBlockHashes({}), true, 101, 1);
    auto policyRequest = makeBatchRequest(98);
    policyRequest.totalTokenCount = 257;
    policyRequest.blockSize = 256;
    policyRequest.blockHashes = makeBlockHashes({42});
    NaviSchedBatchPtr policyBatch;
    ASSERT_TRUE(addRtpLLMBatchRequest(
        agent, policyRequest, 98, 1,
        [&policyBatch](const NaviSchedBatchPtr &batch) { policyBatch = batch; }));
    ASSERT_TRUE(policyBatch);
    ASSERT_EQ(1, policyBatch->_requestStats.size());
    EXPECT_EQ(256, policyBatch->_requestStats[0].cacheHitTokenCount);
    ASSERT_TRUE(startRtpLLMBatch(policyBatch->finishReporter, {98}));
    auto policyStats = makeRequestStats(98);
    policyStats.totalTokenCount = 257;
    policyStats.cacheHitTokenCount = 256;
    ASSERT_TRUE(finishRtpLLMBatch(policyBatch->finishReporter, {policyStats}));

    auto mismatchedBlockSize = makeBatchRequest(99);
    mismatchedBlockSize.blockSize = 128;
    ASSERT_FALSE(addRtpLLMBatchRequest(agent, mismatchedBlockSize, 0, 0,
                                       [](const NaviSchedBatchPtr &) {}));

    auto request = makeRequestStats(1);
    request.totalTokenCount = 2;
    request.cacheHitTokenCount = 1;
    NaviSchedBatchPtr readyBatch;
    auto callback = [&readyBatch](const NaviSchedBatchPtr &batch) { readyBatch = batch; };
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(request), 1, 2, callback));
    auto second = makeRequestStats(2);
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(second), 1, 2, callback));
    ASSERT_TRUE(startRtpLLMBatch(readyBatch->finishReporter, {1, 2}));
    reportRtpLLMCacheInserted(agent, makeBlockHashes({10, 20}));
    reportRtpLLMCacheRemoved(agent, makeBlockHashes({10}));
    ASSERT_TRUE(finishRtpLLMBatch(readyBatch->finishReporter, {request, second}));
}

TEST_F(RtpLLMAdapterTest, testNullAgentIgnoresEvents)
{
    std::shared_ptr<NaviSchedAgent> agent;
    ASSERT_FALSE(addRtpLLMBatchRequest(agent, makeBatchRequest(1), 1, 2,
                                       [](const NaviSchedBatchPtr &) {}));
    ASSERT_FALSE(startRtpLLMBatch(nullptr, {1}));
    ASSERT_FALSE(finishRtpLLMBatch(nullptr, {makeRequestStats(1)}));
    reportRtpLLMCacheInserted(agent, nullptr);
    reportRtpLLMCacheRemoved(agent, nullptr);
    ASSERT_EQ(-1, getRtpLLMAgentGrpcPort(agent));
    stopRtpLLMAgent(agent);
}

TEST_F(RtpLLMAdapterTest, testBatchLatencyAndQueueFeatures)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_adapter_latency_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    config.latencyFitBatchSize = 1;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    agentConfig.runtime.threadNum = 1;
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));
    ASSERT_TRUE(agent->init());
    initializeCachePolicy(agent);
    NaviSchedRequestStats first;
    first.requestId = 1;
    first.totalTokenCount = 8;
    first.cacheHitTokenCount = 2;
    NaviSchedRequestStats second;
    second.requestId = 2;
    second.totalTokenCount = 4;
    NaviSchedBatchPtr readyBatch;
    auto callback = [&readyBatch](const NaviSchedBatchPtr &batch) { readyBatch = batch; };
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(first), 1, 2, callback));
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(second), 1, 2, callback));
    ASSERT_TRUE(startRtpLLMBatch(readyBatch->finishReporter, {1, 2}));
    ASSERT_TRUE(finishRtpLLMBatch(readyBatch->finishReporter, {first, second}));
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testObservedBatchFinishesStartedLogicalBatch)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_observed_batch_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    config.latencyFitBatchSize = 100;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    agentConfig.runtime.threadNum = 1;
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));
    ASSERT_TRUE(agent->init());
    initializeCachePolicy(agent);

    auto handler = std::make_shared<CaptureReadyBatchHandler>();
    auto callback = [handler](const NaviSchedBatchPtr &batch) {
        handler->handleReadyBatch(batch);
    };
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(10), 7, 2, callback));
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(11), 7, 2, callback));
    ASSERT_TRUE(handler->wait(100 * 1000));
    auto batch = handler->getBatch();
    ASSERT_TRUE(startRtpLLMBatch(batch->finishReporter, {10, 11}));

    std::vector<NaviSchedLatencyQuery> queries(2);
    queries[0].requestId = 10;
    queries[0].totalTokenCount = 16;
    queries[1].requestId = 11;
    queries[1].totalTokenCount = 32;
    recordRtpLLMPrefillBatch(agent, queries, 8000, 100,
                             {batch->finishReporter, batch->finishReporter});
    ASSERT_FALSE(startRtpLLMBatch(batch->finishReporter, {10, 11}));
    ASSERT_FALSE(finishRtpLLMBatch(batch->finishReporter,
                                   {makeRequestStats(10), makeRequestStats(11)}));
}

TEST_F(RtpLLMAdapterTest, testRetainedBatchFollowsExecutorLifecycle)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_retained_batch_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    config.latencyFitBatchSize = 100;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    agentConfig.runtime.threadNum = 1;
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));
    ASSERT_TRUE(agent->init());
    initializeCachePolicy(agent);

    NaviSchedBatchPtr readyBatch;
    auto callback = [&readyBatch](const NaviSchedBatchPtr &batch) { readyBatch = batch; };
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(20), 9, 2, callback));
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(21), 9, 2, callback));
    ASSERT_TRUE(readyBatch);
    auto retainResult = retainRtpLLMReadyBatchWithResult(readyBatch);
    ASSERT_TRUE(retainResult);
    ASSERT_EQ(2, retainResult.retainedRequestCount);
    auto retainedLease = acquireRtpLLMRetainedBatchLease(20);
    ASSERT_TRUE(retainedLease);
    auto duplicateResult = retainRtpLLMReadyBatchWithResult(readyBatch);
    ASSERT_EQ(RtpLLMReadyBatchRetainStatus::DUPLICATE_REQUEST_ID, duplicateResult.status);
    ASSERT_EQ(20, duplicateResult.requestId);
    ASSERT_EQ(2, duplicateResult.retainedRequestCount);

    std::weak_ptr<NaviSchedBatchFinishReporter> retainedReporter =
        readyBatch->finishReporter;
    readyBatch.reset();
    ASSERT_FALSE(retainedReporter.expired());

    std::vector<NaviSchedLatencyQuery> queries(2);
    queries[0].requestId = 20;
    queries[0].totalTokenCount = 16;
    queries[1].requestId = 21;
    queries[1].totalTokenCount = 32;
    ASSERT_TRUE(startRetainedRtpLLMBatches(queries));
    recordRtpLLMPrefillBatch(agent, queries, 8000, 100);
    retainedLease.reset();
    ASSERT_FALSE(abortRetainedRtpLLMBatch(20));
    ASSERT_TRUE(retainedReporter.expired());
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testAbortRetainedBatchReleasesWholeBatch)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_abort_retained_batch_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    agentConfig.runtime.threadNum = 1;
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));
    ASSERT_TRUE(agent->init());
    initializeCachePolicy(agent);

    NaviSchedBatchPtr readyBatch;
    auto callback = [&readyBatch](const NaviSchedBatchPtr &batch) { readyBatch = batch; };
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(30), 10, 2, callback));
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(31), 10, 2, callback));
    ASSERT_TRUE(readyBatch);
    ASSERT_TRUE(retainRtpLLMReadyBatch(readyBatch));

    std::weak_ptr<NaviSchedBatchFinishReporter> retainedReporter =
        readyBatch->finishReporter;
    auto firstLease = acquireRtpLLMRetainedBatchLease(30);
    auto secondLease = acquireRtpLLMRetainedBatchLease(31);
    ASSERT_TRUE(firstLease);
    ASSERT_TRUE(secondLease);
    ASSERT_FALSE(acquireRtpLLMRetainedBatchLease(32));
    readyBatch.reset();

    firstLease->abort();
    ASSERT_TRUE(firstLease);
    ASSERT_FALSE(abortRetainedRtpLLMBatch(31));
    ASSERT_TRUE(retainedReporter.expired());
    firstLease.reset();
    secondLease.reset();
    auto *impl = NaviSchedAgentImpl::getImpl(*agent);
    ASSERT_NE(nullptr, impl);
    ASSERT_EQ(0, impl->_agentBatchR->getReadyWithoutRunningStats().currentTimeUs);
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testDismissedRetainedBatchLeaseDoesNotAbort)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_dismiss_retained_batch_lease_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    agentConfig.runtime.threadNum = 1;
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));
    ASSERT_TRUE(agent->init());
    initializeCachePolicy(agent);

    NaviSchedBatchPtr readyBatch;
    auto callback = [&readyBatch](const NaviSchedBatchPtr &batch) { readyBatch = batch; };
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(35), 12, 1, callback));
    ASSERT_TRUE(readyBatch);
    ASSERT_TRUE(retainRtpLLMReadyBatch(readyBatch));

    auto lease = acquireRtpLLMRetainedBatchLease(35);
    ASSERT_TRUE(lease);
    lease->dismiss();
    lease->abort();
    lease.reset();
    ASSERT_TRUE(abortRetainedRtpLLMBatch(35));
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testSplitBatchIsRecordedAsObservedAfterAbort)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_observed_split_batch_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    config.latencyFitBatchSize = 1;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    agentConfig.runtime.threadNum = 1;
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));
    ASSERT_TRUE(agent->init());
    initializeCachePolicy(agent);

    NaviSchedBatchPtr readyBatch;
    auto callback = [&readyBatch](const NaviSchedBatchPtr &batch) { readyBatch = batch; };
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(40), 11, 2, callback));
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(41), 11, 2, callback));
    ASSERT_TRUE(readyBatch);
    ASSERT_TRUE(retainRtpLLMReadyBatch(readyBatch));

    NaviSchedLatencyQuery executed;
    executed.requestId = 40;
    executed.totalTokenCount = 32;
    executed.cacheHitTokenCount = 8;
    recordRtpLLMObservedPrefillBatch(agent, {executed}, 9000, 100);

    ASSERT_FALSE(abortRetainedRtpLLMBatch(41));
    ASSERT_FALSE(startRtpLLMBatch(readyBatch->finishReporter, {40, 41}));
    auto *impl = NaviSchedAgentImpl::getImpl(*agent);
    ASSERT_NE(nullptr, impl);
    ASSERT_EQ(0, impl->_agentBatchR->getReadyWithoutRunningStats().currentTimeUs);
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testIndependentCacheSequences)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_adapter_cache_sequence_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));

    reportRtpLLMCacheInserted(agent, makeBlockHashes({10, 20}));
    reportRtpLLMCacheInserted(agent, makeBlockHashes({30}));
    reportRtpLLMCacheRemoved(agent, makeBlockHashes({20}));
}

TEST_F(RtpLLMAdapterTest, testBatchRequestsAreReleasedTogether)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_adapter_batch_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));
    ASSERT_TRUE(agent->init());
    initializeCachePolicy(agent);
    auto handler = std::make_shared<CaptureReadyBatchHandler>();
    NaviSchedReadyBatchCallback callback = [handler](const NaviSchedBatchPtr &batch) {
        handler->handleReadyBatch(batch);
    };

    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(1), 10, 2, callback));
    ASSERT_FALSE(handler->wait(1000));
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(2), 10, 2, callback));
    ASSERT_TRUE(handler->wait(1000));
    auto batch = handler->getBatch();
    ASSERT_GT(batch->batchId, 0);
    ASSERT_EQ(std::vector<uint64_t>({1, 2}), batch->requestIds);
    ASSERT_EQ(2, batch->expectedBatchSize);
    ASSERT_FALSE(batch->timedOut);
    ASSERT_TRUE(batch->finishReporter);
}

TEST_F(RtpLLMAdapterTest, testForwardedRequestsAreSubmittedAsOneBatch)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_forward_batch_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    config.batchTimeoutMs = 1000;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    agentConfig.runtime.threadNum = 1;
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));
    ASSERT_TRUE(agent->init());
    initializeCachePolicy(agent);

    std::vector<NaviSchedForwardedRequest> forwarded;
    int64_t forwardedBatchId = 0;
    size_t forwardCallCount = 0;
    NaviSchedBatchFinishReporterPtr finishReporter;
    ASSERT_TRUE(setRtpLLMForwardBatchHandler(
        agent, [&](const std::vector<NaviSchedForwardedRequest> &requests,
                   int64_t batchId, const NaviSchedBatchFinishReporterPtr &reporter) {
            ++forwardCallCount;
            forwarded = requests;
            forwardedBatchId = batchId;
            finishReporter = reporter;
            return true;
        }));

    auto firstPipe = std::make_shared<CaptureCompletionPipe>();
    auto secondPipe = std::make_shared<CaptureCompletionPipe>();
    ASSERT_TRUE(agent->_impl->_agentRequestR->submit(
        makeForwardRequest(1, 7, std::string("first\0pb", 8)), firstPipe));
    ASSERT_FALSE(firstPipe->waitForEof(1000));
    ASSERT_TRUE(agent->_impl->_agentRequestR->submit(
        makeForwardRequest(2, 7, "second-pb"), secondPipe));
    ASSERT_EQ(1, forwardCallCount);
    ASSERT_TRUE(firstPipe->waitForEof(100 * 1000));
    ASSERT_TRUE(secondPipe->waitForEof(100 * 1000));

    ASSERT_EQ(7, forwardedBatchId);
    ASSERT_EQ(2, forwarded.size());
    EXPECT_EQ(1, forwarded[0].requestId);
    EXPECT_EQ(std::string("first\0pb", 8), forwarded[0].payload);
    EXPECT_EQ(2, forwarded[0].expectedBatchSize);
    EXPECT_FALSE(forwarded[0].timedOut);
    EXPECT_EQ(1001, forwarded[0].scheduleTimeUs);
    EXPECT_EQ(101, forwarded[0].scheduleLatencyUs);
    EXPECT_GT(forwarded[0].agentArrivalTimeUs, 0);
    EXPECT_GT(forwarded[0].batchReadyTimeUs, 0);
    EXPECT_GE(forwarded[0].callbackDispatchTimeUs, forwarded[0].batchReadyTimeUs);
    EXPECT_GE(forwarded[0].forwardHandlerStartTimeUs,
              forwarded[0].callbackDispatchTimeUs);
    EXPECT_GE(forwarded[0].readyAcceptedCallbackTimeUs, 0);
    EXPECT_GE(forwarded[0].forwardStateLockWaitUs, 0);
    EXPECT_GE(forwarded[0].forwardBatchBuildTimeUs, 0);
    EXPECT_DOUBLE_EQ(1.25, forwarded[0].predictedQueueLatencyMs);
    EXPECT_EQ(2, forwarded[1].requestId);
    EXPECT_EQ("second-pb", forwarded[1].payload);
    EXPECT_EQ(2, forwarded[1].expectedBatchSize);
    EXPECT_FALSE(forwarded[1].timedOut);
    EXPECT_EQ(1002, forwarded[1].scheduleTimeUs);
    EXPECT_EQ(102, forwarded[1].scheduleLatencyUs);
    EXPECT_GT(forwarded[1].agentArrivalTimeUs, 0);
    EXPECT_EQ(forwarded[0].batchReadyTimeUs, forwarded[1].batchReadyTimeUs);
    EXPECT_EQ(forwarded[0].callbackDispatchTimeUs, forwarded[1].callbackDispatchTimeUs);
    EXPECT_EQ(forwarded[0].forwardHandlerStartTimeUs,
              forwarded[1].forwardHandlerStartTimeUs);
    EXPECT_DOUBLE_EQ(2.5, forwarded[1].predictedExecutionLatencyMs);
    ASSERT_TRUE(finishReporter);
    EXPECT_TRUE(firstPipe->hasResponse());
    EXPECT_TRUE(secondPipe->hasResponse());
    EXPECT_TRUE(firstPipe->getResponse()->terminal);
    EXPECT_TRUE(secondPipe->getResponse()->terminal);
    EXPECT_FALSE(agent->publishForwardResponse(1, {"first-output"}, true));
    EXPECT_FALSE(agent->publishForwardResponse(2, {"second-output"}, true));

    auto pendingPipe = std::make_shared<CaptureCompletionPipe>();
    ASSERT_TRUE(agent->_impl->_agentRequestR->submit(
        makeForwardRequest(3, 8, "pending-pb"), pendingPipe));
    ASSERT_TRUE(setRtpLLMForwardBatchHandler(agent, {}));
    ASSERT_TRUE(pendingPipe->waitForEof(100 * 1000));
    auto response = pendingPipe->getResponse();
    ASSERT_TRUE(response);
    EXPECT_EQ(3, response->requestId);
    EXPECT_EQ(AgentResponseKind::ERROR, response->kind);
    EXPECT_TRUE(response->terminal);
    EXPECT_EQ("agent forwarding disabled", response->errorMessage);
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testBatchRequestTimeoutReleasesPartialBatch)
{
    NaviSchedConfig config;
    config.name = "rtp_llm_adapter_batch_timeout_test";
    config.enableKmon = false;
    config.gig.grpcPort = 0;
    config.enableSnapshot = false;
    config.batchTimeoutMs = 1;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "agent_biz";
    agentConfig.runtime.threadNum = 1;
    auto agent =
        (agentConfig.service = config, std::make_shared<NaviSchedAgent>(agentConfig));
    ASSERT_TRUE(agent->init());
    initializeCachePolicy(agent);
    auto handler = std::make_shared<CaptureReadyBatchHandler>();
    NaviSchedReadyBatchCallback callback = [handler](const NaviSchedBatchPtr &batch) {
        handler->handleReadyBatch(batch);
    };

    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(1), 10, 2, callback));
    ASSERT_TRUE(handler->wait(100 * 1000));
    auto batch = handler->getBatch();
    auto firstBatchId = batch->batchId;
    ASSERT_EQ(std::vector<uint64_t>({1}), batch->requestIds);
    ASSERT_TRUE(batch->timedOut);

    handler = std::make_shared<CaptureReadyBatchHandler>();
    callback = [handler](const NaviSchedBatchPtr &readyBatch) {
        handler->handleReadyBatch(readyBatch);
    };
    ASSERT_TRUE(addRtpLLMBatchRequest(agent, makeBatchRequest(2), 10, 2, callback));
    ASSERT_TRUE(handler->wait(1000));
    batch = handler->getBatch();
    ASSERT_NE(firstBatchId, batch->batchId);
    ASSERT_EQ(std::vector<uint64_t>({2}), batch->requestIds);
    ASSERT_TRUE(batch->timedOut);
}

TEST_F(RtpLLMAdapterTest, testEnvironmentInitializationDisabledForNonAgentRole)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "flexlb");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    auto status =
        createRtpLLMAgentFromEnvironment(false, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::OK, status);
    ASSERT_FALSE(agent);
    ASSERT_TRUE(errorMessage.empty());
}

TEST_F(RtpLLMAdapterTest, testEnvironmentInitializationIgnoresMasterType)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "flexlb");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "agent_biz");
    autil::EnvGuard gigPort("RTP_LLM_NAVI_SCHED_AGENT_GIG_PORT", "0");
    autil::EnvGuard logPath("LOG_PATH", GET_TEMP_DATA_PATH() + "rtp_llm_default_agent");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    auto status =
        createRtpLLMAgentFromEnvironment(true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::OK, status);
    ASSERT_TRUE(agent);
    ASSERT_TRUE(errorMessage.empty());
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testEnvironmentInitializationValidatesConfig)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "navi_sched");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "");
    autil::EnvGuard rtpBizName("BIZ_NAME", "");
    autil::EnvGuard modelConfig("MODEL_SERVICE_CONFIG", "");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    auto status =
        createRtpLLMAgentFromEnvironment(true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::INVALID_CONFIG, status);
    ASSERT_FALSE(agent);
    ASSERT_FALSE(errorMessage.empty());
}

TEST_F(RtpLLMAdapterTest, testEnvironmentDerivesBizNameFromModelServiceConfig)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "navi_sched");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "");
    autil::EnvGuard modelConfig(
        "MODEL_SERVICE_CONFIG",
        R"({"service_id":"generation.model","role_endpoints":[{"group":"default","prefill_endpoint":{"address":"model.prefill.vip","protocol":"http"}}]})");
    autil::EnvGuard gigPort("RTP_LLM_NAVI_SCHED_AGENT_GIG_PORT", "0");
    autil::EnvGuard logPath("LOG_PATH", GET_TEMP_DATA_PATH() + "rtp_llm_derived_biz");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    auto status =
        createRtpLLMAgentFromEnvironment(true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::OK, status);
    ASSERT_TRUE(agent);
    ASSERT_TRUE(errorMessage.empty());
    ASSERT_EQ("model.prefill.vip", NaviSchedAgentImpl::getImpl(*agent)->_agentBizName);
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testUnifiedConfigEnablesAgentAndUsesRtpBizName)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "");
    autil::EnvGuard rtpBizName("BIZ_NAME", "generation.model");
    autil::EnvGuard modelConfig(
        "MODEL_SERVICE_CONFIG",
        R"({"service_id":"generation.model","decode_endpoint":{"address":"model.decode.vip","protocol":"http"}})");
    autil::EnvGuard unifiedConfig("RTP_LLM_NAVI_SCHED_CONFIG_JSON",
                                  R"({"agent":{"gig_port":0}})");
    autil::EnvGuard logPath("LOG_PATH", GET_TEMP_DATA_PATH() + "rtp_llm_rtp_biz_name");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    auto status =
        createRtpLLMAgentFromEnvironment(true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::OK, status);
    ASSERT_TRUE(agent);
    ASSERT_TRUE(errorMessage.empty());
    ASSERT_EQ("generation.model", NaviSchedAgentImpl::getImpl(*agent)->_agentBizName);
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testAgentAlwaysUsesRtpKmonServiceName)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "navi_sched");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "agent_biz");
    autil::EnvGuard kmonServiceName("kmonitorServiceName", "deepseek_v4_flash");
    autil::EnvGuard deploymentName("DEPLOYMENT_NAME", "");
    autil::EnvGuard fingerprint("RTP_LLM_NAVI_SCHED_LATENCY_MODEL_FINGERPRINT", "");
    autil::EnvGuard accelerator("RTP_LLM_NAVI_SCHED_ACCELERATOR_TYPE", "rtx_pro_5000");
    autil::EnvGuard tpSize("TP_SIZE", "4");
    autil::EnvGuard modelType("MODEL_TYPE", "qwen3_moe");
    autil::EnvGuard checkpointPath("CHECKPOINT_PATH", "/models/qwen3/revision-a");
    autil::EnvGuard tokenizerPath("TOKENIZER_PATH", "/tokenizers/qwen3/revision-a");
    autil::EnvGuard modelRevision("MODEL_REVISION", "model-revision-a");
    autil::EnvGuard weightType("WEIGHT_TYPE", "BF16");
    autil::EnvGuard quantization("QUANTIZATION", "fp8");
    autil::EnvGuard speculativeMethod("SPECULATIVE_METHOD", "mtp");
    autil::EnvGuard speculativeTokens("NUM_SPECULATIVE_TOKENS", "3");
    autil::EnvGuard gigPort("RTP_LLM_NAVI_SCHED_AGENT_GIG_PORT", "0");
    autil::EnvGuard logPath("LOG_PATH", GET_TEMP_DATA_PATH() + "rtp_llm_kmon_service");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    auto status =
        createRtpLLMAgentFromEnvironment(true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::OK, status);
    ASSERT_TRUE(agent);
    const auto *impl = NaviSchedAgentImpl::getImpl(*agent);
    ASSERT_NE(nullptr, impl);
    const auto &config = NaviSchedRoleRuntimeAccess::getConfig(*impl);
    ASSERT_EQ("sched_agent", config.name);
    ASSERT_EQ("./logs/navi_sched.log", config.logFileName);
    ASSERT_EQ("navi_sched", config.snapshotDir);
    ASSERT_EQ("navi_sched", config.gig.snapshotLogDir);
    ASSERT_TRUE(config.enableKmon);
    ASSERT_EQ("deepseek_v4_flash", config.kmonServiceName);
    ASSERT_TRUE(config.kmonMetricsPath.empty());
    ASSERT_EQ("agent", config.kmonTags.at("role"));
    ASSERT_EQ(2, config.kmonTags.size());
    ASSERT_EQ("127.0.0.1", config.kmonTags.at("agent_ip"));
    EXPECT_NE(std::string::npos, impl->_latencyModelFingerprint.find("rtx_pro_5000"));
    EXPECT_NE(std::string::npos, impl->_latencyModelFingerprint.find("qwen3_moe"));
    EXPECT_NE(std::string::npos, impl->_latencyModelFingerprint.find("BF16"));
    EXPECT_NE(std::string::npos, impl->_latencyModelFingerprint.find("fp8"));
    EXPECT_NE(std::string::npos, impl->_latencyModelFingerprint.find("mtp"));
    EXPECT_EQ(std::string::npos, impl->_latencyModelFingerprint.find("model-revision-a"));
    const auto historyFingerprint =
        autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
            autil::legacy::json::ParseJson(impl->_latencyHistoryFingerprint));
    EXPECT_EQ("model-revision-a",
              autil::legacy::AnyCast<std::string>(
                  historyFingerprint.at("model_revision")));
    EXPECT_EQ("/models/qwen3/revision-a",
              autil::legacy::AnyCast<std::string>(
                  historyFingerprint.at("checkpoint_path")));
    EXPECT_EQ("/tokenizers/qwen3/revision-a",
              autil::legacy::AnyCast<std::string>(
                  historyFingerprint.at("tokenizer_path")));
    EXPECT_EQ(std::string::npos,
              impl->_latencyModelFingerprint.find("/models/qwen3/revision-a"));
    EXPECT_NE(std::string::npos, impl->_latencyModelFingerprint.find("\"4\""));
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testEnvironmentInitializationValidatesBatchTimeout)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "navi_sched");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "agent_biz");
    autil::EnvGuard batchTimeout("RTP_LLM_NAVI_SCHED_BATCH_TIMEOUT_MS", "0");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    auto status =
        createRtpLLMAgentFromEnvironment(true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::INVALID_CONFIG, status);
    ASSERT_FALSE(agent);
    ASSERT_EQ("RTP_LLM_NAVI_SCHED_BATCH_TIMEOUT_MS must be positive", errorMessage);
}

TEST_F(RtpLLMAdapterTest, testEnvironmentLatencyModelConfig)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "navi_sched");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "agent_biz");
    autil::EnvGuard fitBatchSize("RTP_LLM_NAVI_SCHED_LATENCY_FIT_BATCH_SIZE", "8");
    autil::EnvGuard fitInterval("RTP_LLM_NAVI_SCHED_LATENCY_FIT_INTERVAL", "1");
    autil::EnvGuard learningRate("RTP_LLM_NAVI_SCHED_LATENCY_LEARNING_RATE", "0.004");
    autil::EnvGuard onlineLearningRateScale(
        "RTP_LLM_NAVI_SCHED_LATENCY_ONLINE_LEARNING_RATE_SCALE", "2.5");
    autil::EnvGuard currentSampleWeight(
        "RTP_LLM_NAVI_SCHED_LATENCY_CURRENT_SAMPLE_WEIGHT", "3.0");
    autil::EnvGuard nonlinearScale(
        "RTP_LLM_NAVI_SCHED_LATENCY_NONLINEAR_SCALE", "2048.0");
    autil::EnvGuard queueLearningRate("RTP_LLM_NAVI_SCHED_QUEUE_LEARNING_RATE", "0.003");
    autil::EnvGuard batchTimeout("RTP_LLM_NAVI_SCHED_BATCH_TIMEOUT_MS", "20");
    autil::EnvGuard coldStartFitEpochs("RTP_LLM_NAVI_SCHED_LATENCY_COLD_START_FIT_EPOCHS",
                                       "9");
    autil::EnvGuard historyLimit("RTP_LLM_NAVI_SCHED_LATENCY_HISTORY_LIMIT", "1000");
    autil::EnvGuard historyBucketCapacity(
        "RTP_LLM_NAVI_SCHED_LATENCY_HISTORY_BUCKET_CAPACITY", "17");
    autil::EnvGuard residualLearningRate(
        "RTP_LLM_NAVI_SCHED_RESIDUAL_LEARNING_RATE", "0.07");
    autil::EnvGuard residualUpdateErrorFloor(
        "RTP_LLM_NAVI_SCHED_RESIDUAL_UPDATE_ERROR_FLOOR_MS", "25.0");
    autil::EnvGuard residualUpdateErrorRatio(
        "RTP_LLM_NAVI_SCHED_RESIDUAL_UPDATE_ERROR_RATIO", "0.2");
    autil::EnvGuard residualDecayStart(
        "RTP_LLM_NAVI_SCHED_RESIDUAL_IDLE_DECAY_START_MS", "4000");
    autil::EnvGuard residualDecayEnd(
        "RTP_LLM_NAVI_SCHED_RESIDUAL_IDLE_DECAY_END_MS", "9000");
    autil::EnvGuard persistInterval("RTP_LLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL", "32");
    autil::EnvGuard persistIntervalMs(
        "RTP_LLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL_MS", "4000");
    autil::EnvGuard stateFile("RTP_LLM_NAVI_SCHED_LATENCY_MODEL_STATE_FILE",
                              GET_TEMP_DATA_PATH() + "rtp_llm_environment_latency.state");
    autil::EnvGuard fingerprint("RTP_LLM_NAVI_SCHED_LATENCY_MODEL_FINGERPRINT",
                                R"({"framework":"rtp_llm","hardware":"mi308x","tp":2})");
    autil::EnvGuard logPath("LOG_PATH",
                            GET_TEMP_DATA_PATH() + "rtp_llm_environment_test");
    autil::EnvGuard gigPort("RTP_LLM_NAVI_SCHED_AGENT_GIG_PORT", "0");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    auto status =
        createRtpLLMAgentFromEnvironment(true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::OK, status);
    ASSERT_TRUE(agent);
    ASSERT_TRUE(errorMessage.empty());
    const auto *impl = NaviSchedAgentImpl::getImpl(*agent);
    ASSERT_NE(nullptr, impl);
    const auto &config = NaviSchedRoleRuntimeAccess::getConfig(*impl);
    EXPECT_EQ(32, config.latencyPersistInterval);
    EXPECT_EQ(4000, config.latencyPersistIntervalMs);
    EXPECT_DOUBLE_EQ(2.5, config.latencyOnlineLearningRateScale);
    EXPECT_DOUBLE_EQ(3.0, config.latencyCurrentSampleWeight);
    EXPECT_DOUBLE_EQ(2048.0, config.latencyNonlinearScale);
    EXPECT_EQ(17, config.latencyHistoryBucketCapacity);
    EXPECT_DOUBLE_EQ(0.07, config.residualLearningRate);
    EXPECT_DOUBLE_EQ(25.0, config.residualUpdateErrorFloorMs);
    EXPECT_DOUBLE_EQ(0.2, config.residualUpdateErrorRatio);
    EXPECT_EQ(4000, config.residualIdleDecayStartMs);
    EXPECT_EQ(9000, config.residualIdleDecayEndMs);
    EXPECT_EQ(R"({"framework":"rtp_llm","hardware":"mi308x","tp":2})",
              impl->_latencyModelFingerprint);
    agent->stop();
}

TEST_F(RtpLLMAdapterTest, testEnvironmentRejectsInvalidLatencyNonlinearScale)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "navi_sched");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "agent_biz");
    autil::EnvGuard nonlinearScale(
        "RTP_LLM_NAVI_SCHED_LATENCY_NONLINEAR_SCALE", "0");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    const auto status = createRtpLLMAgentFromEnvironment(
        true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::INVALID_CONFIG, status);
    ASSERT_FALSE(agent);
    ASSERT_EQ(
        "RTP_LLM_NAVI_SCHED_LATENCY_NONLINEAR_SCALE must be positive and finite",
        errorMessage);
}

TEST_F(RtpLLMAdapterTest, testEnvironmentRejectsInvalidLatencyPersistIntervalMs)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "navi_sched");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "agent_biz");
    autil::EnvGuard persistIntervalMs(
        "RTP_LLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL_MS", "0");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    const auto status = createRtpLLMAgentFromEnvironment(
        true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::INVALID_CONFIG, status);
    ASSERT_FALSE(agent);
    ASSERT_EQ(
        "RTP_LLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL_MS must be positive",
        errorMessage);
}

TEST_F(RtpLLMAdapterTest, testUnifiedConfigOverridesLegacyAgentConfig)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "navi_sched");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "agent_biz");
    autil::EnvGuard legacyBatchTimeout("RTP_LLM_NAVI_SCHED_BATCH_TIMEOUT_MS", "20");
    autil::EnvGuard unifiedConfig(
        "RTP_LLM_NAVI_SCHED_CONFIG_JSON",
        R"({"log_path":"logs","agent":{"batch_timeout_ms":0,"latency_learning_rate":0.004}})");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    auto status =
        createRtpLLMAgentFromEnvironment(true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::INVALID_CONFIG, status);
    ASSERT_FALSE(agent);
    ASSERT_EQ("RTP_LLM_NAVI_SCHED_BATCH_TIMEOUT_MS must be positive", errorMessage);
}

TEST_F(RtpLLMAdapterTest, testUnifiedConfigRejectsInvalidJson)
{
    autil::EnvGuard masterType("RTP_LLM_MASTER_TYPE", "navi_sched");
    autil::EnvGuard bizName("RTP_LLM_NAVI_SCHED_BIZ_NAME", "agent_biz");
    autil::EnvGuard unifiedConfig("RTP_LLM_NAVI_SCHED_CONFIG_JSON", "{");
    std::shared_ptr<NaviSchedAgent> agent;
    std::string errorMessage;
    auto status =
        createRtpLLMAgentFromEnvironment(true, "127.0.0.1", 9000, agent, errorMessage);
    ASSERT_EQ(RtpLLMAgentInitStatus::INVALID_CONFIG, status);
    ASSERT_FALSE(agent);
    ASSERT_NE(std::string::npos,
              errorMessage.find("invalid RTP_LLM_NAVI_SCHED_CONFIG_JSON"));
}
