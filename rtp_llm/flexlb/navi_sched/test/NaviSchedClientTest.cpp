#include "rtp_llm/flexlb/navi_sched/data/MasterResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/data/ClientRequestToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/graph/ClientGraphRunner.h"
#include "rtp_llm/flexlb/navi_sched/resource/ClientResponseR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterStateR.h"
#include "rtp_llm/flexlb/navi_sched/resource/ReplayClientR.h"
#include "rtp_llm/flexlb/navi_sched/service/NaviSchedClient.h"
#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/MetricsReporterR.h"
#include "kmonitor/client/core/MetricsCollector.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/core/MetricsRecord.h"
#include "kmonitor/client/test/MetricsTestUtil.h"
#include "unittest/unittest.h"

#include <set>
#include <thread>
#include <vector>

namespace navi_sched {

class NaviSchedClientTest : public TESTBASE {};

TEST_F(NaviSchedClientTest, testNormalScheduleRecordsHistoryWithoutMasterReplay)
{
    NaviSchedClient client(NaviSchedClientConfig{});
    client._runner.reset();
    NaviSchedScheduleRequest request;
    request.requestId = 101;
    request.bizName = "target";
    request.timeoutMs = 1000;
    ASSERT_TRUE(request.valid());

    EXPECT_EQ(navi::EC_ABORT, client.schedule(request, "master", false).errorCode);

    request.requestId = 102;
    EXPECT_EQ(navi::EC_ABORT, client.schedule(request, "master", true).errorCode);
    client._replayClientR->stopHistoryWriter();
    EXPECT_EQ(2u, client._replayClientR->getSummary().sampleCount);
    const auto samples = client._replayClientR->selectRecentSamples(2);
    ASSERT_EQ(2u, samples.size());
    EXPECT_EQ(101u, samples.front().request.requestId);
    EXPECT_EQ(102u, samples.back().request.requestId);
}

TEST_F(NaviSchedClientTest, testClientSampleConfigIsIndependentOfMasterReplay)
{
    NaviSchedClientConfig config;
    ASSERT_TRUE(config.enableSampleCollection);
    config.enableSampleCollection = false;
    NaviSchedClient client(config);
    client._runner.reset();
    NaviSchedScheduleRequest request;
    request.requestId = 103;
    request.bizName = "target";
    request.timeoutMs = 1000;
    EXPECT_EQ(navi::EC_ABORT, client.schedule(request, "master", true).errorCode);
    client._replayClientR->stopHistoryWriter();
    EXPECT_EQ(0u, client._replayClientR->getSummary().sampleCount);
    EXPECT_FALSE(client._replayClientR->_historyThread.joinable());
}

TEST_F(NaviSchedClientTest, testReplayTrafficDoesNotBecomeNewSamples)
{
    NaviSchedClient client(NaviSchedClientConfig{});
    client._runner.reset();
    auto request = std::make_shared<ClientRequestToMasterData>();
    request->requestId = 104;
    request->bizName = "target";
    request->timeoutMs = 1000;
    for (size_t i = 0; i < 100; ++i) {
        client.scheduleReplayRequest(request, "master", 1000);
    }
    client._replayClientR->stopHistoryWriter();
    EXPECT_EQ(100u, client._scheduleCount.load());
    EXPECT_EQ(0u, client._replayClientR->getSummary().sampleCount);
    EXPECT_FALSE(client._replayClientR->_historyThread.joinable());
    EXPECT_FALSE(request->collectReplay);
}

TEST_F(NaviSchedClientTest, testHistoryWriterDoesNotBlockSchedulingOnDisk)
{
    ReplayClientR resource;
    {
        // Block disk/store work while all scheduling-side enqueues still complete.
        std::lock_guard<std::mutex> diskLock(resource._historyStore._mutex);
        for (uint64_t id = 1; id <= 301; ++id) {
            auto request = std::make_shared<ClientRequestToMasterData>();
            request->requestId = id;
            request->bizName = "target";
            request->timeoutMs = 1000;
            resource.enqueueNormalQuery(request, id);
            EXPECT_FALSE(request->collectReplay);
        }
        std::lock_guard<std::mutex> lock(resource._historyMutex);
        ASSERT_EQ(resource.HISTORY_QUEUE_LIMIT, resource._historyQueue.size());
        EXPECT_EQ(46u, resource._historyQueue.front().first->requestId);
        EXPECT_EQ(301u, resource._historyQueue.back().first->requestId);
        EXPECT_LE(resource._historyQueueBytes, resource.HISTORY_QUEUE_BYTES_LIMIT);
        EXPECT_GE(resource._historyDroppedCount, 44u);
        EXPECT_LE(resource._historyDroppedCount, 45u);
    }
    resource.stopHistoryWriter();
    EXPECT_EQ(0u, resource._historyQueue.size());
    EXPECT_EQ(0u, resource._historyQueueBytes);
    const auto samples = resource.selectRecentSamples(256);
    ASSERT_EQ(256u, samples.size());
    EXPECT_EQ(46u, samples.front().request.requestId);
    EXPECT_EQ(301u, samples.back().request.requestId);
}

TEST_F(NaviSchedClientTest, testHistoryQueueBoundsPayloadMemory)
{
    ReplayClientR resource;
    auto request = std::make_shared<ClientRequestToMasterData>();
    request->requestId = 1;
    request->bizName = "target";
    request->timeoutMs = 1000;
    request->forwardPayload.resize(resource.HISTORY_QUEUE_BYTES_LIMIT + 1);
    resource.enqueueNormalQuery(request, 1);
    EXPECT_EQ(1u, resource._historyDroppedCount);
    EXPECT_FALSE(resource._historyThread.joinable());
    EXPECT_EQ(0u, resource._historyQueueBytes);
    request->forwardPayload = std::string(1024 * 1024, 'x');
    {
        std::lock_guard<std::mutex> diskLock(resource._historyStore._mutex);
        for (int64_t time = 2; time < 32; ++time) {
            resource.enqueueNormalQuery(request, time);
        }
        std::lock_guard<std::mutex> lock(resource._historyMutex);
        EXPECT_EQ(15u, resource._historyQueue.size());
        EXPECT_EQ(15 * resource.historyRequestBytes(*request), resource._historyQueueBytes);
        EXPECT_LE(resource._historyQueueBytes, resource.HISTORY_QUEUE_BYTES_LIMIT);
    }
    resource.stopHistoryWriter();
    EXPECT_EQ(0u, resource._historyQueueBytes);
    EXPECT_EQ(15u, resource.selectRecentSamples(15).size());
}

TEST_F(NaviSchedClientTest, testStoppingReplayAbortsOnlyReplayRequests)
{
    NaviSchedClient client(NaviSchedClientConfig{});
    const auto initialAbortGeneration = client._replayAbortGeneration;
    ASSERT_TRUE(client._responseR->begin(101, false, false));
    ASSERT_TRUE(client._responseR->begin(102, false, true));
    client._activeReplayRequestIds.insert(101);

    client.abortReplayRequests();
    EXPECT_TRUE(client.replayAbortOccurredSince(initialAbortGeneration));
    EXPECT_FALSE(client.replayAbortOccurredSince(client._replayAbortGeneration));
    EXPECT_TRUE(client._activeReplayRequestIds.empty());
    auto replayState = client._responseR->find(101);
    auto normalState = client._responseR->find(102);
    EXPECT_NE(nullptr, replayState);
    EXPECT_NE(nullptr, normalState);
    if (replayState && normalState) {
        EXPECT_TRUE(replayState->abortRequested);
        EXPECT_FALSE(normalState->abortRequested);
    }

    client.abortReplayRequests();
    EXPECT_EQ(initialAbortGeneration + 2, client._replayAbortGeneration);
    if (normalState) {
        EXPECT_FALSE(normalState->abortRequested);
    }
    client._responseR->finishGraph(101, navi::EC_ABORT, "test complete", true);
    client._responseR->finishGraph(102, navi::EC_ABORT, "test complete", true);
}

TEST_F(NaviSchedClientTest, testReplayStopBeforeGraphCreationAbortsLateGraph)
{
    NaviSchedClient client(NaviSchedClientConfig{});
    const auto launchGeneration = client._replayAbortGeneration;
    client._activeReplayRequestIds.insert(103);

    client.abortReplayRequests();
    ASSERT_TRUE(client._responseR->begin(103, false, false));
    client.abortReplayRequestLaunchedAcrossStop(103, launchGeneration);

    auto state = client._responseR->find(103);
    ASSERT_NE(nullptr, state);
    EXPECT_TRUE(state->abortRequested);
    EXPECT_FALSE(state->abortIssued);
    client._responseR->finishGraph(103, navi::EC_ABORT, "test complete", true);

    ASSERT_TRUE(client._responseR->begin(104, false, false));
    client.abortReplayRequestLaunchedAcrossStop(104,
                                                 client._replayAbortGeneration);
    auto unaffectedState = client._responseR->find(104);
    ASSERT_NE(nullptr, unaffectedState);
    EXPECT_FALSE(unaffectedState->abortRequested);
    client._responseR->finishGraph(104, navi::EC_ABORT, "test complete", true);
}

class TestNaviSchedRole : public NaviSchedRole {
public:
    explicit TestNaviSchedRole(const NaviSchedConfig &config) : NaviSchedRole(config) {}

    void reportCurrentMetric(const std::string &name, double value)
    {
        reportMetric(name, value);
    }

    void reportSample(const std::string &name, double value)
    {
        reportSampleMetric(name, value);
    }

    void reportTaggedSample(const std::string &name, double value,
                            const kmonitor::MetricsTags &tags)
    {
        reportSampleMetric(name, value, tags);
    }

    uint64_t reportCounterAt(const std::string &name, uint64_t total,
                             int64_t reportTimeUs)
    {
        return reportCounterQpsMetricAt(name, total, reportTimeUs);
    }
};

NaviSchedConfig makeKmonRoleConfig(const std::string &name)
{
    NaviSchedConfig config;
    config.name = name;
    config.logFileName.clear();
    config.enableKmon = true;
    config.enableSnapshot = false;
    return config;
}

TEST_F(NaviSchedClientTest, testNaviSchedRoleKeepsOwnedKmonUntilProcessExit)
{
    kmonitor::KMonitorFactory::Shutdown();
    TestNaviSchedRole first(makeKmonRoleConfig("first_kmon_role"));
    TestNaviSchedRole second(makeKmonRoleConfig("second_kmon_role"));

    ASSERT_TRUE(first.initKmon());
    ASSERT_TRUE(second.initKmon());
    ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());

    first.stop();
    EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    second.stop();
    EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    second.stop();
    kmonitor::KMonitorFactory::Shutdown();
}

TEST_F(NaviSchedClientTest, testNaviSchedRoleDoesNotStopHostOwnedKmon)
{
    kmonitor::KMonitorFactory::Shutdown();
    kmonitor::MetricsConfig kmonConfig;
    kmonConfig.set_tenant_name("default");
    kmonConfig.set_service_name("host_service");
    kmonConfig.set_sink_address("127.0.0.1:4141");
    kmonConfig.set_inited(true);
    ASSERT_TRUE(kmonitor::KMonitorFactory::Init(kmonConfig));
    kmonitor::KMonitorFactory::Start();
    ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());

    TestNaviSchedRole role(makeKmonRoleConfig("host_kmon_role"));
    ASSERT_TRUE(role.initKmon());
    role.stop();
    EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    kmonitor::KMonitorFactory::Shutdown();
}

TEST_F(NaviSchedClientTest, testCurrentMetricSurvivesKmonSampleWithoutRefresh)
{
    kmonitor::KMonitorFactory::Shutdown();
    kmonitor::MetricsConfig kmonConfig;
    kmonConfig.set_tenant_name("default");
    kmonConfig.set_service_name("status_test_service");
    kmonConfig.set_sink_address("127.0.0.1:4141");
    kmonConfig.set_manually_mode(true);
    kmonConfig.set_inited(true);
    ASSERT_TRUE(kmonitor::KMonitorFactory::Init(kmonConfig));
    kmonitor::KMonitorFactory::Start();

    TestNaviSchedRole role(makeKmonRoleConfig("navi_sched_status_test"));
    ASSERT_TRUE(role.initKmon());
    role.reportCurrentMetric("test.current_state", 0.625);
    const auto &monitor = role._metricsReporter->_monitor;

    const std::set<kmonitor::MetricLevel> levels = {kmonitor::MetricLevel::FATAL};
    const auto sampleValueCount = [&]() {
        kmonitor::MetricsCollector collector;
        monitor->GetMetrics(&collector, levels, 1);
        size_t valueCount = 0;
        for (const auto *record : collector.GetRecords().getRecords()) {
            valueCount += record->Values().size();
        }
        return valueCount;
    };

    EXPECT_EQ(1, sampleValueCount());
    EXPECT_EQ(1, sampleValueCount());

    role.reportSample("test.one_second_sample", 42.0);
    EXPECT_EQ(2, sampleValueCount());
    EXPECT_EQ(1, sampleValueCount());
    role.stop();
    kmonitor::KMonitorFactory::Shutdown();
}

TEST_F(NaviSchedClientTest, testTaggedSampleMetricDoesNotPersistWithoutRefresh)
{
    kmonitor::KMonitorFactory::Shutdown();
    kmonitor::MetricsConfig kmonConfig;
    kmonConfig.set_tenant_name("default");
    kmonConfig.set_service_name("tagged_sample_test_service");
    kmonConfig.set_sink_address("127.0.0.1:4141");
    kmonConfig.set_manually_mode(true);
    kmonConfig.set_inited(true);
    ASSERT_TRUE(kmonitor::KMonitorFactory::Init(kmonConfig));
    kmonitor::KMonitorFactory::Start();

    TestNaviSchedRole role(makeKmonRoleConfig("navi_sched_tagged_sample_test"));
    ASSERT_TRUE(role.initKmon());
    kmonitor::MetricsTags tags("bucket_id", "3");
    role.reportTaggedSample("test.tagged_window", 17.0, tags);
    const auto &monitor = role._metricsReporter->_monitor;
    const std::set<kmonitor::MetricLevel> levels = {kmonitor::MetricLevel::FATAL};
    const auto sampleValueCount = [&]() {
        kmonitor::MetricsCollector collector;
        monitor->GetMetrics(&collector, levels, 1);
        size_t valueCount = 0;
        for (const auto *record : collector.GetRecords().getRecords()) {
            valueCount += record->Values().size();
        }
        return valueCount;
    };

    EXPECT_EQ(1, sampleValueCount());
    EXPECT_EQ(0, sampleValueCount());
    role.stop();
    kmonitor::KMonitorFactory::Shutdown();
}

TEST_F(NaviSchedClientTest, testCounterQpsUsesActualReportInterval)
{
    kmonitor::KMonitorFactory::Shutdown();
    kmonitor::MetricsConfig kmonConfig;
    kmonConfig.set_tenant_name("default");
    kmonConfig.set_service_name("normalized_qps_test_service");
    kmonConfig.set_sink_address("127.0.0.1:4141");
    kmonConfig.set_manually_mode(true);
    kmonConfig.set_inited(true);
    ASSERT_TRUE(kmonitor::KMonitorFactory::Init(kmonConfig));
    kmonitor::KMonitorFactory::Start();

    TestNaviSchedRole role(makeKmonRoleConfig("navi_sched_normalized_qps_test"));
    ASSERT_TRUE(role.initKmon());
    EXPECT_EQ(0u, role.reportCounterAt("test.normalized_qps", 100, 1000000));
    EXPECT_EQ(110u, role.reportCounterAt("test.normalized_qps", 210, 2100000));
    EXPECT_EQ(50u, role.reportCounterAt("test.normalized_qps", 260, 2600000));

    const auto &monitor = role._metricsReporter->_monitor;
    const std::set<kmonitor::MetricLevel> levels = {kmonitor::MetricLevel::FATAL};
    kmonitor::MetricsCollector collector;
    monitor->GetMetrics(&collector, levels, 1);
    bool found = false;
    for (const auto *record : collector.GetRecords().getRecords()) {
        if (record->info_->name_.find("test.normalized_qps") ==
            std::string::npos) {
            continue;
        }
        ASSERT_EQ(1u, record->Values().size());
        const auto &value = record->Values()[0]->Value();
        EXPECT_DOUBLE_EQ(100.0, std::stod(value.substr(0, value.find('_'))));
        found = true;
    }
    EXPECT_TRUE(found);

    EXPECT_EQ(0u, role.reportCounterAt("test.normalized_qps", 260, 3600000));
    collector.Clear();
    monitor->GetMetrics(&collector, levels, 1);
    size_t valueCount = 0;
    for (const auto *record : collector.GetRecords().getRecords()) {
        if (record->info_->name_.find("test.normalized_qps") !=
            std::string::npos) {
            valueCount += record->Values().size();
        }
    }
    EXPECT_EQ(0u, valueCount);
    role.stop();
    kmonitor::KMonitorFactory::Shutdown();
}

TEST_F(NaviSchedClientTest, testMasterReportsAcceptedRequestQpsDirectly)
{
    kmonitor::KMonitorFactory::Shutdown();
    kmonitor::MetricsConfig kmonConfig;
    kmonConfig.set_tenant_name("default");
    kmonConfig.set_service_name("master_request_qps_test_service");
    kmonConfig.set_sink_address("127.0.0.1:4141");
    kmonConfig.set_manually_mode(true);
    kmonConfig.set_inited(true);
    ASSERT_TRUE(kmonitor::KMonitorFactory::Init(kmonConfig));
    kmonitor::KMonitorFactory::Start();

    {
        auto reporter = std::make_shared<kmonitor::MetricsReporter>(
            "", kmonitor::MetricsTags());
        kmonitor::MetricsReporterR reporterR(reporter);
        MasterStateR state;
        state._metricsReporterR__ = &reporterR;

        ClientRequestToMasterData request;
        request.requestId = 1;
        request.bizName = "prefill_biz";
        EXPECT_FALSE(state.receiveClientRequest(request));

        request.timeoutMs = 1000;
        EXPECT_TRUE(state.receiveClientRequest(request));
        request.requestId = 2;
        EXPECT_TRUE(state.receiveClientRequest(request));
        EXPECT_EQ(2u, state.getClientRequestCount());
        kmonitor::MetricsTestUtil::checkMetricWithValue(
            reporter.get(),
            "master_request_qps_test_service.master.client_request_received_qps",
            nullptr, 2.0);
    }

    kmonitor::KMonitorFactory::Shutdown();
}

TEST_F(NaviSchedClientTest, testClassifiesMasterFailures)
{
    ClientGraphRunResult result;

    result.errorCode = navi::EC_TIMEOUT;
    EXPECT_EQ(NaviSchedClient::FailureType::TIMEOUT,
              NaviSchedClient::classifyFailure(result));

    result.errorCode = navi::EC_STREAM_RECEIVE;
    EXPECT_EQ(NaviSchedClient::FailureType::TRANSPORT,
              NaviSchedClient::classifyFailure(result));

    result.errorCode = navi::EC_LACK_RESOURCE_BIZ;
    EXPECT_EQ(NaviSchedClient::FailureType::ROUTING,
              NaviSchedClient::classifyFailure(result));

    result.errorCode = navi::EC_NO_DATA;
    EXPECT_EQ(NaviSchedClient::FailureType::NO_DECISION,
              NaviSchedClient::classifyFailure(result));

    result.errorCode = navi::EC_ABORT;
    EXPECT_EQ(NaviSchedClient::FailureType::OTHER,
              NaviSchedClient::classifyFailure(result));
}

TEST_F(NaviSchedClientTest, testMissingSuccessfulDecisionIsNoDecision)
{
    ClientGraphRunResult result;
    result.errorCode = navi::EC_NONE;

    EXPECT_EQ(NaviSchedClient::FailureType::NO_DECISION,
              NaviSchedClient::classifyFailure(result));
}

TEST_F(NaviSchedClientTest, testCountsOnlyReceivedMasterRateLimitDecisions)
{
    NaviSchedClientConfig config;
    config.service.enableKmon = false;
    NaviSchedClient client(config);
    auto request = std::make_shared<ClientRequestToMasterData>();
    request->requestId = 101;
    const auto record = [&](navi::ErrorCode ec, const std::string &reason,
                            bool hasDecision = true) {
        ClientGraphRunResult result;
        result.errorCode = ec;
        if (hasDecision) {
            result.decision = std::make_shared<MasterResponseToClientData>();
            result.decision->reason = reason;
        }
        client.recordScheduleResult(request, "master", result, 100);
    };
    record(navi::EC_NONE, "rate_limited");
    record(navi::EC_NONE, "");
    record(navi::EC_NONE, "no_available_node");
    record(navi::EC_TIMEOUT, "rate_limited");
    record(navi::EC_NONE, "", false);
    EXPECT_NE(std::string::npos,
              client.makeSnapshot().find("\n  master_rate_limited_count: 1\n"));

    std::vector<std::thread> readers;
    for (size_t i = 0; i < 8; ++i) {
        readers.emplace_back([&]() {
            for (size_t j = 0; j < 32; ++j) {
                record(navi::EC_NONE, "rate_limited");
            }
        });
    }
    for (auto &reader : readers) {
        reader.join();
    }
    EXPECT_NE(std::string::npos,
              client.makeSnapshot().find("\n  master_rate_limited_count: 257\n"));
    EXPECT_EQ(259, client._scheduleSuccessCount.load());
    EXPECT_EQ(2, client._scheduleErrorCount.load());
    client.reportMetrics();
    const auto counter = client._lastQpsCounters.find("client.master_rate_limited_qps");
    ASSERT_NE(client._lastQpsCounters.end(), counter);
    EXPECT_EQ(257, counter->second.total);
}

TEST_F(NaviSchedClientTest, testMissingSuccessfulDecisionReturnsExplicitError)
{
    ClientGraphRunResult result;
    result.errorCode = navi::EC_NONE;
    NaviSchedClient::normalizeGraphResult(result);

    EXPECT_EQ(navi::EC_NO_DATA, result.errorCode);
    EXPECT_EQ("master returned no scheduling decision", result.errorMessage);
}

TEST_F(NaviSchedClientTest, testBuildsSuccessfulMasterLatencySample)
{
    ClientGraphRunResult result;
    result.errorCode = navi::EC_NONE;
    result.decision = std::make_shared<MasterResponseToClientData>();

    const auto sample = NaviSchedClient::makeMasterRequestLatencySample(12345, result);

    EXPECT_EQ(12345, sample.latencyUs);
    EXPECT_TRUE(sample.success);
}

TEST_F(NaviSchedClientTest, testBuildsClassifiedMasterFailureLatencySample)
{
    ClientGraphRunResult result;
    result.errorCode = navi::EC_TIMEOUT;

    const auto sample = NaviSchedClient::makeMasterRequestLatencySample(-1, result);

    EXPECT_EQ(0, sample.latencyUs);
    EXPECT_FALSE(sample.success);
    EXPECT_EQ(NaviSchedClient::FailureType::TIMEOUT, sample.failureType);
}

TEST_F(NaviSchedClientTest, testMasterBizIsAbsentBeforeDiscoveryStarts)
{
    NaviSchedClientConfig config;
    config.masterBizName = "master";
    NaviSchedClient client(config);

    EXPECT_FALSE(client.hasMasterBiz());
    EXPECT_FALSE(client.hasMasterBiz("other_master"));
}

} // namespace navi_sched
