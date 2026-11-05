#include "aios/navi_sched/service/NaviSchedClient.h"
#include "aios/navi_sched/config/NaviSchedConfigBuilder.h"
#include "aios/navi_sched/data/ClientRequestToMasterData.h"
#include "aios/navi_sched/graph/ClientGraphRunner.h"
#include "aios/navi_sched/graph/FlexlbScheduleGraph.h"
#include "aios/navi_sched/graph/ReplayControlGraph.h"
#include "aios/navi_sched/kernel/ReplayClientControlKernel.h"
#include "aios/navi_sched/resource/ReplayClientR.h"
#include "aios/navi_sched/resource/ClientResponseR.h"
#include "aios/navi_sched/service/NaviSchedClientSummary.h"
#include "autil/NetUtil.h"
#include "autil/TimeUtility.h"
#include "autil/HashAlgorithm.h"
#include "autil/legacy/fast_jsonizable_dec.h"
#include "navi/engine/ResourceMap.h"
#include "navi/resource/GigClientR.h"
#include "navi/util/CommonUtil.h"
#include <multi_call/interface/SearchService.h>
#include <multi_call/service/SearchServiceSnapshot.h>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <atomic>

namespace navi_sched {
AUTIL_LOG_SETUP(navi_sched, NaviSchedClient);

namespace {

constexpr int64_t FAILURE_LOG_INTERVAL_US = 1000 * 1000;

}

NaviSchedClient::FailureType NaviSchedClient::classifyFailure(
    const ClientGraphRunResult &result)
{
    if (!result.decision && navi::EC_NONE == result.errorCode) {
        return FailureType::NO_DECISION;
    }
    switch (result.errorCode) {
        case navi::EC_TIMEOUT:
            return FailureType::TIMEOUT;
        case navi::EC_RPC_INPUT_PORT:
        case navi::EC_DESERIALIZE:
        case navi::EC_SERIALIZE:
        case navi::EC_STREAM_RPC:
        case navi::EC_STREAM_RECEIVE:
        case navi::EC_STREAM_SEND:
        case navi::EC_SERIALIZE_OVERRIDE_DATA:
        case navi::EC_SERIALIZE_NAMED_DATA:
        case navi::EC_ARPC_RESULT:
            return FailureType::TRANSPORT;
        case navi::EC_PART_ID_OVERFLOW:
        case navi::EC_PART_ID_NOT_EXIST:
        case navi::EC_PART_COUNT_MISMATCH:
        case navi::EC_LACK_REGISTRY:
        case navi::EC_LACK_RESOURCE_BIZ:
        case navi::EC_LACK_RESOURCE_BIZ_PART:
        case navi::EC_LACK_RESOURCE_GRAPH:
        case navi::EC_LACK_RESOURCE_SUB_GRAPH:
            return FailureType::ROUTING;
        case navi::EC_NO_DATA:
        case navi::EC_NO_GRAPH_OUTPUT:
            return FailureType::NO_DECISION;
        default:
            return FailureType::OTHER;
    }
}

void NaviSchedClient::normalizeGraphResult(ClientGraphRunResult &result)
{
    if (navi::EC_NONE == result.errorCode && !result.decision) {
        result.errorCode = navi::EC_NO_DATA;
        result.errorMessage = "master returned no scheduling decision";
    }
}

NaviSchedClient::MasterRequestLatencySample
NaviSchedClient::makeMasterRequestLatencySample(int64_t latencyUs,
                                                const ClientGraphRunResult &result)
{
    MasterRequestLatencySample sample;
    sample.latencyUs = std::max<int64_t>(0, latencyUs);
    sample.success = navi::EC_NONE == result.errorCode;
    if (!sample.success) {
        sample.failureType = classifyFailure(result);
    }
    return sample;
}

void NaviSchedClient::recordMasterFailure(const ClientRequestToMasterData &request,
                                          const std::string &masterBizName,
                                          const ClientGraphRunResult &result,
                                          int64_t masterRequestLatencyUs)
{
    switch (classifyFailure(result)) {
        case FailureType::TIMEOUT:
            _masterTimeoutCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case FailureType::TRANSPORT:
            _masterTransportErrorCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case FailureType::ROUTING:
            _masterRoutingErrorCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case FailureType::NO_DECISION:
            _masterNoDecisionCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case FailureType::OTHER:
            _masterOtherErrorCount.fetch_add(1, std::memory_order_relaxed);
            break;
    }
    logFailure(request.requestId, request.bizName, masterBizName, request.timeoutMs,
               masterRequestLatencyUs, result.errorCode, result.errorMessage);
}

void NaviSchedClient::logFailure(uint64_t requestId, const std::string &targetBizName,
                                 const std::string &masterBizName, int64_t timeoutMs,
                                 int64_t masterRequestLatencyUs, int32_t errorCode,
                                 const std::string &errorMessage) const
{
    const auto nowTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
    auto previousTimeUs = _lastFailureLogTimeUs.load(std::memory_order_relaxed);
    if (nowTimeUs - previousTimeUs < FAILURE_LOG_INTERVAL_US ||
        !_lastFailureLogTimeUs.compare_exchange_strong(previousTimeUs, nowTimeUs,
                                                       std::memory_order_relaxed)) {
        _suppressedFailureLogCount.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const auto suppressedCount =
        _suppressedFailureLogCount.exchange(0, std::memory_order_relaxed);
    AUTIL_LOG(WARN,
              "NaviSched client request failed: request_id=%lu target_biz=%s "
              "master_biz=%s timeout_ms=%ld master_request_latency_ms=%.3f "
              "error_code=%d error_name=%s "
              "error_message=%s suppressed=%lu",
              requestId, targetBizName.c_str(), masterBizName.c_str(), timeoutMs,
              masterRequestLatencyUs >= 0 ? masterRequestLatencyUs / 1000.0 : -1.0,
              errorCode,
              navi::CommonUtil::getErrorString(static_cast<navi::ErrorCode>(errorCode)),
              errorMessage.c_str(), suppressedCount);
}

NaviSchedDecision NaviSchedClient::makeDecision(const MasterResponseToClientData &input)
{
    NaviSchedDecision decision;
    decision.version = input.version;
    decision.requestId = input.requestId;
    decision.scheduleId = input.scheduleId;
    decision.scheduleLoss = input.scheduleLoss;
    decision.scheduleLatencyUs = input.scheduleLatencyUs;
    decision.scheduleTimeUs = input.scheduleTimeUs;
    decision.reason = input.reason;
    decision.batchId = input.batchId;
    decision.expectedBatchSize = input.expectedBatchSize;
    decision.predictedCacheHitTokenCount = input.predictedCacheHitTokenCount;
    decision.predictedQueueLatencyMs = input.predictedQueueLatencyMs;
    decision.predictedExecutionLatencyMs = input.predictedExecutionLatencyMs;
    decision.latencyPrediction = input.latencyPrediction;
    decision.latencyTrainingPhase = input.latencyTrainingPhase;
    decision.target = input.target;
    return decision;
}

NaviSchedScheduleResult NaviSchedClient::makeScheduleResult(
    const ClientGraphRunResult &input)
{
    NaviSchedScheduleResult result;
    result.errorCode = input.errorCode;
    result.errorMessage = input.errorMessage;
    if (input.decision) {
        result.decision = makeDecision(*input.decision);
        result.requestTiming = input.decision->requestTiming;
    }
    result.replay = input.replay;
    return result;
}

NaviSchedScheduleResult NaviSchedClient::scheduleRequest(
    const ClientRequestToMasterDataConstPtr &request, const std::string &masterBizName,
    int64_t timeoutMs, bool expectResponseStream, const std::string &masterAddress)
{
    if (!_runner) {
        _stoppedRequestCount.fetch_add(1, std::memory_order_relaxed);
        NaviSchedScheduleResult result;
        result.errorCode = navi::EC_ABORT;
        result.errorMessage = "navi sched client is stopped";
        logFailure(request ? request->requestId : 0,
                   request ? request->bizName : std::string(), masterBizName, timeoutMs,
                   -1, result.errorCode, result.errorMessage);
        return result;
    }
    _scheduleCount.fetch_add(1, std::memory_order_relaxed);
    const auto beginTimeUs = autil::TimeUtility::monotonicTimeUs();
    auto result =
        _runner->schedule(request, masterBizName, timeoutMs, expectResponseStream,
                          masterAddress);
    const auto masterRequestLatencyUs =
        autil::TimeUtility::monotonicTimeUs() - beginTimeUs;
    recordScheduleResult(request, masterBizName, result, masterRequestLatencyUs);
    return makeScheduleResult(result);
}

void NaviSchedClient::scheduleReplayRequest(
    const std::shared_ptr<const ClientRequestToMasterData> &request,
    const std::string &masterBizName, int64_t timeoutMs)
{
    uint64_t abortGeneration;
    {
        std::lock_guard<std::mutex> lock(_replayRequestsMutex);
        _activeReplayRequestIds.insert(request->requestId);
        abortGeneration = _replayAbortGeneration;
    }
    _scheduleCount.fetch_add(1, std::memory_order_relaxed);
    const auto beginTimeUs = autil::TimeUtility::monotonicTimeUs();
    auto complete = [this, request, masterBizName, beginTimeUs](
                        ClientGraphRunResult result) mutable {
        {
            std::lock_guard<std::mutex> lock(_replayRequestsMutex);
            _activeReplayRequestIds.erase(request->requestId);
        }
        const auto latencyUs = result.masterRequestLatencyUs > 0
                                   ? result.masterRequestLatencyUs
                                   : autil::TimeUtility::monotonicTimeUs() - beginTimeUs;
        recordScheduleResult(request, masterBizName, result,
                             latencyUs);
    };
    ClientGraphRunResult launchResult;
    if (_runner) {
        launchResult = _runner->scheduleDetached(
            request, masterBizName, timeoutMs, complete, false);
    } else {
        launchResult.errorCode = navi::EC_ABORT;
        launchResult.errorMessage = "navi sched client is stopped";
    }
    if (navi::EC_NONE != launchResult.errorCode) {
        complete(std::move(launchResult));
    } else {
        abortReplayRequestLaunchedAcrossStop(request->requestId, abortGeneration);
    }
}

bool NaviSchedClient::replayAbortOccurredSince(uint64_t generation) const
{
    std::lock_guard<std::mutex> lock(_replayRequestsMutex);
    return _replayAbortGeneration != generation;
}

void NaviSchedClient::abortReplayRequestLaunchedAcrossStop(
    uint64_t requestId, uint64_t generation)
{
    if (!replayAbortOccurredSince(generation) || !_responseR) {
        return;
    }
    const bool didAbort = _responseR->abort(requestId);
    AUTIL_LOG(INFO,
              "abort replay graph launched across stop, request_id [%lu], "
              "aborted [%d]",
              requestId, didAbort);
}

void NaviSchedClient::abortReplayRequests()
{
    std::unordered_set<uint64_t> requestIds;
    {
        std::lock_guard<std::mutex> lock(_replayRequestsMutex);
        ++_replayAbortGeneration;
        requestIds.swap(_activeReplayRequestIds);
    }
    size_t aborted = 0;
    for (auto requestId : requestIds) {
        const bool didAbort = _responseR && _responseR->abort(requestId);
        aborted += didAbort ? 1 : 0;
        AUTIL_LOG(INFO, "abort replay graph, request_id [%lu], aborted [%d]",
                  requestId, didAbort);
    }
    if (!requestIds.empty()) {
        AUTIL_LOG(INFO, "abort replay requests, active [%lu], aborted [%lu]",
                  requestIds.size(), aborted);
    }
}

void NaviSchedClient::recordScheduleResult(
    const std::shared_ptr<const ClientRequestToMasterData> &request,
    const std::string &masterBizName, ClientGraphRunResult &result,
    int64_t masterRequestLatencyUs)
{
    normalizeGraphResult(result);
    if (getConfig().enableKmon) {
        _masterRequestLatencyQueue.Push(
            makeMasterRequestLatencySample(masterRequestLatencyUs, result));
    }
    if (navi::EC_NONE == result.errorCode) {
        _scheduleSuccessCount.fetch_add(1, std::memory_order_relaxed);
        if (result.decision->reason == "rate_limited") {
            _masterRateLimitedCount.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        _scheduleErrorCount.fetch_add(1, std::memory_order_relaxed);
        recordMasterFailure(*request, masterBizName, result, masterRequestLatencyUs);
    }
}

NaviSchedClient::NaviSchedClient(const NaviSchedClientConfig &config)
    : NaviSchedRole(config.service)
    , _defaultMasterBizName(config.masterBizName)
    , _clientConfig(config.runtime)
    , _enableAgentForward(config.enableAgentForward)
    , _enableSampleCollection(config.enableSampleCollection)
    , _responseR(std::make_shared<ClientResponseR>())
    , _replayClientR(std::make_shared<ReplayClientR>())
    , _runner(std::make_shared<ClientGraphRunner>(getNavi(), config.service.enableVis,
                                                  _responseR,
                                                  config.expectAgentResponseStream))
{
}

NaviSchedClient::~NaviSchedClient() { stop(); }

bool NaviSchedClient::init()
{
    if (!NaviSchedRole::init()) {
        return false;
    }
    const auto &config = _clientConfig;
    auto replayClientId = config.bizName + "@" + autil::NetUtil::getBindIp();
    if (getGrpcPort() > 0) {
        replayClientId += ":" + std::to_string(getGrpcPort());
    }
    _replayClientR->setClientId(replayClientId);
    const auto historyHash = autil::HashAlgorithm::hashString64(
        replayClientId.data(), replayClientId.size());
    std::ostringstream historyKey;
    historyKey << std::hex << std::setfill('0') << std::setw(16) << historyHash;
    const auto snapshotRoot = getConfig().snapshotDir.empty()
                                  ? std::string("navi_sched")
                                  : getConfig().snapshotDir;
    const auto historyDirectory = std::filesystem::path(snapshotRoot) /
                                  getConfig().name / "replay_history";
    const auto legacyHistoryDirectory = historyDirectory / historyKey.str();
    if (!_replayClientR->configureHistory(historyDirectory.string(), 20000,
                                           legacyHistoryDirectory.string())) {
        AUTIL_LOG(ERROR, "configure persistent replay history failed, path [%s]",
                  historyDirectory.c_str());
        stop();
        return false;
    }
    _replayClientR->setStopCallback([this]() { abortReplayRequests(); });
    _replayClientR->setScheduleCallback([this](const NaviSchedScheduleRequest &request) {
        auto replay = std::make_shared<ClientRequestToMasterData>();
        replay->requestId = allocateRequestId();
        replay->bizName = request.bizName;
        replay->timeoutMs = request.timeoutMs;
        replay->promptTokenCount = request.promptTokenCount;
        replay->blockSize = request.blockSize;
        replay->blockHashes = request.blockHashes;
        replay->forwardPayload = request.forwardPayload;
        _requestReceivedCount.fetch_add(1, std::memory_order_relaxed);
        scheduleReplayRequest(replay, _defaultMasterBizName, request.timeoutMs);
    });
    NaviSchedConfigBuilder::NamedConfigs kernelConfigs;
    std::vector<std::string> kernels = {"navi_sched.replay_client_control_k",
                                        "navi_sched.client_flexlb_k"};
    if (_enableAgentForward) {
        kernels.emplace_back("^navi_sched\\.client_.*_k$");
        autil::legacy::json::JsonMap forwardConfig;
        forwardConfig["collect_metric"] = getConfig().enableVis;
        kernelConfigs["navi_sched.client_forward_k"] =
            autil::legacy::ToJsonString(forwardConfig, true);
    }
    auto configStr = NaviSchedConfigBuilder::build(config, kernels, true, kernelConfigs);
    auto resourceMap = std::make_shared<navi::ResourceMap>();
    if ((_enableAgentForward && !resourceMap->addResource(_responseR)) ||
        !resourceMap->addResource(_replayClientR) ||
        configStr.empty() || !update(configStr, resourceMap)) {
        stop();
        return false;
    }
    navi::ResourceMap gigResources;
    if (!getNavi()->createResource(config.bizName, 1, 0,
                                   {navi::GigClientR::RESOURCE_ID}, gigResources)) {
        stop();
        return false;
    }
    const auto gigIt = gigResources.getMap().find(navi::GigClientR::RESOURCE_ID);
    const auto gigClientR = gigIt == gigResources.getMap().end()
                                ? nullptr
                                : std::dynamic_pointer_cast<navi::GigClientR>(gigIt->second);
    auto searchService = gigClientR ? gigClientR->getSearchService() : nullptr;
    if (!searchService) {
        stop();
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(_masterSearchServiceMutex);
        _masterSearchService = std::move(searchService);
    }
    {
        std::lock_guard<std::mutex> lock(_flexlbGraphMutex);
        _flexlbGraph = std::make_shared<FlexlbScheduleGraph>(getNavi(), config.bizName);
    }
    _runner->setClientBizName(_enableAgentForward ? config.bizName : "");
    _replayControlGraph = std::make_shared<ReplayControlGraph>(
        getNavi(), _replayClientR, config.bizName, _defaultMasterBizName);
    if (!_replayControlGraph->start()) {
        AUTIL_LOG(WARN,
                  "start NaviSched replay control graph failed; normal scheduling "
                  "remains available");
        _replayControlGraph.reset();
    }
    return true;
}

std::shared_ptr<AgentResponseToClientData>
NaviSchedClient::takeNextForwardResponse(int64_t timeoutMs)
{
    return _enableAgentForward && _responseR
               ? _responseR->takeNextResponse(timeoutMs)
               : nullptr;
}

bool NaviSchedClient::abortForwardRequest(uint64_t requestId)
{
    std::shared_ptr<FlexlbScheduleGraph> flexlb;
    {
        std::lock_guard<std::mutex> lock(_flexlbGraphMutex);
        flexlb = _flexlbGraph;
    }
    const bool abortedFlexlb = flexlb && flexlb->abort(requestId);
    const bool didAbort =
        (_enableAgentForward && _responseR && _responseR->abort(requestId)) || abortedFlexlb;
    AUTIL_LOG(INFO, "abort forward graph, request_id [%lu], aborted [%d]",
              requestId, didAbort);
    return didAbort;
}

std::tuple<int32_t, std::string, std::string> NaviSchedClient::scheduleFlexlb(
    uint64_t requestId, const std::string &address, const std::string &payload,
    int64_t timeoutMs, const std::vector<std::pair<std::string, std::string>> &metadata)
{
    std::shared_ptr<FlexlbScheduleGraph> graph;
    {
        std::lock_guard<std::mutex> lock(_flexlbGraphMutex);
        graph = _flexlbGraph;
    }
    if (!graph) {
        throw std::runtime_error("NaviSched FlexLB client is not initialized");
    }
    return graph->schedule(requestId, address, payload, timeoutMs, metadata);
}

uint64_t NaviSchedClient::allocateRequestId()
{
    auto requestId = _nextRequestId.fetch_add(1, std::memory_order_relaxed);
    if (requestId == 0) {
        requestId = _nextRequestId.fetch_add(1, std::memory_order_relaxed);
    }
    return requestId;
}

bool NaviSchedClient::hasMasterBiz(const std::string &masterBizName) const
{
    const auto &effectiveMasterBizName =
        masterBizName.empty() ? _defaultMasterBizName : masterBizName;
    if (effectiveMasterBizName.empty()) {
        return false;
    }
    std::shared_ptr<multi_call::SearchService> searchService;
    {
        std::lock_guard<std::mutex> lock(_masterSearchServiceMutex);
        searchService = _masterSearchService;
    }
    if (!searchService) {
        return false;
    }
    const auto snapshot = searchService->getSearchServiceSnapshot();
    if (!snapshot) {
        return true;
    }
    return snapshot->hasBiz(effectiveMasterBizName);
}

void NaviSchedClient::reportLocalFallback()
{
    _localFallbackCount.fetch_add(1, std::memory_order_relaxed);
}

uint64_t NaviSchedClient::getRequestReceivedCount() const
{
    return _requestReceivedCount.load(std::memory_order_relaxed);
}

NaviSchedScheduleResult NaviSchedClient::schedule(const NaviSchedScheduleRequest &input,
                                                  const std::string &masterBizName,
                                                  bool collectReplay,
                                                  const std::string &masterAddress)
{
    _requestReceivedCount.fetch_add(1, std::memory_order_relaxed);
    const auto &effectiveMasterBizName =
        masterBizName.empty() ? _defaultMasterBizName : masterBizName;
    if (!input.valid() || effectiveMasterBizName.empty()) {
        _invalidRequestCount.fetch_add(1, std::memory_order_relaxed);
        NaviSchedScheduleResult result;
        result.errorCode = navi::EC_ABORT;
        result.errorMessage = "invalid request or empty master biz name";
        logFailure(input.requestId, input.bizName, effectiveMasterBizName,
                   input.timeoutMs, -1, result.errorCode, result.errorMessage);
        return result;
    }
    auto request = std::make_shared<ClientRequestToMasterData>();
    request->requestId = input.requestId;
    request->bizName = input.bizName;
    request->promptTokenCount = input.promptTokenCount;
    request->blockSize = input.blockSize;
    request->blockHashes = input.blockHashes;
    request->timeoutMs = input.timeoutMs;
    request->collectReplay = collectReplay;
    request->forwardPayload = input.forwardPayload;
    if (_enableSampleCollection && _replayClientR) {
        _replayClientR->enqueueNormalQuery(
            request, autil::TimeUtility::currentTimeInMicroSeconds());
    }
    return scheduleRequest(request, effectiveMasterBizName, input.timeoutMs, true,
                           masterAddress);
}

void NaviSchedClient::stop()
{
    std::shared_ptr<FlexlbScheduleGraph> flexlb;
    {
        std::lock_guard<std::mutex> lock(_flexlbGraphMutex);
        flexlb = std::move(_flexlbGraph);
    }
    if (flexlb) {
        flexlb->stop();
    }
    stopSnapshot();
    if (_replayControlGraph) {
        _replayControlGraph->stop();
        _replayControlGraph.reset();
    }
    if (_replayClientR) {
        _replayClientR->stopReplay();
    }
    if (_responseR) {
        AUTIL_LOG(INFO, "stopping NaviSched client, active request count [%lu]",
                  _responseR->requestCount());
        _responseR->finishAll();
        AUTIL_LOG(INFO, "NaviSched client requests drained");
    }
    {
        std::lock_guard<std::mutex> lock(_masterSearchServiceMutex);
        _masterSearchService.reset();
    }
    NaviSchedRole::stop();
    _runner.reset();
    _replayClientR.reset();
}

std::string NaviSchedClient::makeSnapshot() const
{
    NaviSchedClientSummary summary;
    static_cast<NaviSchedRoleSummary &>(summary) = *makeSnapshotSummary();
    summary.role = "client";
    summary.masterBizName = _defaultMasterBizName;
    summary.requestReceivedCount = _requestReceivedCount.load(std::memory_order_relaxed);
    summary.invalidRequestCount = _invalidRequestCount.load(std::memory_order_relaxed);
    summary.stoppedRequestCount = _stoppedRequestCount.load(std::memory_order_relaxed);
    summary.scheduleCount = _scheduleCount.load(std::memory_order_relaxed);
    summary.scheduleSuccessCount = _scheduleSuccessCount.load(std::memory_order_relaxed);
    summary.scheduleErrorCount = _scheduleErrorCount.load(std::memory_order_relaxed);
    summary.masterRateLimitedCount = _masterRateLimitedCount.load(std::memory_order_relaxed);
    summary.masterTimeoutCount = _masterTimeoutCount.load(std::memory_order_relaxed);
    summary.masterTransportErrorCount =
        _masterTransportErrorCount.load(std::memory_order_relaxed);
    summary.masterRoutingErrorCount =
        _masterRoutingErrorCount.load(std::memory_order_relaxed);
    summary.masterNoDecisionCount =
        _masterNoDecisionCount.load(std::memory_order_relaxed);
    summary.masterOtherErrorCount =
        _masterOtherErrorCount.load(std::memory_order_relaxed);
    return summary.format();
}

void NaviSchedClient::reportMetrics() const
{
    NaviSchedRole::reportMetrics();
    const auto requestReceivedCount =
        _requestReceivedCount.load(std::memory_order_relaxed);
    const auto invalidRequestCount = _invalidRequestCount.load(std::memory_order_relaxed);
    const auto stoppedRequestCount = _stoppedRequestCount.load(std::memory_order_relaxed);
    const auto scheduleCount = _scheduleCount.load(std::memory_order_relaxed);
    const auto successCount = _scheduleSuccessCount.load(std::memory_order_relaxed);
    const auto errorCount = _scheduleErrorCount.load(std::memory_order_relaxed);
    const auto rateLimitedCount = _masterRateLimitedCount.load(std::memory_order_relaxed);
    const auto timeoutCount = _masterTimeoutCount.load(std::memory_order_relaxed);
    const auto transportErrorCount =
        _masterTransportErrorCount.load(std::memory_order_relaxed);
    const auto routingErrorCount =
        _masterRoutingErrorCount.load(std::memory_order_relaxed);
    const auto noDecisionCount = _masterNoDecisionCount.load(std::memory_order_relaxed);
    const auto otherErrorCount = _masterOtherErrorCount.load(std::memory_order_relaxed);
    const auto localFallbackCount =
        _localFallbackCount.load(std::memory_order_relaxed);
    reportMetric("client.schedule_total", scheduleCount);
    reportMetric("client.schedule_success_total", successCount);
    reportMetric("client.schedule_error_total", errorCount);
    reportCounterQpsMetric("client.schedule_qps", scheduleCount);
    reportCounterQpsMetric("client.schedule_success_qps", successCount);
    reportCounterQpsMetric("client.schedule_error_qps", errorCount);
    reportCounterQpsMetric("client.request_received_qps", requestReceivedCount);
    reportCounterQpsMetric("client.invalid_request_qps", invalidRequestCount);
    reportCounterQpsMetric("client.stopped_request_qps", stoppedRequestCount);
    reportCounterQpsMetric("client.master_request_qps", scheduleCount);
    reportCounterQpsMetric("client.master_success_qps", successCount);
    reportCounterQpsMetric("client.master_rate_limited_qps", rateLimitedCount);
    reportCounterQpsMetric("client.master_error_qps", errorCount);
    reportCounterQpsMetric("client.master_timeout_qps", timeoutCount);
    reportCounterQpsMetric("client.master_transport_error_qps", transportErrorCount);
    reportCounterQpsMetric("client.master_routing_error_qps", routingErrorCount);
    reportCounterQpsMetric("client.master_no_decision_qps", noDecisionCount);
    reportCounterQpsMetric("client.master_other_error_qps", otherErrorCount);
    reportCounterQpsMetric("client.local_fallback_qps", localFallbackCount);
    MasterRequestLatencySample latencySample;
    while (_masterRequestLatencyQueue.Pop(&latencySample)) {
        const auto latencyMs = latencySample.latencyUs / 1000.0;
        reportSampleMetric("client.master_request_latency_ms", latencyMs);
        if (latencySample.success) {
            reportSampleMetric("client.master_success_latency_ms", latencyMs);
            continue;
        }
        reportSampleMetric("client.master_error_latency_ms", latencyMs);
        switch (latencySample.failureType) {
            case FailureType::TIMEOUT:
                reportSampleMetric("client.master_timeout_latency_ms", latencyMs);
                break;
            case FailureType::TRANSPORT:
                reportSampleMetric("client.master_transport_error_latency_ms", latencyMs);
                break;
            case FailureType::ROUTING:
                reportSampleMetric("client.master_routing_error_latency_ms", latencyMs);
                break;
            case FailureType::NO_DECISION:
                reportSampleMetric("client.master_no_decision_latency_ms", latencyMs);
                break;
            case FailureType::OTHER:
                reportSampleMetric("client.master_other_error_latency_ms", latencyMs);
                break;
        }
    }
}

}
