#include "rtp_llm/flexlb/navi_sched/common/NaviSchedCommon.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/data/ClientRequestToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/test/simulator/TrafficReplayR.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/Kernel.h"
#include "navi/log/NaviLogger.h"
#include "autil/TimeUtility.h"
#include <algorithm>

namespace navi_sched::test {
namespace {

const std::string REQUEST_NODE = "navi_sched_traffic_replay_schedule";
const std::string RETRY_NODE = "navi_sched_traffic_replay_retry";
const std::string FORWARD_NODE = "navi_sched_traffic_replay_forward";
const std::string RESULT_NODE = "navi_sched_traffic_replay_result";
const std::string REPLAY_RESULT_NODE = "navi_sched_traffic_replay_debug_result";
const std::string DISPATCH_GUARD_NODE = "navi_sched_traffic_replay_dispatch_guard";

std::unique_ptr<navi::GraphDef> buildRequestGraph(const std::string &masterBizName,
                                                  const std::string &clientBizName,
                                                  bool forwardRequest,
                                                  bool collectReplay,
                                                  int64_t entryTimeoutMs,
                                                  navi::GraphId &masterGraphId,
                                                  navi::GraphId &clientGraphId)
{
    const auto rpcTimeoutMs = getNaviSchedRpcTimeoutMs(entryTimeoutMs);
    if (rpcTimeoutMs <= 0) {
        return nullptr;
    }
    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    masterGraphId = builder.newSubGraph(masterBizName);
    builder.subGraph(masterGraphId)
        .location(masterBizName, 1, 0)
        .subGraphTimeoutMs(rpcTimeoutMs);
    builder.subGraph(masterGraphId).gigProbe(false);
    clientGraphId = builder.newSubGraph(clientBizName);
    builder.subGraph(clientGraphId).location(clientBizName, 1, 0);
    auto schedule = builder.node(masterGraphId, REQUEST_NODE)
                        .kernel("navi_sched.schedule_k")
                        .integerAttr("dispatch_guard_enabled",
                                     forwardRequest ? 1 : 0);
    auto retry = builder.node(clientGraphId, RETRY_NODE)
                     .kernel("navi_sched.client_simulate_traffic_retry_k");
    auto result =
        builder.node(clientGraphId, RESULT_NODE)
            .kernel("navi_sched.client_simulate_traffic_result_k")
            .integerAttr("wait_response", forwardRequest ? 1 : 0);
    schedule.out("decision").to(retry.in("decision"));
    if (collectReplay) {
        auto replayResult = builder.node(clientGraphId, REPLAY_RESULT_NODE)
                                .kernel("navi_sched.client_simulate_replay_result_k");
        schedule.out("replay").to(replayResult.in("replay"));
        replayResult.out("finished").asGraphOutput("replay");
    }
    if (forwardRequest) {
        auto forward = builder.node(clientGraphId, FORWARD_NODE)
                           .kernel("navi_sched.client_forward_k")
                           .integerAttr("rpc_timeout_ms", rpcTimeoutMs);
        auto dispatchGuard =
            builder.node(masterGraphId, DISPATCH_GUARD_NODE)
                .kernel("navi_sched.master_dispatch_guard_k")
                .integerAttr("admission_timeout_ms", rpcTimeoutMs);
        retry.out("decision").to(forward.in("decision"));
        retry.out("decision").to(dispatchGuard.in("decision"));
        forward.out("decision").to(result.in("decision"));
        forward.out("decision").to(dispatchGuard.in("forward_decision"));
        forward.out("response").to(result.in("response"));
        dispatchGuard.out("finished").asGraphOutput("dispatch_finished");
    } else {
        retry.out("decision").to(result.in("decision"));
    }
    result.out("decision").asGraphOutput("decision");
    if (!builder.ok()) {
        return nullptr;
    }
    navi::GraphBuilder::fillHashKey(graphDef.get());
    return graphDef;
}

bool addRequestData(navi::ForkGraphParam &param, navi::GraphId masterGraphId,
                    navi::GraphId clientGraphId,
                    const ClientRequestToMasterDataConstPtr &request)
{
    if (!request) {
        return false;
    }
    for (auto graphId : {masterGraphId, clientGraphId}) {
        navi::NamedData namedData;
        namedData.graphId = graphId;
        namedData.name = CLIENT_REQUEST_TO_MASTER_DATA_NAME;
        namedData.data = std::const_pointer_cast<ClientRequestToMasterData>(request);
        param.namedDataVec.emplace_back(std::move(namedData));
    }
    return true;
}

bool addRequestData(navi::RunGraphParams &params, navi::GraphId masterGraphId,
                    navi::GraphId clientGraphId,
                    const ClientRequestToMasterDataConstPtr &request)
{
    if (!request) return false;
    for (auto graphId : {masterGraphId, clientGraphId}) {
        navi::NamedData namedData;
        namedData.graphId = graphId;
        namedData.name = CLIENT_REQUEST_TO_MASTER_DATA_NAME;
        namedData.data = std::const_pointer_cast<ClientRequestToMasterData>(request);
        if (!params.addNamedData(namedData)) return false;
    }
    return true;
}

bool startRequestGraph(TrafficReplayR &trafficReplayR, const std::string &clientBizName,
                       const std::shared_ptr<ClientRequestToMasterData> &request)
{
    navi::GraphId masterGraphId = navi::INVALID_GRAPH_ID;
    navi::GraphId clientGraphId = navi::INVALID_GRAPH_ID;
    auto graphDef = buildRequestGraph(trafficReplayR.getMasterBizName(), clientBizName,
                                      true, request->collectReplay,
                                      request->timeoutMs,
                                      masterGraphId, clientGraphId);
    navi::RunGraphParams params;
    params.setResourceBuildMode(navi::RBM_PARALLEL);
    params.setTimeoutMs(request->timeoutMs);
    params.setTraceLevel("ERROR");
    if (!graphDef || !addRequestData(params, masterGraphId, clientGraphId, request)) {
        return false;
    }
    return trafficReplayR.startRequest(graphDef.release(), params, request->requestId);
}

navi::ErrorCode forkRequestGraph(navi::KernelComputeContext &ctx,
                                 const std::string &masterBizName,
                                 const std::string &clientBizName,
                                 const ClientRequestToMasterDataConstPtr &request,
                                 bool forwardRequest)
{
    navi::GraphId masterGraphId = navi::INVALID_GRAPH_ID;
    navi::GraphId clientGraphId = navi::INVALID_GRAPH_ID;
    auto graphDef = buildRequestGraph(masterBizName, clientBizName, forwardRequest,
                                      request->collectReplay,
                                      request->timeoutMs,
                                      masterGraphId, clientGraphId);
    navi::ForkGraphParam param;
    param.timeoutMs = std::min(ctx.getRemainTimeMs(), request->timeoutMs);
    if (!graphDef || param.timeoutMs <= 0 ||
        !addRequestData(param, masterGraphId, clientGraphId, request)) {
        return navi::EC_ABORT;
    }
    return ctx.fork(graphDef.release(), param);
}

} // namespace

class TrafficReplayRetryKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_NAMED_DATA(ClientRequestToMasterData, _request,
                      CLIENT_REQUEST_TO_MASTER_DATA_NAME);
    KERNEL_DEPEND_ON(TrafficReplayBridgeR, _trafficReplayBridgeR);
    std::shared_ptr<TrafficReplayR> _trafficReplayR;
    navi::AsyncPipePtr _retryPipe;
    ClientRequestToMasterDataConstPtr _waitingRequest;
};

void TrafficReplayRetryKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.client_simulate_traffic_retry_k")
        .input("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
        .output("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID);
}

navi::ErrorCode TrafficReplayRetryKernel::init(navi::KernelInitContext &ctx)
{
    _trafficReplayR = _trafficReplayBridgeR->waitTrafficReplayR(0);
    _retryPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    return _trafficReplayR && _retryPipe ? navi::EC_NONE : navi::EC_INIT_SUB_GRAPH;
}

navi::ErrorCode TrafficReplayRetryKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr retryData;
    bool retryEof = false;
    if (_retryPipe->getData(retryData, retryEof) && _waitingRequest) {
        auto request = std::move(_waitingRequest);
        return forkRequestGraph(ctx, _trafficReplayR->getMasterBizName(),
                                ctx.getThisBizName(), request, false);
    }
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        return navi::EC_NONE;
    }
    auto decision = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
    auto request = std::static_pointer_cast<const ClientRequestToMasterData>(
        _request->shared_from_this());
    if (!request) {
        NAVI_KERNEL_LOG(ERROR, "invalid traffic replay schedule request, eof [%d]", eof);
        return navi::EC_ABORT;
    }
    if (!decision && eof) {
        _waitingRequest = request;
        return ctx.registerPeriodicPipe(_retryPipe, 10, 1) ? navi::EC_NONE
                                                           : navi::EC_ABORT;
    }
    if (!decision) {
        return navi::EC_NONE;
    }
    ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), decision, eof);
    return navi::EC_NONE;
}

class TrafficReplayDebugResultKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(TrafficReplayBridgeR, _trafficReplayBridgeR);
    std::shared_ptr<TrafficReplayR> _trafficReplayR;
};

void TrafficReplayDebugResultKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.client_simulate_replay_result_k")
        .input("replay", MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
        .output("finished", "");
}

navi::ErrorCode TrafficReplayDebugResultKernel::init(navi::KernelInitContext &)
{
    _trafficReplayR = _trafficReplayBridgeR->waitTrafficReplayR(0);
    return _trafficReplayR ? navi::EC_NONE : navi::EC_INIT_SUB_GRAPH;
}

navi::ErrorCode TrafficReplayDebugResultKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        return navi::EC_NONE;
    }
    if (data) {
        auto response = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
        if (!response || !response->replay ||
            !_trafficReplayR->saveReplay(response->replay)) {
            return navi::EC_ABORT;
        }
    }
    if (eof) {
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
    }
    return navi::EC_NONE;
}

class TrafficReplayResultKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_NAMED_DATA(ClientRequestToMasterData, _request,
                      CLIENT_REQUEST_TO_MASTER_DATA_NAME);
    KERNEL_DEPEND_ON(TrafficReplayBridgeR, _trafficReplayBridgeR);
    std::shared_ptr<TrafficReplayR> _trafficReplayR;
    int64_t _waitResponse = 0;
    bool _decisionEof = false;
    bool _responseEof = true;
    bool _finished = false;
    MasterResponseToClientDataPtr _decision;
};

void TrafficReplayResultKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.client_simulate_traffic_result_k")
        .input("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
        .input("response", AGENT_RESPONSE_TO_CLIENT_TYPE_ID, navi::IT_OPTIONAL)
        .output("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID);
}

bool TrafficReplayResultKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "wait_response", _waitResponse, _waitResponse);
    return 0 == _waitResponse || 1 == _waitResponse;
}

navi::ErrorCode TrafficReplayResultKernel::init(navi::KernelInitContext &)
{
    _trafficReplayR = _trafficReplayBridgeR->waitTrafficReplayR(0);
    _responseEof = 0 == _waitResponse;
    return _trafficReplayR ? navi::EC_NONE : navi::EC_INIT_SUB_GRAPH;
}

navi::ErrorCode TrafficReplayResultKernel::compute(navi::KernelComputeContext &ctx)
{
    if (_finished) {
        return navi::EC_NONE;
    }
    navi::DataPtr data;
    bool eof = false;
    if (!_decisionEof &&
        ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        auto decision = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
        if (data && !decision) {
            return navi::EC_ABORT;
        }
        if (decision && _decision) {
            return navi::EC_ABORT;
        }
        if (decision && decision->requestTiming) {
            auto request = std::static_pointer_cast<const ClientRequestToMasterData>(
                _request->shared_from_this());
            _trafficReplayR->recordRequestTiming(request->requestId,
                                                 *decision->requestTiming,
                                                 decision->scheduleId,
                                                 decision->scheduleLoss,
                                                 decision->scheduleLatencyUs);
        }
        _decision = std::move(decision);
        _decisionEof = eof;
    }

    data.reset();
    eof = false;
    if (!_responseEof &&
        ctx.getInput(navi::PortIndex(1, navi::INVALID_INDEX), data, eof)) {
        auto response = std::dynamic_pointer_cast<AgentResponseToClientData>(data);
        auto request = std::static_pointer_cast<const ClientRequestToMasterData>(
            _request->shared_from_this());
        if (data && (!response || !response->valid() ||
                     response->requestId != request->requestId)) {
            return navi::EC_ABORT;
        }
        _responseEof = eof;
    }

    if (_decisionEof && _responseEof) {
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), _decision, true);
        _finished = true;
    }
    return navi::EC_NONE;
}

class TrafficReplayKernel : public navi::Kernel {
public:
    ~TrafficReplayKernel() override;

public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    bool dispatchReady(navi::KernelComputeContext &ctx);
    bool scheduleNext(navi::KernelComputeContext &ctx);

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(TrafficReplayR, _trafficReplayR);
    navi::AsyncPipePtr _timerPipe;
    navi::AsyncPipePtr _controlPipe;
    navi::AsyncPipePtr _completionPipe;
    navi::AsyncPipePtr _startPipe;
    int64_t _startTimeUs = 0;
    bool _stopRequested = false;
};

void TrafficReplayKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.client_simulate_traffic_replay_k").output("finished", "");
}

TrafficReplayKernel::~TrafficReplayKernel()
{
    if (_controlPipe) {
        _trafficReplayR->clearControlPipe(_controlPipe);
    }
    if (_completionPipe) _trafficReplayR->clearCompletionPipe(_completionPipe);
    if (_startPipe) _trafficReplayR->clearStartPipe(_startPipe);
}

navi::ErrorCode TrafficReplayKernel::init(navi::KernelInitContext &ctx)
{
    _timerPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    _controlPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    _completionPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    _startPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_timerPipe || !_controlPipe || !_completionPipe || !_startPipe ||
        !_trafficReplayR->setControlPipe(_controlPipe) ||
        !_trafficReplayR->setCompletionPipe(_completionPipe) ||
        !_trafficReplayR->setStartPipe(_startPipe)) {
        return navi::EC_INIT_SUB_GRAPH;
    }
    NAVI_KERNEL_LOG(INFO, "traffic replay kernel initialized");
    return navi::EC_NONE;
}

navi::ErrorCode TrafficReplayKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (_controlPipe->getData(data, eof) && eof) {
        _stopRequested = true;
        ctx.stopPeriodicAsyncPipe(_timerPipe);
    }
    _completionPipe->getData(data, eof);
    if ((_stopRequested || _trafficReplayR->empty()) &&
        0 == _trafficReplayR->getInflightRequestCount()) {
        _trafficReplayR->logLatencyStats();
        NAVI_KERNEL_LOG(INFO, "traffic replay completed, requestCount [%lu]",
                        _trafficReplayR->getRequestCount());
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
        return navi::EC_NONE;
    }
    if (_stopRequested) return navi::EC_NONE;
    if (0 == _startTimeUs) {
        _startPipe->getData(data, eof);
        int64_t startTimeMs = 0;
        int64_t firstRequestTimeMs = 0;
        if (!_trafficReplayR->getStartTime(startTimeMs, firstRequestTimeMs)) {
            return navi::EC_ABORT;
        }
        _startTimeUs = startTimeMs * 1000;
        NAVI_KERNEL_LOG(INFO, "traffic replay started, startTimeMs [%ld]", startTimeMs);
    }
    _timerPipe->getData(data, eof);
    if (!dispatchReady(ctx)) {
        return navi::EC_ABORT;
    }
    if (_trafficReplayR->empty()) return navi::EC_NONE;
    if (!scheduleNext(ctx)) {
        NAVI_KERNEL_LOG(ERROR, "schedule next traffic replay request failed");
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

bool TrafficReplayKernel::dispatchReady(navi::KernelComputeContext &ctx)
{
    TrafficReplayRequest replayRequest;
    if (!_trafficReplayR->peekNext(replayRequest)) {
        return true;
    }
    int64_t startTimeMs = 0;
    int64_t firstRequestTimeMs = 0;
    if (!_trafficReplayR->getStartTime(startTimeMs, firstRequestTimeMs)) {
        return false;
    }
    auto elapsedLogMs = replayRequest.requestTimeMs - firstRequestTimeMs;
    auto dueTimeUs =
        _startTimeUs + static_cast<int64_t>(elapsedLogMs * 1000.0 /
                                            _trafficReplayR->getTimeCompression());
    if (autil::TimeUtility::currentTime() < dueTimeUs ||
        !_trafficReplayR->takeNext(replayRequest)) {
        return true;
    }
    auto request = std::make_shared<ClientRequestToMasterData>();
    request->bizName = _trafficReplayR->getTargetBizName();
    request->timeoutMs = _trafficReplayR->getRequestTimeoutMs();
    request->promptTokenCount = replayRequest.promptTokenCount;
    request->blockSize = _trafficReplayR->getBlockSize();
    request->requestId = replayRequest.requestId;
    request->blockHashes = replayRequest.blockHashes;
    request->collectReplay = _trafficReplayR->shouldCollectReplay(replayRequest);
    return startRequestGraph(*_trafficReplayR, ctx.getThisBizName(), request);
}

bool TrafficReplayKernel::scheduleNext(navi::KernelComputeContext &ctx)
{
    TrafficReplayRequest request;
    if (!_trafficReplayR->peekNext(request)) {
        return true;
    }
    int64_t startTimeMs = 0;
    int64_t firstRequestTimeMs = 0;
    if (!_trafficReplayR->getStartTime(startTimeMs, firstRequestTimeMs)) {
        return false;
    }
    auto elapsedLogMs = request.requestTimeMs - firstRequestTimeMs;
    auto dueTimeUs =
        _startTimeUs + static_cast<int64_t>(elapsedLogMs * 1000.0 /
                                            _trafficReplayR->getTimeCompression());
    auto delayUs = std::max<int64_t>(0, dueTimeUs - autil::TimeUtility::currentTime());
    auto delayMs = std::max<int64_t>(1, (delayUs + 999) / 1000);
    return ctx.registerPeriodicPipe(_timerPipe, delayMs, 1);
}

REGISTER_KERNEL(TrafficReplayRetryKernel);
REGISTER_KERNEL(TrafficReplayDebugResultKernel);
REGISTER_KERNEL(TrafficReplayResultKernel);
REGISTER_KERNEL(TrafficReplayKernel);

} // namespace navi_sched::test
