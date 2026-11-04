#include "aios/navi_sched/data/AgentResponseToClientData.h"
#include "aios/navi_sched/data/ClientRequestToMasterData.h"
#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "aios/navi_sched/data/NaviSchedRequestTrace.h"
#include "autil/TimeUtility.h"
#include "aios/navi_sched/resource/AgentRequestR.h"
#include "aios/navi_sched/resource/AgentResponseR.h"
#include "aios/navi_sched/resource/LatencyModelR.h"
#include "navi/engine/Kernel.h"
#include "navi/log/NaviLogger.h"
#include <algorithm>

namespace navi_sched {
namespace {

const std::string MASTER_RESPONSE_TO_CLIENT_DATA_NAME = "navi_sched.master_response";

}

class AgentRequestKernel : public navi::Kernel {
public:
    ~AgentRequestKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    static NaviSchedForwardRequest makeForwardRequest(
        const ClientRequestToMasterData &request,
        const MasterResponseToClientData &decision);

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_NAMED_DATA(ClientRequestToMasterData, _request,
                      CLIENT_REQUEST_TO_MASTER_DATA_NAME);
    KERNEL_NAMED_DATA(MasterResponseToClientData, _decision,
                      MASTER_RESPONSE_TO_CLIENT_DATA_NAME);
    KERNEL_DEPEND_ON(AgentRequestR, _agentRequestR);
    KERNEL_DEPEND_ON_FALSE(AgentResponseR, _agentResponseR);
    KERNEL_DEPEND_ON(LatencyModelR, _latencyModelR);
    navi::AsyncPipePtr _completionPipe;
    uint64_t _requestId = 0;
    bool _decisionSent = false;
    bool _terminalReceived = false;
};

AgentRequestKernel::~AgentRequestKernel()
{
    if (_agentResponseR && _requestId != 0 && !_terminalReceived) {
        _agentResponseR->cancel(_requestId, _completionPipe);
    }
}

NaviSchedForwardRequest AgentRequestKernel::makeForwardRequest(
    const ClientRequestToMasterData &request, const MasterResponseToClientData &decision)
{
    NaviSchedForwardRequest forwardRequest;
    auto &scheduleRequest = forwardRequest.request;
    scheduleRequest.version = request.version;
    scheduleRequest.requestId = decision.requestId;
    scheduleRequest.bizName = request.bizName;
    scheduleRequest.promptTokenCount = request.promptTokenCount;
    scheduleRequest.blockSize = request.blockSize;
    scheduleRequest.blockHashes = request.blockHashes;
    scheduleRequest.timeoutMs = request.timeoutMs;
    scheduleRequest.forwardPayload = request.forwardPayload;
    forwardRequest.decision.version = decision.version;
    forwardRequest.decision.requestId = decision.requestId;
    forwardRequest.decision.scheduleId = decision.scheduleId;
    forwardRequest.decision.scheduleLoss = decision.scheduleLoss;
    forwardRequest.decision.scheduleLatencyUs = decision.scheduleLatencyUs;
    forwardRequest.decision.scheduleTimeUs = decision.scheduleTimeUs;
    forwardRequest.decision.reason = decision.reason;
    forwardRequest.decision.batchId = decision.batchId;
    forwardRequest.decision.expectedBatchSize = decision.expectedBatchSize;
    forwardRequest.decision.predictedQueueLatencyMs =
        decision.predictedQueueLatencyMs;
    forwardRequest.decision.predictedCacheHitTokenCount =
        decision.predictedCacheHitTokenCount;
    forwardRequest.decision.predictedExecutionLatencyMs =
        decision.predictedExecutionLatencyMs;
    forwardRequest.decision.latencyPrediction = decision.latencyPrediction;
    forwardRequest.decision.latencyTrainingPhase =
        decision.latencyTrainingPhase;
    forwardRequest.decision.target = decision.target;
    return forwardRequest;
}

void AgentRequestKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.agent_request_k")
        .output("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
        .output("response", AGENT_RESPONSE_TO_CLIENT_TYPE_ID);
}

navi::ErrorCode AgentRequestKernel::init(navi::KernelInitContext &ctx)
{
    const bool tracing = _decision && traceRequestBatch(_decision->batchId);
    const auto initBeginTimeUs = tracing ? autil::TimeUtility::currentTime() : 0;
    _completionPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_completionPipe || !_decision || !_request) {
        NAVI_KERNEL_LOG(ERROR,
                        "invalid agent request input, pipe [%p], decision [%p], request [%p]",
                        _completionPipe.get(), _decision, _request);
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (tracing) {
        NAVI_KERNEL_LOG(INFO,
                        "[navi-request-stage] stage=agent_init_begin request_id=%lu "
                        "schedule_batch_id=%ld expected_batch_size=%ld event_time_us=%ld "
                        "deserialize_begin_us=%ld deserialize_end_us=%ld payload_bytes=%zu",
                        _decision->requestId, _decision->batchId,
                        _decision->expectedBatchSize, initBeginTimeUs,
                        _request->deserializeBeginTimeUs, _request->deserializeEndTimeUs,
                        _request->forwardPayload.size());
    }
    auto forwardRequest = makeForwardRequest(*_request, *_decision);
    const auto copyDoneTimeUs = tracing ? autil::TimeUtility::currentTime() : 0;
    if (_decision->latencyModelGlobal.valid()) {
        const auto result =
            _latencyModelR->applyGlobalZeroPoint(_decision->latencyModelGlobal);
        if (LatencyModelR::GlobalZeroPointApplyResult::INCOMPATIBLE == result) {
            NAVI_KERNEL_LOG(WARN,
                            "ignore incompatible global zero point, masterEpoch [%ld], "
                            "version [%lu], "
                            "formula [%d], binaryVersion [%ld], parameterCount [%u]",
                            _decision->latencyModelGlobal.masterEpoch,
                            _decision->latencyModelGlobal.version,
                            _decision->latencyModelGlobal.formula,
                            _decision->latencyModelGlobal.binaryVersion,
                            _decision->latencyModelGlobal.parameterCount);
        }
    }
    const auto remainingMs = std::min(ctx.getTimeoutChecker()->remainTime(),
                                      getNaviSchedRpcTimeoutMs(_request->timeoutMs));
    if (remainingMs <= 0) {
        return navi::EC_TIMEOUT;
    }
    forwardRequest.request.timeoutMs = remainingMs;
    _requestId = forwardRequest.request.requestId;
    const auto submitBeginTimeUs = tracing ? autil::TimeUtility::currentTime() : 0;
    auto submitResult =
        _agentRequestR->submitWithResult(forwardRequest, _completionPipe);
    if (tracing) {
        const auto endTimeUs = autil::TimeUtility::currentTime();
        NAVI_KERNEL_LOG(INFO,
                        "[navi-request-stage] stage=agent_init_end request_id=%lu "
                        "schedule_batch_id=%ld event_time_us=%ld request_copy_us=%ld "
                        "apply_model_us=%ld submit_us=%ld submit_result=%d",
                        _requestId, _decision->batchId, endTimeUs,
                        copyDoneTimeUs - initBeginTimeUs, submitBeginTimeUs - copyDoneTimeUs,
                        endTimeUs - submitBeginTimeUs, static_cast<int>(submitResult));
    }
    if (AgentRequestR::SubmitResult::ACCEPTED != submitResult) {
        const char *reason =
            AgentRequestR::SubmitResult::COMPLETION_HANDLER_NOT_SET == submitResult
                ? "completion_handler_not_set"
                : "completion_handler_rejected";
        NAVI_KERNEL_LOG(
            ERROR,
            "agent request submission failed, requestId [%lu], payloadSize [%zu], "
            "batchId [%ld], expectedBatchSize [%ld], reason [%s]",
            forwardRequest.request.requestId, forwardRequest.request.forwardPayload.size(),
            forwardRequest.decision.batchId, forwardRequest.decision.expectedBatchSize,
            reason);
        return navi::EC_INIT_SUB_GRAPH;
    }
    return navi::EC_NONE;
}

navi::ErrorCode AgentRequestKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    auto hasData = _completionPipe->getData(data, eof);
    if (!hasData) {
        return navi::EC_NONE;
    }
    if (eof && !data) {
        if (_terminalReceived) {
            return navi::EC_NONE;
        }
        NAVI_KERNEL_LOG(ERROR, "agent response pipe reached eof before terminal frame");
        return navi::EC_ABORT;
    }
    auto response = std::dynamic_pointer_cast<AgentResponseToClientData>(data);
    if (!response || !response->valid() || response->requestId != _requestId) {
        NAVI_KERNEL_LOG(ERROR,
                        "invalid agent response, expected requestId [%lu], actual [%lu]",
                        _requestId, response ? response->requestId : 0);
        return navi::EC_ABORT;
    }
    if (AgentResponseKind::ACK == response->kind) {
        _terminalReceived = response->terminal;
        auto output = _decision->makeForwardResponse();
        output->requestTiming =
            std::make_shared<NaviSchedRequestTiming>(response->requestTiming);
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), output, true);
        _decisionSent = true;
        if (response->terminal) {
            ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), nullptr, true);
        }
    } else {
        _terminalReceived = response->terminal;
        if (response->terminal && !_decisionSent) {
            auto output = _decision->makeForwardResponse();
            if (AgentResponseKind::ERROR == response->kind) {
                output->reason = response->errorMessage;
                output->target = {};
            }
            ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), output, true);
            _decisionSent = true;
        }
        ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), response,
                      response->terminal);
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(AgentRequestKernel);

}
