#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "aios/navi_sched/data/NaviSchedRequestTrace.h"
#include "autil/TimeUtility.h"
#include "aios/navi_sched/resource/MasterScheduleR.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/kernel/MasterDispatchGuardKernel.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {
namespace {

class DispatchReservationResponsePipe final : public navi::AsyncPipe {
public:
    DispatchReservationResponsePipe(const navi::AsyncPipePtr &pipe,
                                    const std::shared_ptr<MasterStateR> &state)
        : navi::AsyncPipe(nullptr, navi::AS_REQUIRED), _pipe(pipe), _state(state)
    {
    }

    ~DispatchReservationResponsePipe() override
    {
        if (_decision && !_decision->target.agentAddress.empty()) {
            if (auto state = _state.lock()) {
                if (_decision->dispatchHandle.valid()) {
                    _decision->dispatchHandle.abandon();
                } else {
                    state->reportDispatchAbandoned(
                        _decision->target, _decision->batchId, _decision->requestId);
                }
            }
        }
    }

    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        _decision = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
        return _pipe->setData(data);
    }

    navi::ErrorCode setEof() override { return _pipe->setEof(); }

private:
    navi::AsyncPipePtr _pipe;
    std::weak_ptr<MasterStateR> _state;
    MasterResponseToClientDataPtr _decision;
};

}

class ScheduleKernel : public MasterDispatchGuardKernel {
public:
    ~ScheduleKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE_BASE(MasterDispatchGuardKernel);

private:
    KERNEL_NAMED_DATA(ClientRequestToMasterData, _request,
                      CLIENT_REQUEST_TO_MASTER_DATA_NAME);
    KERNEL_DEPEND_ON(MasterScheduleR, _masterScheduleR);
    navi::AsyncPipePtr _responsePipe;
    navi::AsyncPipePtr _dispatchResponsePipe;
    MasterScheduleRequestDataPtr _scheduleRequest;
    int64_t _dispatchGuardEnabled = 0;
    bool _fusedDispatchGuard = false;
};

ScheduleKernel::~ScheduleKernel()
{
    if (_scheduleRequest) {
        _scheduleRequest->cancel();
    }
    if (_dispatchGuardEnabled && _masterStateR && _decision &&
        !_decision->target.agentAddress.empty()) {
        if (_decision->dispatchHandle.valid()) {
            _decision->dispatchHandle.abandon();
        } else {
            _masterStateR->reportDispatchAbandoned(
                _decision->target, _decision->batchId, _decision->requestId);
        }
    }
}

void ScheduleKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.schedule_k")
        .input("forward_decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID,
               navi::IT_OPTIONAL, true)
        .output("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
        .output("replay", MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
        .output("dispatch_finished", "");
}

bool ScheduleKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "dispatch_guard_enabled", _dispatchGuardEnabled,
                 _dispatchGuardEnabled);
    NAVI_JSONIZE(ctx, "fused_dispatch_guard", _fusedDispatchGuard,
                 _fusedDispatchGuard);
    return (_dispatchGuardEnabled == 0 || _dispatchGuardEnabled == 1) &&
        (!_fusedDispatchGuard || (_dispatchGuardEnabled && MasterDispatchGuardKernel::config(ctx)));
}

navi::ErrorCode ScheduleKernel::init(navi::KernelInitContext &ctx)
{
    if (!_request) {
        NAVI_KERNEL_LOG(ERROR, "missing Client request named data on Master");
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (_fusedDispatchGuard) {
        _forwardDecisionPort = 0;
        _finishedPort = 2;
        const auto ec = MasterDispatchGuardKernel::init(ctx);
        if (ec != navi::EC_NONE) {
            return ec;
        }
    }
    _responsePipe = ctx.createAsyncPipe();
    if (!_responsePipe) {
        NAVI_KERNEL_LOG(ERROR,
                        "create Master schedule response pipe failed, requestId [%lu]",
                        _request ? _request->requestId : 0);
        return navi::EC_INIT_SUB_GRAPH;
    }
    auto request = std::static_pointer_cast<const ClientRequestToMasterData>(
        _request->shared_from_this());
    if (!_masterStateR->receiveClientRequest(*request)) {
        NAVI_KERNEL_LOG(ERROR,
                        "receive invalid Client request on Master, requestId [%lu], "
                        "bizName [%s], timeoutMs [%ld]",
                        request->requestId, request->bizName.c_str(),
                        request->timeoutMs);
        return navi::EC_ABORT;
    }
    auto responsePipe = _responsePipe;
    if (_dispatchGuardEnabled) {
        _dispatchResponsePipe = std::make_shared<DispatchReservationResponsePipe>(
            _responsePipe,
            std::static_pointer_cast<MasterStateR>(_masterStateR->shared_from_this()));
        responsePipe = _dispatchResponsePipe;
    }
    _scheduleRequest = _masterScheduleR->submit(request, responsePipe);
    if (_scheduleRequest) {
        NAVI_KERNEL_LOG(DEBUG,
                        "[navi-request-stage] stage=master_submit input_request_id=%lu "
                        "request_id=%lu queue_enqueue_mono_us=%ld request_timeout_ms=%ld",
                        request->requestId, _scheduleRequest->request->requestId,
                        _scheduleRequest->enqueueTimeUs, request->timeoutMs);
    }
    if (!_scheduleRequest) {
        NAVI_KERNEL_LOG(WARN,
                        "Master schedule graph unavailable, return retryable decision, "
                        "requestId [%lu], generation [%lu]",
                        request->requestId,
                        _masterScheduleR->getRequestPipeGeneration());
    }
    return navi::EC_NONE;
}

navi::ErrorCode ScheduleKernel::compute(navi::KernelComputeContext &ctx)
{
    if (_fusedDispatchGuard && _decision) {
        return MasterDispatchGuardKernel::compute(ctx);
    }
    navi::DataPtr data;
    bool eof = false;
    if (!_responsePipe->getData(data, eof)) {
        return navi::EC_NONE;
    }
    if (!data) {
        NAVI_KERNEL_LOG(ERROR,
                        "Master schedule response pipe closed without decision, eof [%d]",
                        static_cast<int>(eof));
        return navi::EC_ABORT;
    }
    auto decision = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
    if (!decision) {
        NAVI_KERNEL_LOG(ERROR,
                        "invalid Master schedule response type [%s]",
                        data->getTypeId().c_str());
        return navi::EC_ABORT;
    }
    if (_fusedDispatchGuard) {
        if (!acceptDecision(decision)) {
            return navi::EC_ABORT;
        }
    } else {
        _decision = decision;
    }
    if (traceRequestBatch(decision->batchId)) {
        NAVI_KERNEL_LOG(INFO,
                        "[navi-request-stage] stage=master_output request_id=%lu "
                        "schedule_batch_id=%ld expected_batch_size=%ld event_time_us=%ld "
                        "agent=%s",
                        decision->requestId, decision->batchId,
                        decision->expectedBatchSize, autil::TimeUtility::currentTime(),
                        decision->target.agentAddress.c_str());
    }
    MasterResponseToClientDataPtr replay;
    if (decision->replay) {
        replay = std::make_shared<MasterResponseToClientData>();
        replay->replay = std::move(decision->replay);
    }
    ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), decision, true);
    ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), replay, true);
    return _fusedDispatchGuard ? MasterDispatchGuardKernel::compute(ctx) : navi::EC_NONE;
}

REGISTER_KERNEL(ScheduleKernel);

}
