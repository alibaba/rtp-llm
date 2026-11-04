#include "rtp_llm/flexlb/navi_sched/data/MasterResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/data/NaviSchedRequestTrace.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterScheduleR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterStateR.h"
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

} // namespace

class ScheduleKernel : public navi::Kernel {
public:
    ~ScheduleKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_NAMED_DATA(ClientRequestToMasterData, _request,
                      CLIENT_REQUEST_TO_MASTER_DATA_NAME);
    KERNEL_DEPEND_ON(MasterScheduleR, _masterScheduleR);
    KERNEL_DEPEND_ON(MasterStateR, _masterStateR);
    navi::AsyncPipePtr _responsePipe;
    navi::AsyncPipePtr _dispatchResponsePipe;
    MasterScheduleRequestDataPtr _scheduleRequest;
    MasterResponseToClientDataPtr _decision;
    int64_t _dispatchGuardEnabled = 0;
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
        .output("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
        .output("replay", MASTER_RESPONSE_TO_CLIENT_TYPE_ID);
}

bool ScheduleKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "dispatch_guard_enabled", _dispatchGuardEnabled,
                 _dispatchGuardEnabled);
    return _dispatchGuardEnabled == 0 || _dispatchGuardEnabled == 1;
}

navi::ErrorCode ScheduleKernel::init(navi::KernelInitContext &ctx)
{
    if (!_request) {
        NAVI_KERNEL_LOG(ERROR, "missing Client request named data on Master");
        return navi::EC_INIT_SUB_GRAPH;
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
        // Both the request graph and the scheduler own this wrapper. It also
        // covers a decision discarded before this kernel can consume it.
        _dispatchResponsePipe = std::make_shared<DispatchReservationResponsePipe>(
            _responsePipe,
            std::static_pointer_cast<MasterStateR>(_masterStateR->shared_from_this()));
        responsePipe = _dispatchResponsePipe;
    }
    _scheduleRequest = _masterScheduleR->submit(request, responsePipe);
    if (!_scheduleRequest) {
        // MasterScheduleR publishes a retryable decision before rejecting the submit.
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
    _decision = decision;
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
    return navi::EC_NONE;
}

REGISTER_KERNEL(ScheduleKernel);

} // namespace navi_sched
