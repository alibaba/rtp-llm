#include "rtp_llm/flexlb/navi_sched/data/MasterResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterStateR.h"
#include "rtp_llm/flexlb/navi_sched/resource/SchedulerConfigR.h"
#include "navi/engine/Kernel.h"
#include "navi/log/NaviLogger.h"

namespace navi_sched {

class MasterDispatchGuardKernel : public navi::Kernel {
public:
    ~MasterDispatchGuardKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    void consumeAdmissionTimeout(navi::KernelComputeContext &ctx);
    bool consumeDispatchAck();
    navi::ErrorCode consumeForwardDecision(navi::KernelComputeContext &ctx);
    void stopAdmissionTimeout(navi::KernelComputeContext &ctx);
    void releaseUnacknowledgedReservation();
    navi::ErrorCode finish(navi::KernelComputeContext &ctx);

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(MasterStateR, _masterStateR);
    KERNEL_DEPEND_ON_FALSE(SchedulerConfigR, _schedulerConfigR);
    MasterResponseToClientDataPtr _decision;
    navi::AsyncPipePtr _dispatchAckPipe;
    navi::AsyncPipePtr _admissionTimeoutPipe;
    bool _registered = false;
    bool _acknowledged = false;
    bool _admissionTimedOut = false;
    bool _admissionTimeoutStopped = false;
    bool _forwardDecisionReceived = false;
    bool _forwardDecisionEof = false;
    bool _finished = false;
    int64_t _admissionTimeoutMs = 0;
};

MasterDispatchGuardKernel::~MasterDispatchGuardKernel()
{
    releaseUnacknowledgedReservation();
}

void MasterDispatchGuardKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.master_dispatch_guard_k")
        .input("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
        .input("forward_decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID,
               navi::IT_OPTIONAL)
        .output("finished", "");
}

bool MasterDispatchGuardKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "admission_timeout_ms", _admissionTimeoutMs,
                 _admissionTimeoutMs);
    return _admissionTimeoutMs > 0;
}

navi::ErrorCode MasterDispatchGuardKernel::init(navi::KernelInitContext &ctx)
{
    if (_schedulerConfigR) {
        _admissionTimeoutMs =
            _schedulerConfigR->get().config.admissionTimeoutMs;
    }
    _dispatchAckPipe = ctx.createAsyncPipe(navi::AS_OPTIONAL);
    _admissionTimeoutPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_dispatchAckPipe || !_admissionTimeoutPipe) {
        NAVI_KERNEL_LOG(ERROR,
                        "create Master dispatch guard pipes failed, ackPipe [%p], "
                        "timeoutPipe [%p], admissionTimeoutMs [%ld]",
                        _dispatchAckPipe.get(), _admissionTimeoutPipe.get(),
                        _admissionTimeoutMs);
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (!ctx.registerPeriodicPipe(
            _admissionTimeoutPipe, _admissionTimeoutMs, 1)) {
        NAVI_KERNEL_LOG(ERROR,
                        "register Master dispatch guard timeout failed, "
                        "admissionTimeoutMs [%ld]",
                        _admissionTimeoutMs);
        return navi::EC_INIT_SUB_GRAPH;
    }
    return navi::EC_NONE;
}

navi::ErrorCode
MasterDispatchGuardKernel::compute(navi::KernelComputeContext &ctx)
{
    if (_finished) {
        return navi::EC_NONE;
    }
    consumeAdmissionTimeout(ctx);
    if (!_decision) {
        navi::DataPtr data;
        bool eof = false;
        if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
            return navi::EC_NONE;
        }
        _decision = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
        if (!_decision || !eof) {
            NAVI_KERNEL_LOG(ERROR, "invalid Master dispatch guard decision");
            return navi::EC_ABORT;
        }
        if (_decision->target.agentAddress.empty()) {
            _acknowledged = true;
        } else if (!(_decision->dispatchHandle.valid()
                         ? _decision->dispatchHandle.registerWaiter(_dispatchAckPipe, _acknowledged)
                         : _masterStateR->registerDispatchWaiter(
                               _decision->target, _decision->batchId,
                               _decision->requestId, _dispatchAckPipe, _acknowledged))) {
            NAVI_KERNEL_LOG(
                ERROR,
                "failed to register Master dispatch waiter, requestId [%lu], "
                "batchId [%ld], agentAddress [%s]",
                _decision->requestId, _decision->batchId,
                _decision->target.agentAddress.c_str());
            return navi::EC_ABORT;
        } else {
            _registered = !_acknowledged;
        }
    }

    if (!consumeDispatchAck()) {
        return navi::EC_ABORT;
    }
    if (_acknowledged) {
        stopAdmissionTimeout(ctx);
        return finish(ctx);
    }

    auto ec = consumeForwardDecision(ctx);
    if (navi::EC_NONE != ec || _finished) {
        return ec;
    }
    if (_admissionTimedOut) {
        NAVI_KERNEL_LOG(
            WARN,
            "Agent admission timed out, requestId [%lu], batchId [%ld], "
            "agentAddress [%s]",
            _decision->requestId, _decision->batchId,
            _decision->target.agentAddress.c_str());
        releaseUnacknowledgedReservation();
        return finish(ctx);
    }
    return navi::EC_NONE;
}

void MasterDispatchGuardKernel::consumeAdmissionTimeout(
    navi::KernelComputeContext &ctx)
{
    if (_admissionTimeoutStopped || _admissionTimedOut) {
        return;
    }
    int64_t activatedCount = 0;
    int64_t remainingCount = 0;
    int64_t periodMs = 0;
    int64_t nextTickTimeMs = 0;
    if (ctx.getPeriodicPipeActivationCount(
            _admissionTimeoutPipe, activatedCount, remainingCount, periodMs,
            nextTickTimeMs) &&
        activatedCount > 0) {
        _admissionTimedOut = true;
    }
}

bool MasterDispatchGuardKernel::consumeDispatchAck()
{
    if (_acknowledged) {
        return true;
    }
    navi::DataPtr data;
    bool eof = false;
    if (_dispatchAckPipe->getData(data, eof) && data) {
        NAVI_KERNEL_LOG(ERROR, "unexpected Master dispatch ACK payload");
        return false;
    }
    if (eof) {
        _registered = false;
        _acknowledged = true;
    }
    return true;
}

navi::ErrorCode MasterDispatchGuardKernel::consumeForwardDecision(
    navi::KernelComputeContext &ctx)
{
    if (_forwardDecisionEof) {
        return navi::EC_NONE;
    }
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(1, navi::INVALID_INDEX), data, eof)) {
        return navi::EC_NONE;
    }
    if (data) {
        auto decision = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
        if (!decision || decision->requestId != _decision->requestId ||
            decision->batchId != _decision->batchId) {
            NAVI_KERNEL_LOG(ERROR, "invalid forwarded decision in Master dispatch guard");
            return navi::EC_ABORT;
        }
        _forwardDecisionReceived = !decision->target.address.empty();
    }
    _forwardDecisionEof = eof;
    if (!_forwardDecisionEof || _forwardDecisionReceived) {
        return navi::EC_NONE;
    }

    releaseUnacknowledgedReservation();
    return finish(ctx);
}

void MasterDispatchGuardKernel::stopAdmissionTimeout(
    navi::KernelComputeContext &ctx)
{
    if (_admissionTimeoutStopped) {
        return;
    }
    _admissionTimeoutStopped = true;
    int64_t activatedCount = 0;
    int64_t remainingCount = 0;
    int64_t periodMs = 0;
    int64_t nextTickTimeMs = 0;
    if (!ctx.getPeriodicPipeActivationCount(
            _admissionTimeoutPipe, activatedCount, remainingCount, periodMs,
            nextTickTimeMs) ||
        nextTickTimeMs < 0) {
        return;
    }
    if (!ctx.stopPeriodicAsyncPipe(_admissionTimeoutPipe)) {
        NAVI_KERNEL_LOG(WARN, "stop Agent admission timeout failed");
    }
}

navi::ErrorCode MasterDispatchGuardKernel::finish(navi::KernelComputeContext &ctx)
{
    if (_finished) {
        return navi::EC_NONE;
    }

    _finished = true;
    if (!ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true)) {
        NAVI_KERNEL_LOG(ERROR,
                        "finish Master dispatch guard output failed, requestId [%lu], "
                        "batchId [%ld]",
                        _decision ? _decision->requestId : 0,
                        _decision ? _decision->batchId : 0);
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

void MasterDispatchGuardKernel::releaseUnacknowledgedReservation()
{
    if (_finished || _acknowledged || !_masterStateR || !_decision ||
        _decision->target.agentAddress.empty()) {
        return;
    }
    if (_registered) {
        if (_decision->dispatchHandle.valid()) {
            _decision->dispatchHandle.cancel(_dispatchAckPipe);
        } else {
            _masterStateR->cancelDispatchWaiter(
                _decision->target, _decision->batchId, _decision->requestId,
                _dispatchAckPipe);
        }
        _registered = false;
    }
}

REGISTER_KERNEL(MasterDispatchGuardKernel);

} // namespace navi_sched
