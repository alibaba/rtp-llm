#include "aios/navi_sched/kernel/SchedulerConfigSyncKernel.h"

#include "autil/TimeUtility.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi_sched {

const std::string SchedulerConfigSyncKernel::KERNEL_ID =
    "navi_sched.scheduler_config_sync_k";

SchedulerConfigSyncKernel::~SchedulerConfigSyncKernel()
{
    if (_schedulerConfigR && _activationPipe) {
        _schedulerConfigR->clearSyncPipe(_activationPipe);
    }
}

void SchedulerConfigSyncKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID).output("finished", "");
}

navi::ErrorCode SchedulerConfigSyncKernel::init(navi::KernelInitContext &ctx)
{
    _activationPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    _timeoutPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_activationPipe || !_timeoutPipe ||
        !_schedulerConfigR->setSyncPipe(_activationPipe)) {
        NAVI_KERNEL_LOG(ERROR,
                        "initialize scheduler config sync pipes failed, "
                        "activation [%p], timeout [%p]",
                        _activationPipe.get(), _timeoutPipe.get());
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (_schedulerConfigR->hasActiveAgentSync()) {
        const auto remainingMs =
            _schedulerConfigR->getAgentSyncRemainingTimeoutMs(
                autil::TimeUtility::monotonicTimeUs());
        if (remainingMs <= 0 ||
            !ctx.registerPeriodicPipe(_timeoutPipe, remainingMs, 1)) {
            NAVI_KERNEL_LOG(ERROR,
                            "restore scheduler config sync timeout failed, "
                            "remainingMs [%ld]",
                            remainingMs);
            return navi::EC_INIT_SUB_GRAPH;
        }
        _timeoutActive = true;
    }
    return navi::EC_NONE;
}

navi::ErrorCode SchedulerConfigSyncKernel::compute(
    navi::KernelComputeContext &ctx)
{
    navi::DataPtr activationData;
    bool activationEof = false;
    _activationPipe->getData(activationData, activationEof);
    if (activationData) {
        NAVI_KERNEL_LOG(ERROR,
                        "unexpected scheduler config sync activation payload");
        return navi::EC_ABORT;
    }
    if (activationEof) {
        stopTimeout(ctx);
        if (!_timeoutPipe || navi::EC_NONE != _timeoutPipe->setEof()) {
            NAVI_KERNEL_LOG(ERROR,
                            "close scheduler config timeout pipe failed");
            return navi::EC_ABORT;
        }
        navi::DataPtr timeoutData;
        bool timeoutEof = false;
        _timeoutPipe->getData(timeoutData, timeoutEof);
        if (timeoutData || !timeoutEof) {
            NAVI_KERNEL_LOG(ERROR,
                            "consume scheduler config timeout EOF failed, "
                            "data [%p], eof [%d]",
                            timeoutData.get(), static_cast<int>(timeoutEof));
            return navi::EC_ABORT;
        }
        return ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr,
                             true)
                   ? navi::EC_NONE
                   : navi::EC_ABORT;
    }

    navi::DataPtr timeoutData;
    bool timeoutEof = false;
    _timeoutPipe->getData(timeoutData, timeoutEof);
    if (timeoutData) {
        NAVI_KERNEL_LOG(ERROR,
                        "unexpected scheduler config sync timeout payload");
        return navi::EC_ABORT;
    }
    bool timedOut = false;
    if (_timeoutActive) {
        int64_t activatedCount = 0;
        int64_t remainingCount = 0;
        int64_t periodMs = 0;
        int64_t nextTickTimeMs = 0;
        if (!ctx.getPeriodicPipeActivationCount(
                _timeoutPipe, activatedCount, remainingCount, periodMs,
                nextTickTimeMs)) {
            NAVI_KERNEL_LOG(ERROR,
                            "read scheduler config sync timeout state failed");
            return navi::EC_ABORT;
        }
        timedOut = activatedCount > _timeoutActivationCount;
        _timeoutActivationCount = activatedCount;
    }

    if (_schedulerConfigR->hasActiveAgentSync()) {
        SchedulerConfigResponseDataPtr response;
        navi::AsyncPipePtr responsePipe;
        if (_schedulerConfigR->finishAgentSync(timedOut, response,
                                               responsePipe)) {
            stopTimeout(ctx);
            publishResponse(responsePipe, response);
        } else if (!_timeoutActive && !startTimeout(ctx)) {
            return navi::EC_ABORT;
        }
    }

    SchedulerConfigR::Request request;
    while (_schedulerConfigR->takeRequest(request)) {
        SchedulerConfigResponseDataPtr response;
        bool waitingForAgents = false;
        if (!_schedulerConfigR->processRequest(request, response,
                                               waitingForAgents)) {
            NAVI_KERNEL_LOG(ERROR, "process scheduler config request failed");
            return navi::EC_ABORT;
        }
        if (response) {
            publishResponse(request.responsePipe, response);
        }
        if (waitingForAgents) {
            if (!startTimeout(ctx)) {
                return navi::EC_ABORT;
            }
        }
    }
    return navi::EC_NONE;
}

bool SchedulerConfigSyncKernel::startTimeout(
    navi::KernelComputeContext &ctx)
{
    if (_timeoutActive) {
        return true;
    }
    const auto remainingMs = _schedulerConfigR->getAgentSyncRemainingTimeoutMs(
        autil::TimeUtility::monotonicTimeUs());
    if (remainingMs <= 0 ||
        !ctx.registerPeriodicPipe(_timeoutPipe, remainingMs, 1)) {
        NAVI_KERNEL_LOG(ERROR,
                        "register scheduler config sync timeout failed, "
                        "remainingMs [%ld]",
                        remainingMs);
        return false;
    }
    _timeoutActivationCount = 0;
    _timeoutActive = true;
    return true;
}

void SchedulerConfigSyncKernel::stopTimeout(
    navi::KernelComputeContext &ctx)
{
    if (!_timeoutActive) {
        return;
    }
    _timeoutActive = false;
    int64_t activatedCount = 0;
    int64_t remainingCount = 0;
    int64_t periodMs = 0;
    int64_t nextTickTimeMs = 0;
    if (!ctx.getPeriodicPipeActivationCount(
            _timeoutPipe, activatedCount, remainingCount, periodMs,
            nextTickTimeMs) ||
        nextTickTimeMs < 0) {
        return;
    }
    if (!ctx.stopPeriodicAsyncPipe(_timeoutPipe)) {
        NAVI_KERNEL_LOG(WARN,
                        "stop scheduler config sync timeout failed");
    }
}

bool SchedulerConfigSyncKernel::publishResponse(
    const navi::AsyncPipePtr &responsePipe,
    const SchedulerConfigResponseDataPtr &response)
{
    if (responsePipe && response &&
        navi::EC_NONE == responsePipe->setData(response)) {
        return true;
    }
    NAVI_KERNEL_LOG(ERROR,
                    "publish scheduler config response to request graph failed");
    return true;
}

REGISTER_KERNEL(SchedulerConfigSyncKernel);

}
