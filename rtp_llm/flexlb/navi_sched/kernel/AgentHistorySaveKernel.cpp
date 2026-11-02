#include "aios/navi_sched/resource/AgentHistoryR.h"
#include "aios/navi_sched/resource/AgentSchedulerConfigR.h"
#include "aios/navi_sched/resource/LatencyModelR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class AgentHistorySaveKernel : public navi::Kernel {
public:
    ~AgentHistorySaveKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    bool updatePersistTimer(navi::KernelComputeContext &ctx);
    void stopPersistTimer(navi::KernelComputeContext &ctx);

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(AgentHistoryR, _agentHistoryR);
    KERNEL_DEPEND_ON(AgentSchedulerConfigR, _agentSchedulerConfigR);
    KERNEL_DEPEND_ON(LatencyModelR, _latencyModelR);
    navi::AsyncPipePtr _persistPipe;
    navi::AsyncPipePtr _configNotificationPipe;
    int64_t _persistIntervalMs = 10 * 1000;
    int64_t _persistActivationCount = 0;
    uint64_t _schedulerConfigGeneration = 0;
    bool _persistTimerActive = false;
};

AgentHistorySaveKernel::~AgentHistorySaveKernel()
{
    if (_agentSchedulerConfigR && _configNotificationPipe) {
        _agentSchedulerConfigR->unsubscribe("agent_history_save",
                                            _configNotificationPipe);
    }
}

void AgentHistorySaveKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.agent_history_save_k")
        .input("save", "")
        .output("finished", "");
}

bool AgentHistorySaveKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "persist_interval_ms", _persistIntervalMs,
                 _persistIntervalMs);
    return _persistIntervalMs > 0;
}

navi::ErrorCode AgentHistorySaveKernel::init(navi::KernelInitContext &ctx)
{
    _persistPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    _configNotificationPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_persistPipe || !_configNotificationPipe) {
        NAVI_KERNEL_LOG(ERROR,
                        "create Agent history persist pipes failed, timer [%p], "
                        "config [%p]",
                        _persistPipe.get(), _configNotificationPipe.get());
        return navi::EC_INIT_SUB_GRAPH;
    }
    const auto config = _agentSchedulerConfigR->get();
    _persistIntervalMs = config.config.latencyPersistIntervalMs;
    _schedulerConfigGeneration = config.generation;
    if (!_agentSchedulerConfigR->subscribe("agent_history_save",
                                           _configNotificationPipe)) {
        NAVI_KERNEL_LOG(ERROR, "subscribe Agent history persist config failed");
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (!ctx.registerPeriodicPipe(_persistPipe, _persistIntervalMs, -1)) {
        NAVI_KERNEL_LOG(ERROR,
                        "register Agent history persist timer failed, intervalMs [%ld]",
                        _persistIntervalMs);
        return navi::EC_INIT_SUB_GRAPH;
    }
    _persistTimerActive = true;
    return navi::EC_NONE;
}

navi::ErrorCode AgentHistorySaveKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr configNotification;
    bool configNotificationEof = false;
    _configNotificationPipe->getData(configNotification, configNotificationEof);
    if (configNotification) {
        NAVI_KERNEL_LOG(ERROR,
                        "unexpected Agent scheduler config notification payload");
        return navi::EC_ABORT;
    }
    if (!updatePersistTimer(ctx)) {
        return navi::EC_ABORT;
    }

    navi::DataPtr inputData;
    bool inputEof = false;
    const bool inputReady = ctx.getInput(
        navi::PortIndex(0, navi::INVALID_INDEX), inputData, inputEof);

    navi::DataPtr timerData;
    bool timerEof = false;
    _persistPipe->getData(timerData, timerEof);
    int64_t activatedCount = 0;
    int64_t remainingCount = 0;
    int64_t periodMs = 0;
    int64_t nextTickTimeMs = 0;
    if (!ctx.getPeriodicPipeActivationCount(
            _persistPipe, activatedCount, remainingCount, periodMs,
            nextTickTimeMs)) {
        NAVI_KERNEL_LOG(ERROR,
                        "read Agent history persist timer state failed, "
                        "persistActivationCount [%ld]",
                        _persistActivationCount);
        return navi::EC_ABORT;
    }
    const bool timerActivated = activatedCount > _persistActivationCount;
    _persistActivationCount = activatedCount;
    if (!inputReady && !timerActivated) {
        return navi::EC_NONE;
    }
    if (!_latencyModelR->saveState()) {
        NAVI_KERNEL_LOG(ERROR, "save Agent latency history failed");
    }
    if (inputReady && inputEof) {
        stopPersistTimer(ctx);
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
    }
    return navi::EC_NONE;
}

bool AgentHistorySaveKernel::updatePersistTimer(
    navi::KernelComputeContext &ctx)
{
    const auto config = _agentSchedulerConfigR->get();
    if (config.generation <= _schedulerConfigGeneration) {
        return true;
    }
    _schedulerConfigGeneration = config.generation;
    const auto nextIntervalMs = config.config.latencyPersistIntervalMs;
    if (nextIntervalMs == _persistIntervalMs) {
        return true;
    }
    stopPersistTimer(ctx);
    _persistIntervalMs = nextIntervalMs;
    _persistActivationCount = 0;
    if (!ctx.registerPeriodicPipe(_persistPipe, _persistIntervalMs, -1)) {
        NAVI_KERNEL_LOG(ERROR,
                        "re-register Agent history persist timer failed, "
                        "intervalMs [%ld], generation [%lu]",
                        _persistIntervalMs, _schedulerConfigGeneration);
        return false;
    }
    _persistTimerActive = true;
    NAVI_KERNEL_LOG(INFO,
                    "Agent history persist timer updated, intervalMs [%ld], "
                    "generation [%lu]",
                    _persistIntervalMs, _schedulerConfigGeneration);
    return true;
}

void AgentHistorySaveKernel::stopPersistTimer(navi::KernelComputeContext &ctx)
{
    if (!_persistTimerActive) {
        return;
    }
    _persistTimerActive = false;
    int64_t activatedCount = 0;
    int64_t remainingCount = 0;
    int64_t periodMs = 0;
    int64_t nextTickTimeMs = 0;
    if (!ctx.getPeriodicPipeActivationCount(
            _persistPipe, activatedCount, remainingCount, periodMs,
            nextTickTimeMs) ||
        nextTickTimeMs < 0) {
        return;
    }
    if (!ctx.stopPeriodicAsyncPipe(_persistPipe)) {
        NAVI_KERNEL_LOG(WARN, "stop Agent latency history persist timer failed");
    }
}

REGISTER_KERNEL(AgentHistorySaveKernel);

}
