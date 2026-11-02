#include "aios/navi_sched/data/AgentCacheToMasterData.h"
#include "aios/navi_sched/data/AgentControlToMasterData.h"
#include "aios/navi_sched/data/AgentLatencyToMasterData.h"
#include "aios/navi_sched/data/AgentRequestEventToMasterData.h"
#include "aios/navi_sched/data/AgentUpdateToMasterData.h"
#include "aios/navi_sched/data/MasterPendingCacheSyncToAgentData.h"
#include "aios/navi_sched/data/MasterLatencyModelWarmupToAgentData.h"
#include "aios/navi_sched/data/MasterStopAckToAgentData.h"
#include "aios/navi_sched/data/MasterSchedulerConfigToAgentData.h"
#include "aios/navi_sched/data/NaviSchedAgentEvent.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/resource/MasterLatencyModelR.h"
#include "aios/navi_sched/resource/SchedulerConfigR.h"
#include "autil/TimeUtility.h"
#include "navi/engine/Kernel.h"

#include <algorithm>

namespace navi_sched {

class MasterUpdateKernel : public navi::Kernel {
public:
    ~MasterUpdateKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    navi::ErrorCode processUpdate(navi::KernelComputeContext &ctx,
                                  const AgentUpdateToMasterDataPtr &update,
                                  bool finishEvent, bool configNotified);
    bool applyUpdate(const AgentUpdateToMasterDataPtr &update,
                     MasterStopAckToAgentDataPtr &stopAck,
                     MasterPendingCacheSyncToAgentDataPtr &pendingCacheSync,
                     bool &resyncRequired,
                     MasterStateR::AgentUpdateTiming *timing);
    navi::ErrorCode updateLatencyWarmup(navi::KernelComputeContext &ctx);
    navi::ErrorCode updateSchedulerConfig(navi::KernelComputeContext &ctx);
    bool stopWarmupTimer(navi::KernelComputeContext &ctx);
    navi::ErrorCode closeOutputs(navi::KernelComputeContext &ctx);
    void releaseNode();

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(MasterStateR, _masterStateR);
    KERNEL_DEPEND_ON(MasterLatencyModelR, _masterLatencyModelR);
    KERNEL_DEPEND_ON_FALSE(SchedulerConfigR, _schedulerConfigR);
    std::string _agentId;
    std::string _agentBizName;
    std::string _collectorAddress;
    std::string _collectorId;
    MasterStateR::NodeStatePtr _nodeState;
    std::string _latencyModelFingerprint;
    std::string _warmupFingerprint;
    int32_t _latencyFormula = NAVI_SCHED_LATENCY_FORMULA_UNKNOWN;
    int64_t _latencyBinaryVersion = 0;
    size_t _latencyParameterCount = 0;
    bool _latencyModelTrusted = false;
    bool _warmupSubscribed = false;
    bool _warmupTimerActive = false;
    bool _handshakeAccepted = false;
    bool _stopping = false;
    uint64_t _lastWarmupCandidateVersion = 0;
    int64_t _lastWarmupSendTimeUs = 0;
    int64_t _lastSlowUpdateLogMonoUs = 0;
    uint64_t _suppressedSlowUpdateCount = 0;
    navi::AsyncPipePtr _warmupNotificationPipe;
    navi::AsyncPipePtr _warmupTimerPipe;
    navi::AsyncPipePtr _schedulerConfigNotificationPipe;
};

namespace {
constexpr int64_t WARMUP_SEND_COOLDOWN_US = 1000 * 1000;
constexpr int64_t SLOW_UPDATE_THRESHOLD_US = 100 * 1000;
constexpr int64_t SLOW_UPDATE_LOG_INTERVAL_US = 1000 * 1000;
}

MasterUpdateKernel::~MasterUpdateKernel() { releaseNode(); }

void MasterUpdateKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.master_update_k")
        .input("update", "")
        .output("finished", AGENT_UPDATE_TO_MASTER_TYPE_ID)
        .output("stop_ack", MASTER_STOP_ACK_TO_AGENT_TYPE_ID)
        .output("pending_cache_sync",
                MASTER_PENDING_CACHE_SYNC_TO_AGENT_TYPE_ID)
        .output("latency_warmup",
                MASTER_LATENCY_MODEL_WARMUP_TO_AGENT_TYPE_ID)
        .output("scheduler_config",
                MASTER_SCHEDULER_CONFIG_TO_AGENT_TYPE_ID);
}

bool MasterUpdateKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "agent_id", _agentId);
    NAVI_JSONIZE(ctx, "agent_biz_name", _agentBizName);
    NAVI_JSONIZE(ctx, "collector_address", _collectorAddress);
    NAVI_JSONIZE(ctx, "collector_id", _collectorId);
    return !_agentId.empty() && !_agentBizName.empty() && !_collectorId.empty();
}

navi::ErrorCode MasterUpdateKernel::init(navi::KernelInitContext &ctx)
{
    _warmupNotificationPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    _warmupTimerPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (_schedulerConfigR) {
        _schedulerConfigNotificationPipe =
            ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    }
    if (!_warmupNotificationPipe || !_warmupTimerPipe ||
        (_schedulerConfigR && !_schedulerConfigNotificationPipe)) {
        NAVI_KERNEL_LOG(
            ERROR,
            "create latency warmup pipes failed, agentId [%s], collectorId [%s], "
            "notificationPipe [%p], timerPipe [%p]",
            _agentId.c_str(), _collectorId.c_str(),
            _warmupNotificationPipe.get(), _warmupTimerPipe.get());
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (_schedulerConfigR &&
        !_schedulerConfigR->subscribe(_collectorId,
                                      _schedulerConfigNotificationPipe)) {
        NAVI_KERNEL_LOG(ERROR,
                        "subscribe scheduler config failed, collectorId [%s]",
                        _collectorId.c_str());
        return navi::EC_INIT_SUB_GRAPH;
    }
    return navi::EC_NONE;
}

navi::ErrorCode MasterUpdateKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr notificationData;
    bool notificationEof = false;
    _warmupNotificationPipe->getData(notificationData, notificationEof);
    if (notificationData) {
        NAVI_KERNEL_LOG(ERROR,
                        "unexpected latency warmup notification payload, "
                        "agentId [%s], collectorId [%s]",
                        _agentId.c_str(), _collectorId.c_str());
        return navi::EC_ABORT;
    }
    navi::DataPtr timerData;
    bool timerEof = false;
    _warmupTimerPipe->getData(timerData, timerEof);
    if (timerData) {
        NAVI_KERNEL_LOG(ERROR,
                        "unexpected latency warmup timer payload, agentId [%s], "
                        "collectorId [%s]",
                        _agentId.c_str(), _collectorId.c_str());
        return navi::EC_ABORT;
    }
    navi::DataPtr schedulerConfigNotification;
    bool schedulerConfigNotificationEof = !_schedulerConfigNotificationPipe;
    bool schedulerConfigNotified = false;
    if (_schedulerConfigNotificationPipe) {
        schedulerConfigNotified = _schedulerConfigNotificationPipe->getData(
            schedulerConfigNotification, schedulerConfigNotificationEof);
    }
    if (schedulerConfigNotification) {
        NAVI_KERNEL_LOG(ERROR,
                        "unexpected scheduler config notification payload, "
                        "agentId [%s], collectorId [%s]",
                        _agentId.c_str(), _collectorId.c_str());
        return navi::EC_ABORT;
    }
    if (_warmupTimerActive) {
        int64_t activatedCount = 0;
        int64_t remainingCount = 0;
        int64_t periodMs = 0;
        int64_t nextTickTimeMs = 0;
        if (!ctx.getPeriodicPipeActivationCount(
                _warmupTimerPipe, activatedCount, remainingCount, periodMs,
                nextTickTimeMs)) {
            NAVI_KERNEL_LOG(ERROR,
                            "read latency warmup timer state failed, agentId [%s], "
                            "collectorId [%s]",
                            _agentId.c_str(), _collectorId.c_str());
            return navi::EC_ABORT;
        }
        if (activatedCount > 0) {
            _warmupTimerActive = false;
        }
    }

    navi::DataPtr data;
    bool eof = false;
    const auto inputReady =
        ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof);
    auto update = std::dynamic_pointer_cast<AgentUpdateToMasterData>(data);
    if (data && !update) {
        NAVI_KERNEL_LOG(ERROR,
                        "invalid Agent update payload type, agentId [%s], "
                        "collectorId [%s]",
                        _agentId.c_str(), _collectorId.c_str());
        return navi::EC_ABORT;
    }
    if (update && (update->version == NAVI_SCHED_AGENT_UPDATE_BATCH_VERSION ||
                   update->version == NAVI_SCHED_AGENT_UPDATE_LEGACY_BATCH_VERSION)) {
        if (!update->validEnvelope()) {
            NAVI_KERNEL_LOG(ERROR, "invalid ordered Agent update batch");
            return navi::EC_ABORT;
        }
        for (size_t i = 0; i < update->updates.size(); ++i) {
            const bool last = i + 1 == update->updates.size();
            auto ec = processUpdate(ctx, update->updates[i], !last || !eof,
                                    last && schedulerConfigNotified);
            if (ec != navi::EC_NONE) return ec;
        }
    } else if (update) {
        auto ec = processUpdate(ctx, update, !eof, schedulerConfigNotified);
        if (ec != navi::EC_NONE) return ec;
    }
    if (eof) {
        stopWarmupTimer(ctx);
        releaseNode();
        return closeOutputs(ctx);
    }
    if (update) return navi::EC_NONE;
    if (!inputReady && notificationEof && timerEof &&
        schedulerConfigNotificationEof) {
        return navi::EC_NONE;
    }
    auto ec = schedulerConfigNotified
                  ? updateSchedulerConfig(ctx)
                  : navi::EC_NONE;
    return navi::EC_NONE == ec ? updateLatencyWarmup(ctx) : ec;
}

navi::ErrorCode MasterUpdateKernel::processUpdate(
    navi::KernelComputeContext &ctx, const AgentUpdateToMasterDataPtr &update,
    bool finishEvent, bool configNotified)
{
    MasterStopAckToAgentDataPtr stopAck;
    MasterPendingCacheSyncToAgentDataPtr pendingCacheSync;
    bool resyncRequired = false;
    MasterStateR::AgentUpdateTiming timing;
    const auto applyBeginMonoUs = update ? autil::TimeUtility::monotonicTimeUs() : 0;
    const auto updateApplied =
        update && applyUpdate(update, stopAck, pendingCacheSync, resyncRequired, &timing);
    if (update) {
        const auto applyEndMonoUs = autil::TimeUtility::monotonicTimeUs();
        const auto applyUs = applyEndMonoUs - applyBeginMonoUs;
        const bool receiveTimingValid = update->deserializeBeginMonoUs > 0 &&
            update->deserializeBeginMonoUs <= update->deserializeEndMonoUs &&
            update->deserializeEndMonoUs <= applyBeginMonoUs;
        const auto deserializeUs = receiveTimingValid
            ? update->deserializeEndMonoUs - update->deserializeBeginMonoUs : -1;
        const auto receiveQueueUs = receiveTimingValid
            ? applyBeginMonoUs - update->deserializeEndMonoUs : -1;
        if (applyUs >= SLOW_UPDATE_THRESHOLD_US ||
            deserializeUs >= SLOW_UPDATE_THRESHOLD_US ||
            receiveQueueUs >= SLOW_UPDATE_THRESHOLD_US) {
            if (_lastSlowUpdateLogMonoUs == 0 ||
                applyEndMonoUs - _lastSlowUpdateLogMonoUs >= SLOW_UPDATE_LOG_INTERVAL_US) {
                const bool enteredRuntime = timing.runtimeLockBeginMonoUs > 0;
                const auto preRuntimeUs = timing.beginMonoUs > 0
                    ? (enteredRuntime ? timing.runtimeLockBeginMonoUs : timing.endMonoUs) -
                          timing.beginMonoUs : -1;
                NAVI_KERNEL_LOG(
                    WARN,
                    "[navi-master-agent-update-timing] agent=%s collector=%s "
                    "event=%s event_time_us=%ld workload_revision=%lu "
                    "deserialize_us=%ld receive_queue_us=%ld apply_us=%ld "
                    "pre_runtime_us=%ld runtime_lock_wait_us=%ld "
                    "runtime_lock_hold_us=%ld post_runtime_us=%ld "
                    "state_total_us=%ld applied=%d suppressed_slow_updates=%lu",
                    _agentId.c_str(), _collectorId.c_str(),
                    NaviSchedAgentEvent::getName(update->event).c_str(), update->eventTimeUs,
                    update->workload ? update->workload->revision : 0,
                    deserializeUs, receiveQueueUs, applyUs, preRuntimeUs,
                    enteredRuntime ? timing.runtimeLockAcquiredMonoUs -
                                         timing.runtimeLockBeginMonoUs : -1,
                    enteredRuntime ? timing.runtimeLockReleasedMonoUs -
                                         timing.runtimeLockAcquiredMonoUs : -1,
                    enteredRuntime ? timing.endMonoUs -
                                         timing.runtimeLockReleasedMonoUs : -1,
                    timing.beginMonoUs > 0 ? timing.endMonoUs - timing.beginMonoUs : -1,
                    static_cast<int>(updateApplied), _suppressedSlowUpdateCount);
                _lastSlowUpdateLogMonoUs = applyEndMonoUs;
                _suppressedSlowUpdateCount = 0;
            } else {
                ++_suppressedSlowUpdateCount;
            }
        }
    }
    if (update && !updateApplied) {
        if (resyncRequired) {
            NAVI_KERNEL_LOG(ERROR,
                            "agent cache stream diverged; abort collector to "
                            "request a fresh snapshot");
            releaseNode();
            return navi::EC_ABORT;
        }
        NAVI_KERNEL_LOG(
            ERROR,
            "discard invalid agent update, agentId [%s], collectorId [%s], "
            "event [%s], version [%d], eventTimeUs [%ld], payloadValid [%d]",
            _agentId.c_str(), _collectorId.c_str(),
            NaviSchedAgentEvent::getName(update->event).c_str(), update->version,
            update->eventTimeUs, static_cast<int>(update->valid()));
    }
    if (stopAck) {
        if (!ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), stopAck,
                           false)) {
            NAVI_KERNEL_LOG(ERROR,
                            "publish Agent stop ACK failed, agentId [%s], "
                            "collectorId [%s]",
                            _agentId.c_str(), _collectorId.c_str());
            return navi::EC_ABORT;
        }
    }
    if (pendingCacheSync) {
        if (!ctx.setOutput(navi::PortIndex(2, navi::INVALID_INDEX),
                           pendingCacheSync, false)) {
            NAVI_KERNEL_LOG(ERROR,
                            "publish pending cache sync failed, agentId [%s], "
                            "collectorId [%s]",
                            _agentId.c_str(), _collectorId.c_str());
            return navi::EC_ABORT;
        }
    }
    if (updateApplied && NaviSchedAgentEvent::AGENT_STOPPING == update->event &&
        !ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), update, false)) {
        NAVI_KERNEL_LOG(ERROR,
                        "publish agent stopping collector result failed, "
                        "agentId [%s], collectorId [%s]",
                        _agentId.c_str(), _collectorId.c_str());
        return navi::EC_ABORT;
    }
    if (!finishEvent) return navi::EC_NONE;
    auto ec = configNotified || (updateApplied && update->event == NaviSchedAgentEvent::AGENT_HANDSHAKE)
                  ? updateSchedulerConfig(ctx) : navi::EC_NONE;
    return ec == navi::EC_NONE ? updateLatencyWarmup(ctx) : ec;
}

bool MasterUpdateKernel::applyUpdate(const AgentUpdateToMasterDataPtr &update,
                                     MasterStopAckToAgentDataPtr &stopAck,
                                     MasterPendingCacheSyncToAgentDataPtr &pendingCacheSync,
                                     bool &resyncRequired,
                                     MasterStateR::AgentUpdateTiming *timing)
{
    resyncRequired = false;
    if (_stopping) {
        NAVI_KERNEL_LOG(WARN,
                        "reject Agent update after stopping, agentId [%s], "
                        "collectorId [%s], event [%s]",
                        _agentId.c_str(), _collectorId.c_str(),
                        NaviSchedAgentEvent::getName(update->event).c_str());
        return false;
    }
    if (!_handshakeAccepted &&
        NaviSchedAgentEvent::AGENT_HANDSHAKE != update->event) {
        NAVI_KERNEL_LOG(WARN,
                        "reject Agent update before handshake, agentId [%s], "
                        "collectorId [%s], event [%s]",
                        _agentId.c_str(), _collectorId.c_str(),
                        NaviSchedAgentEvent::getName(update->event).c_str());
        return false;
    }
    if (!_nodeState) {
        _nodeState =
            _masterStateR->createAgent(_agentId, _agentBizName, _collectorAddress, 0);
        if (!_nodeState ||
            !_masterStateR->bindAgentCollector(_collectorId, _agentId, _nodeState) ||
            !_masterLatencyModelR->activateAgent(_agentId, _collectorId)) {
            releaseNode();
            return false;
        }
    }
    bool pendingCacheResyncRequired = false;
    MasterPendingCacheR::ApplyResult pendingCacheApplyResult;
    MasterLatencyModelR::RecordResult latencyRecordResult;
    if (!_masterStateR->reportAgent(_nodeState, update, &resyncRequired,
                                    &pendingCacheResyncRequired,
                                    &pendingCacheApplyResult, _collectorId,
                                    &latencyRecordResult, timing)) {
        return false;
    }
    if (NaviSchedAgentEvent::AGENT_HANDSHAKE == update->event) {
        _handshakeAccepted = true;
        _latencyModelFingerprint = update->latencyModelFingerprint;
        if (_schedulerConfigR && update->schedulerConfigGeneration > 0 &&
            !_schedulerConfigR->activateSubscriber(
                _collectorId, _agentId, update->control->serverInfo.ip,
                update->schedulerConfigGeneration,
                update->schedulerConfigHash, update->schedulerConfigJson,
                update->latencyModelFingerprint)) {
            NAVI_KERNEL_LOG(ERROR,
                            "activate scheduler config subscriber failed, "
                            "agentId [%s], collectorId [%s]",
                            _agentId.c_str(), _collectorId.c_str());
            return false;
        }
    } else if (_schedulerConfigR &&
               NaviSchedAgentEvent::SCHEDULER_CONFIG_UPDATED == update->event) {
        _schedulerConfigR->reportAgentConfig(
            _collectorId, update->schedulerConfigRequestGeneration,
            update->schedulerConfigGeneration, update->schedulerConfigHash,
            update->schedulerConfigJson, update->schedulerConfigError);
    }
    if (update->latency && !_latencyModelFingerprint.empty()) {
        if (latencyRecordResult.model) {
            const auto trusted =
                latencyRecordResult.model->overloadPredictionTrusted;
            const auto subscriptionChanged =
                !_warmupSubscribed || _latencyModelTrusted != trusted ||
                _warmupFingerprint != _latencyModelFingerprint ||
                _latencyFormula != update->latency->formula ||
                _latencyBinaryVersion != update->latency->binaryVersion ||
                _latencyParameterCount != update->latency->parameters.size();
            _latencyModelTrusted = trusted;
            _latencyFormula = update->latency->formula;
            _latencyBinaryVersion = update->latency->binaryVersion;
            _latencyParameterCount = update->latency->parameters.size();
            if (trusted) {
                _warmupSubscribed = false;
            } else if (subscriptionChanged) {
                _lastWarmupCandidateVersion = 0;
                if (!_masterLatencyModelR->updateWarmupSubscriber(
                        _agentId, _collectorId, _latencyModelFingerprint,
                        *update->latency, false, _warmupNotificationPipe)) {
                    NAVI_KERNEL_LOG(
                        ERROR,
                        "register latency warmup candidate pipe failed, agentId [%s], "
                        "collectorId [%s], fingerprint [%s], formula [%d], "
                        "binaryVersion [%ld], parameterCount [%lu]",
                        _agentId.c_str(), _collectorId.c_str(),
                        _latencyModelFingerprint.c_str(), _latencyFormula,
                        _latencyBinaryVersion, _latencyParameterCount);
                    return false;
                }
                _warmupFingerprint = _latencyModelFingerprint;
                _warmupSubscribed = true;
            }
        }
    }
    if (pendingCacheResyncRequired) {
        pendingCacheSync =
            std::make_shared<MasterPendingCacheSyncToAgentData>();
        pendingCacheSync->collectorId = _collectorId;
        pendingCacheSync->reason = pendingCacheApplyResult.reason;
        pendingCacheSync->fullSnapshot = pendingCacheApplyResult.fullSnapshot;
        pendingCacheSync->snapshotReadyBefore =
            pendingCacheApplyResult.snapshotReadyBefore;
        pendingCacheSync->candidateStateComputed =
            pendingCacheApplyResult.candidateStateComputed;
        pendingCacheSync->masterRevisionBefore =
            pendingCacheApplyResult.masterRevisionBefore;
        pendingCacheSync->expectedRevision =
            pendingCacheApplyResult.expectedRevision;
        pendingCacheSync->rejectedRevision =
            pendingCacheApplyResult.receivedRevision;
        pendingCacheSync->masterChecksumBefore =
            pendingCacheApplyResult.masterChecksumBefore;
        pendingCacheSync->candidateChecksum =
            pendingCacheApplyResult.candidateChecksum;
        pendingCacheSync->receivedChecksum =
            pendingCacheApplyResult.receivedChecksum;
        pendingCacheSync->masterBatchCountBefore =
            pendingCacheApplyResult.masterBatchCountBefore;
        pendingCacheSync->candidateBatchCount =
            pendingCacheApplyResult.candidateBatchCount;
        pendingCacheSync->receivedBatchCount =
            pendingCacheApplyResult.receivedBatchCount;
        pendingCacheSync->masterBlockCountBefore =
            pendingCacheApplyResult.masterBlockCountBefore;
        pendingCacheSync->candidateBlockCount =
            pendingCacheApplyResult.candidateBlockCount;
        pendingCacheSync->receivedBlockCount =
            pendingCacheApplyResult.receivedBlockCount;
        pendingCacheSync->masterTailBlockCountBefore =
            pendingCacheApplyResult.masterTailBlockCountBefore;
        pendingCacheSync->candidateTailBlockCount =
            pendingCacheApplyResult.candidateTailBlockCount;
        pendingCacheSync->receivedTailBlockCount =
            pendingCacheApplyResult.receivedTailBlockCount;
        NAVI_KERNEL_LOG(
            WARN,
            "request pending cache full snapshot fallback, agentId [%s], "
            "collectorId [%s], reason [%s], fullSnapshot [%d], "
            "snapshotReadyBefore [%d], masterRevisionBefore [%lu], "
            "expectedRevision [%lu], rejectedRevision [%lu], "
            "masterChecksumBefore [%lu], candidateChecksum [%lu], "
            "receivedChecksum [%lu], candidateStateComputed [%d], "
            "masterBatchCountBefore [%ld], candidateBatchCount [%ld], "
            "receivedBatchCount [%ld], masterBlockCountBefore [%ld], "
            "candidateBlockCount [%ld], receivedBlockCount [%ld], "
            "masterTailBlockCountBefore [%ld], candidateTailBlockCount [%ld], "
            "receivedTailBlockCount [%ld]",
            _agentId.c_str(), _collectorId.c_str(),
            pendingCacheSync->reason.c_str(),
            static_cast<int>(pendingCacheSync->fullSnapshot),
            static_cast<int>(pendingCacheSync->snapshotReadyBefore),
            pendingCacheSync->masterRevisionBefore,
            pendingCacheSync->expectedRevision,
            pendingCacheSync->rejectedRevision,
            pendingCacheSync->masterChecksumBefore,
            pendingCacheSync->candidateChecksum,
            pendingCacheSync->receivedChecksum,
            static_cast<int>(pendingCacheSync->candidateStateComputed),
            pendingCacheSync->masterBatchCountBefore,
            pendingCacheSync->candidateBatchCount,
            pendingCacheSync->receivedBatchCount,
            pendingCacheSync->masterBlockCountBefore,
            pendingCacheSync->candidateBlockCount,
            pendingCacheSync->receivedBlockCount,
            pendingCacheSync->masterTailBlockCountBefore,
            pendingCacheSync->candidateTailBlockCount,
            pendingCacheSync->receivedTailBlockCount);
    }
    if (NaviSchedAgentEvent::AGENT_STOPPING == update->event) {
        _stopping = true;
        _warmupSubscribed = false;
        releaseNode();
        stopAck = std::make_shared<MasterStopAckToAgentData>();
        stopAck->collectorId = _collectorId;
        NAVI_KERNEL_LOG(INFO, "agent stopping confirmed, agentId [%s], collectorId [%s]",
                        _agentId.c_str(), _collectorId.c_str());
    }
    return true;
}

navi::ErrorCode MasterUpdateKernel::updateLatencyWarmup(
    navi::KernelComputeContext &ctx)
{
    if (_stopping || _latencyModelTrusted || !_warmupSubscribed ||
        _latencyModelFingerprint.empty() ||
        _latencyFormula == NAVI_SCHED_LATENCY_FORMULA_UNKNOWN ||
        _latencyBinaryVersion <= 0 || _latencyParameterCount == 0) {
        if ((_stopping || _latencyModelTrusted) && _warmupTimerActive) {
            stopWarmupTimer(ctx);
        }
        return navi::EC_NONE;
    }

    const auto candidateVersion =
        _masterLatencyModelR->getWarmupCandidateVersion(
            _latencyModelFingerprint, _latencyFormula, _latencyParameterCount);
    if (candidateVersion <= _lastWarmupCandidateVersion) {
        return navi::EC_NONE;
    }

    const auto nowUs = autil::TimeUtility::monotonicTimeUs();
    const auto elapsedUs = _lastWarmupSendTimeUs == 0
                               ? WARMUP_SEND_COOLDOWN_US
                               : std::max<int64_t>(
                                     0, nowUs - _lastWarmupSendTimeUs);
    if (elapsedUs < WARMUP_SEND_COOLDOWN_US) {
        if (_warmupTimerActive) {
            return navi::EC_NONE;
        }
        const auto remainingUs = WARMUP_SEND_COOLDOWN_US - elapsedUs;
        const auto remainingMs = std::max<int64_t>(1, (remainingUs + 999) / 1000);
        if (!ctx.registerPeriodicPipe(_warmupTimerPipe, remainingMs, 1)) {
            NAVI_KERNEL_LOG(
                ERROR,
                "register latency warmup one-shot timer failed, agentId [%s], "
                "collectorId [%s], candidateVersion [%lu], lastVersion [%lu], "
                "remainingMs [%ld]",
                _agentId.c_str(), _collectorId.c_str(), candidateVersion,
                _lastWarmupCandidateVersion, remainingMs);
            return navi::EC_ABORT;
        }
        _warmupTimerActive = true;
        return navi::EC_NONE;
    }

    uint64_t emittedVersion = 0;
    auto warmup = _masterLatencyModelR->makeWarmupIfNew(
        _latencyModelFingerprint, _latencyFormula, _latencyBinaryVersion,
        _latencyParameterCount, _lastWarmupCandidateVersion, emittedVersion);
    if (!warmup) {
        _lastWarmupCandidateVersion =
            std::max(_lastWarmupCandidateVersion, emittedVersion);
        return navi::EC_NONE;
    }
    if (!ctx.setOutput(navi::PortIndex(3, navi::INVALID_INDEX), warmup, false)) {
        NAVI_KERNEL_LOG(
            ERROR,
            "publish latency warmup failed, agentId [%s], collectorId [%s], "
            "candidateVersion [%lu], lastVersion [%lu], candidateCount [%lu]",
            _agentId.c_str(), _collectorId.c_str(), emittedVersion,
            _lastWarmupCandidateVersion, warmup->candidates.size());
        return navi::EC_ABORT;
    }
    if (_warmupTimerActive) {
        stopWarmupTimer(ctx);
    }
    _lastWarmupCandidateVersion = emittedVersion;
    _lastWarmupSendTimeUs = nowUs;
    NAVI_KERNEL_LOG(
        INFO,
        "publish latency warmup, agentId [%s], collectorId [%s], "
        "candidateVersion [%lu], candidateCount [%lu]",
        _agentId.c_str(), _collectorId.c_str(), emittedVersion,
        warmup->candidates.size());
    return navi::EC_NONE;
}

navi::ErrorCode MasterUpdateKernel::updateSchedulerConfig(
    navi::KernelComputeContext &ctx)
{
    if (!_schedulerConfigR || _stopping || !_handshakeAccepted) {
        return navi::EC_NONE;
    }
    SchedulerConfigR::PendingAgentUpdate pending;
    if (!_schedulerConfigR->getPendingAgentUpdate(_collectorId, pending)) {
        return navi::EC_NONE;
    }
    auto config = std::make_shared<MasterSchedulerConfigToAgentData>();
    config->generation = pending.requestGeneration;
    config->configHash = pending.configHash;
    config->configJson = pending.configJson;
    if (!config->valid() ||
        !ctx.setOutput(navi::PortIndex(4, navi::INVALID_INDEX), config, false)) {
        NAVI_KERNEL_LOG(ERROR,
                        "publish scheduler config failed, agentId [%s], "
                        "collectorId [%s], generation [%lu]",
                        _agentId.c_str(), _collectorId.c_str(),
                        pending.requestGeneration);
        return navi::EC_ABORT;
    }
    NAVI_KERNEL_LOG(INFO,
                    "publish scheduler config, agentId [%s], collectorId [%s], "
                    "generation [%lu], hash [%s]",
                    _agentId.c_str(), _collectorId.c_str(),
                    pending.requestGeneration, pending.configHash.c_str());
    return navi::EC_NONE;
}

bool MasterUpdateKernel::stopWarmupTimer(navi::KernelComputeContext &ctx)
{
    if (!_warmupTimerActive) {
        return true;
    }
    _warmupTimerActive = false;
    int64_t activatedCount = 0;
    int64_t remainingCount = 0;
    int64_t periodMs = 0;
    int64_t nextTickTimeMs = 0;
    if (!ctx.getPeriodicPipeActivationCount(
            _warmupTimerPipe, activatedCount, remainingCount, periodMs,
            nextTickTimeMs) || nextTickTimeMs < 0) {
        return true;
    }
    if (ctx.stopPeriodicAsyncPipe(_warmupTimerPipe)) {
        return true;
    }
    NAVI_KERNEL_LOG(WARN,
                    "stop latency warmup timer failed, agentId [%s], "
                    "collectorId [%s]",
                    _agentId.c_str(), _collectorId.c_str());
    return false;
}

navi::ErrorCode MasterUpdateKernel::closeOutputs(
    navi::KernelComputeContext &ctx)
{
    for (int32_t index = 0; index < 5; ++index) {
        if (ctx.setOutput(navi::PortIndex(index, navi::INVALID_INDEX), nullptr,
                          true)) {
            continue;
        }
        NAVI_KERNEL_LOG(ERROR,
                        "close Master update output failed, agentId [%s], "
                        "collectorId [%s], outputIndex [%d]",
                        _agentId.c_str(), _collectorId.c_str(), index);
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

void MasterUpdateKernel::releaseNode()
{
    if (_schedulerConfigR && _schedulerConfigNotificationPipe) {
        _schedulerConfigR->unsubscribe(_collectorId,
                                       _schedulerConfigNotificationPipe);
    }
    if (_masterLatencyModelR) {
        _masterLatencyModelR->removeWarmupSubscriber(_collectorId);
    }
    _warmupSubscribed = false;
    if (!_nodeState) {
        return;
    }
    if (!_stopping &&
        !_masterStateR->stopAgentCollector(_agentId, _collectorId)) {
        NAVI_KERNEL_LOG(WARN,
                        "failed to mark Agent offline before releasing collector, "
                        "agentId [%s], collectorId [%s]",
                        _agentId.c_str(), _collectorId.c_str());
    }
    _masterStateR->closeAgentCollector(_agentId, _collectorId);
    _masterStateR->unbindAgentCollector(_collectorId);
    _masterStateR->removeAgent(_nodeState);
    _masterLatencyModelR->removeAgent(_agentId, _collectorId);
    _nodeState.reset();
}

REGISTER_KERNEL(MasterUpdateKernel);

}
