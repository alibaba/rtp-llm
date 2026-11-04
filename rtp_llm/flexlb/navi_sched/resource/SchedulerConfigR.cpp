#include "rtp_llm/flexlb/navi_sched/resource/SchedulerConfigR.h"

#include "rtp_llm/flexlb/navi_sched/kernel/SchedulerConfigKernel.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyResidualModel.h"
#include "rtp_llm/flexlb/navi_sched/proto/SchedulerConfig.pb.h"
#include "autil/legacy/fast_jsonizable_dec.h"
#include "autil/TimeUtility.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/rpc_server/ArpcRegistryParam.h"

#include <limits>
#include <algorithm>
#include <set>
#include <vector>

namespace navi_sched {

namespace {
constexpr int64_t AGENT_CONFIG_SYNC_TIMEOUT_US = 10 * 1000 * 1000;
const std::string SCHEDULER_CONFIG_RESPONSE_TYPE_ID =
    "NaviSchedSchedulerConfigResponseData";

struct AgentConfigSyncFailure : autil::legacy::Jsonizable {
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("ip", ip);
        json.Jsonize("error", error);
        json.Jsonize("effective_hash", effectiveHash);
        json.Jsonize("effective_config", effectiveConfig);
    }

    std::string ip;
    std::string error;
    std::string effectiveHash;
    std::string effectiveConfig;
};

struct LatencyModelFingerprint : autil::legacy::Jsonizable {
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("hardware", hardware, hardware);
    }

    std::string hardware;
};

const std::set<std::string> &agentConfigFields()
{
    static const std::set<std::string> fields = {
        "batch_timeout_ms",
        "closed_batch_keep_ms",
        "latency_fit_batch_size",
        "latency_fit_interval",
        "latency_learning_rate",
        "latency_online_learning_rate_scale",
        "latency_current_sample_weight",
        "latency_parameter_learning_rate_scales",
        "latency_nonlinear_scale",
        "queue_learning_rate",
        "latency_check_gradient",
        "latency_cold_start_fit_epochs",
        "latency_history_limit",
        "latency_history_bucket_capacity",
        "latency_persist_interval",
        "latency_persist_interval_ms",
        "latency_warmup_sample_count",
        "latency_model_trust_min_revision",
        "latency_zero_anchor_max_ms",
        "latency_zero_anchor_min_samples",
        "latency_validation_enabled",
        "residual_learning_rate",
        "residual_update_error_floor_ms",
        "residual_update_error_ratio",
        "residual_idle_decay_start_ms",
        "residual_idle_decay_end_ms",
        "time_compression",
    };
    return fields;
}

bool hasNonAgentConfigFields(const std::string &configJson,
                             std::string &errorMessage)
{
    try {
        const auto patch =
            autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                autil::legacy::json::ParseJson(configJson));
        return std::any_of(
            patch.begin(), patch.end(), [](const auto &entry) {
                return agentConfigFields().find(entry.first) ==
                       agentConfigFields().end();
            });
    } catch (const std::exception &error) {
        errorMessage = error.what();
    } catch (...) {
        errorMessage = "scheduler config must be a JSON object";
    }
    return false;
}
} // namespace

SchedulerConfigResponseData::SchedulerConfigResponseData()
    : navi::Data(SCHEDULER_CONFIG_RESPONSE_TYPE_ID)
{
}

const std::string SchedulerConfigR::RESOURCE_ID = "navi_sched.scheduler_config_r";
const std::string SchedulerConfigR::RPC_REQUEST_NAME =
    "navi_sched.scheduler_config_request";

void SchedulerConfigR::Config::Jsonize(JsonWrapper &json)
{
    cost.Jsonize(json);
    json.Jsonize("max_wait_schedule_count", maxWaitScheduleCount,
                 maxWaitScheduleCount);
    json.Jsonize("max_wait_schedule_time_ms", maxWaitScheduleTimeMs,
                 maxWaitScheduleTimeMs);
    json.Jsonize("queue_overload_start_cycle_multiplier",
                 queueOverloadStartCycleMultiplier,
                 queueOverloadStartCycleMultiplier);
    json.Jsonize("queue_overload_full_cycle_multiplier",
                 queueOverloadFullCycleMultiplier,
                 queueOverloadFullCycleMultiplier);
    json.Jsonize("admission_timeout_ms", admissionTimeoutMs,
                 admissionTimeoutMs);
    agent.Jsonize(json);
}

bool SchedulerConfigR::Config::valid() const
{
    if (!cost.valid() || maxWaitScheduleCount == 0 || maxWaitScheduleTimeMs < 10 ||
        maxWaitScheduleTimeMs > std::numeric_limits<int64_t>::max() / 1000 ||
        queueOverloadStartCycleMultiplier < 0 ||
        queueOverloadFullCycleMultiplier < 0 || admissionTimeoutMs <= 0 ||
        !agent.valid()) {
        return false;
    }
    if (queueOverloadStartCycleMultiplier == 0 ||
        queueOverloadFullCycleMultiplier == 0) {
        return true;
    }
    return queueOverloadFullCycleMultiplier >
               queueOverloadStartCycleMultiplier &&
           queueOverloadFullCycleMultiplier <=
               std::numeric_limits<int64_t>::max() / maxWaitScheduleTimeMs;
}

void SchedulerConfigR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ)
        .dependOn(navi::NaviRpcServerR::RESOURCE_ID, true,
                  BIND_RESOURCE_TO(_naviRpcServerR));
}

bool SchedulerConfigR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "config_json", _initialConfigJson, _initialConfigJson);
    return true;
}

navi::ErrorCode SchedulerConfigR::init(navi::ResourceInitContext &ctx)
{
    _bizName = ctx.getBizName();
    if (!_initialConfigJson.empty()) {
        try {
            autil::legacy::FromJsonString(_snapshot.config,
                                          _initialConfigJson);
        } catch (const std::exception &error) {
            NAVI_KERNEL_LOG(ERROR, "invalid initial scheduler config: %s",
                            error.what());
            return navi::EC_ABORT;
        }
        if (!_snapshot.config.valid()) {
            NAVI_KERNEL_LOG(ERROR, "invalid initial scheduler config values");
            return navi::EC_ABORT;
        }
        _snapshot.config.cost.optimizationTimeBudgetUs =
            _snapshot.config.maxWaitScheduleTimeMs * 800;
    }
    _snapshot.generation = std::max<uint64_t>(
        _snapshot.generation,
        static_cast<uint64_t>(autil::TimeUtility::currentTime()));
    _snapshot.agentConfigHash =
        AgentSchedulerConfigR::hashConfig(_snapshot.config.agent);
    navi::ArpcRegistryParam param;
    param.arpcParam.method = "scheduler_config";
    param.arpcParam.httpAliasVec.emplace_back("/scheduler-config");
    param.arpcParam.httpAliasVec.emplace_back("/health");
    param.arpcParam.logQuery = true;
    param.graphParam.requestDataName = RPC_REQUEST_NAME;
    if (!buildGraph(param.graphParam.graph)) {
        NAVI_KERNEL_LOG(ERROR, "build scheduler config rpc graph failed");
        return navi::EC_ABORT;
    }
    auto service = _naviRpcServerR->registerArpcService<SchedulerConfigService>(
        param, this);
    if (!service) {
        NAVI_KERNEL_LOG(ERROR, "register scheduler config rpc service failed");
        return navi::EC_ABORT;
    }
    NAVI_KERNEL_LOG(INFO,
                    "scheduler_config_effective generation [%lu] config [%s]",
                    _snapshot.generation, toJson(_snapshot.config).c_str());
    return navi::EC_NONE;
}

bool SchedulerConfigR::buildGraph(navi::GraphDef &graphDef) const
{
    navi::GraphBuilder builder(&graphDef);
    auto graphId = builder.newSubGraph(_bizName);
    builder.subGraph(graphId).location(_bizName, 1, 0);
    builder.node("scheduler_config")
        .kernel(SchedulerConfigKernel::KERNEL_ID)
        .out(SchedulerConfigKernel::OUTPUT_PORT)
        .asGraphOutput("out");
    navi::GraphBuilder::fillHashKey(&graphDef);
    return builder.ok();
}

SchedulerConfigR::Snapshot SchedulerConfigR::get() const
{
    autil::ScopedLock lock(_mutex);
    return _snapshot;
}

bool SchedulerConfigR::setSyncPipe(const navi::AsyncPipePtr &syncPipe)
{
    if (!syncPipe) {
        return false;
    }
    bool activate = false;
    {
        autil::ScopedLock lock(_mutex);
        _syncPipe = syncPipe;
        activate = !_requests.empty() || static_cast<bool>(_agentSync);
    }
    return !activate || notifySyncPipe(syncPipe);
}

void SchedulerConfigR::clearSyncPipe(const navi::AsyncPipePtr &syncPipe)
{
    autil::ScopedLock lock(_mutex);
    if (!syncPipe || _syncPipe == syncPipe) {
        _syncPipe.reset();
    }
}

bool SchedulerConfigR::stopSyncPipe()
{
    navi::AsyncPipePtr syncPipe;
    {
        autil::ScopedLock lock(_mutex);
        syncPipe = _syncPipe;
    }
    return syncPipe && navi::EC_NONE == syncPipe->setEof();
}

bool SchedulerConfigR::submit(Request request)
{
    if (!request.responsePipe) {
        return false;
    }
    const auto responsePipe = request.responsePipe;
    navi::AsyncPipePtr syncPipe;
    SchedulerConfigResponseDataPtr rejectedResponse;
    {
        autil::ScopedLock lock(_mutex);
        syncPipe = _syncPipe;
        if (!syncPipe) {
            return false;
        }
        if (request.update) {
            if (_activeUpdateToken != 0) {
                const std::string errorMessage =
                    "another scheduler config update is still active";
                rejectedResponse = makeResponseLocked(false, errorMessage);
                logRejectedUpdateLocked(request.configJson, errorMessage);
            } else {
                request.updateToken = acquireUpdateTokenLocked();
            }
        }
        if (rejectedResponse) {
            syncPipe.reset();
        } else {
            _requests.emplace_back(std::move(request));
        }
    }
    if (rejectedResponse) {
        return navi::EC_NONE == responsePipe->setData(rejectedResponse);
    }
    if (notifySyncPipe(syncPipe)) {
        return true;
    }
    autil::ScopedLock lock(_mutex);
    auto it = std::find_if(
        _requests.begin(), _requests.end(),
        [&responsePipe](const auto &queued) {
            return queued.responsePipe == responsePipe;
        });
    if (it == _requests.end()) {
        return true;
    }
    releaseUpdateTokenLocked(it->updateToken);
    _requests.erase(it);
    return false;
}

bool SchedulerConfigR::takeRequest(Request &request)
{
    autil::ScopedLock lock(_mutex);
    if (_requests.empty()) {
        return false;
    }
    request = std::move(_requests.front());
    _requests.pop_front();
    return true;
}

bool SchedulerConfigR::processRequest(
    const Request &request, SchedulerConfigResponseDataPtr &response,
    bool &waitingForAgents)
{
    response.reset();
    waitingForAgents = false;
    if (!request.update) {
        autil::ScopedLock lock(_mutex);
        response = makeResponseLocked(true, "");
        response->hardware = request.hardware;
        return true;
    }

    std::vector<navi::AsyncPipePtr> notificationPipes;
    bool allTargetsCompleted = true;
    {
        autil::ScopedLock lock(_mutex);
        uint64_t updateToken = request.updateToken;
        if (updateToken == 0) {
            if (_activeUpdateToken != 0) {
                const std::string errorMessage =
                    "another scheduler config update is still active";
                response = makeResponseLocked(false, errorMessage);
                logRejectedUpdateLocked(request.configJson, errorMessage);
                return true;
            }
            updateToken = acquireUpdateTokenLocked();
        } else if (_activeUpdateToken != updateToken) {
            const std::string errorMessage =
                "scheduler config update token is no longer active";
            response = makeResponseLocked(false, errorMessage);
            logRejectedUpdateLocked(request.configJson, errorMessage);
            return true;
        }
        Config updated;
        std::string errorMessage;
        if (!applyPatch(_snapshot.config, request.configJson, updated,
                        errorMessage)) {
            response = makeResponseLocked(false, errorMessage);
            response->hardware = request.hardware;
            logRejectedUpdateLocked(request.configJson, errorMessage);
            releaseUpdateTokenLocked(updateToken);
            return true;
        }
        const bool hardwareScoped = !request.hardware.empty();
        if (request.hardware.size() > 256) {
            errorMessage = "hardware selector is longer than 256 bytes";
            response = makeResponseLocked(false, errorMessage);
            response->hardware = request.hardware;
            logRejectedUpdateLocked(request.configJson, errorMessage);
            releaseUpdateTokenLocked(updateToken);
            return true;
        }
        if (hardwareScoped &&
            hasNonAgentConfigFields(request.configJson, errorMessage)) {
            errorMessage =
                "hardware-scoped updates may only contain Agent config fields";
            response = makeResponseLocked(false, errorMessage);
            response->hardware = request.hardware;
            logRejectedUpdateLocked(request.configJson, errorMessage);
            releaseUpdateTokenLocked(updateToken);
            return true;
        }
        AgentSchedulerConfig ignoredAgentConfig;
        bool hasAgentFields = false;
        if (!applyAgentPatch(_snapshot.config.agent, request.configJson,
                             ignoredAgentConfig, hasAgentFields,
                             errorMessage)) {
            response = makeResponseLocked(false, errorMessage);
            response->hardware = request.hardware;
            logRejectedUpdateLocked(request.configJson, errorMessage);
            releaseUpdateTokenLocked(updateToken);
            return true;
        }
        if (hardwareScoped && !hasAgentFields) {
            errorMessage =
                "hardware-scoped update contains no Agent config fields";
            response = makeResponseLocked(false, errorMessage);
            response->hardware = request.hardware;
            logRejectedUpdateLocked(request.configJson, errorMessage);
            releaseUpdateTokenLocked(updateToken);
            return true;
        }
        std::string currentAgentPatchJson;
        if (hardwareScoped) {
            const auto it =
                _hardwareAgentConfigPatchJson.find(request.hardware);
            if (it != _hardwareAgentConfigPatchJson.end()) {
                currentAgentPatchJson = it->second;
            }
        } else {
            currentAgentPatchJson = _agentConfigPatchJson;
        }
        std::string updatedAgentPatchJson;
        if (!mergeAgentPatch(currentAgentPatchJson, request.configJson,
                             updatedAgentPatchJson, hasAgentFields,
                             errorMessage)) {
            response = makeResponseLocked(false, errorMessage);
            response->hardware = request.hardware;
            logRejectedUpdateLocked(request.configJson, errorMessage);
            releaseUpdateTokenLocked(updateToken);
            return true;
        }

        if (hardwareScoped) {
            AgentSchedulerConfig validated;
            if (!applyStoredAgentPatchesLocked(
                    _snapshot.config.agent, request.hardware, nullptr,
                    &updatedAgentPatchJson, validated, errorMessage)) {
                response = makeResponseLocked(false, errorMessage);
                response->hardware = request.hardware;
                logRejectedUpdateLocked(request.configJson, errorMessage);
                releaseUpdateTokenLocked(updateToken);
                return true;
            }
            updated = _snapshot.config;
        }

        const auto previousGeneration = _snapshot.generation;
        const auto previousAgentConfigHash = _snapshot.agentConfigHash;
        const auto previousConfigJson = toJson(_snapshot.config);
        _snapshot.config = std::move(updated);
        _snapshot.agentConfigHash =
            AgentSchedulerConfigR::hashConfig(_snapshot.config.agent);
        ++_snapshot.generation;
        NAVI_KERNEL_LOG(
            INFO,
            "[navi-scheduler-config-update] role [master], status [accepted], "
            "previousGeneration [%lu], generation [%lu], "
            "previousAgentHash [%s], agentHash [%s], hasAgentFields [%d], "
            "hardware [%s], patch [%s], previousConfig [%s], "
            "effectiveConfig [%s]",
            previousGeneration, _snapshot.generation,
            previousAgentConfigHash.c_str(), _snapshot.agentConfigHash.c_str(),
            static_cast<int>(hasAgentFields), request.hardware.c_str(),
            request.configJson.c_str(),
            previousConfigJson.c_str(), toJson(_snapshot.config).c_str());

        if (!hasAgentFields) {
            response = makeResponseLocked(true, "");
            response->hardware = request.hardware;
            logCompletedUpdateLocked(_snapshot.generation, true, false, 0, 0,
                                     0, "");
            releaseUpdateTokenLocked(updateToken);
            return true;
        }
        if (hardwareScoped) {
            _hardwareAgentConfigPatchJson[request.hardware] =
                std::move(updatedAgentPatchJson);
        } else {
            _agentConfigPatchJson = std::move(updatedAgentPatchJson);
        }
        _agentConfigGeneration = _snapshot.generation;

        auto sync = std::make_unique<AgentSyncState>();
        sync->updateToken = updateToken;
        sync->requestGeneration = _snapshot.generation;
        sync->hardware = request.hardware;
        sync->deadlineTimeUs = autil::TimeUtility::monotonicTimeUs() +
                               AGENT_CONFIG_SYNC_TIMEOUT_US;
        sync->responsePipe = request.responsePipe;
        for (auto it = _subscribers.begin(); it != _subscribers.end();) {
            auto notificationPipe = it->second.notificationPipe.lock();
            if (!notificationPipe) {
                it = _subscribers.erase(it);
                continue;
            }
            const auto subscriberId = it->first;
            auto &subscriber = it->second;
            ++it;
            if (!subscriber.active || subscriber.configHash.empty() ||
                subscriber.configJson.empty()) {
                continue;
            }
            if (hardwareScoped && subscriber.hardware != request.hardware) {
                continue;
            }
            AgentSyncTarget target;
            target.agentIp = subscriber.agentIp.empty()
                                 ? subscriber.agentId
                                 : subscriber.agentIp;
            target.hardware = subscriber.hardware;
            target.effectiveHash = subscriber.configHash;
            target.effectiveJson = subscriber.configJson;
            AgentSchedulerConfig current;
            AgentSchedulerConfig expected;
            if (!AgentSchedulerConfigR::parse(subscriber.configJson, current,
                                              errorMessage) ||
                !applyStoredAgentPatchesLocked(current, subscriber.hardware,
                                               nullptr, nullptr, expected,
                                               errorMessage)) {
                target.errorMessage = errorMessage;
                target.completed = true;
                sync->targets.emplace(subscriberId, std::move(target));
                continue;
            }
            const bool hasPendingCatchup = subscriber.pendingConfigGeneration != 0;
            subscriber.pendingConfigGeneration = 0;
            subscriber.pendingConfigHash.clear();
            subscriber.pendingConfigJson.clear();
            subscriber.pendingConfigSent = false;
            target.expectedJson = AgentSchedulerConfigR::toJson(expected);
            target.expectedHash =
                AgentSchedulerConfigR::hashJson(target.expectedJson);
            // A reported hash predates any in-flight catch-up. Require a fresh
            // ACK even when the new desired config equals that reported hash.
            if (!hasPendingCatchup && target.expectedHash == subscriber.configHash) {
                target.completed = true;
            } else {
                allTargetsCompleted = false;
                notificationPipes.emplace_back(std::move(notificationPipe));
            }
            sync->targets.emplace(subscriberId, std::move(target));
        }
        if (sync->targets.empty()) {
            response = makeResponseLocked(true, "");
            response->hardware = request.hardware;
            logCompletedUpdateLocked(_snapshot.generation, true, false, 0, 0,
                                     0, "");
            releaseUpdateTokenLocked(updateToken);
            return true;
        }
        _agentSync = std::move(sync);
        waitingForAgents = true;
    }

    for (const auto &pipe : notificationPipes) {
        // Keep one consumable signal so collectors need not poll on every Agent report.
        if (navi::EC_NONE != pipe->setDataOverride(nullptr, true)) {
            NAVI_KERNEL_LOG(WARN,
                            "notify Agent scheduler config subscriber failed");
        }
    }
    if (allTargetsCompleted) {
        waitingForAgents = false;
        navi::AsyncPipePtr ignoredResponsePipe;
        return finishAgentSync(false, response, ignoredResponsePipe);
    }
    return true;
}

bool SchedulerConfigR::finishAgentSync(
    bool timedOut, SchedulerConfigResponseDataPtr &response,
    navi::AsyncPipePtr &responsePipe)
{
    autil::ScopedLock lock(_mutex);
    if (!_agentSync) {
        return false;
    }
    const auto allCompleted = std::all_of(
        _agentSync->targets.begin(), _agentSync->targets.end(),
        [](const auto &entry) { return entry.second.completed; });
    if (!allCompleted && !timedOut) {
        return false;
    }
    std::vector<AgentConfigSyncFailure> failures;
    for (auto &[subscriberId, target] : _agentSync->targets) {
        if (!target.completed) {
            target.completed = true;
            target.errorMessage = "scheduler config sync timed out";
        }
        if (!target.errorMessage.empty()) {
            AgentConfigSyncFailure failure;
            failure.ip = target.agentIp;
            failure.error = target.errorMessage;
            failure.effectiveHash = target.effectiveHash;
            failure.effectiveConfig = target.effectiveJson;
            failures.emplace_back(std::move(failure));
        }
    }
    const auto accepted = failures.empty();
    const auto errorMessage = accepted
                                  ? std::string()
                                  : autil::legacy::ToJsonString(failures, true);
    response = makeResponseLocked(accepted, errorMessage);
    response->hardware = _agentSync->hardware;
    response->targetAgentCount = _agentSync->targets.size();
    response->failedAgentCount = failures.size();
    response->updatedAgentCount =
        response->targetAgentCount - response->failedAgentCount;
    response->agentResults.reserve(_agentSync->targets.size());
    for (const auto &[subscriberId, target] : _agentSync->targets) {
        SchedulerConfigResponseData::AgentResult result;
        result.ip = target.agentIp;
        result.hardware = target.hardware;
        result.updated = target.errorMessage.empty();
        result.error = target.errorMessage;
        result.effectiveHash = target.effectiveHash;
        result.effectiveConfigJson = target.effectiveJson;
        response->agentResults.emplace_back(std::move(result));
    }
    logCompletedUpdateLocked(
        _agentSync->requestGeneration, accepted, timedOut,
        response->targetAgentCount, response->updatedAgentCount,
        response->failedAgentCount, response->error);
    responsePipe = _agentSync->responsePipe;
    releaseUpdateTokenLocked(_agentSync->updateToken);
    _agentSync.reset();
    return true;
}

bool SchedulerConfigR::hasActiveAgentSync() const
{
    autil::ScopedLock lock(_mutex);
    return static_cast<bool>(_agentSync);
}

int64_t SchedulerConfigR::getAgentSyncRemainingTimeoutMs(
    int64_t currentTimeUs) const
{
    autil::ScopedLock lock(_mutex);
    if (!_agentSync) {
        return 0;
    }
    return std::max<int64_t>(
        1, (_agentSync->deadlineTimeUs - currentTimeUs + 999) / 1000);
}

bool SchedulerConfigR::subscribe(const std::string &subscriberId,
                                 const navi::AsyncPipePtr &notificationPipe)
{
    if (subscriberId.empty() || !notificationPipe) {
        return false;
    }
    navi::AsyncPipePtr syncPipe;
    {
        autil::ScopedLock lock(_mutex);
        auto &subscriber = _subscribers[subscriberId];
        if (_agentSync) {
            auto targetIt = _agentSync->targets.find(subscriberId);
            if (targetIt != _agentSync->targets.end() &&
                !targetIt->second.completed) {
                targetIt->second.completed = true;
                targetIt->second.errorMessage =
                    "Agent collector replaced during scheduler config sync";
                targetIt->second.effectiveHash = subscriber.configHash;
                targetIt->second.effectiveJson = subscriber.configJson;
                syncPipe = _syncPipe;
            }
        }
        subscriber = Subscriber();
        subscriber.notificationPipe = notificationPipe;
    }
    if (syncPipe) {
        notifySyncPipe(syncPipe);
    }
    return true;
}

bool SchedulerConfigR::activateSubscriber(
    const std::string &subscriberId, const std::string &agentId,
    const std::string &agentIp, uint64_t configGeneration,
    const std::string &configHash, const std::string &configJson,
    const std::string &latencyModelFingerprint)
{
    if (subscriberId.empty() || agentId.empty() || configGeneration == 0 ||
        configHash.empty() || configJson.empty()) {
        return false;
    }
    autil::ScopedLock lock(_mutex);
    auto it = _subscribers.find(subscriberId);
    if (it == _subscribers.end()) {
        return false;
    }
    auto &subscriber = it->second;
    subscriber.active = true;
    subscriber.agentId = agentId;
    subscriber.agentIp = agentIp;
    subscriber.hardware = extractHardware(latencyModelFingerprint);
    subscriber.configGeneration = configGeneration;
    subscriber.configHash = configHash;
    subscriber.configJson = configJson;
    const auto hardwarePatchIt =
        _hardwareAgentConfigPatchJson.find(subscriber.hardware);
    if (_agentConfigPatchJson.empty() &&
        hardwarePatchIt == _hardwareAgentConfigPatchJson.end()) {
        return true;
    }
    AgentSchedulerConfig current;
    AgentSchedulerConfig expected;
    std::string errorMessage;
    if (!AgentSchedulerConfigR::parse(configJson, current, errorMessage) ||
        !applyStoredAgentPatchesLocked(current, subscriber.hardware, nullptr,
                                       nullptr, expected, errorMessage)) {
        NAVI_KERNEL_LOG(ERROR,
                        "prepare scheduler config for new Agent failed, "
                        "agentId [%s], collectorId [%s], error [%s]",
                        agentId.c_str(), subscriberId.c_str(),
                        errorMessage.c_str());
        return false;
    }
    const auto expectedJson = AgentSchedulerConfigR::toJson(expected);
    const auto expectedHash = AgentSchedulerConfigR::hashJson(expectedJson);
    if (expectedHash == configHash) {
        return true;
    }
    subscriber.pendingConfigGeneration = _agentConfigGeneration;
    subscriber.pendingConfigHash = expectedHash;
    subscriber.pendingConfigJson = expectedJson;
    subscriber.pendingConfigSent = false;
    return true;
}

void SchedulerConfigR::reportAgentConfig(
    const std::string &subscriberId, uint64_t requestGeneration,
    uint64_t configGeneration, const std::string &configHash,
    const std::string &configJson, const std::string &errorMessage)
{
    navi::AsyncPipePtr syncPipe;
    {
        autil::ScopedLock lock(_mutex);
        auto it = _subscribers.find(subscriberId);
        if (it == _subscribers.end()) {
            return;
        }
        auto &subscriber = it->second;
        if (configGeneration >= subscriber.configGeneration) {
            subscriber.configGeneration = configGeneration;
            subscriber.configHash = configHash;
            subscriber.configJson = configJson;
        }
        if (subscriber.pendingConfigGeneration == requestGeneration) {
            subscriber.pendingConfigGeneration = 0;
            subscriber.pendingConfigHash.clear();
            subscriber.pendingConfigJson.clear();
            subscriber.pendingConfigSent = false;
        }
        if (!_agentSync ||
            requestGeneration != _agentSync->requestGeneration) {
            if (errorMessage.empty()) {
                NAVI_KERNEL_LOG(
                    INFO,
                    "[navi-scheduler-config-update] role [master], "
                    "status [agent_ack], agentId [%s], agentIp [%s], "
                    "requestGeneration [%lu], effectiveGeneration [%lu], "
                    "effectiveHash [%s], effectiveConfig [%s], error [%s]",
                    subscriber.agentId.c_str(), subscriber.agentIp.c_str(),
                    requestGeneration, configGeneration, configHash.c_str(),
                    configJson.c_str(), errorMessage.c_str());
            } else {
                NAVI_KERNEL_LOG(
                    WARN,
                    "[navi-scheduler-config-update] role [master], "
                    "status [agent_ack], agentId [%s], agentIp [%s], "
                    "requestGeneration [%lu], effectiveGeneration [%lu], "
                    "effectiveHash [%s], effectiveConfig [%s], error [%s]",
                    subscriber.agentId.c_str(), subscriber.agentIp.c_str(),
                    requestGeneration, configGeneration, configHash.c_str(),
                    configJson.c_str(), errorMessage.c_str());
            }
            return;
        }
        auto targetIt = _agentSync->targets.find(subscriberId);
        if (targetIt == _agentSync->targets.end() ||
            targetIt->second.completed) {
            return;
        }
        auto &target = targetIt->second;
        target.effectiveHash = configHash;
        target.effectiveJson = configJson;
        target.errorMessage = errorMessage;
        if (target.errorMessage.empty() && configHash != target.expectedHash) {
            target.errorMessage = "expected hash " + target.expectedHash +
                                  ", got " + configHash;
        }
        target.completed = true;
        if (target.errorMessage.empty()) {
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-scheduler-config-update] role [master], "
                "status [agent_ack], agentId [%s], agentIp [%s], "
                "requestGeneration [%lu], effectiveGeneration [%lu], "
                "expectedHash [%s], effectiveHash [%s], "
                "effectiveConfig [%s], error [%s]",
                subscriber.agentId.c_str(), subscriber.agentIp.c_str(),
                requestGeneration, configGeneration,
                target.expectedHash.c_str(), configHash.c_str(),
                configJson.c_str(), target.errorMessage.c_str());
        } else {
            NAVI_KERNEL_LOG(
                WARN,
                "[navi-scheduler-config-update] role [master], "
                "status [agent_ack], agentId [%s], agentIp [%s], "
                "requestGeneration [%lu], effectiveGeneration [%lu], "
                "expectedHash [%s], effectiveHash [%s], "
                "effectiveConfig [%s], error [%s]",
                subscriber.agentId.c_str(), subscriber.agentIp.c_str(),
                requestGeneration, configGeneration,
                target.expectedHash.c_str(), configHash.c_str(),
                configJson.c_str(), target.errorMessage.c_str());
        }
        syncPipe = _syncPipe;
    }
    if (syncPipe) {
        notifySyncPipe(syncPipe);
    }
}

bool SchedulerConfigR::getPendingAgentUpdate(
    const std::string &subscriberId, PendingAgentUpdate &update)
{
    autil::ScopedLock lock(_mutex);
    if (_agentSync) {
        auto it = _agentSync->targets.find(subscriberId);
        if (it != _agentSync->targets.end()) {
            if (it->second.completed || it->second.sent) {
                return false;
            }
            it->second.sent = true;
            update.requestGeneration = _agentSync->requestGeneration;
            update.configHash = it->second.expectedHash;
            update.configJson = it->second.expectedJson;
            return true;
        }
    }
    auto subscriberIt = _subscribers.find(subscriberId);
    if (subscriberIt == _subscribers.end()) {
        return false;
    }
    auto &subscriber = subscriberIt->second;
    if (subscriber.pendingConfigGeneration == 0 ||
        subscriber.pendingConfigSent) {
        return false;
    }
    subscriber.pendingConfigSent = true;
    update.requestGeneration = subscriber.pendingConfigGeneration;
    update.configHash = subscriber.pendingConfigHash;
    update.configJson = subscriber.pendingConfigJson;
    return true;
}

void SchedulerConfigR::unsubscribe(const std::string &subscriberId,
                                   const navi::AsyncPipePtr &notificationPipe)
{
    navi::AsyncPipePtr syncPipe;
    {
        autil::ScopedLock lock(_mutex);
        auto it = _subscribers.find(subscriberId);
        if (it == _subscribers.end()) {
            return;
        }
        auto current = it->second.notificationPipe.lock();
        if (notificationPipe && current && current != notificationPipe) {
            return;
        }
        if (_agentSync) {
            auto targetIt = _agentSync->targets.find(subscriberId);
            if (targetIt != _agentSync->targets.end() &&
                !targetIt->second.completed) {
                targetIt->second.completed = true;
                targetIt->second.errorMessage =
                    "Agent disconnected during scheduler config sync";
                targetIt->second.effectiveHash = it->second.configHash;
                targetIt->second.effectiveJson = it->second.configJson;
                syncPipe = _syncPipe;
            }
        }
        _subscribers.erase(it);
    }
    if (syncPipe) {
        notifySyncPipe(syncPipe);
    }
}

uint64_t SchedulerConfigR::acquireUpdateTokenLocked()
{
    auto updateToken = _nextUpdateToken++;
    if (updateToken == 0) {
        updateToken = _nextUpdateToken++;
    }
    _activeUpdateToken = updateToken;
    return updateToken;
}

void SchedulerConfigR::releaseUpdateTokenLocked(uint64_t updateToken)
{
    if (updateToken != 0 && _activeUpdateToken == updateToken) {
        _activeUpdateToken = 0;
    }
}

void SchedulerConfigR::logRejectedUpdateLocked(
    const std::string &configJson, const std::string &errorMessage) const
{
    NAVI_KERNEL_LOG(
        WARN,
        "[navi-scheduler-config-update] role [master], status [rejected], "
        "generation [%lu], activeUpdateToken [%lu], patch [%s], error [%s]",
        _snapshot.generation, _activeUpdateToken, configJson.c_str(),
        errorMessage.c_str());
}

void SchedulerConfigR::logCompletedUpdateLocked(
    uint64_t generation, bool accepted, bool timedOut, size_t targetAgentCount,
    size_t updatedAgentCount, size_t failedAgentCount,
    const std::string &errorMessage) const
{
    if (accepted) {
        NAVI_KERNEL_LOG(
            INFO,
            "[navi-scheduler-config-update] role [master], status [completed], "
            "generation [%lu], accepted [1], timedOut [%d], "
            "targetAgentCount [%lu], updatedAgentCount [%lu], "
            "failedAgentCount [%lu], error [%s]",
            generation, static_cast<int>(timedOut), targetAgentCount,
            updatedAgentCount, failedAgentCount, errorMessage.c_str());
    } else {
        NAVI_KERNEL_LOG(
            WARN,
            "[navi-scheduler-config-update] role [master], status [completed], "
            "generation [%lu], accepted [0], timedOut [%d], "
            "targetAgentCount [%lu], updatedAgentCount [%lu], "
            "failedAgentCount [%lu], error [%s]",
            generation, static_cast<int>(timedOut), targetAgentCount,
            updatedAgentCount, failedAgentCount, errorMessage.c_str());
    }
}

SchedulerConfigResponseDataPtr SchedulerConfigR::makeResponseLocked(
    bool accepted, const std::string &errorMessage) const
{
    auto response = std::make_shared<SchedulerConfigResponseData>();
    response->configJson = toJson(_snapshot.config);
    response->generation = _snapshot.generation;
    response->accepted = accepted;
    response->error = errorMessage;
    return response;
}

bool SchedulerConfigR::notifySyncPipe(
    const navi::AsyncPipePtr &syncPipe) const
{
    return syncPipe &&
           navi::EC_NONE == syncPipe->setDataOverride(nullptr, true, true);
}

bool SchedulerConfigR::applyPatch(const Config &current,
                                  const std::string &configJson,
                                  Config &updated,
                                  std::string &errorMessage)
{
    static const std::set<std::string> supportedFields = {
        "lambda",
        "alpha",
        "alpha_decay",
        "min_alpha",
        "max_loop_count",
        "min_loop_count",
        "optimization_time_budget_us",
        "parallelism",
        "latency_weight_concavity",
        "latency_token_normalization_power",
        "initial_weight_seed",
        "discrete_check_interval",
        "discrete_gap_threshold",
        "discrete_repair_alpha_decay",
        "max_wait_schedule_count",
        "max_wait_schedule_time_ms",
        "queue_overload_start_cycle_multiplier",
        "queue_overload_full_cycle_multiplier",
        "admission_timeout_ms",
        "batch_timeout_ms",
        "closed_batch_keep_ms",
        "latency_fit_batch_size",
        "latency_fit_interval",
        "latency_learning_rate",
        "latency_online_learning_rate_scale",
        "latency_current_sample_weight",
        "latency_parameter_learning_rate_scales",
        "latency_nonlinear_scale",
        "queue_learning_rate",
        "latency_check_gradient",
        "latency_cold_start_fit_epochs",
        "latency_history_limit",
        "latency_history_bucket_capacity",
        "latency_persist_interval",
        "latency_persist_interval_ms",
        "latency_warmup_sample_count",
        "latency_model_trust_min_revision",
        "latency_zero_anchor_max_ms",
        "latency_zero_anchor_min_samples",
        "latency_validation_enabled",
        "residual_learning_rate",
        "residual_update_error_floor_ms",
        "residual_update_error_ratio",
        "residual_idle_decay_start_ms",
        "residual_idle_decay_end_ms",
        "time_compression",
    };
    try {
        auto patch = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
            autil::legacy::json::ParseJson(configJson));
        for (const auto &[name, value] : patch) {
            if (supportedFields.find(name) == supportedFields.end()) {
                errorMessage = "unsupported scheduler config field: " + name;
                return false;
            }
        }
        auto merged = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
            autil::legacy::json::ParseJson(toJson(current)));
        for (auto &[name, value] : patch) {
            merged[name] = std::move(value);
        }
        updated = current;
        autil::legacy::FromJsonString(
            updated, autil::legacy::ToJsonString(merged, true));
        if (patch.find("max_wait_schedule_time_ms") != patch.end() &&
            patch.find("optimization_time_budget_us") == patch.end()) {
            updated.cost.optimizationTimeBudgetUs =
                updated.maxWaitScheduleTimeMs * 800;
        }
    } catch (const std::exception &error) {
        errorMessage = error.what();
        return false;
    } catch (...) {
        errorMessage = "scheduler config must be a JSON object";
        return false;
    }
    if (!updated.valid()) {
        errorMessage = "invalid scheduler config values";
        return false;
    }
    return true;
}

bool SchedulerConfigR::applyAgentPatch(
    const AgentSchedulerConfig &current, const std::string &configJson,
    AgentSchedulerConfig &updated, bool &hasAgentFields,
    std::string &errorMessage)
{
    hasAgentFields = false;
    try {
        const auto patch =
            autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                autil::legacy::json::ParseJson(configJson));
        auto merged =
            autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                autil::legacy::json::ParseJson(
                    AgentSchedulerConfigR::toJson(current)));
        for (const auto &[name, value] : patch) {
            if (agentConfigFields().find(name) == agentConfigFields().end()) {
                continue;
            }
            merged[name] = value;
            hasAgentFields = true;
        }
        updated = current;
        autil::legacy::FromJsonString(
            updated, autil::legacy::ToJsonString(merged, true));
    } catch (const std::exception &error) {
        errorMessage = error.what();
        return false;
    } catch (...) {
        errorMessage = "scheduler config must be a JSON object";
        return false;
    }
    if (!updated.valid()) {
        errorMessage = "invalid Agent scheduler config values";
        return false;
    }
    return true;
}

bool SchedulerConfigR::mergeAgentPatch(
    const std::string &currentPatchJson, const std::string &configJson,
    std::string &updatedPatchJson, bool &hasAgentFields,
    std::string &errorMessage)
{
    hasAgentFields = false;
    try {
        autil::legacy::json::JsonMap merged;
        if (!currentPatchJson.empty()) {
            merged = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                autil::legacy::json::ParseJson(currentPatchJson));
        }
        const auto patch =
            autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                autil::legacy::json::ParseJson(configJson));
        for (const auto &[name, value] : patch) {
            if (agentConfigFields().find(name) == agentConfigFields().end()) {
                continue;
            }
            merged[name] = value;
            hasAgentFields = true;
        }
        updatedPatchJson = autil::legacy::ToJsonString(merged, true);
    } catch (const std::exception &error) {
        errorMessage = error.what();
        return false;
    } catch (...) {
        errorMessage = "scheduler config must be a JSON object";
        return false;
    }
    return true;
}

std::string SchedulerConfigR::extractHardware(
    const std::string &latencyModelFingerprint)
{
    if (latencyModelFingerprint.empty()) {
        return {};
    }
    try {
        LatencyModelFingerprint fingerprint;
        autil::legacy::FromJsonString(fingerprint, latencyModelFingerprint);
        if (fingerprint.hardware.size() <= 256) {
            return fingerprint.hardware;
        }
    } catch (...) {
    }
    return {};
}

bool SchedulerConfigR::applyStoredAgentPatchesLocked(
    const AgentSchedulerConfig &current, const std::string &hardware,
    const std::string *globalPatchOverride,
    const std::string *hardwarePatchOverride, AgentSchedulerConfig &updated,
    std::string &errorMessage) const
{
    updated = current;
    const auto &globalPatch =
        globalPatchOverride ? *globalPatchOverride : _agentConfigPatchJson;
    bool hasAgentFields = false;
    if (!globalPatch.empty() &&
        !applyAgentPatch(updated, globalPatch, updated, hasAgentFields,
                         errorMessage)) {
        return false;
    }

    const std::string *hardwarePatch = hardwarePatchOverride;
    if (!hardwarePatch) {
        const auto it = _hardwareAgentConfigPatchJson.find(hardware);
        if (it != _hardwareAgentConfigPatchJson.end()) {
            hardwarePatch = &it->second;
        }
    }
    return !hardwarePatch || hardwarePatch->empty() ||
           applyAgentPatch(updated, *hardwarePatch, updated, hasAgentFields,
                           errorMessage);
}

std::string SchedulerConfigR::toJson(const Config &config)
{
    return autil::legacy::ToJsonString(config, true);
}

std::string SchedulerConfigR::toAgentJson(const Config &config)
{
    return AgentSchedulerConfigR::toJson(config.agent);
}

REGISTER_RESOURCE(SchedulerConfigR);

} // namespace navi_sched
