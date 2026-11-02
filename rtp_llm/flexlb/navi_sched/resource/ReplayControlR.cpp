#include "aios/navi_sched/resource/ReplayControlR.h"

#include "aios/navi_sched/kernel/ReplayControlSyncKernel.h"
#include "aios/navi_sched/proto/ReplayControl.pb.h"
#include "autil/TimeUtility.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/rpc_server/ArpcRegistryParam.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace navi_sched {
namespace {

const std::string RESPONSE_TYPE_ID = "NaviSchedReplayControlResponseType";
constexpr int64_t REALTIME_START_LEAD_US = 1'000'000;

bool parseMode(const std::string &value, ReplayMode &mode)
{
    if (value == "fixed_qps") {
        mode = ReplayMode::FIXED_QPS;
        return true;
    }
    if (value == "realtime") {
        mode = ReplayMode::REALTIME;
        return true;
    }
    return false;
}

bool parseHistoryMode(const std::string &value, ReplayHistoryMode &mode)
{
    if (value.empty() || value == "time_range") {
        mode = ReplayHistoryMode::TIME_RANGE;
        return true;
    }
    if (value == "recent_count") {
        mode = ReplayHistoryMode::RECENT_COUNT;
        return true;
    }
    return false;
}

}

ReplayControlResponseData::ReplayControlResponseData()
    : navi::Data(RESPONSE_TYPE_ID)
{
}

const std::string ReplayControlR::RESOURCE_ID = "navi_sched.replay_control_r";
const std::string ReplayControlR::RPC_REQUEST_NAME =
    "navi_sched.replay_control_request";

void ReplayControlR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ)
        .dependOn(navi::NaviRpcServerR::RESOURCE_ID, true,
                  BIND_RESOURCE_TO(_naviRpcServerR));
}

bool ReplayControlR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "config_json", _initialConfigJson, _initialConfigJson);
    return true;
}

navi::ErrorCode ReplayControlR::init(navi::ResourceInitContext &ctx)
{
    _bizName = ctx.getBizName();
    navi::ArpcRegistryParam param;
    param.arpcParam.method = "replay_control";
    param.arpcParam.httpAliasVec.emplace_back("/replay-control");
    param.arpcParam.logQuery = true;
    param.graphParam.requestDataName = RPC_REQUEST_NAME;
    if (!buildGraph(param.graphParam.graph)) {
        NAVI_KERNEL_LOG(ERROR, "build replay control rpc graph failed");
        return navi::EC_ABORT;
    }
    auto service = _naviRpcServerR->registerArpcService<ReplayControlService>(
        param, this);
    if (!service) {
        NAVI_KERNEL_LOG(ERROR, "register replay control rpc service failed");
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

bool ReplayControlR::buildGraph(navi::GraphDef &graphDef) const
{
    navi::GraphBuilder builder(&graphDef);
    auto graphId = builder.newSubGraph(_bizName);
    builder.subGraph(graphId).location(_bizName, 1, 0);
    builder.node("replay_control_sync")
        .kernel(ReplayControlSyncKernel::KERNEL_ID)
        .out(ReplayControlSyncKernel::OUTPUT_PORT)
        .asGraphOutput("out");
    navi::GraphBuilder::fillHashKey(&graphDef);
    return builder.ok();
}

bool ReplayControlR::validateStart(const Request &request, ReplayMode &mode,
                                   ReplayHistoryMode &historyMode,
                                   int64_t &endTimeUs, std::string &error) const
{
    if (!parseMode(request.mode, mode)) {
        error = "mode must be fixed_qps or realtime";
        return false;
    }
    if (!parseHistoryMode(request.historyMode, historyMode)) {
        error = "history_mode must be time_range or recent_count";
        return false;
    }
    if (request.durationMs > std::numeric_limits<int64_t>::max() / 1000) {
        error = "duration_ms is too large";
        return false;
    }
    if (historyMode == ReplayHistoryMode::TIME_RANGE) {
        if (request.recentQueryCount != 0) {
            error = "recent_query_count is only valid in recent_count mode";
            return false;
        }
        if (request.startTimeUs <= 0) {
            error = "start_time_us must be positive";
            return false;
        }
        if (request.endTimeUs != 0) {
            if (request.endTimeUs < 0) {
                error = "end_time_us must be positive";
                return false;
            }
            endTimeUs = request.endTimeUs;
        } else if (request.durationMs > 0) {
            const auto durationUs = request.durationMs * 1000;
            if (request.startTimeUs >
                std::numeric_limits<int64_t>::max() - durationUs) {
                error = "start_time_us plus duration_ms overflows";
                return false;
            }
            endTimeUs = request.startTimeUs + durationUs;
        } else {
            error = "end_time_us or positive duration_ms is required";
            return false;
        }
        if (endTimeUs <= request.startTimeUs) {
            error = "end_time_us must be greater than start_time_us";
            return false;
        }
    } else {
        if (request.recentQueryCount == 0) {
            error = "recent_query_count must be positive in recent_count mode";
            return false;
        }
        if (request.startTimeUs != 0 || request.endTimeUs != 0) {
            error = "start_time_us and end_time_us are invalid in recent_count mode";
            return false;
        }
    }
    if (request.loopCount == 0 && request.durationMs <= 0) {
        error = "loop_count or positive duration_ms is required";
        return false;
    }
    if (mode == ReplayMode::FIXED_QPS &&
        (!std::isfinite(request.totalQps) || request.totalQps <= 0.0)) {
        error = "total_qps must be positive in fixed_qps mode";
        return false;
    }
    if (mode == ReplayMode::REALTIME && request.totalQps != 0.0) {
        error = "total_qps must be zero in realtime mode";
        return false;
    }
    return true;
}

bool ReplayControlR::isSelectionCandidate(
    const ReplayHistorySummary &summary, const CommandState &command) const
{
    if (!summary.replayable || summary.sampleCount == 0) {
        return false;
    }
    if (command.historyMode == ReplayHistoryMode::RECENT_COUNT) {
        return summary.sampleCount >= command.recentQueryCount;
    }
    return summary.earliestEventTimeUs <= command.endTimeUs &&
           summary.latestEventTimeUs >= command.startTimeUs;
}

bool ReplayControlR::isEligible(const ReplayHistorySummary &summary,
                                const CommandState &command) const
{
    if (!isSelectionCandidate(summary, command)) {
        return false;
    }
    if (command.selectionGeneration != 0 &&
        summary.selectionGeneration == command.selectionGeneration) {
        return summary.selectedSampleCount > 0;
    }
    return true;
}

ReplayControlCommandDataPtr ReplayControlR::makeCommandLocked(
    const ReplayHistorySummary &summary) const
{
    size_t eligible = 0;
    size_t pacingIndex = std::numeric_limits<size_t>::max();
    if (_command.active && isEligible(summary, _command)) {
        for (const auto &[clientId, subscriber] : _subscribers) {
            if (isEligible(subscriber.summary, _command)) {
                if (clientId == summary.clientId) {
                    pacingIndex = eligible;
                }
                ++eligible;
            }
        }
    }
    return makeCommandLocked(summary, pacingIndex, eligible);
}

ReplayControlCommandDataPtr ReplayControlR::makeCommandLocked(
    const ReplayHistorySummary &summary, size_t pacingIndex,
    size_t eligibleCount) const
{
    auto command = std::make_shared<ReplayControlCommandData>();
    command->generation = _command.generation;
    if (!_command.active || !isEligible(summary, _command)) {
        command->action = ReplayAction::STOP;
        if (_command.active &&
            summary.selectionGeneration == _command.selectionGeneration) {
            command->selectionGeneration = _command.selectionGeneration;
        }
        return command;
    }
    command->action = ReplayAction::START;
    command->selectionGeneration = _command.selectionGeneration;
    command->mode = _command.mode;
    command->historyMode = _command.historyMode;
    command->startTimeUs = _command.startTimeUs;
    command->endTimeUs = _command.endTimeUs;
    command->durationMs = remainingDurationMsLocked();
    command->totalQps = _command.totalQps;
    command->loopCount = _command.loopCount;
    command->recentQueryCount = _command.recentQueryCount;
    command->pacingEpochUs = _command.pacingEpochUs;
    const auto subscriberIt = _subscribers.find(summary.clientId);
    if (subscriberIt != _subscribers.end() &&
        subscriberIt->second.hasClockOffset) {
        command->pacingClockOffsetUs = subscriberIt->second.clockOffsetUs;
    }
    if (pacingIndex < eligibleCount) {
        command->pacingIndex = pacingIndex;
    }
    command->perClientQps = _command.mode == ReplayMode::FIXED_QPS && eligibleCount > 0
                                ? _command.totalQps / eligibleCount
                                : 0.0;
    if (_command.mode == ReplayMode::FIXED_QPS && pacingIndex < eligibleCount) {
        command->pacingClientCount = eligibleCount;
    }
    return command;
}

void ReplayControlR::notifySubscribersLocked()
{
    bool removedEligibleClient = false;
    do {
        removedEligibleClient = false;
        const auto eligibleCount = _command.active
            ? std::count_if(_subscribers.begin(), _subscribers.end(),
                            [&](const auto &entry) {
                                return isEligible(entry.second.summary, _command);
                            })
            : 0;
        size_t pacingIndex = 0;
        for (auto it = _subscribers.begin(); it != _subscribers.end();) {
            const bool eligible = _command.active &&
                                  isEligible(it->second.summary, _command);
            const auto clientPacingIndex = pacingIndex;
            pacingIndex += eligible;
            auto pipe = it->second.commandPipe.lock();
            if (!pipe) {
                if (_pendingHistoryQuery &&
                    _pendingHistoryQuery->responses.count(it->first) == 0) {
                    _pendingHistoryQuery->failures[it->first] =
                        "client disconnected during history query";
                }
                removedEligibleClient = removedEligibleClient || eligible;
                it = _subscribers.erase(it);
                continue;
            }
            auto command = makeCommandLocked(it->second.summary,
                                              clientPacingIndex, eligibleCount);
            if (command->action == ReplayAction::START) {
                NAVI_KERNEL_LOG(INFO,
                                "send replay pacing command, client [%s], "
                                "generation [%lu], mode [%d], total qps [%f], "
                                "pacing index [%lu], client count [%lu], master "
                                "epoch [%ld], clock offset [%ld]",
                                it->first.c_str(), command->generation,
                                static_cast<int>(command->mode), command->totalQps,
                                command->pacingIndex,
                                command->pacingClientCount,
                                command->pacingEpochUs,
                                command->pacingClockOffsetUs);
            }
            if (navi::EC_NONE != pipe->setData(command)) {
                if (_pendingHistoryQuery &&
                    _pendingHistoryQuery->responses.count(it->first) == 0) {
                    _pendingHistoryQuery->failures[it->first] =
                        "client command delivery failed during history query";
                }
                removedEligibleClient = removedEligibleClient || eligible;
                it = _subscribers.erase(it);
                continue;
            }
            ++it;
        }
        if (removedEligibleClient) {
            resetPacingEpochLocked();
            ++_command.generation;
        }
    } while (removedEligibleClient);
}

void ReplayControlR::notifyClockOffsetLocked(const Subscriber &subscriber)
{
    if (auto pipe = subscriber.commandPipe.lock()) {
        if (navi::EC_NONE == pipe->setData(makeCommandLocked(subscriber.summary))) {
            return;
        }
    }
    notifySubscribersLocked();
}

void ReplayControlR::resetPacingEpochLocked()
{
    if (_command.active && _command.mode == ReplayMode::FIXED_QPS) {
        _command.pacingEpochUs =
            autil::TimeUtility::currentTimeInMicroSeconds();
    }
}

void ReplayControlR::rebasePacingEpochLocked(double newTotalQps)
{
    const auto nowUs = autil::TimeUtility::currentTimeInMicroSeconds();
    if (!_command.active || _command.mode != ReplayMode::FIXED_QPS ||
        _command.pacingEpochUs <= 0 || !std::isfinite(_command.totalQps) ||
        _command.totalQps <= 0.0 || !std::isfinite(newTotalQps) ||
        newTotalQps <= 0.0) {
        _command.pacingEpochUs = nowUs;
        return;
    }

    const auto oldEpochUs = _command.pacingEpochUs;
    const auto elapsedUs = static_cast<long double>(nowUs) -
                           static_cast<long double>(oldEpochUs);
    const auto rebasedEpochUs = static_cast<long double>(nowUs) -
                                elapsedUs * _command.totalQps / newTotalQps;
    if (!std::isfinite(rebasedEpochUs) || rebasedEpochUs < 1.0L ||
        rebasedEpochUs >
            static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        NAVI_KERNEL_LOG(WARN,
                        "cannot preserve fixed replay phase while changing total "
                        "qps from [%f] to [%f], reset epoch from [%ld] to [%ld]",
                        _command.totalQps, newTotalQps, oldEpochUs, nowUs);
        _command.pacingEpochUs = nowUs;
        return;
    }
    _command.pacingEpochUs = static_cast<int64_t>(std::llround(rebasedEpochUs));
    NAVI_KERNEL_LOG(INFO,
                    "rebased fixed replay pacing epoch, old qps [%f], new qps "
                    "[%f], old epoch [%ld], new epoch [%ld], update time [%ld]",
                    _command.totalQps, newTotalQps, oldEpochUs,
                    _command.pacingEpochUs, nowUs);
}

bool ReplayControlR::updateClockOffsetLocked(Subscriber &subscriber,
                                             int64_t clientReportTimeUs,
                                             int64_t masterReceiveTimeUs)
{
    if (clientReportTimeUs <= 0 || masterReceiveTimeUs <= 0) {
        return false;
    }
    const int64_t observedOffsetUs = clientReportTimeUs >= masterReceiveTimeUs
                                         ? clientReportTimeUs - masterReceiveTimeUs
                                         : -(masterReceiveTimeUs - clientReportTimeUs);
    const bool changed = !subscriber.hasClockOffset ||
                         observedOffsetUs > subscriber.clockOffsetUs;
    if (changed) {
        subscriber.clockOffsetUs = observedOffsetUs;
        subscriber.hasClockOffset = true;
    }
    NAVI_KERNEL_LOG(INFO,
                    "observed replay client clock, client [%s], client report "
                    "time [%ld], master receive time [%ld], observed offset "
                    "[%ld], selected offset [%ld], changed [%d]",
                    subscriber.summary.clientId.c_str(), clientReportTimeUs,
                    masterReceiveTimeUs, observedOffsetUs,
                    subscriber.clockOffsetUs, changed);
    return changed;
}

void ReplayControlR::initializeReplayEpochLocked()
{
    if (!_command.active) {
        _command.pacingEpochUs = 0;
        return;
    }
    const auto nowUs = autil::TimeUtility::currentTimeInMicroSeconds();
    _command.pacingEpochUs =
        _command.mode == ReplayMode::REALTIME &&
                nowUs <= std::numeric_limits<int64_t>::max() -
                             REALTIME_START_LEAD_US
            ? nowUs + REALTIME_START_LEAD_US
            : nowUs;
}

void ReplayControlR::clearSelectionStateLocked()
{
    for (auto &entry : _subscribers) {
        auto &subscriber = entry.second;
        subscriber.summary.selectionGeneration = 0;
        subscriber.summary.selectedSampleCount = 0;
        subscriber.summary.completedSelectionGeneration = 0;
    }
}

int64_t ReplayControlR::remainingDurationMsLocked() const
{
    if (_command.durationMs <= 0 || _command.deadlineTimeUs <= 0) {
        return _command.durationMs;
    }
    const auto remainingUs =
        _command.deadlineTimeUs - autil::TimeUtility::monotonicTimeUs();
    if (remainingUs <= 0) {
        return 0;
    }
    const auto remainingMs =
        remainingUs / 1000 + (remainingUs % 1000 == 0 ? 0 : 1);
    return std::min(_command.durationMs, remainingMs);
}

bool ReplayControlR::expireCommandIfNeededLocked()
{
    if (!_command.active || _command.deadlineTimeUs <= 0 ||
        autil::TimeUtility::monotonicTimeUs() < _command.deadlineTimeUs) {
        return false;
    }
    _command.active = false;
    _command.deadlineTimeUs = 0;
    clearSelectionStateLocked();
    ++_command.generation;
    notifySubscribersLocked();
    NAVI_KERNEL_LOG(INFO, "replay command expired at generation [%lu]",
                    _command.generation);
    return true;
}

bool ReplayControlR::finishCommandIfAllClientsCompletedLocked()
{
    if (!_command.active || _command.selectionGeneration == 0) {
        return false;
    }
    bool hasSelectionCandidate = false;
    for (const auto &[clientId, subscriber] : _subscribers) {
        const auto &summary = subscriber.summary;
        if (!isSelectionCandidate(summary, _command)) {
            continue;
        }
        hasSelectionCandidate = true;
        if (summary.selectionGeneration != _command.selectionGeneration) {
            return false;
        }
        if (summary.selectedSampleCount > 0 &&
            summary.completedSelectionGeneration !=
                _command.selectionGeneration) {
            return false;
        }
    }
    if (!hasSelectionCandidate) {
        return false;
    }
    _command.active = false;
    _command.deadlineTimeUs = 0;
    clearSelectionStateLocked();
    ++_command.generation;
    notifySubscribersLocked();
    NAVI_KERNEL_LOG(INFO,
                    "replay selection [%lu] completed at generation [%lu]",
                    _command.selectionGeneration, _command.generation);
    return true;
}

ReplayControlResponseDataPtr ReplayControlR::makeResponseLocked(
    bool accepted, const std::string &error) const
{
    auto response = std::make_shared<ReplayControlResponseData>();
    response->accepted = accepted;
    response->error = error;
    response->generation = _command.generation;
    response->action = _command.active ? ReplayAction::START : ReplayAction::STOP;
    response->mode = _command.mode;
    response->historyMode = _command.historyMode;
    response->recentQueryCount = _command.recentQueryCount;
    for (const auto &[clientId, subscriber] : _subscribers) {
        response->clients.emplace_back(subscriber.summary);
        if (_command.active && isEligible(subscriber.summary, _command)) {
            ++response->eligibleClientCount;
        }
    }
    response->perClientQps = _command.mode == ReplayMode::FIXED_QPS &&
                                     response->eligibleClientCount > 0
                                 ? _command.totalQps / response->eligibleClientCount
                                 : 0.0;
    return response;
}

bool ReplayControlR::processRequest(const Request &request,
                                    ReplayControlResponseDataPtr &response)
{
    std::string error;
    {
        autil::ScopedLock lock(_mutex);
        if (_stopped) {
            response = makeResponseLocked(false, "replay control is stopped");
        } else if (request.action.empty() || request.action == "query") {
            expireCommandIfNeededLocked();
            response = makeResponseLocked(true, "");
        } else if (request.action == "stop") {
            _command.active = false;
            _command.deadlineTimeUs = 0;
            clearSelectionStateLocked();
            ++_command.generation;
            notifySubscribersLocked();
            response = makeResponseLocked(true, "");
        } else if (request.action == "update" && request.mode.empty()) {
            expireCommandIfNeededLocked();
            if (!_command.active) {
                response = makeResponseLocked(false, "no active replay to update");
            } else if (_command.mode != ReplayMode::FIXED_QPS) {
                response = makeResponseLocked(
                    false, "qps-only update requires an active fixed_qps replay");
            } else if (!request.historyMode.empty() || request.startTimeUs != 0 ||
                       request.endTimeUs != 0 || request.durationMs != 0 ||
                       request.loopCount != 0 || request.recentQueryCount != 0) {
                response = makeResponseLocked(
                    false, "qps-only update cannot change replay selection or limits");
            } else if (!std::isfinite(request.totalQps) ||
                       request.totalQps <= 0.0) {
                response = makeResponseLocked(
                    false, "total_qps must be positive in fixed_qps mode");
            } else {
                rebasePacingEpochLocked(request.totalQps);
                _command.totalQps = request.totalQps;
                ++_command.generation;
                notifySubscribersLocked();
                response = makeResponseLocked(true, "");
            }
        } else if (request.action == "start" || request.action == "update") {
            ReplayMode mode;
            ReplayHistoryMode historyMode;
            int64_t endTimeUs = 0;
            if (!validateStart(request, mode, historyMode, endTimeUs, error)) {
                response = makeResponseLocked(false, error);
            } else {
                CommandState candidate;
                candidate.active = true;
                candidate.mode = mode;
                candidate.historyMode = historyMode;
                candidate.startTimeUs = request.startTimeUs;
                candidate.endTimeUs = endTimeUs;
                candidate.recentQueryCount = request.recentQueryCount;
                const bool hasSelectionCandidate = std::any_of(
                    _subscribers.begin(), _subscribers.end(), [&](const auto &entry) {
                        return isSelectionCandidate(entry.second.summary, candidate);
                    });
                if (!hasSelectionCandidate) {
                    response = makeResponseLocked(
                        false, "no replayable samples in requested selection");
                    return static_cast<bool>(response);
                }
                _command.active = true;
                _command.mode = mode;
                _command.historyMode = historyMode;
                _command.startTimeUs = request.startTimeUs;
                _command.endTimeUs = endTimeUs;
                _command.durationMs = request.durationMs;
                _command.totalQps = request.totalQps;
                _command.loopCount = request.loopCount;
                _command.recentQueryCount = request.recentQueryCount;
                initializeReplayEpochLocked();
                const auto nowUs = autil::TimeUtility::monotonicTimeUs();
                const auto durationUs = request.durationMs * 1000;
                const auto startLeadUs =
                    mode == ReplayMode::REALTIME ? REALTIME_START_LEAD_US : 0;
                const auto replayWindowUs =
                    durationUs <= std::numeric_limits<int64_t>::max() - startLeadUs
                        ? durationUs + startLeadUs
                        : std::numeric_limits<int64_t>::max();
                _command.deadlineTimeUs =
                    request.durationMs > 0 &&
                            replayWindowUs <=
                                std::numeric_limits<int64_t>::max() - nowUs
                        ? nowUs + replayWindowUs
                        : request.durationMs > 0
                              ? std::numeric_limits<int64_t>::max()
                              : 0;
                if (++_command.selectionGeneration == 0) {
                    ++_command.selectionGeneration;
                }
                clearSelectionStateLocked();
                ++_command.generation;
                notifySubscribersLocked();
                response = makeResponseLocked(true, "");
            }
        } else {
            response = makeResponseLocked(false, "action must be query, start, update, or stop");
        }
    }
    return static_cast<bool>(response);
}

bool ReplayControlR::processRequestAsync(
    const Request &request, const navi::AsyncPipePtr &responsePipe,
    uint64_t &historyQueryId,
    ReplayControlResponseDataPtr &immediateResponse)
{
    historyQueryId = 0;
    immediateResponse.reset();
    if (request.action != "query" && !request.action.empty()) {
        return processRequest(request, immediateResponse);
    }
    if (!responsePipe) {
        return false;
    }
    autil::ScopedLock lock(_mutex);
    if (_stopped) {
        immediateResponse = makeResponseLocked(false, "replay control is stopped");
        return true;
    }
    expireCommandIfNeededLocked();
    if (_pendingHistoryQuery) {
        immediateResponse = makeResponseLocked(
            false, "another replay history query is already running");
        return true;
    }
    auto pending = std::make_shared<PendingHistoryQuery>();
    pending->queryId = ++_nextHistoryQueryId;
    if (pending->queryId == 0) {
        pending->queryId = ++_nextHistoryQueryId;
    }
    pending->responsePipe = responsePipe;
    auto query = std::make_shared<ReplayHistoryQueryData>();
    query->queryId = pending->queryId;
    for (auto it = _subscribers.begin(); it != _subscribers.end();) {
        auto commandPipe = it->second.commandPipe.lock();
        auto queryPipe = it->second.historyQueryPipe.lock();
        if (!commandPipe || !queryPipe) {
            it = _subscribers.erase(it);
            continue;
        }
        pending->expectedClients.push_back(it->first);
        if (navi::EC_NONE != queryPipe->setData(query)) {
            pending->failures[it->first] = "history query delivery failed";
        }
        ++it;
    }
    historyQueryId = pending->queryId;
    _pendingHistoryQuery = pending;
    NAVI_KERNEL_LOG(INFO,
                    "started replay history query [%lu], expected client count [%lu]",
                    historyQueryId, pending->expectedClients.size());
    if (historyQueryCompleteLocked()) {
        immediateResponse = finishHistoryQueryLocked("");
    }
    return true;
}

bool ReplayControlR::historyQueryCompleteLocked() const
{
    return _pendingHistoryQuery &&
           _pendingHistoryQuery->responses.size() +
                   _pendingHistoryQuery->failures.size() >=
               _pendingHistoryQuery->expectedClients.size();
}

ReplayControlResponseDataPtr ReplayControlR::finishHistoryQueryLocked(
    const std::string &timeoutError)
{
    if (!_pendingHistoryQuery) {
        return nullptr;
    }
    auto pending = _pendingHistoryQuery;
    auto response = makeResponseLocked(false, "");
    response->historyQueryId = pending->queryId;
    response->expectedClientCount = pending->expectedClients.size();
    response->respondedClientCount = pending->responses.size();
    response->clients.clear();
    size_t timeoutCount = 0;
    for (const auto &clientId : pending->expectedClients) {
        const auto responseIt = pending->responses.find(clientId);
        if (responseIt != pending->responses.end()) {
            auto summary = responseIt->second;
            summary.status = "ok";
            response->clients.emplace_back(std::move(summary));
            continue;
        }
        ReplayHistorySummary summary;
        summary.clientId = clientId;
        const auto failureIt = pending->failures.find(clientId);
        if (failureIt != pending->failures.end()) {
            summary.status = "offline";
            summary.error = failureIt->second;
        } else {
            summary.status = "timeout";
            summary.error = timeoutError.empty() ? "history query timed out"
                                                 : timeoutError;
            ++timeoutCount;
        }
        response->clients.emplace_back(std::move(summary));
    }
    response->accepted = pending->responses.size() ==
                         pending->expectedClients.size();
    if (!response->accepted) {
        response->error = timeoutCount > 0
                              ? "replay history query timed out"
                              : "replay history query failed for one or more clients";
    }
    _pendingHistoryQuery.reset();
    return response;
}

void ReplayControlR::publishAsyncResponse(
    const navi::AsyncPipePtr &pipe,
    const ReplayControlResponseDataPtr &response)
{
    if (pipe && response) {
        pipe->setData(response);
    }
}

void ReplayControlR::timeoutHistoryQuery(uint64_t historyQueryId)
{
    navi::AsyncPipePtr pipe;
    ReplayControlResponseDataPtr response;
    {
        autil::ScopedLock lock(_mutex);
        if (!_pendingHistoryQuery ||
            _pendingHistoryQuery->queryId != historyQueryId) {
            return;
        }
        expireCommandIfNeededLocked();
        pipe = _pendingHistoryQuery->responsePipe;
        response = finishHistoryQueryLocked("history query timed out after 10 seconds");
    }
    publishAsyncResponse(pipe, response);
}

bool ReplayControlR::subscribe(const std::string &clientId,
                               const navi::AsyncPipePtr &commandPipe,
                               const navi::AsyncPipePtr &historyQueryPipe)
{
    if (clientId.empty() || !commandPipe || !historyQueryPipe) {
        return false;
    }
    autil::ScopedLock lock(_mutex);
    if (_stopped) {
        return false;
    }
    expireCommandIfNeededLocked();
    auto &subscriber = _subscribers[clientId];
    subscriber.commandPipe = commandPipe;
    subscriber.historyQueryPipe = historyQueryPipe;
    subscriber.summary = {};
    subscriber.summary.clientId = clientId;
    subscriber.clockOffsetUs = 0;
    subscriber.hasClockOffset = false;
    auto command = makeCommandLocked(subscriber.summary);
    if (navi::EC_NONE != commandPipe->setData(command)) {
        _subscribers.erase(clientId);
        return false;
    }
    NAVI_KERNEL_LOG(INFO, "registered replay client [%s]", clientId.c_str());
    if (_command.active) {
        auto query = std::make_shared<ReplayHistoryQueryData>();
        query->queryId = ++_nextHistoryQueryId;
        if (query->queryId == 0) {
            query->queryId = ++_nextHistoryQueryId;
        }
        if (navi::EC_NONE != historyQueryPipe->setData(query)) {
            _subscribers.erase(clientId);
            return false;
        }
    }
    return true;
}

void ReplayControlR::unsubscribe(const std::string &clientId,
                                 const navi::AsyncPipePtr &commandPipe,
                                 const navi::AsyncPipePtr &historyQueryPipe)
{
    navi::AsyncPipePtr responsePipe;
    ReplayControlResponseDataPtr response;
    {
        autil::ScopedLock lock(_mutex);
        expireCommandIfNeededLocked();
        auto it = _subscribers.find(clientId);
        if (it != _subscribers.end() &&
            (!commandPipe || it->second.commandPipe.lock() == commandPipe) &&
            (!historyQueryPipe ||
             it->second.historyQueryPipe.lock() == historyQueryPipe)) {
            const bool wasEligible = _command.active &&
                                     isEligible(it->second.summary, _command);
            _subscribers.erase(it);
            if (_pendingHistoryQuery &&
                std::find(_pendingHistoryQuery->expectedClients.begin(),
                          _pendingHistoryQuery->expectedClients.end(), clientId) !=
                    _pendingHistoryQuery->expectedClients.end() &&
                _pendingHistoryQuery->responses.count(clientId) == 0) {
                _pendingHistoryQuery->failures[clientId] =
                    "client disconnected during history query";
                if (historyQueryCompleteLocked()) {
                    responsePipe = _pendingHistoryQuery->responsePipe;
                    response = finishHistoryQueryLocked("");
                }
            }
            if (wasEligible) {
                resetPacingEpochLocked();
                ++_command.generation;
                notifySubscribersLocked();
            }
        }
    }
    publishAsyncResponse(responsePipe, response);
}

void ReplayControlR::completeHistoryQuery(
    const ReplayHistoryQueryResponseData &queryResponse)
{
    const auto masterReceiveTimeUs =
        autil::TimeUtility::currentTimeInMicroSeconds();
    navi::AsyncPipePtr responsePipe;
    ReplayControlResponseDataPtr response;
    {
        autil::ScopedLock lock(_mutex);
        expireCommandIfNeededLocked();
        const auto clientId = queryResponse.summary.clientId;
        auto subscriberIt = _subscribers.find(clientId);
        if (subscriberIt == _subscribers.end()) {
            return;
        }
        auto summary = queryResponse.summary;
        summary.status.clear();
        summary.error.clear();
        if (!_command.active) {
            summary.selectionGeneration = 0;
            summary.selectedSampleCount = 0;
        }
        const bool wasEligible =
            _command.active && isEligible(subscriberIt->second.summary, _command);
        const bool clockOffsetChanged = updateClockOffsetLocked(
            subscriberIt->second, summary.reportTimeUs, masterReceiveTimeUs);
        subscriberIt->second.summary = summary;
        const bool nowEligible =
            _command.active && isEligible(subscriberIt->second.summary, _command);
        if (wasEligible != nowEligible) {
            resetPacingEpochLocked();
            ++_command.generation;
            notifySubscribersLocked();
        } else if (clockOffsetChanged && nowEligible) {
            ++_command.generation;
            notifyClockOffsetLocked(subscriberIt->second);
        }
        finishCommandIfAllClientsCompletedLocked();
        if (!_pendingHistoryQuery ||
            _pendingHistoryQuery->queryId != queryResponse.queryId ||
            std::find(_pendingHistoryQuery->expectedClients.begin(),
                      _pendingHistoryQuery->expectedClients.end(), clientId) ==
                _pendingHistoryQuery->expectedClients.end()) {
            return;
        }
        _pendingHistoryQuery->responses[clientId] = std::move(summary);
        if (historyQueryCompleteLocked()) {
            responsePipe = _pendingHistoryQuery->responsePipe;
            response = finishHistoryQueryLocked("");
        }
    }
    publishAsyncResponse(responsePipe, response);
}

void ReplayControlR::updateHistory(const ReplayHistorySummary &summary)
{
    if (summary.clientId.empty()) {
        return;
    }
    const auto masterReceiveTimeUs =
        autil::TimeUtility::currentTimeInMicroSeconds();
    autil::ScopedLock lock(_mutex);
    expireCommandIfNeededLocked();
    auto it = _subscribers.find(summary.clientId);
    if (it == _subscribers.end()) {
        return;
    }
    const bool wasEligible = _command.active &&
                             isEligible(it->second.summary, _command);
    const bool clockOffsetChanged = updateClockOffsetLocked(
        it->second, summary.reportTimeUs, masterReceiveTimeUs);
    it->second.summary = summary;
    if (!_command.active) {
        it->second.summary.selectionGeneration = 0;
        it->second.summary.selectedSampleCount = 0;
    }
    const bool isNowEligible = _command.active &&
                               isEligible(it->second.summary, _command);
    if (wasEligible != isNowEligible) {
        resetPacingEpochLocked();
        ++_command.generation;
        notifySubscribersLocked();
    } else if (clockOffsetChanged && isNowEligible) {
        ++_command.generation;
        notifyClockOffsetLocked(it->second);
    }
    finishCommandIfAllClientsCompletedLocked();
}

std::vector<ReplayHistorySummary> ReplayControlR::getHistory() const
{
    autil::ScopedLock lock(_mutex);
    std::vector<ReplayHistorySummary> result;
    result.reserve(_subscribers.size());
    for (const auto &[clientId, subscriber] : _subscribers) {
        result.emplace_back(subscriber.summary);
    }
    return result;
}

void ReplayControlR::stop()
{
    std::vector<navi::AsyncPipePtr> pipes;
    navi::AsyncPipePtr pendingResponsePipe;
    ReplayControlResponseDataPtr pendingResponse;
    {
        autil::ScopedLock lock(_mutex);
        if (_stopped) {
            return;
        }
        _stopped = true;
        _command.active = false;
        _command.deadlineTimeUs = 0;
        ++_command.generation;
        pipes.reserve(_subscribers.size());
        for (const auto &[clientId, subscriber] : _subscribers) {
            if (auto pipe = subscriber.commandPipe.lock()) {
                pipes.emplace_back(std::move(pipe));
            }
            if (auto pipe = subscriber.historyQueryPipe.lock()) {
                pipes.emplace_back(std::move(pipe));
            }
        }
        if (_pendingHistoryQuery) {
            pendingResponsePipe = _pendingHistoryQuery->responsePipe;
            pendingResponse = finishHistoryQueryLocked("replay control stopped");
        }
        _subscribers.clear();
    }
    publishAsyncResponse(pendingResponsePipe, pendingResponse);
    for (const auto &pipe : pipes) {
        pipe->setEof();
    }
}

REGISTER_RESOURCE(ReplayControlR);

}
