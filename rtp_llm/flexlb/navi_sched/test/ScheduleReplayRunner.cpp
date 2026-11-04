#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include "rtp_llm/flexlb/navi_sched/data/BlockHashUtil.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterScheduleRequestData.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentSchedulerConfigR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterPendingCacheR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterBatchReservation.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/ScheduleContextInternal.h"
#include "autil/Lock.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#define private public
#include "rtp_llm/flexlb/navi_sched/scheduler/ScheduleContext.h"
#undef private

#include "rtp_llm/flexlb/navi_sched/test/ScheduleReplayRunner.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerColdStartStrategy.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerKernel.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/MasterScheduleContextData.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/MasterScheduleContextKernel.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerGraph.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerKernelInternal.h"

namespace navi_sched {
namespace {

using JsonMap = autil::legacy::json::JsonMap;

template <typename T>
T auditValue(const JsonMap &info, const std::string &name)
{
    T value;
    autil::legacy::FromJson(value, info.at(name));
    return value;
}

bool auditNear(double actual, double expected)
{
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(actual - expected) <= 1e-8 * std::max(1.0, std::abs(expected));
}

class RecordedDraws {
public:
    explicit RecordedDraws(const std::vector<JsonMap> &records)
    {
        for (const auto &record : records) {
            const auto probability = auditValue<double>(record, "probability");
            const auto sample = auditValue<std::string>(record, "sample");
            if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0 ||
                sample.empty() || sample.find_first_not_of("0123456789") != std::string::npos) {
                throw std::invalid_argument("invalid replay random draw");
            }
            const auto value = std::stoull(sample);
            if (probability > 0.0 && probability < 1.0) {
                _draws.push_back(value);
            }
        }
    }

    uint64_t next()
    {
        if (_position == _draws.size()) {
            throw std::invalid_argument("missing replay random draw");
        }
        return _draws[_position++];
    }
    bool exhausted() const { return _position == _draws.size(); }

private:
    std::vector<uint64_t> _draws;
    size_t _position = 0;
};

template <typename Decisions, typename Outcome>
bool randomDecisionsMatch(const Decisions &actual, const std::vector<JsonMap> &recorded,
                          const std::string &outcomeName, Outcome outcome)
{
    if (actual.size() != recorded.size()) {
        return false;
    }
    for (size_t index = 0; index < actual.size(); ++index) {
        const auto &left = actual[index];
        const auto &right = recorded[index];
        if (left.requestIndex != auditValue<size_t>(right, "request_index") ||
            std::to_string(left.sample) != auditValue<std::string>(right, "sample") ||
            !auditNear(left.probability, auditValue<double>(right, "probability")) ||
            outcome(left) != auditValue<bool>(right, outcomeName)) {
            return false;
        }
    }
    return true;
}

void fillSnapshotFromNodeInfo(const NaviSchedNodeInfo &info,
                              NaviSchedNodeSnapshot &snapshot)
{
    snapshot.info = std::make_shared<const NaviSchedNodeInfo>(info);
    if (info.latencyModelRevision > 0) {
        auto latencyModel = std::make_shared<NaviSchedLatencyModel>();
        latencyModel->formula = info.latencyFormula;
        latencyModel->binaryVersion = info.latencyBinaryVersion;
        latencyModel->revision = info.latencyModelRevision;
        latencyModel->parameters = info.latencyParameters;
        latencyModel->residualVersion = info.latencyResidualVersion;
        latencyModel->residualParameters = info.latencyResidualParameters;
        latencyModel->residualLastUpdateTimeUs =
            info.latencyResidualLastUpdateTimeUs;
        latencyModel->overloadPredictionTrusted = info.latencyModelTrusted;
        latencyModel->warmStarted = info.latencyModelWarmStarted;
        snapshot.latencyModel = std::move(latencyModel);
    }
    snapshot.load.totalBlocks = info.totalBlocks;
    snapshot.load.inflightRequests = info.inflightRequests;
    snapshot.load.batchWaitingRequests = info.batchWaitingRequests;
    snapshot.load.queuedBatchRequests = info.queuedBatchRequests;
    snapshot.load.runningRequests = info.runningRequests;
    snapshot.load.batchTimeoutCount = info.batchTimeoutCount;
    snapshot.load.abnormalBatchFinishCount = info.abnormalBatchFinishCount;
    snapshot.load.queueTotalPredictedLatencyMs =
        info.queueTotalPredictedLatencyMs;
    snapshot.load.queueDecayHeadRemainingMs =
        info.queueDecayHeadRemainingMs;
    snapshot.load.queueReportReceiveTimeUs = info.queueReportReceiveTimeUs;
    snapshot.workload = info.workload;
    snapshot.unifiedLatencyPrediction = info.unifiedLatencyPrediction;
    if (snapshot.unifiedLatencyPrediction) {
        snapshot.reportedWorkloadFeatures.appendPrefillWorkload(info.workload);
        snapshot.reservedWorkloadFeatures = info.reservedWorkloadFeatures;
        if (info.workloadFeatureSnapshotValid) {
            snapshot.workloadFeatures = info.workloadFeatures;
            snapshot.maxRunningRequestCount = info.maxRunningRequestCount;
            snapshot.workloadRunningCapacitySlotCount =
                info.workloadRunningCapacitySlotCount;
            snapshot.reservedRunningCapacitySlotCount =
                info.reservedRunningCapacitySlotCount;
        } else {
            snapshot.workloadFeatures.append(
                snapshot.reportedWorkloadFeatures);
            snapshot.workloadFeatures.append(
                snapshot.reservedWorkloadFeatures);
        }
    }
    snapshot.workloadReportReceiveTimeUs = info.workloadReportReceiveTimeUs;
}

class ReplayNode final : public NaviSchedNode {
public:
    explicit ReplayNode(const NaviSchedNodeInfo &node)
        : _node(std::make_shared<const NaviSchedNodeInfo>(node))
    {
    }

public:
    bool buildPendingCache(const JsonMap &state,
                           const std::shared_ptr<MasterPendingCacheR> &pending)
    {
        try {
            auto restoreHashes = [&state](const std::string &name, auto &output) {
                const auto hashes = auditValue<std::vector<uint64_t>>(state, name);
                output.insert(hashes.begin(), hashes.end());
            };
            restoreHashes("cached_block_hashes", _cachedBlockHashes);
            restoreHashes("full_ready_block_hashes", _fullReadyBlockHashes);
            restoreHashes("tail_ready_block_hashes", _tailReadyBlockHashes);
            _agentIdHash = auditValue<uint64_t>(state, "agent_id_hash");
            _blockSize = auditValue<int64_t>(state, "block_size");
            _hybridTailRequired = auditValue<bool>(state, "hybrid_tail_required");
            for (const auto &record : auditValue<std::vector<JsonMap>>(state, "reservations")) {
                MasterBatchReservation reservation;
                if (!reservation.restoreReplayState(record)) return false;
                _reservations.emplace_back(std::move(reservation));
            }
            _pending = pending;
            return _pending->restoreReplayState(_agentIdHash,
                                                auditValue<JsonMap>(state, "pending_cache"));
        } catch (const std::exception &) {
            return false;
        }
    }

    bool buildCache(const std::vector<NaviSchedScheduleItem> &requests,
                    const std::vector<std::vector<NaviSchedReplayNodeMatch>> &matches,
                    size_t nodeIndex)
    {
        if (requests.size() != matches.size()) {
            return false;
        }
        for (size_t requestIndex = 0; requestIndex < requests.size(); ++requestIndex) {
            if (nodeIndex >= matches[requestIndex].size()) {
                return false;
            }
            const auto &blockHashes =
                BlockHashUtil::get(requests[requestIndex].request.blockHashes);
            auto matchedBlockCount = matches[requestIndex][nodeIndex].matchedBlockCount;
            if (matchedBlockCount < 0 ||
                static_cast<size_t>(matchedBlockCount) > blockHashes.size()) {
                return false;
            }
            _cachedBlockHashes.insert(blockHashes.begin(),
                                      blockHashes.begin() + matchedBlockCount);
        }
        for (size_t requestIndex = 0; requestIndex < requests.size(); ++requestIndex) {
            const auto &blockHashes =
                BlockHashUtil::get(requests[requestIndex].request.blockHashes);
            if (getMatchedBlockCount(blockHashes) !=
                matches[requestIndex][nodeIndex].matchedBlockCount) {
                return false;
            }
        }
        return true;
    }
    NaviSchedNodeInfo getInfo() const override { return *_node; }
    bool getScheduleSnapshot(const std::string &bizName,
                             NaviSchedNodeSnapshot &snapshot) const override
    {
        if (_node->bizName != bizName) {
            return false;
        }
        fillSnapshotFromNodeInfo(*_node, snapshot);
        return true;
    }
    int64_t getMatchedBlockCount(const NaviSchedBlockHashes &blockHashes) const override
    {
        autil::ScopedReadLock lock(_cacheLock);
        if (_pending) {
            return _pending->getMatchedBlockCount(_agentIdHash, blockHashes,
                _cachedBlockHashes, _fullReadyBlockHashes, _tailReadyBlockHashes,
                _hybridTailRequired);
        }
        int64_t matchedBlockCount = 0;
        for (auto hash : blockHashes) {
            if (_cachedBlockHashes.end() == _cachedBlockHashes.find(hash)) {
                break;
            }
            ++matchedBlockCount;
        }
        return matchedBlockCount;
    }
    void getMatchedBlockCounts(
        const std::vector<NaviSchedScheduleItem> &requests,
        int64_t *matchedBlockCounts) const override
    {
        if (_pending) {
            autil::ScopedReadLock lock(_cacheLock);
            _pending->getMatchedBlockCounts(_agentIdHash, requests, _blockSize,
                _cachedBlockHashes, _fullReadyBlockHashes, _tailReadyBlockHashes,
                _hybridTailRequired, matchedBlockCounts);
            return;
        }
        for (size_t requestIndex = 0; requestIndex < requests.size(); ++requestIndex) {
            matchedBlockCounts[requestIndex] = getMatchedBlockCount(
                BlockHashUtil::get(requests[requestIndex].request.blockHashes));
        }
    }

    int64_t getMatchedBlockCountForRequest(const NaviSchedScheduleRequest &request) const override
    {
        if (!_pending) return getMatchedBlockCount(BlockHashUtil::get(request.blockHashes));
        autil::ScopedReadLock lock(_cacheLock);
        if (_blockSize <= 0 || request.blockSize != _blockSize) return 0;
        return _pending->getMatchedBlockCount(_agentIdHash,
            BlockHashUtil::get(request.blockHashes), _cachedBlockHashes,
            _fullReadyBlockHashes, _tailReadyBlockHashes, _hybridTailRequired,
            BlockHashUtil::getMatchableBlockCount(request.promptTokenCount,
                                                 request.blockSize, request.blockHashes));
    }

private:
    std::shared_ptr<const NaviSchedNodeInfo> _node;
    MasterPendingCacheR::PhysicalBlockSet _cachedBlockHashes;
    MasterPendingCacheR::PhysicalBlockSet _fullReadyBlockHashes;
    MasterPendingCacheR::PhysicalBlockSet _tailReadyBlockHashes;
    mutable autil::ReadWriteLock _cacheLock;
    std::shared_ptr<MasterPendingCacheR> _pending;
    std::vector<MasterBatchReservation> _reservations;
    uint64_t _agentIdHash = 0;
    int64_t _blockSize = -1;
    bool _hybridTailRequired = false;
};

} // namespace

bool ScheduleReplayRunner::makeContext(const NaviSchedScheduleReplay &replay,
                                       NaviSchedScheduleContext &context,
                                       std::shared_ptr<MasterPendingCacheR> pendingCache)
{
    if (replay.version != NAVI_SCHED_DATA_VERSION || replay.requests.empty() ||
        replay.nodeMatches.size() != replay.requests.size()) {
        return false;
    }
    auto requests = replay.requests;
    for (auto &item : requests) {
        item.request.bizName = replay.bizName;
        if (!item.request.valid() || item.waitTimeMs < 0 ||
            item.remainingTimeoutMs < 0) {
            return false;
        }
    }
    context = NaviSchedScheduleContext();
    context._scheduleTimeUs = replay.scheduleTimeUs;
    if (replay.schedulerInfo.count("queue_prediction_time_us")) {
        try {
            context._queuePredictionTimeUs =
                auditValue<int64_t>(replay.schedulerInfo, "queue_prediction_time_us");
            if (context._queuePredictionTimeUs < context._scheduleTimeUs) {
                return false;
            }
        } catch (const std::exception &) {
            return false;
        }
    }
    std::vector<JsonMap> nodeRuntime;
    const bool audited = replay.schedulerInfo.count("audit_version") != 0;
    if (audited) {
        try {
            if (auditValue<int64_t>(replay.schedulerInfo, "audit_version") != 1) {
                return false;
            }
            context._timeCompression = auditValue<double>(replay.schedulerInfo, "time_compression");
            nodeRuntime = auditValue<std::vector<JsonMap>>(replay.schedulerInfo, "replay_node_runtime");
            if (!std::isfinite(context._timeCompression) || context._timeCompression <= 0.0 ||
                nodeRuntime.size() != replay.nodes.size()) {
                return false;
            }
        } catch (const std::exception &) {
            return false;
        }
    }
    if (!replay.decisions.empty()) {
        context._scheduleId = replay.decisions.front().scheduleId;
        for (const auto &decision : replay.decisions) {
            if (decision.scheduleId != context._scheduleId) {
                return false;
            }
        }
    }
    context._requests = std::move(requests);
    std::vector<NaviSchedNodeConstPtr> nodes;
    nodes.reserve(replay.nodes.size());
    std::vector<NaviSchedNodeSnapshot> snapshots;
    snapshots.reserve(replay.nodes.size());
    std::vector<JsonMap> pendingStates;
    if (replay.schedulerInfo.count("pending_state")) {
        try {
            pendingStates = auditValue<std::vector<JsonMap>>(replay.schedulerInfo, "pending_state");
            if (pendingStates.size() != replay.nodes.size()) return false;
        } catch (const std::exception &) {
            return false;
        }
    }
    // Production shares the pending node index while each node owns its cache lock.
    if (!pendingStates.empty() && !pendingCache) {
        pendingCache = std::make_shared<MasterPendingCacheR>();
    }
    for (size_t nodeIndex = 0; nodeIndex < replay.nodes.size(); ++nodeIndex) {
        for (const auto &requestMatches : replay.nodeMatches) {
            if (requestMatches.size() != replay.nodes.size()) {
                return false;
            }
        }
        auto nodeInfo = replay.nodes[nodeIndex];
        nodeInfo.bizName = replay.bizName;
        auto node = std::make_shared<ReplayNode>(nodeInfo);
        if (!(pendingStates.empty() || pendingStates[nodeIndex].empty()
                  ? node->buildCache(context._requests, replay.nodeMatches, nodeIndex)
                  : node->buildPendingCache(pendingStates[nodeIndex], pendingCache))) {
            return false;
        }
        nodes.emplace_back(node);
        NaviSchedNodeSnapshot snapshot;
        if (!node->getScheduleSnapshot(replay.bizName, snapshot)) {
            return false;
        }
        if (!nodeInfo.schedulerConfigJson.empty()) {
            AgentSchedulerConfig config;
            std::string errorMessage;
            if (!AgentSchedulerConfigR::parse(nodeInfo.schedulerConfigJson, config,
                                              errorMessage)) {
                return false;
            }
            snapshot.residualRuntimeConfig = config.residualRuntimeConfig();
        }
        // Legacy fixtures default to steady-state nodes; audited records restore
        // the actual start time and model runtime below.
        snapshot.agentStartTimeUs = replay.scheduleTimeUs - 300'000'000;
        if (audited) {
            try {
                const auto &runtime = nodeRuntime[nodeIndex];
                if (auditValue<size_t>(runtime, "node_index") != nodeIndex) {
                    return false;
                }
                snapshot.agentStartTimeUs = auditValue<int64_t>(runtime, "agent_start_time_us");
                const auto sampleCount = auditValue<int64_t>(runtime, "latency_session_sample_count");
                if (sampleCount < 0 || (!snapshot.latencyModel && sampleCount != 0)) {
                    return false;
                }
                if (snapshot.latencyModel) {
                    auto model = std::make_shared<NaviSchedLatencyModel>(*snapshot.latencyModel);
                    model->sessionSampleCount = sampleCount;
                    snapshot.latencyModel = std::move(model);
                }
                auto &config = snapshot.residualRuntimeConfig;
                config.learningRate = auditValue<double>(runtime, "residual_learning_rate");
                config.updateErrorFloorMs = auditValue<double>(runtime, "residual_update_error_floor_ms");
                config.updateErrorRatio = auditValue<double>(runtime, "residual_update_error_ratio");
                config.idleDecayStartUs = auditValue<int64_t>(runtime, "residual_idle_decay_start_us");
                config.idleDecayEndUs = auditValue<int64_t>(runtime, "residual_idle_decay_end_us");
                if (!config.valid()) {
                    return false;
                }
            } catch (const std::exception &) {
                return false;
            }
        }
        snapshots.emplace_back(std::move(snapshot));
    }
    context.setNodes(std::move(nodes), std::move(snapshots));
    if (audited) {
        try {
            for (size_t index = 0; index < nodeRuntime.size(); ++index) {
                // MasterStateR records the features after schedule-time progress.
                // The constructor advances reported workload separately; do not
                // advance the recorded aggregate a second time.
                if (replay.nodes[index].workloadFeatureSnapshotValid) {
                    context._scheduleNodes[index]._workloadFeatures =
                        replay.nodes[index].workloadFeatures;
                }
                const auto &node = context.getNode(index);
                const auto &runtime = nodeRuntime[index];
                if (node.isBacklogPredictionTrusted() != auditValue<bool>(runtime, "backlog_prediction_trusted") ||
                    CostSchedulerColdStartStrategy::isRampFinished(
                        replay.scheduleTimeUs, node.getAgentStartTimeUs(), node.getLatencySessionSampleCount()) !=
                        auditValue<bool>(runtime, "ramp_finished") ||
                    !auditNear(node.getEstimatedBacklogLatencyMs(), auditValue<double>(runtime, "estimated_backlog_latency_ms")) ||
                    !auditNear(node.getReportedRunningElapsedTimeMs(), auditValue<double>(runtime, "reported_running_elapsed_time_ms")) ||
                    !auditNear(node.getWorkloadReportAgeMs(), auditValue<double>(runtime, "workload_report_age_ms"))) {
                    return false;
                }
            }
        } catch (const std::exception &) {
            return false;
        }
    }
    return true;
}

NaviSchedNodeSnapshot ScheduleReplayRunner::makeNodeSnapshot(
    const NaviSchedNodeInfo &info)
{
    NaviSchedNodeSnapshot snapshot;
    fillSnapshotFromNodeInfo(info, snapshot);
    return snapshot;
}

bool ScheduleReplayRunner::prepare(NaviSchedScheduleContext &context)
{
    return context.prepare();
}

bool ScheduleReplayRunner::prepareContext(
    NaviSchedScheduleContext &context)
{
    return prepare(context);
}

bool ScheduleReplayRunner::collectDecisions(NaviSchedScheduleContext &context,
                                            std::vector<NaviSchedDecision> &decisions)
{
    if (!context.prepareResult(context._queuePredictionTimeUs)) {
        return false;
    }
    decisions = context._decisions;
    for (const auto &response : context._deferredResponses) {
        auto decision = response.decision;
        decision.scheduleLoss = context._scheduleLoss;
        decision.scheduleLatencyUs = context._scheduleLatencyUs;
        decisions.emplace_back(std::move(decision));
    }
    return true;
}

void ScheduleReplayRunner::addResponseRequest(NaviSchedScheduleContext &context,
                                              const MasterScheduleRequestDataPtr &request)
{
    context.addRequest(request);
}

void ScheduleReplayRunner::setReplay(
    NaviSchedScheduleContext &context,
    const std::shared_ptr<NaviSchedScheduleReplay> &replay)
{
    context.setReplay(replay);
}

bool ScheduleReplayRunner::finish(NaviSchedScheduleContext &context)
{
    return context.finish();
}

const autil::legacy::json::JsonMap *ScheduleReplayRunner::getSchedulerInfo(
    const NaviSchedScheduleContext &context)
{
    return context._replay ? &context._replay->schedulerInfo : nullptr;
}

bool ScheduleReplayRunner::decisionsEqual(const std::vector<NaviSchedDecision> &expected,
                                          const std::vector<NaviSchedDecision> &actual,
                                          const std::string &sharedBizName)
{
    if (expected.size() != actual.size()) {
        return false;
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        const auto &left = expected[index];
        const auto &right = actual[index];
        const auto &leftBizName = left.target.bizName.empty()
                                      ? sharedBizName
                                      : left.target.bizName;
        const auto &rightBizName = right.target.bizName.empty()
                                      ? sharedBizName
                                      : right.target.bizName;
        if (left.version != right.version || left.requestId != right.requestId ||
            left.reason != right.reason || left.batchId != right.batchId ||
            left.expectedBatchSize != right.expectedBatchSize ||
            left.latencyTrainingPhase != right.latencyTrainingPhase ||
            left.target.version != right.target.version ||
            leftBizName != rightBizName ||
            left.target.address != right.target.address ||
            left.target.agentAddress != right.target.agentAddress) {
            return false;
        }
    }
    return true;
}

void ScheduleReplayRunner::initialize(NaviSchedScheduleContext &context,
                                      int64_t scheduleTimeUs,
                                      std::vector<NaviSchedScheduleItem> requests,
                                      std::vector<NaviSchedNodeConstPtr> nodes)
{
    context = NaviSchedScheduleContext();
    context._scheduleTimeUs = scheduleTimeUs;
    context._requests = std::move(requests);
    std::vector<NaviSchedNodeSnapshot> snapshots;
    snapshots.reserve(nodes.size());
    for (const auto &node : nodes) {
        snapshots.emplace_back(makeNodeSnapshot(node->getInfo()));
    }
    context.setNodes(std::move(nodes), std::move(snapshots));
}

int64_t ScheduleReplayRunner::getScheduleTimeUs(const NaviSchedScheduleContext &context)
{
    return context._scheduleTimeUs;
}

double ScheduleReplayRunner::makeEstimatedQueueWaitMs(int64_t scheduleTimeUs,
                                                      double reportedQueueWaitMs,
                                                      int64_t reportReceiveTimeUs,
                                                      double decayHeadRemainingMs,
                                                      double timeCompression)
{
    NaviSchedNodeInfo info;
    info.agentId = "agent";
    NaviSchedNodeSnapshot snapshot;
    snapshot.info = std::make_shared<const NaviSchedNodeInfo>(info);
    snapshot.load.queueTotalPredictedLatencyMs = reportedQueueWaitMs;
    snapshot.load.queueDecayHeadRemainingMs = decayHeadRemainingMs;
    snapshot.load.queueReportReceiveTimeUs = reportReceiveTimeUs;
    NaviSchedScheduleNode node(nullptr, std::move(snapshot), scheduleTimeUs,
                               timeCompression);
    return node.getEstimatedQueueWaitMs();
}

double ScheduleReplayRunner::makeUnifiedDecayedLatencyMs(
    int64_t scheduleTimeUs, int64_t reportReceiveTimeUs,
    int64_t reportedBatchCount, double latencyWithReportedWorkloadMs,
    double latencyWithoutReportedWorkloadMs, double timeCompression,
    bool *usesReportedWorkload)
{
    NaviSchedNodeInfo info;
    info.agentId = "agent";
    NaviSchedNodeSnapshot snapshot;
    snapshot.info = std::make_shared<const NaviSchedNodeInfo>(info);
    snapshot.unifiedLatencyPrediction = true;
    snapshot.reportedWorkloadFeatures.appendBatchCount(reportedBatchCount);
    snapshot.workloadReportReceiveTimeUs = reportReceiveTimeUs;
    NaviSchedScheduleNode node(nullptr, std::move(snapshot), scheduleTimeUs,
                               timeCompression);
    return node.decayUnifiedLatency(latencyWithReportedWorkloadMs,
                                    latencyWithoutReportedWorkloadMs,
                                    usesReportedWorkload);
}

bool ScheduleReplayRunner::initializeCostSchedulerWeights(
    size_t requestCount, size_t nodeCount,
    CostSchedulerPreparedState &preparedState)
{
    auto schedInfoCount = requestCount * nodeCount;
    std::vector<double> weights(schedInfoCount);
    CostSchedulerKernelInternal::initializeWeights(
        weights.data(), requestCount, nodeCount);

    preparedState = CostSchedulerPreparedState();
    preparedState.requestCount = requestCount;
    preparedState.nodeCount = nodeCount;
    preparedState.weights.reserve(schedInfoCount);
    preparedState.parameterGradientsValid = true;
    AutoGradTape autoGrad(schedInfoCount, schedInfoCount);
    std::vector<AutoGradTape::Value> parameterWeights;
    parameterWeights.reserve(schedInfoCount);
    for (size_t index = 0; index < schedInfoCount; ++index) {
        parameterWeights.emplace_back(
            autoGrad.parameter(weights[index], index));
    }
    std::vector<double> gradients;
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            auto parameterIndex = requestIndex + nodeIndex * requestCount;
            preparedState.weights.emplace_back(weights[parameterIndex]);
            if (!autoGrad.backward(
                    parameterWeights[parameterIndex], gradients)) {
                return false;
            }
            for (size_t index = 0; index < gradients.size(); ++index) {
                auto expected = parameterIndex == index ? 1.0 : 0.0;
                if (gradients[index] != expected) {
                    preparedState.parameterGradientsValid = false;
                }
            }
        }
    }
    return true;
}

bool ScheduleReplayRunner::prepareCostScheduler(
    NaviSchedScheduleContext &context,
    CostSchedulerPreparedState &preparedState)
{
    auto requestCount = context.requestCount();
    auto nodeCount = context.nodeCount();
    auto schedInfoCount = requestCount * nodeCount;
    auto requestTokenCounts = std::make_unique<int64_t[]>(requestCount);
    auto totalTokenCount =
        CostSchedulerKernelInternal::fillRequestTokenCounts(
            context, requestTokenCounts.get());
    CostSchedulerKernelInternal::ReplayInfo replayInfo;
    auto costs = CostSchedulerKernelInternal::prepare(
        context, requestTokenCounts.get(), 0.5, 0.01, 1.0, 0.0, 100,
        0, replayInfo);
    if (!costs) {
        return false;
    }
    std::vector<double> weights(schedInfoCount);
    CostSchedulerKernelInternal::initializeWeights(
        weights.data(), requestCount, nodeCount);
    preparedState = CostSchedulerPreparedState();
    preparedState.requestCount = requestCount;
    preparedState.nodeCount = nodeCount;
    preparedState.requestTokenCounts.assign(
        requestTokenCounts.get(), requestTokenCounts.get() + requestCount);
    preparedState.totalTokenCount = totalTokenCount;
    preparedState.costs.reserve(schedInfoCount);
    preparedState.queueWaits.reserve(schedInfoCount);
    preparedState.weights.reserve(schedInfoCount);
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        auto queueWaitMs = context.getNode(
            nodeIndex).getEstimatedQueueWaitMs();
        auto rowOffset = nodeIndex * requestCount;
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            auto index = rowOffset + requestIndex;
            preparedState.costs.emplace_back(costs[index]);
            preparedState.queueWaits.emplace_back(queueWaitMs);
            preparedState.weights.emplace_back(weights[index]);
        }
    }
    return true;
}

bool ScheduleReplayRunner::runCostScheduler(
    const NaviSchedScheduleReplay &replay,
    const NaviSchedScheduleContextPtr &context, double fallbackLambda,
    double fallbackAlpha, double fallbackAlphaDecay,
    double fallbackMinAlpha, int32_t fallbackMaxLoopCount,
    double latencyWeightConcavity,
    std::vector<NaviSchedDecision> &decisions,
    double latencyTokenNormalizationPower,
    double latencyWeightConcavityOverride,
    const std::function<void(const NaviSchedScheduleContext &)> &inspectInput)
{
    CostSchedulerConfig config;
    config.lambda = fallbackLambda;
    config.alpha = fallbackAlpha;
    config.alphaDecay = fallbackAlphaDecay;
    config.minAlpha = fallbackMinAlpha;
    config.maxLoopCount = fallbackMaxLoopCount;
    config.optimizationTimeBudgetUs = -1;
    config.latencyWeightConcavity = latencyWeightConcavity;
    size_t workerCount = 1;
    const std::vector<double> *initialWeights = nullptr;
    CostSchedulerKernelInternal::ReplayInfo replayInfo;
    if (!replay.schedulerInfo.empty()) {
        try {
            autil::legacy::FromJson(
                replayInfo, autil::legacy::Any(replay.schedulerInfo));
        } catch (const std::exception &) {
            decisions.clear();
            return false;
        }
        if ("cost" == replayInfo.scheduler) {
            if (replayInfo.maxLoopCount >
                    static_cast<size_t>(CostSchedulerConfig::MAX_LOOP_COUNT) ||
                replayInfo.parallelism >
                    static_cast<size_t>(CostSchedulerConfig::MAX_PARALLELISM)) {
                decisions.clear();
                return false;
            }
            config.lambda = replayInfo.lambda;
            config.alpha = replayInfo.alpha;
            config.alphaDecay = replayInfo.alphaDecay;
            config.minAlpha = replayInfo.minAlpha;
            config.minLoopCount = static_cast<int32_t>(replayInfo.minLoopCount);
            config.maxLoopCount =
                static_cast<int32_t>(replayInfo.maxLoopCount);
            config.optimizationTimeBudgetUs = -1;
            config.latencyWeightConcavity =
                replayInfo.latencyWeightConcavity;
            config.latencyTokenNormalizationPower =
                replayInfo.latencyTokenNormalizationPower;
            config.initialWeightSeed = replayInfo.initialWeightSeed;
            config.discreteCheckInterval = static_cast<int32_t>(
                replayInfo.discreteCheckInterval);
            config.discreteGapThreshold =
                replayInfo.discreteGapThreshold;
            config.discreteRepairAlphaDecay =
                replayInfo.discreteRepairAlphaDecay;
            if (!replayInfo.optimizationRuns.empty() &&
                replayInfo.optimizationRuns.front().actualLoopCount > 0) {
                config.maxLoopCount = static_cast<int32_t>(
                    replayInfo.optimizationRuns.front().actualLoopCount);
            }
            workerCount = replayInfo.parallelism;
            if (replayInfo.optimizationRuns.empty() &&
                !replay.schedulerInfo.count("audit_version")) {
                decisions.clear();
                return false;
            }
            if (!replayInfo.optimizationRuns.empty()) {
                initialWeights =
                    &replayInfo.optimizationRuns.front().initialWeights;
            }
        }
    }
    if (std::isfinite(latencyTokenNormalizationPower)) {
        config.latencyTokenNormalizationPower =
            latencyTokenNormalizationPower;
    }
    if (std::isfinite(latencyWeightConcavityOverride)) {
        config.latencyWeightConcavity = latencyWeightConcavityOverride;
    }
    if (workerCount >
        static_cast<size_t>(CostSchedulerConfig::MAX_PARALLELISM)) {
        decisions.clear();
        return false;
    }
    config.parallelism = static_cast<int32_t>(workerCount);
    if (!context || !config.valid() || 0 == workerCount) {
        decisions.clear();
        return false;
    }
    const bool audited = replay.schedulerInfo.count("audit_version") != 0;
    bool costScheduleRequired = true;
    if (audited) {
        // Offline audit compares recorded random decisions; request tracing explicitly.
        context->setReplay(std::make_shared<NaviSchedScheduleReplay>());
        try {
            const auto &info = replay.schedulerInfo;
            MasterScheduleR scheduleResource;
            MasterScheduleContextKernel admission;
            admission._masterScheduleR__ = &scheduleResource;
            admission._schedulerKernelId = CostSchedulerKernel::KERNEL_ID;
            admission._maxWaitScheduleTimeMs = auditValue<int64_t>(info, "queue_overload_schedule_cycle_ms");
            admission._queueOverloadStartCycleMultiplier = auditValue<int64_t>(info, "queue_overload_start_cycle_multiplier");
            admission._queueOverloadFullCycleMultiplier = auditValue<int64_t>(info, "queue_overload_full_cycle_multiplier");
            if (admission._maxWaitScheduleTimeMs <= 0 ||
                admission._queueOverloadStartCycleMultiplier <= 0 ||
                admission._queueOverloadFullCycleMultiplier <= admission._queueOverloadStartCycleMultiplier) {
                decisions.clear();
                return false;
            }
            auto queueRecords = auditValue<std::vector<JsonMap>>(info, "queue_overload_random_decisions");
            RecordedDraws queueDraws(queueRecords);
            context->_responseRequests.resize(context->requestCount());
            admission.applyQueueOverloadLimit(
                std::make_shared<MasterScheduleContextData>(context),
                [&queueDraws]() { return queueDraws.next(); });
            const auto check = [&info](double actual, const std::string &name) {
                return auditNear(actual, auditValue<double>(info, name));
            };
            if (!queueDraws.exhausted() ||
                !randomDecisionsMatch(context->_queueOverloadRandomDecisions, queueRecords, "accepted",
                                      [](const auto &decision) { return decision.accepted; }) ||
                context->_queueOverloadEvaluated != auditValue<bool>(info, "queue_overload_evaluated") ||
                !check(context->_queueOverloadAveragePredictedLatencyMs, "queue_overload_average_predicted_latency_ms") ||
                !check(context->_queueOverloadDropProbability, "queue_overload_drop_probability") ||
                !check(context->_queueOverloadRequestCount, "queue_overload_request_count") ||
                !check(context->_queueOverloadRejectedRequestCount, "queue_overload_rejected_request_count")) {
                decisions.clear();
                return false;
            }
            costScheduleRequired = context->requestCount() > 0;
            if (!prepare(*context) && costScheduleRequired) {
                decisions.clear();
                return false;
            }
            const auto coldEvaluated = auditValue<bool>(info, "cold_start_evaluated");
            if (costScheduleRequired && context->nodeCount() > 0) {
                auto coldRecords = auditValue<std::vector<JsonMap>>(info, "cold_start_random_decisions");
                RecordedDraws coldDraws(coldRecords);
                multi_call::RandomGenerator random(0);
                CostSchedulerColdStartStrategy::Result result;
                if (!coldEvaluated || !CostSchedulerColdStartStrategy::apply(
                        *context, random, result, [&coldDraws]() { return coldDraws.next(); }) ||
                    !coldDraws.exhausted() ||
                    !randomDecisionsMatch(result.randomDecisions, coldRecords, "selected",
                                          [](const auto &decision) { return decision.selected; }) ||
                    !check(result.targetInflight, "cold_start_target_inflight") ||
                    !check(result.trustedNodeCount, "cold_start_trusted_node_count") ||
                    !check(result.untrustedNodeCount, "cold_start_untrusted_node_count") ||
                    !check(result.trustedRequestCount, "cold_start_trusted_request_count") ||
                    !check(result.untrustedRequestCount, "cold_start_untrusted_request_count") ||
                    !check(result.rejectedRequestCount, "cold_start_rejected_request_count") ||
                    !check(result.untrustedCapacity, "cold_start_untrusted_capacity") ||
                    !check(result.rampingNodeCount, "cold_start_ramping_node_count") ||
                    !check(result.unsafeWaitingMarginalNodeCount, "cold_start_unsafe_waiting_marginal_node_count") ||
                    !check(result.minimumRampProgress, "cold_start_minimum_ramp_progress") ||
                    !check(result.minimumRampedTargetInflight, "cold_start_minimum_ramped_target_inflight")) {
                    decisions.clear();
                    return false;
                }
                costScheduleRequired = result.costScheduleRequired;
            } else if (coldEvaluated) {
                decisions.clear();
                return false;
            }
            if (!costScheduleRequired) {
                if (!replayInfo.optimizationRuns.empty()) {
                    decisions.clear();
                    return false;
                }
                context->setScheduleLoss(0.0);
                return collectDecisions(*context, decisions);
            }
            if (!initialWeights) {
                decisions.clear();
                return false;
            }
        } catch (const std::exception &) {
            decisions.clear();
            return false;
        }
    } else if (!prepare(*context)) {
        decisions.clear();
        return false;
    }
    auto state = CostSchedulerGraphState::create(
        context, config, workerCount);
    if (!state) {
        decisions.clear();
        return false;
    }
    if (initialWeights) {
        auto expectedCount = context->requestCount() * context->nodeCount();
        if (initialWeights->size() != expectedCount) {
            decisions.clear();
            return false;
        }
        for (auto weight : *initialWeights) {
            if (!std::isfinite(weight)) {
                decisions.clear();
                return false;
            }
        }
        if (!state->resetWeights(initialWeights->data(),
                                 initialWeights->size())) {
            decisions.clear();
            return false;
        }
    }
    if (inspectInput) {
        inspectInput(*context);
    }
    for (size_t iteration = 0;
         iteration < static_cast<size_t>(config.maxLoopCount);
         ++iteration) {
        for (size_t workerIndex = 0; workerIndex < workerCount;
             ++workerIndex) {
            if (!state->runRow(workerIndex, iteration)) {
                decisions.clear();
                return false;
            }
        }
        if (!state->finishRow(iteration)) {
            decisions.clear();
            return false;
        }
        for (size_t workerIndex = 0; workerIndex < workerCount;
             ++workerIndex) {
            if (!state->runProjection(workerIndex, iteration)) {
                decisions.clear();
                return false;
            }
        }
        if (!state->finishProjection(iteration)) {
            decisions.clear();
            return false;
        }
    }
    for (size_t workerIndex = 0; workerIndex < workerCount;
         ++workerIndex) {
        if (!state->runFinalRow(workerIndex)) {
            decisions.clear();
            return false;
        }
    }
    // Replay the actual amount of discrete work, not a new wall-clock budget.
    if (!replayInfo.optimizationRuns.empty() &&
        replayInfo.optimizationRuns.front().refinementEvaluations >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        decisions.clear();
        return false;
    }
    const auto refinementEvaluations = replayInfo.optimizationRuns.empty()
        ? int64_t{-1}
        : static_cast<int64_t>(replayInfo.optimizationRuns.front().refinementEvaluations);
    if (!state->finalize(refinementEvaluations) || !collectDecisions(*context, decisions)) {
        decisions.clear();
        return false;
    }
    return true;
}

} // namespace navi_sched
