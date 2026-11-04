#include "rtp_llm/flexlb/navi_sched/resource/MasterLatencyModelR.h"

#include "rtp_llm/flexlb/navi_sched/data/SummaryUtil.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyResidualModel.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include "autil/CRC32C.h"
#include "autil/TimeUtility.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <tuple>

namespace navi_sched {
namespace {
constexpr int32_t STATE_VERSION = 5;
static_assert(NonLinearPrefillModelR::BINARY_VERSION == 24);
constexpr double ZERO_POINT_RELATIVE_TOLERANCE = 1e-9;
const std::string STATE_MAGIC = "navi_sched_master_latency_models";

bool zeroPointsEqual(double left, double right)
{
    return std::abs(left - right) <= ZERO_POINT_RELATIVE_TOLERANCE *
                                         std::max({1.0, std::abs(left), std::abs(right)});
}
} // namespace

const char *MasterLatencyModelR::getRejectReasonName(RejectReason reason)
{
    switch (reason) {
    case RejectReason::NONE:
        return "none";
    case RejectReason::INVALID_MODEL:
        return "invalid_model";
    case RejectReason::SOURCE_MISMATCH:
        return "source_mismatch";
    case RejectReason::TRUSTED_REGRESSION:
        return "trusted_regression";
    case RejectReason::STALE_REVISION:
        return "stale_revision";
    }
    return "unknown";
}

const std::string MasterLatencyModelR::RESOURCE_ID =
    "navi_sched.master_latency_model_r";

MasterLatencyModelR::~MasterLatencyModelR()
{
    _globalZeroPointThread.reset();
    save();
}

bool MasterLatencyModelR::ModelKey::operator<(const ModelKey &other) const
{
    return std::tie(fingerprint, formula, binaryVersion, parameterCount) <
           std::tie(other.fingerprint, other.formula, other.binaryVersion,
                    other.parameterCount);
}

bool MasterLatencyModelR::WarmupKey::operator<(const WarmupKey &other) const
{
    return std::tie(fingerprint, formula, parameterCount) <
           std::tie(other.fingerprint, other.formula, other.parameterCount);
}

bool MasterLatencyModelR::ModelSignature::operator<(
    const ModelSignature &other) const
{
    return std::tie(formula, binaryVersion, parameterCount) <
           std::tie(other.formula, other.binaryVersion, other.parameterCount);
}

void MasterLatencyModelR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

bool MasterLatencyModelR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "state_file", _stateFile, _stateFile);
    NAVI_JSONIZE(ctx, "trust_min_revision", _trustMinRevision,
                 _trustMinRevision);
    NAVI_JSONIZE(ctx, "source_min_revision", _sourceMinRevision,
                 _sourceMinRevision);
    NAVI_JSONIZE(ctx, "max_candidates", _maxCandidates, _maxCandidates);
    NAVI_JSONIZE(ctx, "persist_interval_ms", _persistIntervalMs,
                 _persistIntervalMs);
    NAVI_JSONIZE(ctx, "global_zero_point_refresh_interval_ms",
                 _globalZeroPointRefreshIntervalMs, _globalZeroPointRefreshIntervalMs);
    return _trustMinRevision > 1 && _sourceMinRevision > 1 &&
           _maxCandidates > 0 && _persistIntervalMs > 0 &&
           _globalZeroPointRefreshIntervalMs > 0;
}

navi::ErrorCode MasterLatencyModelR::init(navi::ResourceInitContext &)
{
    {
        autil::ScopedLock lock(_lock);
        _masterEpoch = std::max<int64_t>(1, autil::TimeUtility::currentTime());
        std::atomic_store_explicit(
            &_liveEntries, std::make_shared<const LiveEntryList>(),
            std::memory_order_release);
        std::atomic_store_explicit(&_globalZeroPoint,
                                   std::make_shared<const GlobalZeroPointMap>(),
                                   std::memory_order_release);
        if (!loadLocked()) {
            _models.clear();
            _dirty = true;
            ++_stats.persistFailureCount;
        }
    }
    _globalZeroPointThread = autil::LoopThread::createLoopThread(
        [this]() { refreshGlobalZeroPoint(); }, _globalZeroPointRefreshIntervalMs * 1000,
        "navi_zero_point");
    if (!_globalZeroPointThread) {
        return navi::EC_INIT_RESOURCE;
    }
    return navi::EC_NONE;
}

bool MasterLatencyModelR::validFingerprint(const std::string &fingerprint)
{
    return !fingerprint.empty() && fingerprint.size() <= 4096;
}

bool MasterLatencyModelR::activateAgent(const std::string &agentId,
                                         const std::string &sourceId)
{
    if (agentId.empty() || sourceId.empty()) {
        return false;
    }
    bool replaced = false;
    {
        autil::ScopedLock lock(_lock);
        auto it = _liveModels.find(agentId);
        if (it != _liveModels.end() && it->second->sourceId == sourceId) {
            return true;
        }
        if (it != _liveModels.end()) {
            it->second->active.store(false, std::memory_order_release);
            std::atomic_store_explicit(
                &it->second->model,
                std::shared_ptr<const LiveModel>(),
                std::memory_order_release);
            removeWarmupSubscriberLocked(it->second->sourceId);
            removeLiveCandidateLocked(agentId);
            replaced = true;
        }
        auto entry = std::make_shared<LiveEntry>();
        entry->sourceId = sourceId;
        _liveModels[agentId] = std::move(entry);
        publishLiveEntriesLocked();
    }
    if (replaced) {
        refreshGlobalZeroPoint();
    }
    return true;
}

MasterLatencyModelR::LiveModelSlotPtr MasterLatencyModelR::getLiveModelSlot(
    const std::string &agentId, const std::string &sourceId) const
{
    if (agentId.empty()) {
        return nullptr;
    }
    autil::ScopedLock lock(_lock);
    auto it = _liveModels.find(agentId);
    if (it == _liveModels.end() ||
        (!sourceId.empty() && it->second->sourceId != sourceId)) {
        return nullptr;
    }
    return LiveModelSlotPtr(it->second,
                            static_cast<LiveModelSlot *>(it->second.get()));
}

bool MasterLatencyModelR::updateWarmupSubscriber(
    const std::string &agentId, const std::string &sourceId,
    const std::string &fingerprint, const AgentLatencyToMasterData &latency,
    bool trusted, const navi::AsyncPipePtr &notificationPipe)
{
    if (sourceId.empty()) {
        return false;
    }
    autil::ScopedLock lock(_lock);
    removeWarmupSubscriberLocked(sourceId);
    if (trusted) {
        return true;
    }
    if (agentId.empty() || !validFingerprint(fingerprint) || !notificationPipe ||
        latency.formula != NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL ||
        latency.binaryVersion != NonLinearPrefillModelR::BINARY_VERSION ||
        latency.parameters.empty() ||
        latency.parameters.size() > std::numeric_limits<uint32_t>::max() ||
        !NonLinearPrefillModelR::validateParameters(latency.parameters) ||
        !latency.optimizerState.valid(latency.parameters.size()) ||
        latency.residualVersion != LatencyResidualModel::VERSION ||
        !LatencyResidualModel::validateParameters(latency.residualParameters) ||
        latency.residualLastUpdateTimeUs < 0) {
        return false;
    }
    WarmupSubscriber subscriber;
    subscriber.agentId = agentId;
    subscriber.notificationPipe = notificationPipe;
    WarmupKey key{fingerprint, latency.formula,
                  static_cast<uint32_t>(latency.parameters.size())};
    _warmupGroups[key].subscribers[sourceId] = std::move(subscriber);
    _warmupSubscriberKeys[sourceId] = std::move(key);
    return true;
}

void MasterLatencyModelR::removeWarmupSubscriber(const std::string &sourceId)
{
    if (sourceId.empty()) {
        return;
    }
    autil::ScopedLock lock(_lock);
    removeWarmupSubscriberLocked(sourceId);
}

void MasterLatencyModelR::removeWarmupSubscriberLocked(
    const std::string &sourceId)
{
    auto keyIt = _warmupSubscriberKeys.find(sourceId);
    if (keyIt == _warmupSubscriberKeys.end()) {
        return;
    }
    auto groupIt = _warmupGroups.find(keyIt->second);
    if (groupIt != _warmupGroups.end()) {
        groupIt->second.subscribers.erase(sourceId);
        if (groupIt->second.subscribers.empty() && groupIt->second.version == 0) {
            _warmupGroups.erase(groupIt);
        }
    }
    _warmupSubscriberKeys.erase(keyIt);
}

void MasterLatencyModelR::removeLiveCandidateLocked(
    const std::string &agentId)
{
    auto keyIt = _liveCandidateKeys.find(agentId);
    if (keyIt == _liveCandidateKeys.end()) {
        return;
    }
    auto candidatesIt = _liveCandidates.find(keyIt->second);
    if (candidatesIt != _liveCandidates.end()) {
        candidatesIt->second.erase(agentId);
        if (candidatesIt->second.empty()) {
            _liveCandidates.erase(candidatesIt);
        }
    }
    _liveCandidateKeys.erase(keyIt);
}

void MasterLatencyModelR::pruneLiveCandidatesLocked(
    const ModelKey &key, CandidateMap &candidates)
{
    while (candidates.size() > _maxCandidates) {
        auto oldest = std::min_element(
            candidates.begin(), candidates.end(), [](const auto &left,
                                                      const auto &right) {
                return std::tie(left.second.sourceRevision,
                                left.second.eventTimeUs) <
                       std::tie(right.second.sourceRevision,
                                right.second.eventTimeUs);
            });
        _liveCandidateKeys.erase(oldest->first);
        candidates.erase(oldest);
    }
    for (const auto &[agentId, candidate] : candidates) {
        (void)candidate;
        _liveCandidateKeys[agentId] = key;
    }
}

void MasterLatencyModelR::markRejectedModelLocked(
    const std::string &agentId, const std::shared_ptr<LiveEntry> &entry,
    RejectReason reason,
    RecordResult *recordResult)
{
    if (!entry) {
        return;
    }
    auto current = std::atomic_load_explicit(&entry->model,
                                             std::memory_order_acquire);
    if (!current) {
        return;
    }
    entry->consecutiveRejectedCount = std::min<uint32_t>(10,
                                                          entry->consecutiveRejectedCount + 1);
    auto rejected = std::make_shared<LiveModel>(*current);
    rejected->consecutiveRejectedCount = entry->consecutiveRejectedCount;
    rejected->lastRejectReason = getRejectReasonName(reason);
    if (rejected->consecutiveRejectedCount >= 10) {
        rejected->overloadPredictionTrusted = false;
        removeLiveCandidateLocked(agentId);
    }
    const auto forcedUntrusted = current->overloadPredictionTrusted &&
                                 !rejected->overloadPredictionTrusted;
    std::atomic_store_explicit(
        &entry->model, std::static_pointer_cast<const LiveModel>(rejected),
        std::memory_order_release);
    if (recordResult) {
        recordResult->forcedUntrusted = forcedUntrusted;
        recordResult->consecutiveRejectedCount =
            rejected->consecutiveRejectedCount;
        recordResult->model = rejected;
    }
}

bool MasterLatencyModelR::record(const std::string &agentId,
                                 const std::string &fingerprint,
                                 const AgentLatencyToMasterData &latency,
                                 bool trusted, int64_t eventTimeUs,
                                 const std::string &sourceId,
                                 bool *liveUpdateAccepted, bool warmStarted,
                                 RecordResult *recordResult, bool rampFinished)
{
    const auto revisionEligible = latency.modelRevision >= _trustMinRevision;
    const auto effectiveTrusted = trusted && rampFinished && revisionEligible;
    if (liveUpdateAccepted) {
        *liveUpdateAccepted = false;
    }
    if (recordResult) {
        *recordResult = RecordResult();
    }
    const bool parametersValid =
        !agentId.empty() && latency.modelRevision > 0 &&
        latency.formula == NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL &&
        latency.binaryVersion == NonLinearPrefillModelR::BINARY_VERSION &&
        NonLinearPrefillModelR::validateParameters(latency.parameters) &&
        latency.optimizerState.valid(latency.parameters.size()) &&
        latency.residualVersion == LatencyResidualModel::VERSION &&
        LatencyResidualModel::validateParameters(latency.residualParameters) &&
        latency.residualLastUpdateTimeUs >= 0;
    const auto calculatedZeroPoint =
        parametersValid ? NonLinearPrefillModelR::calculateZeroPoint(latency.parameters)
                        : 0.0;
    const bool latencyValid = parametersValid && std::isfinite(latency.zeroPoint) &&
                              latency.zeroPoint >= 0.0 &&
                              zeroPointsEqual(latency.zeroPoint, calculatedZeroPoint);
    std::vector<navi::AsyncPipePtr> notificationPipes;
    bool result = false;
    {
        autil::ScopedLock lock(_lock);
        if (!latencyValid) {
            ++_stats.rejectedUpdateCount;
            if (recordResult) {
                recordResult->rejectReason = RejectReason::INVALID_MODEL;
            }
            auto liveIt = _liveModels.find(agentId);
            if (liveIt != _liveModels.end() &&
                (sourceId.empty() || liveIt->second->sourceId == sourceId)) {
                markRejectedModelLocked(agentId, liveIt->second,
                                         RejectReason::INVALID_MODEL,
                                         recordResult);
            }
            return false;
        }
        ModelSignature modelSignature{
            latency.formula, latency.binaryVersion,
            static_cast<uint32_t>(latency.parameters.size())};
        ModelKey modelKey{fingerprint, latency.formula, latency.binaryVersion,
                          static_cast<uint32_t>(latency.parameters.size())};
        WarmupKey warmupKey{fingerprint, latency.formula,
                            static_cast<uint32_t>(latency.parameters.size())};
        auto liveIt = _liveModels.find(agentId);
        if (liveIt != _liveModels.end() && liveIt->second->sourceId != sourceId) {
            ++_stats.rejectedUpdateCount;
            if (recordResult) {
                recordResult->rejectReason = RejectReason::SOURCE_MISMATCH;
                recordResult->consecutiveRejectedCount =
                    liveIt->second->consecutiveRejectedCount;
                recordResult->model = std::atomic_load_explicit(
                    &liveIt->second->model, std::memory_order_acquire);
            }
            return false;
        }
        auto current = liveIt == _liveModels.end()
                           ? std::shared_ptr<const LiveModel>()
                           : std::atomic_load_explicit(
                                 &liveIt->second->model, std::memory_order_acquire);
        if (current && current->overloadPredictionTrusted &&
            current->revision < _trustMinRevision) {
            // Normalize a model published by an older process before applying
            // the new gate; otherwise an invalid trusted window can persist.
            auto downgraded = std::make_shared<LiveModel>(*current);
            downgraded->overloadPredictionTrusted = false;
            downgraded->lastRejectReason = "revision_below_trust_minimum";
            std::atomic_store_explicit(
                &liveIt->second->model,
                std::static_pointer_cast<const LiveModel>(downgraded),
                std::memory_order_release);
            removeLiveCandidateLocked(agentId);
            current = std::move(downgraded);
        }
        const auto sameKey = current && !(current->modelKey < modelKey) &&
                             !(modelKey < current->modelKey);
        const auto trustedRegression =
            sameKey && current && current->overloadPredictionTrusted &&
            !effectiveTrusted;
        const auto sameRevision =
            sameKey && current && current->revision == latency.modelRevision;
        const auto sameParameters =
            sameRevision && current->parameters == latency.parameters &&
            current->optimizerState.adamMoment1 == latency.optimizerState.adamMoment1 &&
            current->optimizerState.adamMoment2 == latency.optimizerState.adamMoment2 &&
            current->optimizerState.step == latency.optimizerState.step &&
            current->residualVersion == latency.residualVersion &&
            current->residualParameters == latency.residualParameters &&
            current->residualLastUpdateTimeUs == latency.residualLastUpdateTimeUs &&
            current->overloadPredictionTrusted == effectiveTrusted &&
            current->warmStarted == warmStarted;
        const auto duplicate =
            sameParameters && current->sessionSampleCount == latency.sessionSampleCount;
        const auto sessionProgress =
            sameParameters && current->sessionSampleCount < latency.sessionSampleCount;
        const auto hasFreshModel =
            !current || latency.modelRevision > current->revision ||
            latency.sessionSampleCount > current->sessionSampleCount ||
            latency.parameters != current->parameters ||
            latency.optimizerState.adamMoment1 !=
                current->optimizerState.adamMoment1 ||
            latency.optimizerState.adamMoment2 !=
                current->optimizerState.adamMoment2 ||
            latency.optimizerState.step != current->optimizerState.step ||
            latency.residualParameters != current->residualParameters ||
            latency.residualLastUpdateTimeUs != current->residualLastUpdateTimeUs;
        const auto forcedUntrusted =
            current && current->consecutiveRejectedCount >= 10;
        const auto trustedPromotion =
            sameKey && current && !current->overloadPredictionTrusted &&
            effectiveTrusted &&
            (!forcedUntrusted || hasFreshModel);
        if (duplicate) {
            liveIt->second->consecutiveRejectedCount = 0;
            auto acceptedModel = current;
            if (current->consecutiveRejectedCount != 0 ||
                !current->lastRejectReason.empty()) {
                auto cleared = std::make_shared<LiveModel>(*current);
                cleared->consecutiveRejectedCount = 0;
                cleared->lastRejectReason.clear();
                acceptedModel = cleared;
                std::atomic_store_explicit(
                    &liveIt->second->model,
                    std::static_pointer_cast<const LiveModel>(cleared),
                    std::memory_order_release);
            }
            if (liveUpdateAccepted) {
                *liveUpdateAccepted = true;
            }
            if (recordResult) {
                recordResult->accepted = true;
                recordResult->model = acceptedModel;
            }
            ++_stats.acceptedUpdateCount;
            return false;
        }
        const auto staleRevision =
            sameKey && !trustedPromotion &&
            (current->revision > latency.modelRevision ||
             (sameRevision && !sessionProgress));
        const auto rampTrustDowngrade = trustedRegression && !rampFinished;
        if (rampTrustDowngrade && staleRevision) {
            // A reconnect can report an older untrusted snapshot while ramping.
            // Publish the trust downgrade, but never roll back the live model.
            auto downgraded = std::make_shared<LiveModel>(*current);
            downgraded->overloadPredictionTrusted = false;
            downgraded->eventTimeUs = eventTimeUs;
            downgraded->sessionSampleCount =
                std::max(current->sessionSampleCount, latency.sessionSampleCount);
            std::atomic_store_explicit(
                &liveIt->second->model,
                std::static_pointer_cast<const LiveModel>(downgraded),
                std::memory_order_release);
            removeLiveCandidateLocked(agentId);
            liveIt->second->consecutiveRejectedCount = 0;
            if (liveUpdateAccepted) {
                *liveUpdateAccepted = true;
            }
            if (recordResult) {
                recordResult->accepted = true;
                recordResult->model = downgraded;
            }
            ++_stats.acceptedUpdateCount;
            return false;
        }
        if (trustedRegression || staleRevision) {
            ++_stats.rejectedUpdateCount;
            auto &entry = liveIt->second;
            const auto rejectReason = trustedRegression
                                          ? RejectReason::TRUSTED_REGRESSION
                                          : RejectReason::STALE_REVISION;
            if (recordResult) {
                recordResult->rejectReason = rejectReason;
            }
            markRejectedModelLocked(agentId, entry, rejectReason, recordResult);
            return false;
        }
        const auto currentCandidateEligible =
            sameKey && current && current->overloadPredictionTrusted &&
            current->revision >= _sourceMinRevision;
        const auto candidateEligible =
            effectiveTrusted && latency.modelRevision >= _sourceMinRevision;
        const auto becameCandidateEligible =
            candidateEligible && !currentCandidateEligible;
        auto entry = liveIt == _liveModels.end() ? std::make_shared<LiveEntry>()
                                                 : liveIt->second;
        if (liveIt == _liveModels.end()) {
            entry->sourceId = sourceId;
            _liveModels[agentId] = entry;
            publishLiveEntriesLocked();
        }
        entry->consecutiveRejectedCount = 0;
        auto model = std::make_shared<LiveModel>();
        model->signature = modelSignature;
        model->modelKey = modelKey;
        model->formula = latency.formula;
        model->binaryVersion = latency.binaryVersion;
        model->revision = latency.modelRevision;
        model->eventTimeUs = eventTimeUs;
        model->zeroPoint = latency.zeroPoint;
        model->overloadPredictionTrusted = effectiveTrusted;
        model->warmStarted = warmStarted;
        model->sessionSampleCount = latency.sessionSampleCount;
        model->parameters = latency.parameters;
        model->optimizerState = latency.optimizerState;
        model->residualVersion = latency.residualVersion;
        model->residualParameters = latency.residualParameters;
        model->residualLastUpdateTimeUs = latency.residualLastUpdateTimeUs;
        std::atomic_store_explicit(
            &entry->model, std::static_pointer_cast<const LiveModel>(model),
            std::memory_order_release);
        if (liveUpdateAccepted) {
            *liveUpdateAccepted = true;
        }
        if (recordResult) {
            recordResult->accepted = true;
            recordResult->model = model;
        }

        if (effectiveTrusted) {
            removeWarmupSubscriberLocked(sourceId);
        }

        bool uniqueWarmupCandidate = true;
        const auto inspectCandidates = [&](const auto &candidateModels) {
            for (const auto &[key, candidates] : candidateModels) {
                if (key.fingerprint != fingerprint || key.formula != latency.formula ||
                    key.parameterCount != latency.parameters.size()) {
                    continue;
                }
                if (std::any_of(candidates.begin(), candidates.end(),
                                [&](const auto &candidatePair) {
                                    return candidatePair.second.parameters ==
                                               latency.parameters &&
                                           candidatePair.second.residualParameters ==
                                               latency.residualParameters;
                                })) {
                    uniqueWarmupCandidate = false;
                    return;
                }
            }
        };
        if (becameCandidateEligible) {
            inspectCandidates(_models);
            if (uniqueWarmupCandidate) {
                inspectCandidates(_liveCandidates);
            }
        }
        if (candidateEligible) {
            removeLiveCandidateLocked(agentId);
            auto &liveCandidates = _liveCandidates[modelKey];
            auto &candidate = liveCandidates[agentId];
            candidate.agentId = agentId;
            candidate.sourceRevision = latency.modelRevision;
            candidate.eventTimeUs = eventTimeUs;
            candidate.parameters = latency.parameters;
            candidate.optimizerState = latency.optimizerState;
            candidate.residualVersion = latency.residualVersion;
            candidate.residualParameters = latency.residualParameters;
            candidate.residualLastUpdateTimeUs = latency.residualLastUpdateTimeUs;
            pruneLiveCandidatesLocked(modelKey, liveCandidates);
            if (becameCandidateEligible && uniqueWarmupCandidate &&
                liveCandidates.find(agentId) != liveCandidates.end()) {
                auto &group = _warmupGroups[warmupKey];
                group.version = _nextWarmupVersion++;
                for (auto it = group.subscribers.begin();
                     it != group.subscribers.end();) {
                    auto pipe = it->second.notificationPipe.lock();
                    if (!pipe) {
                        _warmupSubscriberKeys.erase(it->first);
                        it = group.subscribers.erase(it);
                        continue;
                    }
                    if (it->second.agentId != agentId) {
                        notificationPipes.emplace_back(std::move(pipe));
                    }
                    ++it;
                }
            }
        } else {
            removeLiveCandidateLocked(agentId);
        }

        const bool persistentValid =
            validFingerprint(fingerprint) && effectiveTrusted &&
            latency.modelRevision >= _sourceMinRevision;
        if (!persistentValid) {
            ++_stats.acceptedUpdateCount;
            // The live model update succeeded even when it is not yet eligible
            // for cross-node persistence. Do not report that normal untrusted
            // state as a failed record operation.
            result = true;
        } else {
            auto &candidates = _models[modelKey];
            auto &candidate = candidates[agentId];
            candidate.agentId = agentId;
            candidate.sourceRevision = latency.modelRevision;
            candidate.eventTimeUs = eventTimeUs;
            candidate.parameters = latency.parameters;
            candidate.optimizerState = latency.optimizerState;
            candidate.residualVersion = latency.residualVersion;
            candidate.residualParameters = latency.residualParameters;
            candidate.residualLastUpdateTimeUs = latency.residualLastUpdateTimeUs;
            pruneLocked(candidates);
            ++_stats.acceptedUpdateCount;
            _dirty = true;
            const auto currentTimeUs = autil::TimeUtility::currentTime();
            result = _stateFile.empty() ||
                     currentTimeUs - _lastPersistTimeUs <
                         _persistIntervalMs * 1000 ||
                     saveLocked();
        }
    }
    notifyWarmupSubscribers(notificationPipes);
    return result;
}

void MasterLatencyModelR::removeAgent(const std::string &agentId,
                                      const std::string &sourceId)
{
    if (agentId.empty()) {
        return;
    }
    bool removed = false;
    {
        autil::ScopedLock lock(_lock);
        auto it = _liveModels.find(agentId);
        if (it == _liveModels.end() ||
            (!sourceId.empty() && it->second->sourceId != sourceId)) {
            return;
        }
        removeWarmupSubscriberLocked(it->second->sourceId);
        removeLiveCandidateLocked(agentId);
        it->second->active.store(false, std::memory_order_release);
        std::atomic_store_explicit(
            &it->second->model,
            std::shared_ptr<const LiveModel>(),
            std::memory_order_release);
        _liveModels.erase(it);
        publishLiveEntriesLocked();
        removed = true;
    }
    if (removed) {
        refreshGlobalZeroPoint();
    }
}

LatencyModelGlobalData MasterLatencyModelR::getGlobalZeroPoint(
    int32_t formula, int64_t binaryVersion, size_t parameterCount) const
{
    if (parameterCount > std::numeric_limits<uint32_t>::max()) {
        return {};
    }
    auto targets =
        std::atomic_load_explicit(&_globalZeroPoint, std::memory_order_acquire);
    if (!targets) {
        return {};
    }
    ModelSignature key{formula, binaryVersion,
                       static_cast<uint32_t>(parameterCount)};
    auto it = targets->find(key);
    return it == targets->end() ? LatencyModelGlobalData() : it->second;
}

MasterLatencyModelR::GlobalZeroPointDiagnostics
MasterLatencyModelR::getGlobalZeroPointDiagnostics(
    int32_t formula, int64_t binaryVersion, size_t parameterCount) const
{
    GlobalZeroPointDiagnostics diagnostics;
    if (parameterCount > std::numeric_limits<uint32_t>::max()) {
        diagnostics.invalidReason =
            GlobalZeroPointInvalidReason::PARAMETER_COUNT_INVALID;
        return diagnostics;
    }
    const ModelSignature key{formula, binaryVersion,
                             static_cast<uint32_t>(parameterCount)};
    auto entries =
        std::atomic_load_explicit(&_liveEntries, std::memory_order_acquire);
    if (entries) {
        for (const auto &entry : *entries) {
            if (!entry || !entry->active.load(std::memory_order_acquire)) {
                continue;
            }
            auto model = std::atomic_load_explicit(
                &entry->model, std::memory_order_acquire);
            if (!model || !entry->active.load(std::memory_order_acquire) ||
                model->signature < key || key < model->signature) {
                continue;
            }
            ++diagnostics.groupLiveNodeCount;
            if (model->overloadPredictionTrusted) {
                ++diagnostics.groupTrustedLiveNodeCount;
            }
        }
    }
    auto targets =
        std::atomic_load_explicit(&_globalZeroPoint, std::memory_order_acquire);
    if (!targets) {
        return diagnostics;
    }
    auto it = targets->find(key);
    if (it == targets->end()) {
        diagnostics.invalidReason =
            GlobalZeroPointInvalidReason::GROUP_NOT_FOUND;
        return diagnostics;
    }
    diagnostics.data = it->second;
    const auto &data = diagnostics.data;
    if (data.masterEpoch <= 0) {
        diagnostics.invalidReason =
            GlobalZeroPointInvalidReason::MASTER_EPOCH_NOT_SET;
    } else if (data.version == 0) {
        diagnostics.invalidReason = GlobalZeroPointInvalidReason::VERSION_NOT_SET;
    } else if (data.formula != formula) {
        diagnostics.invalidReason = GlobalZeroPointInvalidReason::FORMULA_MISMATCH;
    } else if (data.binaryVersion <= 0 || data.binaryVersion != binaryVersion) {
        diagnostics.invalidReason =
            GlobalZeroPointInvalidReason::BINARY_VERSION_INVALID;
    } else if (data.parameterCount == 0 || data.parameterCount != parameterCount) {
        diagnostics.invalidReason =
            GlobalZeroPointInvalidReason::PARAMETER_COUNT_INVALID;
    } else if (!std::isfinite(data.globalZeroPoint)) {
        diagnostics.invalidReason =
            GlobalZeroPointInvalidReason::ZERO_POINT_NON_FINITE;
    } else if (data.globalZeroPoint < 0.0) {
        diagnostics.invalidReason =
            GlobalZeroPointInvalidReason::ZERO_POINT_NEGATIVE;
    } else {
        diagnostics.invalidReason = GlobalZeroPointInvalidReason::NONE;
    }
    return diagnostics;
}

void MasterLatencyModelR::publishLiveEntriesLocked()
{
    auto entries = std::make_shared<LiveEntryList>();
    entries->reserve(_liveModels.size());
    for (const auto &[agentId, entry] : _liveModels) {
        (void)agentId;
        entries->emplace_back(entry);
    }
    std::shared_ptr<const LiveEntryList> published = std::move(entries);
    std::atomic_store_explicit(&_liveEntries, std::move(published),
                               std::memory_order_release);
}

void MasterLatencyModelR::refreshGlobalZeroPoint()
{
    struct Accumulator {
        double stableSum = 0.0;
        size_t stableCount = 0;
        double coldWeightedSum = 0.0;
        double coldWeight = 0.0;
    };
    std::map<ModelSignature, Accumulator> accumulators;
    auto entries =
        std::atomic_load_explicit(&_liveEntries, std::memory_order_acquire);
    if (entries) {
        for (const auto &entry : *entries) {
            if (!entry || !entry->active.load(std::memory_order_acquire)) {
                continue;
            }
            auto model = std::atomic_load_explicit(
                &entry->model, std::memory_order_acquire);
            if (!model || !entry->active.load(std::memory_order_acquire)) {
                continue;
            }
            auto &accumulator = accumulators[model->signature];
            if (model->overloadPredictionTrusted) {
                accumulator.stableSum += model->zeroPoint;
                ++accumulator.stableCount;
                continue;
            }
            const auto weight = std::min(
                1.0, static_cast<double>(model->revision) /
                         _sourceMinRevision);
            accumulator.coldWeightedSum += weight * model->zeroPoint;
            accumulator.coldWeight += weight;
        }
    }
    auto previous =
        std::atomic_load_explicit(&_globalZeroPoint, std::memory_order_acquire);
    auto updated = std::make_shared<GlobalZeroPointMap>();
    for (const auto &[key, accumulator] : accumulators) {
        const auto hasStable = accumulator.stableCount > 0;
        if (!hasStable && accumulator.coldWeight <= 0.0) {
            continue;
        }
        const auto zeroPoint = hasStable
                                   ? accumulator.stableSum / accumulator.stableCount
                                   : accumulator.coldWeightedSum / accumulator.coldWeight;
        if (!std::isfinite(zeroPoint) || zeroPoint < 0.0) {
            continue;
        }
        LatencyModelGlobalData target;
        target.masterEpoch = _masterEpoch;
        target.formula = key.formula;
        target.binaryVersion = key.binaryVersion;
        target.parameterCount = key.parameterCount;
        target.globalZeroPoint = zeroPoint;
        auto old = previous ? previous->find(key) : GlobalZeroPointMap::const_iterator();
        if (previous && old != previous->end() &&
            old->second.globalZeroPoint == target.globalZeroPoint) {
            target.version = old->second.version;
        } else {
            target.version =
                _nextGlobalZeroPointVersion.fetch_add(1, std::memory_order_relaxed);
        }
        updated->emplace(key, target);
    }
    std::shared_ptr<const GlobalZeroPointMap> published = std::move(updated);
    std::atomic_store_explicit(&_globalZeroPoint, std::move(published),
                               std::memory_order_release);
    _globalZeroPointRefreshCount.fetch_add(1, std::memory_order_relaxed);
}

void MasterLatencyModelR::pruneLocked(CandidateMap &candidates)
{
    while (candidates.size() > _maxCandidates) {
        auto oldest = std::min_element(
            candidates.begin(), candidates.end(), [](const auto &left,
                                                      const auto &right) {
                return std::tie(left.second.sourceRevision,
                                left.second.eventTimeUs) <
                       std::tie(right.second.sourceRevision,
                                right.second.eventTimeUs);
            });
        candidates.erase(oldest);
    }
}

MasterLatencyModelWarmupToAgentDataPtr MasterLatencyModelR::makeWarmup(
    const std::string &fingerprint, int32_t formula, int64_t binaryVersion,
    size_t parameterCount)
{
    autil::ScopedLock lock(_lock);
    uint64_t candidateVersion = 0;
    return makeWarmupLocked(fingerprint, formula, binaryVersion, parameterCount,
                            0, candidateVersion);
}

MasterLatencyModelWarmupToAgentDataPtr MasterLatencyModelR::makeWarmupIfNew(
    const std::string &fingerprint, int32_t formula, int64_t binaryVersion,
    size_t parameterCount, uint64_t lastCandidateVersion,
    uint64_t &candidateVersion)
{
    autil::ScopedLock lock(_lock);
    return makeWarmupLocked(fingerprint, formula, binaryVersion, parameterCount,
                            lastCandidateVersion, candidateVersion);
}

uint64_t MasterLatencyModelR::getWarmupCandidateVersion(
    const std::string &fingerprint, int32_t formula,
    size_t parameterCount) const
{
    if (!validFingerprint(fingerprint) ||
        parameterCount > std::numeric_limits<uint32_t>::max()) {
        return 0;
    }
    autil::ScopedLock lock(_lock);
    const WarmupKey key{fingerprint, formula,
                        static_cast<uint32_t>(parameterCount)};
    auto it = _warmupGroups.find(key);
    return it == _warmupGroups.end() ? 0 : it->second.version;
}

MasterLatencyModelWarmupToAgentDataPtr MasterLatencyModelR::makeWarmupLocked(
    const std::string &fingerprint, int32_t formula, int64_t binaryVersion,
    size_t parameterCount, uint64_t lastCandidateVersion,
    uint64_t &candidateVersion)
{
    candidateVersion = 0;
    if (!validFingerprint(fingerprint) ||
        binaryVersion != NonLinearPrefillModelR::BINARY_VERSION ||
        parameterCount > std::numeric_limits<uint32_t>::max()) {
        return nullptr;
    }
    const WarmupKey warmupKey{fingerprint, formula,
                              static_cast<uint32_t>(parameterCount)};
    auto groupIt = _warmupGroups.find(warmupKey);
    if (groupIt == _warmupGroups.end()) {
        return nullptr;
    }
    candidateVersion = groupIt->second.version;
    if (candidateVersion == 0 || candidateVersion <= lastCandidateVersion) {
        return nullptr;
    }
    struct CandidateRef {
        const StoredCandidate *candidate = nullptr;
    };
    std::vector<CandidateRef> ordered;
    const auto collectCandidates = [&](const auto &candidateModels) {
        for (const auto &[key, candidates] : candidateModels) {
            if (key.fingerprint != fingerprint || key.formula != formula ||
                key.binaryVersion != binaryVersion ||
                key.parameterCount != parameterCount) {
                continue;
            }
            for (const auto &[agentId, candidate] : candidates) {
                (void)agentId;
                ordered.push_back({&candidate});
            }
        }
    };
    collectCandidates(_models);
    collectCandidates(_liveCandidates);
    if (ordered.empty()) {
        return nullptr;
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto &left,
                                                 const auto &right) {
        return std::tie(left.candidate->sourceRevision,
                        left.candidate->eventTimeUs) >
               std::tie(right.candidate->sourceRevision,
                        right.candidate->eventTimeUs);
    });
    auto warmup = std::make_shared<MasterLatencyModelWarmupToAgentData>();
    warmup->fingerprint = fingerprint;
    warmup->formula = formula;
    warmup->binaryVersion = binaryVersion;
    for (const auto &ref : ordered) {
        const auto *candidate = ref.candidate;
        if (std::any_of(warmup->candidates.begin(), warmup->candidates.end(),
                        [candidate](const auto &current) {
                            return current.parameters == candidate->parameters &&
                                   current.residualParameters ==
                                       candidate->residualParameters;
                        })) {
            continue;
        }
        warmup->candidates.push_back({candidate->sourceRevision,
                                      candidate->parameters,
                                      candidate->optimizerState,
                                      candidate->residualVersion,
                                      candidate->residualParameters,
                                      candidate->residualLastUpdateTimeUs});
        if (warmup->candidates.size() >= _maxCandidates) {
            break;
        }
    }
    if (warmup->candidates.empty()) {
        return nullptr;
    }
    ++_stats.warmupResponseCount;
    return warmup;
}

void MasterLatencyModelR::notifyWarmupSubscribers(
    const std::vector<navi::AsyncPipePtr> &notificationPipes)
{
    uint64_t successCount = 0;
    uint64_t failureCount = 0;
    for (const auto &pipe : notificationPipes) {
        if (pipe && navi::EC_NONE ==
                        pipe->setDataOverride(nullptr, true, true)) {
            ++successCount;
        } else {
            ++failureCount;
        }
    }
    if (successCount == 0 && failureCount == 0) {
        return;
    }
    autil::ScopedLock lock(_lock);
    _stats.warmupNotificationCount += successCount;
    _stats.warmupNotificationFailureCount += failureCount;
}

bool MasterLatencyModelR::save()
{
    autil::ScopedLock lock(_lock);
    return saveLocked();
}

bool MasterLatencyModelR::saveLocked()
{
    if (!_dirty || _stateFile.empty()) {
        return true;
    }
    autil::DataBuffer payloadBuffer;
    payloadBuffer.write(static_cast<uint32_t>(_models.size()));
    for (const auto &[key, candidates] : _models) {
        payloadBuffer.write(key.fingerprint);
        payloadBuffer.write(key.formula);
        payloadBuffer.write(key.binaryVersion);
        payloadBuffer.write(key.parameterCount);
        payloadBuffer.write(static_cast<uint32_t>(candidates.size()));
        for (const auto &[agentId, candidate] : candidates) {
            payloadBuffer.write(agentId);
            payloadBuffer.write(candidate.sourceRevision);
            payloadBuffer.write(candidate.eventTimeUs);
            payloadBuffer.write(candidate.parameters);
            payloadBuffer.write(candidate.optimizerState);
            payloadBuffer.write(candidate.residualVersion);
            payloadBuffer.write(candidate.residualParameters);
            payloadBuffer.write(candidate.residualLastUpdateTimeUs);
        }
    }
    const std::string payload(payloadBuffer.getData(),
                              payloadBuffer.getDataLen());
    const auto checksum = autil::CRC32C::Mask(
        autil::CRC32C::Value(payload.data(), payload.size()));
    autil::DataBuffer buffer;
    buffer.write(STATE_MAGIC);
    buffer.write(STATE_VERSION);
    buffer.write(checksum);
    buffer.write(payload);
    std::filesystem::path path(_stateFile);
    std::error_code error;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            ++_stats.persistFailureCount;
            return false;
        }
    }
    const auto temporary = _stateFile + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output.write(buffer.getData(), buffer.getDataLen())) {
            ++_stats.persistFailureCount;
            return false;
        }
    }
    std::filesystem::rename(temporary, _stateFile, error);
    if (error) {
        std::filesystem::remove(_stateFile, error);
        error.clear();
        std::filesystem::rename(temporary, _stateFile, error);
    }
    if (error) {
        ++_stats.persistFailureCount;
        return false;
    }
    _dirty = false;
    _lastPersistTimeUs = autil::TimeUtility::currentTime();
    ++_stats.persistSuccessCount;
    return true;
}

bool MasterLatencyModelR::loadLocked()
{
    if (_stateFile.empty()) {
        return true;
    }
    std::ifstream input(_stateFile, std::ios::binary);
    if (!input) {
        return true;
    }
    input.seekg(0, std::ios::end);
    const auto length = input.tellg();
    if (length <= 0 || length > 64 * 1024 * 1024) {
        return false;
    }
    input.seekg(0, std::ios::beg);
    std::string content(static_cast<size_t>(length), '\0');
    if (!input.read(content.data(), length)) {
        return false;
    }
    try {
        autil::DataBuffer buffer(content.data(), content.size());
        std::string magic;
        int32_t version = 0;
        buffer.read(magic);
        buffer.read(version);
        if (magic != STATE_MAGIC) {
            return false;
        }
        if (version != STATE_VERSION) {
            return true;
        }
        std::string payload;
        uint32_t expectedChecksum = 0;
        buffer.read(expectedChecksum);
        buffer.read(payload);
        const auto actualChecksum = autil::CRC32C::Mask(
            autil::CRC32C::Value(payload.data(), payload.size()));
        if (buffer.getDataLen() != 0 || actualChecksum != expectedChecksum) {
            return false;
        }
        autil::DataBuffer stateBuffer(payload.data(), payload.size());
        uint32_t modelCount = 0;
        stateBuffer.read(modelCount);
        if (modelCount > 4096) {
            return false;
        }
        std::map<ModelKey, CandidateMap> loaded;
        bool stateRepaired = false;
        for (uint32_t modelIndex = 0; modelIndex < modelCount; ++modelIndex) {
            ModelKey key;
            uint32_t candidateCount = 0;
            stateBuffer.read(key.fingerprint);
            stateBuffer.read(key.formula);
            stateBuffer.read(key.binaryVersion);
            stateBuffer.read(key.parameterCount);
            stateBuffer.read(candidateCount);
            if (!validFingerprint(key.fingerprint) || candidateCount > 1024) {
                return false;
            }
            const auto supportedParameterCount =
                key.parameterCount == NonLinearPrefillModelR::PARAMETER_COUNT ||
                key.parameterCount ==
                    NonLinearPrefillModelR::CAPACITY_SPLIT_PARAMETER_COUNT;
            const auto currentModelKey =
                key.formula == NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL &&
                key.binaryVersion == NonLinearPrefillModelR::BINARY_VERSION &&
                supportedParameterCount;
            if (!currentModelKey) {
                return true;
            }
            CandidateMap candidates;
            for (uint32_t candidateIndex = 0; candidateIndex < candidateCount;
                 ++candidateIndex) {
                StoredCandidate candidate;
                stateBuffer.read(candidate.agentId);
                stateBuffer.read(candidate.sourceRevision);
                stateBuffer.read(candidate.eventTimeUs);
                stateBuffer.read(candidate.parameters);
                stateBuffer.read(candidate.optimizerState);
                stateBuffer.read(candidate.residualVersion);
                stateBuffer.read(candidate.residualParameters);
                stateBuffer.read(candidate.residualLastUpdateTimeUs);
                if (candidate.agentId.empty() ||
                    candidate.sourceRevision < _sourceMinRevision ||
                    candidate.parameters.size() != key.parameterCount ||
                    !NonLinearPrefillModelR::validateParameters(
                        candidate.parameters) ||
                    !candidate.optimizerState.valid(key.parameterCount)) {
                    return false;
                }
                if (candidate.residualVersion != LatencyResidualModel::VERSION ||
                    !LatencyResidualModel::validateParameters(
                        candidate.residualParameters) ||
                    candidate.residualLastUpdateTimeUs < 0) {
                    candidate.residualVersion = LatencyResidualModel::VERSION;
                    candidate.residualParameters =
                        LatencyResidualModel::makeInitialParameters();
                    candidate.residualLastUpdateTimeUs = 0;
                    stateRepaired = true;
                }
                candidates[candidate.agentId] = std::move(candidate);
            }
            pruneLocked(candidates);
            loaded.emplace(std::move(key), std::move(candidates));
        }
        if (stateBuffer.getDataLen() != 0) {
            return false;
        }
        _models.swap(loaded);
        for (const auto &[key, candidates] : _models) {
            if (candidates.empty()) {
                continue;
            }
            WarmupKey warmupKey{key.fingerprint, key.formula,
                                key.parameterCount};
            auto &group = _warmupGroups[warmupKey];
            if (group.version == 0) {
                group.version = _nextWarmupVersion++;
            }
        }
        _dirty = _dirty || stateRepaired;
    } catch (const autil::DataBufferCorruptedException &) {
        return false;
    }
    _lastPersistTimeUs = autil::TimeUtility::currentTime();
    return true;
}

MasterLatencyModelR::Stats MasterLatencyModelR::getStats() const
{
    autil::ScopedLock lock(_lock);
    auto stats = _stats;
    stats.globalZeroPointLiveNodeCount = _liveModels.size();
    stats.warmupSubscriberCount = _warmupSubscriberKeys.size();
    stats.globalZeroPointRefreshCount =
        _globalZeroPointRefreshCount.load(std::memory_order_relaxed);
    auto globalZeroPoints =
        std::atomic_load_explicit(&_globalZeroPoint, std::memory_order_acquire);
    stats.globalZeroPointGroupCount = globalZeroPoints ? globalZeroPoints->size() : 0;
    std::set<std::string> fingerprints;
    for (const auto &[key, candidates] : _models) {
        fingerprints.emplace(key.fingerprint);
        stats.candidateCount += candidates.size();
    }
    stats.fingerprintCount = fingerprints.size();
    return stats;
}

navi::SummaryErrorCode MasterLatencyModelR::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    const auto stats = getStats();
    auto table = builder.newTable("MasterLatencyModelOverview");
    TABLE_ADD_COLUMN_VALUE(table, "fingerprint_count", stats.fingerprintCount);
    TABLE_ADD_COLUMN_VALUE(table, "candidate_count", stats.candidateCount);
    TABLE_ADD_COLUMN_VALUE(table, "accepted_update_count",
                           stats.acceptedUpdateCount);
    TABLE_ADD_COLUMN_VALUE(table, "rejected_update_count",
                           stats.rejectedUpdateCount);
    TABLE_ADD_COLUMN_VALUE(table, "warmup_response_count",
                           stats.warmupResponseCount);
    TABLE_ADD_COLUMN_VALUE(table, "warmup_notification_count",
                           stats.warmupNotificationCount);
    TABLE_ADD_COLUMN_VALUE(table, "warmup_notification_failure_count",
                           stats.warmupNotificationFailureCount);
    TABLE_ADD_COLUMN_VALUE(table, "warmup_subscriber_count",
                           stats.warmupSubscriberCount);
    TABLE_ADD_COLUMN_VALUE(table, "persist_success_count",
                           stats.persistSuccessCount);
    TABLE_ADD_COLUMN_VALUE(table, "persist_failure_count",
                           stats.persistFailureCount);
    TABLE_ADD_COLUMN_VALUE(table, "global_zero_point_live_node_count",
                           stats.globalZeroPointLiveNodeCount);
    TABLE_ADD_COLUMN_VALUE(table, "global_zero_point_group_count",
                           stats.globalZeroPointGroupCount);
    TABLE_ADD_COLUMN_VALUE(table, "global_zero_point_refresh_count",
                           stats.globalZeroPointRefreshCount);
    return navi::SEC_OK;
}

REGISTER_RESOURCE(MasterLatencyModelR);

} // namespace navi_sched
