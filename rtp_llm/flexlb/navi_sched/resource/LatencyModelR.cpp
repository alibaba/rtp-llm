#include "aios/navi_sched/resource/LatencyModelR.h"
#include "autil/TimeUtility.h"
#include <algorithm>
#include <cmath>
#include <tuple>

namespace navi_sched {
bool NaviSchedLatencyModelSnapshot::valid() const
{
    return NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL == formula &&
           NonLinearPrefillModelR::supportsBinaryVersion(binaryVersion) && revision > 0 &&
           NonLinearPrefillModelR::validateParametersForVersion(parameters,
                                                                binaryVersion) &&
           NonLinearPrefillModelR::validateOptimizerState(optimizerState,
                                                          parameters.size()) &&
           LatencyResidualModel::VERSION == residualVersion &&
           LatencyResidualModel::validateParameters(residualParameters) &&
           residualLastUpdateTimeUs >= 0;
}

const std::string LatencyModelR::RESOURCE_ID = "navi_sched.latency_model_r";

void LatencyModelR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

bool LatencyModelR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "latency_model_fingerprint", _latencyModelFingerprint,
                 _latencyModelFingerprint);
    NAVI_JSONIZE(ctx, "cold_start_fit_epochs", _coldStartFitEpochs, _coldStartFitEpochs);
    NAVI_JSONIZE(ctx, "trust_min_revision", _trustMinRevision, _trustMinRevision);
    NAVI_JSONIZE(ctx, "zero_anchor_min_samples", _zeroAnchorMinSamples,
                 _zeroAnchorMinSamples);
    return _latencyModelFingerprint.size() <= 4096 && _trustMinRevision > 1 &&
           _zeroAnchorMinSamples > 0;
}

navi::ErrorCode LatencyModelR::init(navi::ResourceInitContext &ctx)
{
    _coldStartFitEpochs = std::max<size_t>(1, _coldStartFitEpochs);
    if (_agentSchedulerConfigR) {
        const auto config = _agentSchedulerConfigR->get().config;
        _nonLinearPrefillModelR->applyRuntimeConfig(config);
        _zeroAnchorMaxMs = config.latencyZeroAnchorMaxMs;
        _zeroAnchorMinSamples = config.latencyZeroAnchorMinSamples;
        _fixedOnlineLearningRateScale = config.latencyOnlineLearningRateScale;
        _fixedResidualLearningRate = config.residualLearningRate;
        std::atomic_store_explicit(
            &_residualRuntimeConfig,
            std::make_shared<const LatencyResidualModel::RuntimeConfig>(
                config.residualRuntimeConfig()),
            std::memory_order_release);
    }
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> restoredSnapshot;
    if (!_agentHistoryR->load(*_nonLinearPrefillModelR, restoredSnapshot, _fitOnStart,
                              getResidualRuntimeConfig())) {
        return navi::EC_INIT_RESOURCE;
    }
    if (restoredSnapshot) {
        const auto initialParameters = _nonLinearPrefillModelR->getInitialParameters();
        const auto zeroPointIndex = NonLinearPrefillModelR::ZERO_POINT_PARAMETER_INDEX;
        if (restoredSnapshot->parameters.size() == initialParameters.size() &&
            restoredSnapshot->parameters[zeroPointIndex] <= 0.0 &&
            initialParameters[zeroPointIndex] > 0.0) {
            auto repaired =
                std::make_shared<NaviSchedLatencyModelSnapshot>(*restoredSnapshot);
            repaired->parameters[zeroPointIndex] = initialParameters[zeroPointIndex];
            repaired->optimizerState.adamMoment1[zeroPointIndex] = 0.0;
            repaired->optimizerState.adamMoment2[zeroPointIndex] = 0.0;
            if (!repaired->valid() || !_nonLinearPrefillModelR->restoreOptimizerState(
                                          repaired->optimizerState)) {
                NAVI_KERNEL_LOG(
                    ERROR,
                    "failed to repair restored latency zero point, value [%f], "
                    "initial [%f]",
                    restoredSnapshot->parameters[zeroPointIndex],
                    initialParameters[zeroPointIndex]);
                return navi::EC_INIT_RESOURCE;
            }
            NAVI_KERNEL_LOG(
                WARN, "repair restored latency zero point, value [%f], restored [%f]",
                restoredSnapshot->parameters[zeroPointIndex],
                repaired->parameters[zeroPointIndex]);
            restoredSnapshot = std::move(repaired);
        }
    }
    const auto validateRestoredParameters =
        restoredSnapshot && _agentHistoryR->requiresParameterValidation();
    if (restoredSnapshot && !validateRestoredParameters) {
        _restoredTrustedModel = restoredSnapshot->overloadPredictionTrusted;
        auto startupSnapshot =
            std::make_shared<NaviSchedLatencyModelSnapshot>(*restoredSnapshot);
        startupSnapshot->overloadPredictionTrusted = false;
        startupSnapshot->onlineFitUpdate = false;
        publishSnapshot(startupSnapshot);
        return navi::EC_NONE;
    }
    auto parameters = _nonLinearPrefillModelR->getInitialParameters();
    if (!_nonLinearPrefillModelR->validateConfiguredParameters(parameters)) {
        return navi::EC_INIT_RESOURCE;
    }
    auto initialSnapshot = std::make_shared<NaviSchedLatencyModelSnapshot>();
    initialSnapshot->formula = _nonLinearPrefillModelR->getFormula();
    initialSnapshot->binaryVersion = _nonLinearPrefillModelR->getBinaryVersion();
    initialSnapshot->revision = 1;
    initialSnapshot->parameters = std::move(parameters);
    initialSnapshot->optimizerState = _nonLinearPrefillModelR->getOptimizerState();
    initialSnapshot->residualParameters = LatencyResidualModel::makeInitialParameters();
    initialSnapshot->sessionSampleCount = _agentHistoryR->getSessionSampleCount();
    publishSnapshot(initialSnapshot);
    if (validateRestoredParameters) {
        NaviSchedLatencyModelCandidate candidate;
        candidate.sourceRevision = restoredSnapshot->revision;
        candidate.parameters = restoredSnapshot->parameters;
        candidate.optimizerState = NonLinearPrefillModelR::makeOptimizerStateData(
            restoredSnapshot->optimizerState, restoredSnapshot->parameters.size());
        candidate.residualVersion = restoredSnapshot->residualVersion;
        candidate.residualParameters = restoredSnapshot->residualParameters;
        candidate.residualLastUpdateTimeUs = restoredSnapshot->residualLastUpdateTimeUs;
        if (!addColdStartCandidates({candidate})) {
            return navi::EC_INIT_RESOURCE;
        }
        _coldStartAppliedCount.fetch_add(1, std::memory_order_relaxed);
    }
    return navi::EC_NONE;
}

bool LatencyModelR::recordSample(
    const NaviSchedLatencySample &sample,
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot,
    bool &saveNeeded, NaviSchedLatencyTrainingPhase trainingPhase)
{
    updatedSnapshot.reset();
    saveNeeded = false;
    if (!std::isfinite(sample.latencyMs) || sample.latencyMs < 0.0 ||
        !validLatencyTrainingPhase(trainingPhase) ||
        (_nonLinearPrefillModelR->usesExecutionCacheSource() &&
         !sample.features.hasExecutionCacheSource()) ||
        (_nonLinearPrefillModelR->usesExecutionGeometry() &&
         !sample.features.hasExecutionGeometry())) {
        return false;
    }
    autil::ScopedLock fitLock(_fitMutex);
    if (NaviSchedLatencyTrainingPhase::NORMAL == trainingPhase) {
        _rampFinished = true;
    }
    AgentHistoryR::AppendResult appendResult;
    const auto appended = _agentHistoryR->append(sample, trainingPhase, appendResult);
    if (appendResult.trainingDeferred) {
        _rampDeferredSampleCount.fetch_add(1, std::memory_order_relaxed);
        auto current = getSnapshot();
        const auto sessionSampleCount = _agentHistoryR->getSessionSampleCount();
        if (!current || sessionSampleCount < current->sessionSampleCount) {
            return false;
        }
        if (sessionSampleCount > current->sessionSampleCount) {
            auto progress = std::make_shared<NaviSchedLatencyModelSnapshot>(*current);
            progress->onlineFitUpdate = false;
            progress->sessionSampleCount = sessionSampleCount;
            publishSnapshot(progress);
            updatedSnapshot = std::move(progress);
        }
    }
    if (appendResult.lateRampSample) {
        _lateRampSampleCount.fetch_add(1, std::memory_order_relaxed);
    }
    const auto evaluatingColdStart = !_coldStartCandidates.empty();
    ColdStartSelectionResult selectionResult = ColdStartSelectionResult::PENDING;
    if (appended && !appendResult.trainingDeferred && evaluatingColdStart &&
        !selectColdStartCandidate(updatedSnapshot, selectionResult)) {
        return false;
    }
    const auto coldStartSelectionCompleted =
        evaluatingColdStart && selectionResult != ColdStartSelectionResult::PENDING;
    const auto historyAlreadyReplayed =
        selectionResult == ColdStartSelectionResult::IMPORTED_ACCEPTED;
    auto rampReplayCompleted = appendResult.rampReplay && historyAlreadyReplayed;
    if (historyAlreadyReplayed) {
        _agentHistoryR->clearPendingFitSamples();
    }
    if (appendResult.fitReady && (!evaluatingColdStart || (coldStartSelectionCompleted &&
                                                           !historyAlreadyReplayed))) {
        const auto fitted = fit(getOnlineFitBatches(), updatedSnapshot);
        _agentHistoryR->clearPendingFitSamples();
        if (!fitted) {
            return false;
        }
        _onlineFitCount.fetch_add(1, std::memory_order_relaxed);
        rampReplayCompleted = appendResult.rampReplay;
    }
    if (rampReplayCompleted) {
        _rampReplayCount.fetch_add(1, std::memory_order_relaxed);
    }
    saveNeeded = (coldStartSelectionCompleted ||
                  (!evaluatingColdStart && appendResult.persistReady)) &&
                 _agentHistoryR->enabled();
    refreshTrustLocked(updatedSnapshot);
    return true;
}

bool LatencyModelR::fitHistoryOnStart(
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot)
{
    updatedSnapshot.reset();
    autil::ScopedLock fitLock(_fitMutex);
    if (!_fitOnStart) {
        return true;
    }
    _fitOnStart = false;
    auto history = _agentHistoryR->getHistory();
    if (history->empty()) {
        return true;
    }
    auto current = getSnapshot();
    auto parameters =
        current ? current->parameters : _nonLinearPrefillModelR->getInitialParameters();
    if (!_nonLinearPrefillModelR->validateConfiguredParameters(parameters)) {
        return false;
    }
    std::vector<double> updatedParameters;
    if (!fitHistoryParameters(*history, parameters, updatedParameters)) {
        return false;
    }
    auto snapshot = std::make_shared<NaviSchedLatencyModelSnapshot>();
    snapshot->formula = _nonLinearPrefillModelR->getFormula();
    snapshot->binaryVersion = _nonLinearPrefillModelR->getBinaryVersion();
    snapshot->revision = current ? current->revision + 1 : 1;
    snapshot->parameters = std::move(updatedParameters);
    snapshot->optimizerState = _nonLinearPrefillModelR->getOptimizerState();
    snapshot->overloadPredictionTrusted = shouldTrustLocked(snapshot->revision);
    snapshot->warmStarted = current && current->warmStarted;
    snapshot->onlineFitUpdate = true;
    snapshot->sessionSampleCount = _agentHistoryR->getSessionSampleCount();
    publishSnapshot(snapshot);
    updatedSnapshot = std::move(snapshot);
    return true;
}

double LatencyModelR::predict(const NaviSchedLatencyFeatures &features) const
{
    auto snapshot = getSnapshot();
    if (!snapshot || !snapshot->valid()) {
        return 0.0;
    }
    const auto basePrediction = predict(features, snapshot->parameters);
    return LatencyResidualModel::evaluate(basePrediction, snapshot->residualParameters,
                                          snapshot->residualLastUpdateTimeUs,
                                          autil::TimeUtility::currentTime(),
                                          getResidualRuntimeConfig())
        .predictionMs;
}

double LatencyModelR::predict(const NaviSchedLatencyFeatures &features,
                              const std::vector<double> &parameters) const
{
    return _nonLinearPrefillModelR->predictConfigured(features, parameters);
}

std::optional<NonLinearPrefillModelR::NonlinearOperatingPoint>
LatencyModelR::getNonlinearOperatingPoint(const NaviSchedLatencyFeatures &features,
                                          const std::vector<double> &parameters) const
{
    return _nonLinearPrefillModelR->getNonlinearOperatingPointConfigured(features,
                                                                         parameters);
}

NonLinearPrefillModelR::OptimizerState LatencyModelR::getOptimizerState() const
{
    return _nonLinearPrefillModelR->getOptimizerState();
}

bool LatencyModelR::saveState()
{
    autil::ScopedLock persistLock(_persistMutex);
    std::string payload;
    int64_t binaryVersion = 0;
    {
        autil::ScopedLock fitLock(_fitMutex);
        auto snapshot = getSnapshot();
        if (!snapshot || !snapshot->overloadPredictionTrusted) {
            return true;
        }
        binaryVersion = snapshot->binaryVersion;
        if (!_agentHistoryR->makeSavePayload(snapshot, snapshot->optimizerState,
                                             payload)) {
            return false;
        }
    }
    return _agentHistoryR->savePayload(payload, binaryVersion);
}

bool LatencyModelR::applyColdStartCandidates(
    const MasterLatencyModelWarmupToAgentData &warmup,
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot)
{
    updatedSnapshot.reset();
    if (!warmup.valid() || (!_latencyModelFingerprint.empty() &&
                            warmup.fingerprint != _latencyModelFingerprint)) {
        return false;
    }
    const auto selected =
        std::max_element(warmup.candidates.begin(), warmup.candidates.end(),
                         [](const auto &left, const auto &right) {
                             return left.sourceRevision < right.sourceRevision;
                         });
    if (selected == warmup.candidates.end() ||
        selected->sourceRevision < COLD_START_SOURCE_MIN_REVISION) {
        return true;
    }
    autil::ScopedLock fitLock(_fitMutex);
    auto current = getSnapshot();
    if (!current || current->formula != warmup.formula ||
        current->binaryVersion != warmup.binaryVersion ||
        current->parameters.size() != selected->parameters.size() ||
        !_nonLinearPrefillModelR->validateConfiguredParameters(selected->parameters) ||
        selected->residualVersion != LatencyResidualModel::VERSION ||
        !LatencyResidualModel::validateParameters(selected->residualParameters) ||
        selected->residualLastUpdateTimeUs < 0) {
        return false;
    }
    if (current->overloadPredictionTrusted || current->warmStarted ||
        _restoredTrustedModel) {
        return true;
    }
    NonLinearPrefillModelR::OptimizerState selectedOptimizer;
    if (!NonLinearPrefillModelR::makeOptimizerState(selected->optimizerState,
                                                    selectedOptimizer) ||
        !_nonLinearPrefillModelR->restoreOptimizerState(selectedOptimizer)) {
        return false;
    }
    auto selectedParameters = selected->parameters;
    auto snapshot = std::make_shared<NaviSchedLatencyModelSnapshot>();
    snapshot->formula = current->formula;
    snapshot->binaryVersion = current->binaryVersion;
    snapshot->revision = current->revision;
    snapshot->parameters = std::move(selectedParameters);
    snapshot->optimizerState = selectedOptimizer;
    snapshot->residualVersion = selected->residualVersion;
    snapshot->residualParameters = selected->residualParameters;
    snapshot->residualLastUpdateTimeUs = autil::TimeUtility::currentTime();
    if (!LatencyResidualModel::applyIdleDecay(
            snapshot->residualParameters, selected->residualLastUpdateTimeUs,
            snapshot->residualLastUpdateTimeUs, getResidualRuntimeConfig())) {
        return false;
    }
    snapshot->overloadPredictionTrusted = shouldTrustLocked(snapshot->revision);
    snapshot->warmStarted = true;
    snapshot->onlineFitUpdate = true;
    snapshot->sessionSampleCount = _agentHistoryR->getSessionSampleCount();
    clearColdStartCandidates();
    publishSnapshot(snapshot);
    updatedSnapshot = snapshot;
    _coldStartAppliedCount.fetch_add(1, std::memory_order_relaxed);
    _coldStartSelectedCount.fetch_add(1, std::memory_order_relaxed);
    NAVI_KERNEL_LOG(INFO,
                    "adopt cold-start latency model directly, sourceRevision [%ld], "
                    "localRevision [%ld], parameterCount [%lu]",
                    selected->sourceRevision, snapshot->revision,
                    snapshot->parameters.size());
    return true;
}

LatencyModelR::GlobalZeroPointApplyResult LatencyModelR::applyGlobalZeroPoint(
    const LatencyModelGlobalData &global)
{
    if (!global.valid()) {
        _globalZeroPointIncompatibleCount.fetch_add(1, std::memory_order_relaxed);
        return GlobalZeroPointApplyResult::INCOMPATIBLE;
    }
    auto current = getSnapshot();
    if (!current || current->formula != global.formula ||
        current->binaryVersion != global.binaryVersion ||
        current->parameters.size() != global.parameterCount) {
        _globalZeroPointIncompatibleCount.fetch_add(1, std::memory_order_relaxed);
        return GlobalZeroPointApplyResult::INCOMPATIBLE;
    }
    auto applied =
        std::atomic_load_explicit(&_globalZeroPoint, std::memory_order_acquire);
    if (applied && std::tie(global.masterEpoch, global.version) <=
                       std::tie(applied->masterEpoch, applied->version)) {
        return GlobalZeroPointApplyResult::STALE;
    }
    autil::ScopedLock fitLock(_fitMutex);
    applied = std::atomic_load_explicit(&_globalZeroPoint, std::memory_order_acquire);
    if (applied && std::tie(global.masterEpoch, global.version) <=
                       std::tie(applied->masterEpoch, applied->version)) {
        return GlobalZeroPointApplyResult::STALE;
    }
    std::atomic_store_explicit(&_globalZeroPoint,
                               std::make_shared<const LatencyModelGlobalData>(global),
                               std::memory_order_release);
    _globalZeroPointAppliedCount.fetch_add(1, std::memory_order_relaxed);
    return GlobalZeroPointApplyResult::APPLIED;
}

LatencyModelGlobalData LatencyModelR::getGlobalZeroPoint() const
{
    auto global = std::atomic_load_explicit(&_globalZeroPoint, std::memory_order_acquire);
    return global ? *global : LatencyModelGlobalData();
}

bool LatencyModelR::selectColdStartCandidate(
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot,
    ColdStartSelectionResult &result)
{
    result = ColdStartSelectionResult::PENDING;
    if (_coldStartCandidates.empty()) {
        return true;
    }
    const auto history = _agentHistoryR->getHistory();
    if (_agentHistoryR->getSessionSampleCount() < COLD_START_MIN_HISTORY_SAMPLE_COUNT) {
        return true;
    }
    if (_coldStartCandidates.size() != _coldStartCandidateImported.size()) {
        return false;
    }
    auto current = getSnapshot();
    if (!current) {
        return false;
    }
    std::vector<double> candidateErrors(_coldStartCandidates.size(), 0.0);
    for (size_t index = 0; index < _coldStartCandidates.size(); ++index) {
        if (!calculateAbsoluteError(*history, _coldStartCandidates[index].parameters,
                                    _coldStartCandidates[index].residualParameters,
                                    candidateErrors[index])) {
            return false;
        }
    }
    size_t best = 0;
    for (size_t index = 1; index < candidateErrors.size(); ++index) {
        if (candidateErrors[index] < candidateErrors[best]) {
            best = index;
        }
    }
    const auto imported = _coldStartCandidateImported[best];
    const auto selectedRawError = candidateErrors[best];
    auto selectedError = selectedRawError;
    auto selectedSource = "local";
    if (imported) {
        auto candidateParameters = _coldStartCandidates[best].parameters;
        auto candidateOptimizer = _coldStartCandidates[best].optimizerState;
        const auto candidateResidualParameters =
            _coldStartCandidates[best].residualParameters;
        const auto candidateResidualLastUpdateTimeUs =
            _coldStartCandidates[best].residualLastUpdateTimeUs;
        const auto previousOptimizer = _nonLinearPrefillModelR->getOptimizerState();
        std::vector<double> selectedParameters;
        std::vector<double> selectedResidualParameters;
        int64_t selectedResidualLastUpdateTimeUs = 0;
        int64_t localUpdateCount = 0;
        auto refitValid = refitColdStartCandidate(
            candidateParameters, candidateOptimizer, candidateResidualParameters,
            candidateResidualLastUpdateTimeUs, selectedParameters,
            selectedResidualParameters, selectedResidualLastUpdateTimeUs,
            localUpdateCount);
        double refittedError = 0.0;
        refitValid = refitValid &&
                     calculateAbsoluteError(*history, selectedParameters,
                                            selectedResidualParameters, refittedError);
        if (refitValid && refittedError <= selectedRawError) {
            auto snapshot = std::make_shared<NaviSchedLatencyModelSnapshot>();
            snapshot->formula = current->formula;
            snapshot->binaryVersion = current->binaryVersion;
            snapshot->revision = calculateRefittedCandidateRevision(
                _coldStartCandidates[best].sourceRevision, localUpdateCount);
            snapshot->parameters = std::move(selectedParameters);
            snapshot->optimizerState = _nonLinearPrefillModelR->getOptimizerState();
            snapshot->residualParameters = std::move(selectedResidualParameters);
            snapshot->residualLastUpdateTimeUs = selectedResidualLastUpdateTimeUs;
            snapshot->overloadPredictionTrusted = shouldTrustLocked(snapshot->revision);
            snapshot->warmStarted = true;
            snapshot->onlineFitUpdate = true;
            snapshot->sessionSampleCount = _agentHistoryR->getSessionSampleCount();
            selectedError = refittedError;
            selectedSource = "refitted_candidate";
            publishSnapshot(snapshot);
            updatedSnapshot = std::move(snapshot);
            result = ColdStartSelectionResult::IMPORTED_ACCEPTED;
        } else {
            if (!_nonLinearPrefillModelR->restoreOptimizerState(previousOptimizer)) {
                return false;
            }
            if (!calculateAbsoluteError(*history, current->parameters,
                                        current->residualParameters, selectedError)) {
                return false;
            }
            selectedSource = "local_after_refit_rejected";
            result = ColdStartSelectionResult::IMPORTED_REJECTED;
            _coldStartRefitRejectedCount.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        result = ColdStartSelectionResult::LOCAL_SELECTED;
    }
    const auto selectedSnapshot = getSnapshot();
    NAVI_KERNEL_LOG(INFO,
                    "select cold-start latency model, source [%s], rawError [%.6f], "
                    "selectedError [%.6f], historySamples [%lu], revision [%ld], "
                    "trusted [%d], warmStarted [%d]",
                    selectedSource, selectedRawError, selectedError, history->size(),
                    selectedSnapshot->revision,
                    static_cast<int>(selectedSnapshot->overloadPredictionTrusted),
                    static_cast<int>(selectedSnapshot->warmStarted));
    clearColdStartCandidates();
    _coldStartSelectedCount.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool LatencyModelR::addColdStartCandidates(
    const std::vector<NaviSchedLatencyModelCandidate> &candidates)
{
    auto current = getSnapshot();
    if (!current) {
        return false;
    }
    for (const auto &candidate : candidates) {
        if (candidate.parameters.size() != current->parameters.size() ||
            !_nonLinearPrefillModelR->validateConfiguredParameters(
                candidate.parameters) ||
            !candidate.optimizerState.valid(candidate.parameters.size()) ||
            candidate.residualVersion != LatencyResidualModel::VERSION ||
            !LatencyResidualModel::validateParameters(candidate.residualParameters) ||
            candidate.residualLastUpdateTimeUs < 0) {
            return false;
        }
    }
    const auto currentTimeUs = autil::TimeUtility::currentTime();
    if (_coldStartCandidates.empty()) {
        NaviSchedLatencyModelCandidate localCandidate{
            current->revision,
            current->parameters,
            NonLinearPrefillModelR::makeOptimizerStateData(current->optimizerState,
                                                           current->parameters.size()),
            current->residualVersion,
            current->residualParameters,
            current->residualLastUpdateTimeUs};
        if (!LatencyResidualModel::applyIdleDecay(localCandidate.residualParameters,
                                                  localCandidate.residualLastUpdateTimeUs,
                                                  currentTimeUs,
                                                  getResidualRuntimeConfig())) {
            return false;
        }
        localCandidate.residualLastUpdateTimeUs = currentTimeUs;
        _coldStartCandidates.push_back(std::move(localCandidate));
        _coldStartCandidateImported.push_back(false);
    }
    auto sorted = candidates;
    for (auto &candidate : sorted) {
        if (!LatencyResidualModel::applyIdleDecay(
                candidate.residualParameters, candidate.residualLastUpdateTimeUs,
                currentTimeUs, getResidualRuntimeConfig())) {
            return false;
        }
        candidate.residualLastUpdateTimeUs = currentTimeUs;
    }
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const auto &left, const auto &right) {
                         return left.sourceRevision > right.sourceRevision;
                     });
    for (auto &candidate : sorted) {
        const auto duplicate =
            std::find_if(_coldStartCandidates.begin(), _coldStartCandidates.end(),
                         [&candidate](const auto &existing) {
                             return existing.parameters == candidate.parameters;
                         });
        if (duplicate != _coldStartCandidates.end()) {
            const auto index = static_cast<size_t>(
                std::distance(_coldStartCandidates.begin(), duplicate));
            if (!_coldStartCandidateImported[index] ||
                candidate.sourceRevision > duplicate->sourceRevision) {
                *duplicate = std::move(candidate);
            }
            _coldStartCandidateImported[index] = true;
            continue;
        }
        _coldStartCandidates.emplace_back(std::move(candidate));
        _coldStartCandidateImported.push_back(true);
    }
    return true;
}

bool LatencyModelR::fitParameters(const std::vector<NaviSchedLatencySample> &samples,
                                  const std::vector<double> &initialParameters,
                                  std::vector<double> &updatedParameters,
                                  double learningRateScale, double currentSampleWeight,
                                  const std::vector<double> *parameterLearningRateScales,
                                  bool rejectNonPositiveWaitingCandidate)
{
    return _nonLinearPrefillModelR->validateConfiguredParameters(initialParameters) &&
           _nonLinearPrefillModelR->fit(samples, initialParameters, updatedParameters,
                                        learningRateScale, currentSampleWeight,
                                        parameterLearningRateScales,
                                        rejectNonPositiveWaitingCandidate) &&
           _nonLinearPrefillModelR->validateConfiguredParameters(updatedParameters);
}

bool LatencyModelR::fitOnlineParameters(
    const std::vector<NaviSchedLatencySample> &samples,
    const std::vector<double> &initialParameters, std::vector<double> &updatedParameters,
    double learningRateScale, double currentSampleWeight,
    const std::vector<double> *parameterLearningRateScales,
    bool rejectNonPositiveWaitingCandidate)
{
    return fitParameters(samples, initialParameters, updatedParameters, learningRateScale,
                         currentSampleWeight, parameterLearningRateScales,
                         rejectNonPositiveWaitingCandidate);
}

bool LatencyModelR::fitHistoryParameters(
    const std::deque<NaviSchedLatencySample> &history,
    const std::vector<double> &initialParameters, std::vector<double> &updatedParameters)
{
    return _nonLinearPrefillModelR->validateConfiguredParameters(initialParameters) &&
           _nonLinearPrefillModelR->fitHistory(history, _agentHistoryR->getFitBatchSize(),
                                               _coldStartFitEpochs, initialParameters,
                                               updatedParameters) &&
           _nonLinearPrefillModelR->validateConfiguredParameters(updatedParameters);
}

bool LatencyModelR::refitColdStartCandidate(
    const std::vector<double> &initialParameters,
    const NaviSchedLatencyOptimizerState &initialOptimizerState,
    const std::vector<double> &initialResidualParameters,
    int64_t initialResidualLastUpdateTimeUs, std::vector<double> &updatedParameters,
    std::vector<double> &updatedResidualParameters,
    int64_t &updatedResidualLastUpdateTimeUs, int64_t &localUpdateCount)
{
    localUpdateCount = 0;
    const auto replayBatches = _agentHistoryR->getOnlineFitReplayBatches(true);
    if (replayBatches.empty()) {
        return false;
    }
    const auto previousOptimizer = _nonLinearPrefillModelR->getOptimizerState();
    NonLinearPrefillModelR::OptimizerState candidateOptimizer;
    if (!NonLinearPrefillModelR::makeOptimizerState(initialOptimizerState,
                                                    candidateOptimizer) ||
        !_nonLinearPrefillModelR->restoreOptimizerState(candidateOptimizer)) {
        return false;
    }
    auto parameters = initialParameters;
    auto residualParameters = initialResidualParameters;
    auto residualLastUpdateTimeUs = initialResidualLastUpdateTimeUs;
    const auto runtimeConfig = _agentSchedulerConfigR
                                   ? _agentSchedulerConfigR->get().config
                                   : AgentSchedulerConfig();
    for (const auto &batch : replayBatches) {
        std::vector<double> nextParameters;
        std::vector<double> nextResidualParameters;
        int64_t nextResidualLastUpdateTimeUs = 0;
        if (!fitOnlineBatch(batch, parameters, residualParameters,
                            residualLastUpdateTimeUs, runtimeConfig, nextParameters,
                            nextResidualParameters, nextResidualLastUpdateTimeUs)) {
            if (!_nonLinearPrefillModelR->restoreOptimizerState(previousOptimizer)) {
                NAVI_KERNEL_LOG(ERROR,
                                "restore latency optimizer after cold-start refit "
                                "failure failed");
            }
            return false;
        }
        parameters = std::move(nextParameters);
        residualParameters = std::move(nextResidualParameters);
        residualLastUpdateTimeUs = nextResidualLastUpdateTimeUs;
        ++localUpdateCount;
    }
    if (!_nonLinearPrefillModelR->validateConfiguredParameters(parameters) ||
        !LatencyResidualModel::validateParameters(residualParameters) ||
        residualLastUpdateTimeUs < 0) {
        if (!_nonLinearPrefillModelR->restoreOptimizerState(previousOptimizer)) {
            NAVI_KERNEL_LOG(ERROR,
                            "restore latency optimizer after invalid cold-start refit "
                            "failed");
        }
        return false;
    }
    updatedParameters = std::move(parameters);
    updatedResidualParameters = std::move(residualParameters);
    updatedResidualLastUpdateTimeUs = residualLastUpdateTimeUs;
    return true;
}

bool LatencyModelR::fitOnlineBatch(const std::vector<NaviSchedLatencySample> &samples,
                                   const std::vector<double> &initialParameters,
                                   const std::vector<double> &initialResidualParameters,
                                   int64_t initialResidualLastUpdateTimeUs,
                                   const AgentSchedulerConfig &runtimeConfig,
                                   std::vector<double> &updatedParameters,
                                   std::vector<double> &updatedResidualParameters,
                                   int64_t &updatedResidualLastUpdateTimeUs,
                                   bool rejectNonPositiveWaitingCandidate)
{
    if (samples.empty() ||
        !LatencyResidualModel::validateParameters(initialResidualParameters) ||
        initialResidualLastUpdateTimeUs < 0) {
        return false;
    }
    const auto &currentSample = samples.back();
    auto residualConfig = runtimeConfig.residualRuntimeConfig();
    std::vector<double> nextParameters;
    const auto &parameterScales = runtimeConfig.latencyParameterLearningRateScales;
    if (!fitOnlineParameters(samples, initialParameters, nextParameters,
                             runtimeConfig.latencyOnlineLearningRateScale,
                             runtimeConfig.latencyCurrentSampleWeight,
                             parameterScales.empty() ? nullptr : &parameterScales,
                             rejectNonPositiveWaitingCandidate)) {
        return false;
    }
    auto nextResidualParameters = initialResidualParameters;
    auto nextResidualLastUpdateTimeUs = initialResidualLastUpdateTimeUs;
    const auto basePrediction = _nonLinearPrefillModelR->predictConfigured(
        currentSample.features, nextParameters);
    if (!std::isfinite(basePrediction) || basePrediction < 0.0 ||
        basePrediction > NonLinearPrefillModelR::MAX_FIT_PREDICTION_MS) {
        NAVI_KERNEL_LOG(WARN, "reject latency fit with invalid base prediction [%f]",
                        basePrediction);
        return false;
    }
    if (residualConfig.learningRate > 0.0) {
        if (!LatencyResidualModel::update(basePrediction, currentSample.latencyMs,
                                          nextResidualParameters,
                                          nextResidualLastUpdateTimeUs,
                                          currentSample.eventTimeUs, residualConfig)) {
            return false;
        }
    }
    updatedParameters = std::move(nextParameters);
    updatedResidualParameters = std::move(nextResidualParameters);
    updatedResidualLastUpdateTimeUs = nextResidualLastUpdateTimeUs;
    return true;
}

bool LatencyModelR::calculateAbsoluteError(const AgentHistoryR::History &history,
                                           const std::vector<double> &parameters,
                                           const std::vector<double> &residualParameters,
                                           double &error) const
{
    error = 0.0;
    if (history.empty() ||
        !_nonLinearPrefillModelR->validateConfiguredParameters(parameters) ||
        !LatencyResidualModel::validateParameters(residualParameters)) {
        return false;
    }
    for (const auto &sample : history) {
        const auto basePrediction =
            _nonLinearPrefillModelR->predictConfigured(sample.features, parameters);
        const auto prediction =
            LatencyResidualModel::evaluate(basePrediction, residualParameters, 0, 0,
                                           getResidualRuntimeConfig())
                .predictionMs;
        if (!std::isfinite(prediction)) {
            return false;
        }
        error += std::abs(prediction - sample.latencyMs);
    }
    return std::isfinite(error);
}

int64_t LatencyModelR::calculateRefittedCandidateRevision(int64_t sourceRevision,
                                                          int64_t localUpdateCount) const
{
    const auto cappedSourceRevision = std::min(_trustMinRevision, sourceRevision);
    const auto remainingRevision = _trustMinRevision - cappedSourceRevision;
    return cappedSourceRevision +
           std::min(remainingRevision, std::max<int64_t>(0, localUpdateCount));
}

bool LatencyModelR::shouldTrustLocked(int64_t revision) const
{
    return _rampFinished && revision >= _trustMinRevision;
}

void LatencyModelR::refreshTrustLocked(
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot)
{
    auto current = getSnapshot();
    if (!current ||
        current->overloadPredictionTrusted == shouldTrustLocked(current->revision)) {
        return;
    }
    auto refreshed = std::make_shared<NaviSchedLatencyModelSnapshot>(*current);
    refreshed->overloadPredictionTrusted = shouldTrustLocked(current->revision);
    refreshed->onlineFitUpdate = false;
    publishSnapshot(refreshed);
    updatedSnapshot = std::move(refreshed);
}

bool LatencyModelR::fit(
    const std::vector<NaviSchedLatencySample> &samples,
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot)
{
    if (samples.empty()) {
        return false;
    }
    return fit(std::vector<std::vector<NaviSchedLatencySample>>{samples},
               updatedSnapshot);
}

bool LatencyModelR::fit(
    const std::vector<std::vector<NaviSchedLatencySample>> &batches,
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot)
{
    if (batches.empty()) {
        return false;
    }
    auto current = getSnapshot();
    std::vector<double> currentParameters;
    if (current && current->formula == _nonLinearPrefillModelR->getFormula() &&
        current->binaryVersion == _nonLinearPrefillModelR->getBinaryVersion()) {
        currentParameters = current->parameters;
    }
    if (currentParameters.empty() ||
        !_nonLinearPrefillModelR->validateConfiguredParameters(currentParameters)) {
        return false;
    }
    const auto previousOptimizer = _nonLinearPrefillModelR->getOptimizerState();
    const auto runtimeConfig = _agentSchedulerConfigR
                                   ? _agentSchedulerConfigR->get().config
                                   : AgentSchedulerConfig();
    auto newParameters = currentParameters;
    auto residualParameters = current->residualParameters;
    auto residualLastUpdateTimeUs = current->residualLastUpdateTimeUs;
    for (const auto &batch : batches) {
        std::vector<double> nextParameters;
        std::vector<double> nextResidualParameters;
        int64_t nextResidualLastUpdateTimeUs = 0;
        if (batch.empty() ||
            !fitOnlineBatch(batch, newParameters, residualParameters,
                            residualLastUpdateTimeUs, runtimeConfig, nextParameters,
                            nextResidualParameters, nextResidualLastUpdateTimeUs,
                            current->overloadPredictionTrusted)) {
            if (!_nonLinearPrefillModelR->restoreOptimizerState(previousOptimizer)) {
                NAVI_KERNEL_LOG(
                    ERROR, "restore latency optimizer after online fit failure failed");
            }
            return false;
        }
        newParameters = std::move(nextParameters);
        residualParameters = std::move(nextResidualParameters);
        residualLastUpdateTimeUs = nextResidualLastUpdateTimeUs;
    }
    auto snapshot = std::make_shared<NaviSchedLatencyModelSnapshot>();
    snapshot->formula = _nonLinearPrefillModelR->getFormula();
    snapshot->binaryVersion = _nonLinearPrefillModelR->getBinaryVersion();
    snapshot->revision = current ? current->revision + 1 : 1;
    snapshot->parameters = std::move(newParameters);
    snapshot->optimizerState = _nonLinearPrefillModelR->getOptimizerState();
    snapshot->residualParameters = std::move(residualParameters);
    snapshot->residualLastUpdateTimeUs = residualLastUpdateTimeUs;
    snapshot->overloadPredictionTrusted = shouldTrustLocked(snapshot->revision);
    snapshot->warmStarted = current && current->warmStarted;
    snapshot->onlineFitUpdate = true;
    snapshot->sessionSampleCount = _agentHistoryR->getSessionSampleCount();
    publishSnapshot(snapshot);
    updatedSnapshot = std::move(snapshot);
    return true;
}

void LatencyModelR::clearColdStartCandidates()
{
    _coldStartCandidates.clear();
    _coldStartCandidateImported.clear();
}

std::vector<NaviSchedLatencySample> LatencyModelR::getOnlineFitBatch() const
{
    return _agentHistoryR->getOnlineFitBatch();
}

std::vector<std::vector<NaviSchedLatencySample>> LatencyModelR::getOnlineFitBatches()
    const
{
    return _agentHistoryR->getOnlineFitBatches();
}

std::shared_ptr<const NaviSchedLatencyModelSnapshot> LatencyModelR::getSnapshot() const
{
    return std::atomic_load_explicit(&_snapshot, std::memory_order_acquire);
}

bool LatencyModelR::applyRuntimeConfig(const AgentSchedulerConfig &config)
{
    autil::ScopedLock fitLock(_fitMutex);
    _coldStartFitEpochs = config.latencyColdStartFitEpochs;
    _trustMinRevision = config.latencyModelTrustMinRevision;
    _zeroAnchorMaxMs = config.latencyZeroAnchorMaxMs;
    _zeroAnchorMinSamples = config.latencyZeroAnchorMinSamples;
    const bool fixedLearningRateChanged =
        _fixedOnlineLearningRateScale != config.latencyOnlineLearningRateScale ||
        _fixedResidualLearningRate != config.residualLearningRate;
    _fixedOnlineLearningRateScale = config.latencyOnlineLearningRateScale;
    _fixedResidualLearningRate = config.residualLearningRate;
    std::atomic_store_explicit(
        &_residualRuntimeConfig,
        std::make_shared<const LatencyResidualModel::RuntimeConfig>(
            config.residualRuntimeConfig()),
        std::memory_order_release);
    auto current = getSnapshot();
    if (!current) {
        return false;
    }
    auto sanitizedParameters = current->parameters;
    auto sanitizedOptimizer = current->optimizerState;
    if (!_nonLinearPrefillModelR->sanitizeInactiveParameters(sanitizedParameters,
                                                             sanitizedOptimizer)) {
        return false;
    }
    const auto parameterStateChanged =
        sanitizedParameters != current->parameters ||
        sanitizedOptimizer.adamMoment1 != current->optimizerState.adamMoment1 ||
        sanitizedOptimizer.adamMoment2 != current->optimizerState.adamMoment2;
    if (parameterStateChanged &&
        !_nonLinearPrefillModelR->restoreOptimizerState(sanitizedOptimizer)) {
        return false;
    }
    const auto trusted = shouldTrustLocked(current->revision);
    if (trusted == current->overloadPredictionTrusted && !parameterStateChanged &&
        !fixedLearningRateChanged) {
        return false;
    }
    auto updated = std::make_shared<NaviSchedLatencyModelSnapshot>(*current);
    if (parameterStateChanged) {
        updated->revision = current->revision + 1;
        updated->parameters = std::move(sanitizedParameters);
        updated->optimizerState = sanitizedOptimizer;
    }
    updated->overloadPredictionTrusted = trusted;
    updated->onlineFitUpdate = parameterStateChanged;
    publishSnapshot(updated);
    return true;
}

LatencyResidualModel::RuntimeConfig LatencyModelR::getResidualRuntimeConfig() const
{
    const auto config =
        std::atomic_load_explicit(&_residualRuntimeConfig, std::memory_order_acquire);
    return config ? *config : LatencyResidualModel::RuntimeConfig();
}

navi::SummaryErrorCode LatencyModelR::getSummary(const std::string &query,
                                                 navi::SummaryBuilder &builder) const
{
    auto snapshot = getSnapshot();
    size_t coldStartCandidateCount = 0;
    size_t coldStartEvaluatedSampleCount = 0;
    size_t zeroAnchorRetainedSampleCount = 0;
    const auto globalZeroPoint = getGlobalZeroPoint();
    {
        autil::ScopedLock fitLock(_fitMutex);
        coldStartCandidateCount = _coldStartCandidates.size();
        if (_agentHistoryR) {
            const auto history = _agentHistoryR->getHistory();
            coldStartEvaluatedSampleCount = history ? history->size() : 0;
            zeroAnchorRetainedSampleCount =
                _agentHistoryR->getRetainedSampleCountAtMost(_zeroAnchorMaxMs);
        }
    }
    auto overview = builder.newTable("NaviSchedLatencyModelOverview");
    TABLE_ADD_COLUMN_VALUE(overview, "enabled", true);
    TABLE_ADD_COLUMN_VALUE(overview, "formula_id", snapshot ? snapshot->formula : 0);
    TABLE_ADD_COLUMN_VALUE(
        overview, "formula_name",
        getNaviSchedLatencyFormulaName(snapshot ? snapshot->formula : 0));
    TABLE_ADD_COLUMN_VALUE(overview, "binary_version",
                           snapshot ? snapshot->binaryVersion : 0);
    TABLE_ADD_COLUMN_VALUE(overview, "revision", snapshot ? snapshot->revision : 0);
    TABLE_ADD_COLUMN_VALUE(overview, "parameter_count",
                           snapshot ? snapshot->parameters.size() : 0);
    TABLE_ADD_COLUMN_VALUE(overview, "residual_version",
                           snapshot ? snapshot->residualVersion : 0);
    TABLE_ADD_COLUMN_VALUE(overview, "residual_parameter_count",
                           snapshot ? snapshot->residualParameters.size() : 0);
    TABLE_ADD_COLUMN_VALUE(overview, "residual_last_update_time_us",
                           snapshot ? snapshot->residualLastUpdateTimeUs : 0);
    const auto currentTimeUs = autil::TimeUtility::currentTime();
    const auto residualDecay = snapshot ? LatencyResidualModel::decayFactor(
                                              snapshot->residualLastUpdateTimeUs,
                                              currentTimeUs, getResidualRuntimeConfig())
                                        : 0.0;
    TABLE_ADD_COLUMN_VALUE(overview, "residual_decay_factor", residualDecay);
    TABLE_ADD_COLUMN_VALUE(overview, "online_fit_count",
                           _onlineFitCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "ramp_deferred_sample_count",
                           _rampDeferredSampleCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "ramp_replay_count",
                           _rampReplayCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "late_ramp_sample_count",
                           _lateRampSampleCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "overload_prediction_trusted",
                           snapshot ? snapshot->overloadPredictionTrusted : false);
    TABLE_ADD_COLUMN_VALUE(overview, "warm_started",
                           snapshot ? snapshot->warmStarted : false);
    TABLE_ADD_COLUMN_VALUE(overview, "cold_start_candidate_count",
                           coldStartCandidateCount);
    TABLE_ADD_COLUMN_VALUE(overview, "cold_start_evaluated_sample_count",
                           coldStartEvaluatedSampleCount);
    TABLE_ADD_COLUMN_VALUE(overview, "zero_anchor_max_ms", _zeroAnchorMaxMs);
    TABLE_ADD_COLUMN_VALUE(overview, "zero_anchor_min_samples", _zeroAnchorMinSamples);
    TABLE_ADD_COLUMN_VALUE(overview, "zero_anchor_retained_sample_count",
                           zeroAnchorRetainedSampleCount);
    TABLE_ADD_COLUMN_VALUE(overview, "cold_start_applied_count",
                           _coldStartAppliedCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "cold_start_selected_count",
                           _coldStartSelectedCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "cold_start_refit_rejected_count",
                           _coldStartRefitRejectedCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "global_zero_point_valid", globalZeroPoint.valid());
    TABLE_ADD_COLUMN_VALUE(overview, "global_zero_point_master_epoch",
                           globalZeroPoint.masterEpoch);
    TABLE_ADD_COLUMN_VALUE(overview, "global_zero_point_version",
                           globalZeroPoint.version);
    TABLE_ADD_COLUMN_VALUE(overview, "global_zero_point",
                           globalZeroPoint.globalZeroPoint);
    TABLE_ADD_COLUMN_VALUE(overview, "global_zero_point_applied_count",
                           _globalZeroPointAppliedCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(
        overview, "global_zero_point_incompatible_count",
        _globalZeroPointIncompatibleCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "online_fit_equal_bucket_weight", true);
    const auto runtimeConfig = _agentSchedulerConfigR
                                   ? _agentSchedulerConfigR->get().config
                                   : AgentSchedulerConfig();
    TABLE_ADD_COLUMN_VALUE(overview, "online_fit_learning_rate_scale",
                           runtimeConfig.latencyOnlineLearningRateScale);
    TABLE_ADD_COLUMN_VALUE(overview, "online_fit_current_sample_weight",
                           runtimeConfig.latencyCurrentSampleWeight);
    if (query == "0" || !snapshot) {
        return navi::SEC_OK;
    }
    auto parameters = builder.newTable("NaviSchedLatencyModelParameters");
    auto limit = query == "3" ? snapshot->parameters.size()
                              : std::min<size_t>(snapshot->parameters.size(), 64);
    for (size_t index = 0; index < limit; ++index) {
        TABLE_ADD_COLUMN_VALUE(parameters, "index", index);
        TABLE_ADD_COLUMN_VALUE(parameters, "parameter", snapshot->parameters[index]);
    }
    auto residualParameters = builder.newTable("NaviSchedLatencyResidualParameters");
    const auto residualLimit =
        query == "3" ? snapshot->residualParameters.size()
                     : std::min<size_t>(snapshot->residualParameters.size(), 64);
    for (size_t index = 0; index < residualLimit; ++index) {
        TABLE_ADD_COLUMN_VALUE(residualParameters, "index", index);
        TABLE_ADD_COLUMN_VALUE(residualParameters, "parameter",
                               snapshot->residualParameters[index]);
        TABLE_ADD_COLUMN_VALUE(residualParameters, "effective_parameter",
                               snapshot->residualParameters[index] * residualDecay);
    }
    return navi::SEC_OK;
}

void LatencyModelR::publishSnapshot(
    const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot)
{
    std::atomic_store_explicit(&_snapshot, snapshot, std::memory_order_release);
}

REGISTER_RESOURCE(LatencyModelR);

}
