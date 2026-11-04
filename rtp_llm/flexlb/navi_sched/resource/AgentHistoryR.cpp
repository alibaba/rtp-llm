#include "rtp_llm/flexlb/navi_sched/resource/AgentHistoryR.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyModelR.h"
#include "autil/CRC32C.h"
#include "autil/DataBuffer.h"
#include "autil/TimeUtility.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace navi_sched {
const int32_t LATENCY_STATE_VERSION = 6;
const std::string LATENCY_STATE_MAGIC = "navi_sched_latency_state";

const std::string AgentHistoryR::RESOURCE_ID = "navi_sched.agent_history_r";

void AgentHistoryR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

bool AgentHistoryR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "state_file", _stateFile, _stateFile);
    NAVI_JSONIZE(ctx, "formula_fingerprint", _formulaFingerprint, _formulaFingerprint);
    NAVI_JSONIZE(ctx, "history_fingerprint", _historyFingerprint, _historyFingerprint);
    NAVI_JSONIZE(ctx, "binary_fingerprint", _binaryFingerprint, _binaryFingerprint);
    NAVI_JSONIZE(ctx, "history_limit", _historyLimit, _historyLimit);
    NAVI_JSONIZE(ctx, "history_bucket_capacity", _historyBucketCapacity,
                 _historyBucketCapacity);
    NAVI_JSONIZE(ctx, "fit_batch_size", _fitBatchSize, _fitBatchSize);
    NAVI_JSONIZE(ctx, "fit_interval", _fitInterval, _fitInterval);
    NAVI_JSONIZE(ctx, "persist_interval", _persistInterval, _persistInterval);
    NAVI_JSONIZE(ctx, "warmup_sample_count", _warmupSampleCount, _warmupSampleCount);
    return true;
}

navi::ErrorCode AgentHistoryR::init(navi::ResourceInitContext &ctx)
{
    _fitBatchSize = std::max<size_t>(1, _fitBatchSize);
    _historyLimit = std::max(_fitBatchSize, _historyLimit);
    _historyBucketCapacity = std::max<size_t>(1, _historyBucketCapacity);
    _persistInterval = std::max<size_t>(1, _persistInterval);
    _sampler.configure(_historyLimit, _historyBucketCapacity);
    return navi::EC_NONE;
}

bool AgentHistoryR::load(NonLinearPrefillModelR &model,
                         std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
                         bool &fitOnStart)
{
    return load(model, snapshot, fitOnStart,
                LatencyResidualModel::RuntimeConfig());
}

bool AgentHistoryR::load(NonLinearPrefillModelR &model,
                         std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
                         bool &fitOnStart,
                         const LatencyResidualModel::RuntimeConfig &residualConfig)
{
    autil::ScopedLock lock(_lock);
    _sampler.clear();
    snapshot.reset();
    fitOnStart = false;
    _historyRestored = false;
    _parametersRestoredWithoutHistory = false;
    _pendingFitSampleCount = 0;
    _rampCollectionObserved = false;
    _rampCollectionCompleted = false;
    if (_stateFile.empty()) {
        return true;
    }
    std::ifstream input(_stateFile, std::ios::binary);
    if (!input) {
        return true;
    }
    input.seekg(0, std::ios::end);
    auto length = input.tellg();
    if (length <= 0 || length > 64 * 1024 * 1024) {
        return false;
    }
    input.seekg(0, std::ios::beg);
    std::string content(static_cast<size_t>(length), '\0');
    if (!input.read(content.data(), length)) {
        return false;
    }
    try {
        autil::DataBuffer dataBuffer(content.data(), content.size());
        std::string magic;
        int32_t stateVersion = 0;
        int32_t formula = NAVI_SCHED_LATENCY_FORMULA_UNKNOWN;
        int64_t binaryVersion = 0;
        int64_t revision = 0;
        std::string formulaFingerprint;
        std::string historyFingerprint;
        std::string binaryFingerprint;
        std::vector<double> parameters;
        std::vector<double> adamMoment1;
        std::vector<double> adamMoment2;
        int64_t optimizerStep = 0;
        dataBuffer.read(magic);
        dataBuffer.read(stateVersion);
        if (LATENCY_STATE_MAGIC != magic) {
            return false;
        }
        if (stateVersion != LATENCY_STATE_VERSION) {
            return true;
        }
        std::string payload;
        uint32_t expectedChecksum = 0;
        dataBuffer.read(expectedChecksum);
        dataBuffer.read(payload);
        const auto actualChecksum = autil::CRC32C::Mask(
            autil::CRC32C::Value(payload.data(), payload.size()));
        if (dataBuffer.getDataLen() != 0 || actualChecksum != expectedChecksum) {
            return true;
        }
        autil::DataBuffer stateBuffer(payload.data(), payload.size());
        stateBuffer.read(formulaFingerprint);
        stateBuffer.read(historyFingerprint);
        stateBuffer.read(binaryFingerprint);
        stateBuffer.read(formula);
        stateBuffer.read(binaryVersion);
        stateBuffer.read(revision);
        bool overloadPredictionTrusted = false;
        bool warmStarted = false;
        stateBuffer.read(overloadPredictionTrusted);
        stateBuffer.read(warmStarted);
        stateBuffer.read(parameters);
        if (!overloadPredictionTrusted || _formulaFingerprint.empty() ||
            formulaFingerprint != _formulaFingerprint ||
            formula != model.getFormula() ||
            binaryVersion != model.getBinaryVersion() ||
            revision <= 0 ||
            !model.validateParameters(parameters) ||
            parameters.size() != model.getInitialParameters().size()) {
            return true;
        }
        model.resetOptimizerState();
        _parametersRestoredWithoutHistory = true;
        auto loaded = std::make_shared<NaviSchedLatencyModelSnapshot>();
        loaded->formula = formula;
        loaded->binaryVersion = model.getBinaryVersion();
        loaded->revision = revision;
        loaded->parameters = std::move(parameters);
        loaded->optimizerState = model.getOptimizerState();
        snapshot = loaded;
        stateBuffer.read(adamMoment1);
        stateBuffer.read(adamMoment2);
        stateBuffer.read(optimizerStep);
        stateBuffer.read(loaded->residualVersion);
        stateBuffer.read(loaded->residualParameters);
        stateBuffer.read(loaded->residualLastUpdateTimeUs);
        if (loaded->residualVersion != LatencyResidualModel::VERSION ||
            !LatencyResidualModel::validateParameters(loaded->residualParameters) ||
            loaded->residualLastUpdateTimeUs < 0) {
            loaded->residualVersion = LatencyResidualModel::VERSION;
            loaded->residualParameters =
                LatencyResidualModel::makeInitialParameters();
            loaded->residualLastUpdateTimeUs = 0;
        }
        if (LatencyResidualModel::expired(
                loaded->residualLastUpdateTimeUs,
                autil::TimeUtility::currentTime(), residualConfig)) {
            loaded->residualParameters =
                LatencyResidualModel::makeInitialParameters();
            loaded->residualLastUpdateTimeUs = 0;
        }
        const auto sameHistory =
            !_historyFingerprint.empty() && historyFingerprint == _historyFingerprint;
        const auto sameBinary =
            !_binaryFingerprint.empty() && binaryFingerprint == _binaryFingerprint;
        if (sameHistory && sameBinary) {
            if (adamMoment1.size() != loaded->parameters.size() ||
                adamMoment2.size() != loaded->parameters.size() ||
                !_sampler.deserialize(stateBuffer) ||
                stateBuffer.getDataLen() != 0) {
                _sampler.clear();
                return true;
            }
            NonLinearPrefillModelR::OptimizerState optimizerState;
            std::copy(adamMoment1.begin(), adamMoment1.end(),
                      optimizerState.adamMoment1.begin());
            std::copy(adamMoment2.begin(), adamMoment2.end(),
                      optimizerState.adamMoment2.begin());
            optimizerState.step = optimizerStep;
            if (!model.restoreOptimizerState(optimizerState)) {
                _sampler.clear();
                return true;
            }
            loaded->optimizerState = optimizerState;
            _historyRestored = true;
            _parametersRestoredWithoutHistory = false;
        }
        loaded->overloadPredictionTrusted = overloadPredictionTrusted;
        loaded->warmStarted = warmStarted;
    } catch (const autil::DataBufferCorruptedException &) {
        snapshot.reset();
        _sampler.clear();
        model.resetOptimizerState();
        _historyRestored = false;
        _parametersRestoredWithoutHistory = false;
        return true;
    }
    return true;
}

bool AgentHistoryR::append(const NaviSchedLatencySample &sample,
                           NaviSchedLatencyTrainingPhase trainingPhase,
                           AppendResult &result)
{
    autil::ScopedLock lock(_lock);
    result = AppendResult();
    if (!std::isfinite(sample.latencyMs) || sample.latencyMs < 0.0 ||
        !validLatencyTrainingPhase(trainingPhase)) {
        return false;
    }
    if (_warmupSamplesSkipped < _warmupSampleCount) {
        ++_warmupSamplesSkipped;
        if (_warmupSamplesSkipped == _warmupSampleCount) {
            _warmupLocalBatchIdCutoff = std::max<int64_t>(
                0, std::max(sample.localBatchId,
                            sample.latestAllocatedLocalBatchId));
        }
        return false;
    }
    if (_warmupLocalBatchIdCutoff > 0 && sample.localBatchId > 0 &&
        sample.localBatchId <= _warmupLocalBatchIdCutoff) {
        return false;
    }
    _sampler.append(sample);
    result.appended = true;
    ++_pendingFitSampleCount;
    result.persistReady = ++_samplesSincePersist >= _persistInterval;
    if (result.persistReady) {
        _samplesSincePersist = 0;
    }
    if (NaviSchedLatencyTrainingPhase::RAMP == trainingPhase &&
        !_rampCollectionCompleted) {
        _rampCollectionObserved = true;
        result.trainingDeferred = true;
        return true;
    }
    if (NaviSchedLatencyTrainingPhase::NORMAL == trainingPhase &&
        _rampCollectionObserved && !_rampCollectionCompleted) {
        _rampCollectionCompleted = true;
        _samplesSinceFit = 0;
        result.fitReady = true;
        result.rampReplay = true;
        return true;
    }
    result.lateRampSample =
        NaviSchedLatencyTrainingPhase::RAMP == trainingPhase &&
        _rampCollectionCompleted;
    result.fitReady = advanceFitCadence(_sampler.historySize(), _samplesSinceFit);
    return true;
}

bool AgentHistoryR::append(const NaviSchedLatencySample &sample, bool &fitReady,
                           bool &persistReady)
{
    AppendResult result;
    const auto success = append(sample, NaviSchedLatencyTrainingPhase::NORMAL, result);
    fitReady = result.fitReady;
    persistReady = result.persistReady;
    return success;
}

std::vector<NaviSchedLatencySample> AgentHistoryR::getFitBatch() const
{
    autil::ScopedLock lock(_lock);
    return _sampler.getFitBatch(_fitBatchSize);
}

std::vector<NaviSchedLatencySample> AgentHistoryR::getOnlineFitBatch() const
{
    autil::ScopedLock lock(_lock);
    return _sampler.getOnlineFitBatch();
}

std::vector<std::vector<NaviSchedLatencySample>>
AgentHistoryR::getOnlineFitBatches() const
{
    autil::ScopedLock lock(_lock);
    return _sampler.getOnlineFitBatches(_pendingFitSampleCount);
}

void AgentHistoryR::clearPendingFitSamples()
{
    autil::ScopedLock lock(_lock);
    _pendingFitSampleCount = 0;
}

std::vector<std::vector<NaviSchedLatencySample>>
AgentHistoryR::getOnlineFitReplayBatches(bool includePendingTail) const
{
    autil::ScopedLock lock(_lock);
    std::vector<std::vector<NaviSchedLatencySample>> batches;
    LatencyHistorySampler replay(1);
    replay.configure(_historyLimit, _historyBucketCapacity);
    size_t samplesSinceFit = 0;
    size_t pendingFitSampleCount = 0;
    for (const auto &sample : *_sampler.getHistory()) {
        replay.append(sample);
        ++pendingFitSampleCount;
        if (advanceFitCadence(replay.historySize(), samplesSinceFit)) {
            auto current = replay.getOnlineFitBatches(pendingFitSampleCount);
            batches.insert(batches.end(), std::make_move_iterator(current.begin()),
                           std::make_move_iterator(current.end()));
            pendingFitSampleCount = 0;
        }
    }
    if (includePendingTail && pendingFitSampleCount > 0) {
        auto current = replay.getOnlineFitBatches(pendingFitSampleCount);
        batches.insert(batches.end(), std::make_move_iterator(current.begin()),
                       std::make_move_iterator(current.end()));
    }
    return batches;
}

bool AgentHistoryR::advanceFitCadence(size_t historySize,
                                      size_t &samplesSinceFit) const
{
    const auto fitInterval = _fitInterval > 0 ? _fitInterval : _fitBatchSize;
    ++samplesSinceFit;
    const auto fitReady =
        historySize >= _fitBatchSize && samplesSinceFit >= fitInterval;
    if (fitReady) {
        samplesSinceFit = 0;
    }
    return fitReady;
}

AgentHistoryR::HistoryConstPtr AgentHistoryR::getHistory() const
{
    autil::ScopedLock lock(_lock);
    return _sampler.getHistory();
}

size_t AgentHistoryR::getSessionSampleCount() const
{
    autil::ScopedLock lock(_lock);
    return _sampler.sessionSampleCount();
}

size_t AgentHistoryR::getRetainedSampleCountAtMost(double latencyMs) const
{
    autil::ScopedLock lock(_lock);
    return _sampler.sampleCountAtMost(latencyMs);
}

size_t AgentHistoryR::getFitBatchSize() const
{
    autil::ScopedLock lock(_lock);
    return _fitBatchSize;
}

bool AgentHistoryR::isWarmupPending() const
{
    autil::ScopedLock lock(_lock);
    return _warmupSamplesSkipped < _warmupSampleCount;
}

int64_t AgentHistoryR::getWarmupLocalBatchIdCutoff() const
{
    autil::ScopedLock lock(_lock);
    return _warmupLocalBatchIdCutoff;
}

bool AgentHistoryR::requiresParameterValidation() const
{
    autil::ScopedLock lock(_lock);
    return _parametersRestoredWithoutHistory;
}

bool AgentHistoryR::makeSavePayload(
    const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
    const NonLinearPrefillModelR::OptimizerState &optimizerState,
    std::string &payload) const
{
    payload.clear();
    if (_stateFile.empty()) {
        return true;
    }
    if (!snapshot || !snapshot->valid()) {
        return false;
    }
    const auto parameterCount = snapshot->parameters.size();
    if (optimizerState.step <= 0) {
        return false;
    }
    for (size_t index = 0; index < parameterCount; ++index) {
        if (!std::isfinite(optimizerState.adamMoment1[index]) ||
            !std::isfinite(optimizerState.adamMoment2[index]) ||
            optimizerState.adamMoment2[index] < 0.0) {
            return false;
        }
    }
    autil::DataBuffer payloadBuffer;
    payloadBuffer.write(_formulaFingerprint);
    payloadBuffer.write(_historyFingerprint);
    payloadBuffer.write(_binaryFingerprint);
    payloadBuffer.write(snapshot->formula);
    payloadBuffer.write(snapshot->binaryVersion);
    payloadBuffer.write(snapshot->revision);
    payloadBuffer.write(snapshot->overloadPredictionTrusted);
    payloadBuffer.write(snapshot->warmStarted);
    payloadBuffer.write(snapshot->parameters);
    payloadBuffer.write(
        std::vector<double>(optimizerState.adamMoment1.begin(),
                            optimizerState.adamMoment1.begin() + parameterCount));
    payloadBuffer.write(
        std::vector<double>(optimizerState.adamMoment2.begin(),
                            optimizerState.adamMoment2.begin() + parameterCount));
    payloadBuffer.write(optimizerState.step);
    payloadBuffer.write(snapshot->residualVersion);
    payloadBuffer.write(snapshot->residualParameters);
    payloadBuffer.write(snapshot->residualLastUpdateTimeUs);
    {
        autil::ScopedLock lock(_lock);
        _sampler.serialize(payloadBuffer);
    }
    payload.assign(payloadBuffer.getData(), payloadBuffer.getDataLen());
    return true;
}

bool AgentHistoryR::savePayload(const std::string &payload) const
{
    if (_stateFile.empty()) {
        return true;
    }
    if (payload.empty()) {
        return false;
    }
    const auto checksum = autil::CRC32C::Mask(
        autil::CRC32C::Value(payload.data(), payload.size()));
    autil::DataBuffer dataBuffer;
    dataBuffer.write(LATENCY_STATE_MAGIC);
    dataBuffer.write(LATENCY_STATE_VERSION);
    dataBuffer.write(checksum);
    dataBuffer.write(payload);
    std::filesystem::path statePath(_stateFile);
    std::error_code error;
    if (statePath.has_parent_path()) {
        std::filesystem::create_directories(statePath.parent_path(), error);
        if (error) {
            return false;
        }
    }
    auto temporaryFile = _stateFile + ".tmp";
    {
        std::ofstream output(temporaryFile, std::ios::binary | std::ios::trunc);
        if (!output.write(dataBuffer.getData(), dataBuffer.getDataLen())) {
            return false;
        }
    }
    std::filesystem::rename(temporaryFile, _stateFile, error);
    if (!error) {
        return true;
    }
    std::filesystem::remove(_stateFile, error);
    error.clear();
    std::filesystem::rename(temporaryFile, _stateFile, error);
    return !error;
}

bool AgentHistoryR::save(
    const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
    const NonLinearPrefillModelR::OptimizerState &optimizerState) const
{
    std::string payload;
    return makeSavePayload(snapshot, optimizerState, payload) && savePayload(payload);
}

bool AgentHistoryR::enabled() const { return !_stateFile.empty(); }

void AgentHistoryR::applyRuntimeConfig(const AgentSchedulerConfig &config)
{
    autil::ScopedLock lock(_lock);
    _fitBatchSize = config.latencyFitBatchSize;
    _fitInterval = config.latencyFitInterval;
    _historyLimit = config.latencyHistoryLimit;
    _historyBucketCapacity = config.latencyHistoryBucketCapacity;
    _persistInterval = config.latencyPersistInterval;
    _warmupSampleCount = config.latencyWarmupSampleCount;
    _sampler.configure(_historyLimit, _historyBucketCapacity);
}

navi::SummaryErrorCode AgentHistoryR::getSummary(const std::string &query,
                                                 navi::SummaryBuilder &builder) const
{
    autil::ScopedLock lock(_lock);
    auto table = builder.newTable("NaviSchedAgentHistoryOverview");
    TABLE_ADD_COLUMN_VALUE(table, "history_size", _sampler.historySize());
    TABLE_ADD_COLUMN_VALUE(table, "history_bucket_count", _sampler.bucketCount());
    TABLE_ADD_COLUMN_VALUE(table, "history_limit", _historyLimit);
    TABLE_ADD_COLUMN_VALUE(table, "history_bucket_capacity", _historyBucketCapacity);
    TABLE_ADD_COLUMN_VALUE(table, "formula_fingerprint", _formulaFingerprint);
    TABLE_ADD_COLUMN_VALUE(table, "history_fingerprint", _historyFingerprint);
    TABLE_ADD_COLUMN_VALUE(table, "binary_fingerprint", _binaryFingerprint);
    TABLE_ADD_COLUMN_VALUE(table, "history_restored", _historyRestored);
    TABLE_ADD_COLUMN_VALUE(table, "parameters_restored_without_history",
                           _parametersRestoredWithoutHistory);
    TABLE_ADD_COLUMN_VALUE(table, "fit_batch_size", _fitBatchSize);
    TABLE_ADD_COLUMN_VALUE(table, "fit_interval",
                           _fitInterval > 0 ? _fitInterval : _fitBatchSize);
    TABLE_ADD_COLUMN_VALUE(table, "persist_interval", _persistInterval);
    TABLE_ADD_COLUMN_VALUE(table, "warmup_sample_count", _warmupSampleCount);
    TABLE_ADD_COLUMN_VALUE(table, "warmup_samples_skipped", _warmupSamplesSkipped);
    TABLE_ADD_COLUMN_VALUE(table, "warmup_local_batch_id_cutoff",
                           _warmupLocalBatchIdCutoff);
    TABLE_ADD_COLUMN_VALUE(table, "ramp_collection_observed",
                           _rampCollectionObserved);
    TABLE_ADD_COLUMN_VALUE(table, "ramp_collection_completed",
                           _rampCollectionCompleted);
    TABLE_ADD_COLUMN_VALUE(table, "samples_since_fit", _samplesSinceFit);
    TABLE_ADD_COLUMN_VALUE(table, "samples_since_persist", _samplesSincePersist);
    return navi::SEC_OK;
}

REGISTER_RESOURCE(AgentHistoryR);

} // namespace navi_sched
