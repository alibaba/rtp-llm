#include "aios/navi_sched/resource/AgentSchedulerConfigR.h"

#include "autil/legacy/fast_jsonizable_dec.h"
#include "autil/HashAlgorithm.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "aios/navi_sched/resource/LatencyResidualModel.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>
#include <sstream>
#include <vector>

namespace navi_sched {

const std::string AgentSchedulerConfigR::RESOURCE_ID =
    "navi_sched.agent_scheduler_config_r";

void AgentSchedulerConfig::Jsonize(JsonWrapper &json)
{
    json.Jsonize("batch_timeout_ms", batchTimeoutMs, batchTimeoutMs);
    json.Jsonize("closed_batch_keep_ms", closedBatchKeepMs, closedBatchKeepMs);
    json.Jsonize("latency_fit_batch_size", latencyFitBatchSize, latencyFitBatchSize);
    json.Jsonize("latency_fit_interval", latencyFitInterval, latencyFitInterval);
    json.Jsonize("latency_learning_rate", latencyLearningRate, latencyLearningRate);
    json.Jsonize("latency_online_learning_rate_scale", latencyOnlineLearningRateScale,
                 latencyOnlineLearningRateScale);
    json.Jsonize("latency_current_sample_weight", latencyCurrentSampleWeight,
                 latencyCurrentSampleWeight);
    json.Jsonize("latency_error_clip_ms", latencyErrorClipMs, latencyErrorClipMs);
    json.Jsonize("latency_parameter_learning_rate_scales",
                 latencyParameterLearningRateScales, latencyParameterLearningRateScales);
    json.Jsonize("latency_nonlinear_scale", latencyNonlinearScale, latencyNonlinearScale);
    json.Jsonize("queue_learning_rate", queueLearningRate, queueLearningRate);
    json.Jsonize("latency_check_gradient", latencyCheckGradient, latencyCheckGradient);
    json.Jsonize("latency_cold_start_fit_epochs", latencyColdStartFitEpochs,
                 latencyColdStartFitEpochs);
    json.Jsonize("latency_history_limit", latencyHistoryLimit, latencyHistoryLimit);
    json.Jsonize("latency_history_bucket_capacity", latencyHistoryBucketCapacity,
                 latencyHistoryBucketCapacity);
    json.Jsonize("latency_persist_interval", latencyPersistInterval,
                 latencyPersistInterval);
    json.Jsonize("latency_persist_interval_ms", latencyPersistIntervalMs,
                 latencyPersistIntervalMs);
    json.Jsonize("latency_warmup_sample_count", latencyWarmupSampleCount,
                 latencyWarmupSampleCount);
    json.Jsonize("latency_model_trust_min_revision", latencyModelTrustMinRevision,
                 latencyModelTrustMinRevision);
    json.Jsonize("latency_zero_anchor_max_ms", latencyZeroAnchorMaxMs,
                 latencyZeroAnchorMaxMs);
    json.Jsonize("latency_zero_anchor_min_samples", latencyZeroAnchorMinSamples,
                 latencyZeroAnchorMinSamples);
    json.Jsonize("latency_validation_enabled", latencyValidationEnabled,
                 latencyValidationEnabled);
    json.Jsonize("residual_learning_rate", residualLearningRate, residualLearningRate);
    json.Jsonize("residual_update_error_floor_ms", residualUpdateErrorFloorMs,
                 residualUpdateErrorFloorMs);
    json.Jsonize("residual_update_error_ratio", residualUpdateErrorRatio,
                 residualUpdateErrorRatio);
    json.Jsonize("residual_idle_decay_start_ms", residualIdleDecayStartMs,
                 residualIdleDecayStartMs);
    json.Jsonize("residual_idle_decay_end_ms", residualIdleDecayEndMs,
                 residualIdleDecayEndMs);
    json.Jsonize("time_compression", timeCompression, timeCompression);
}

bool AgentSchedulerConfig::valid() const { return validationError().empty(); }

std::string AgentSchedulerConfig::validationError() const
{
    if (batchTimeoutMs <= 0) return "batch_timeout_ms must be positive";
    if (closedBatchKeepMs < batchTimeoutMs) {
        return "closed_batch_keep_ms must be >= batch_timeout_ms";
    }
    if (latencyFitBatchSize == 0) return "latency_fit_batch_size must be positive";
    if (!std::isfinite(latencyLearningRate) || latencyLearningRate <= 0.0) {
        return "latency_learning_rate must be finite and positive";
    }
    if (!std::isfinite(latencyOnlineLearningRateScale) ||
        latencyOnlineLearningRateScale <= 0.0) {
        return "latency_online_learning_rate_scale must be finite and positive";
    }
    if (!std::isfinite(latencyCurrentSampleWeight) || latencyCurrentSampleWeight <= 0.0) {
        return "latency_current_sample_weight must be finite and positive";
    }
    if (!std::isfinite(latencyErrorClipMs) || latencyErrorClipMs < 0.0) {
        return "latency_error_clip_ms must be finite and nonnegative";
    }
    const auto scaleCount = latencyParameterLearningRateScales.size();
    if (scaleCount != 0 && scaleCount != NonLinearPrefillModelR::PARAMETER_COUNT &&
        scaleCount != NonLinearPrefillModelR::CAPACITY_SPLIT_PARAMETER_COUNT &&
        scaleCount != NonLinearPrefillModelR::CACHE_SOURCE_PARAMETER_COUNT) {
        return "latency_parameter_learning_rate_scales has invalid length";
    }
    for (size_t index = 0; index < scaleCount; ++index) {
        const double scale = latencyParameterLearningRateScales[index];
        if (scaleCount == NonLinearPrefillModelR::CACHE_SOURCE_PARAMETER_COUNT) {
            const bool inactive = index >= NonLinearPrefillModelR::PARAMETER_COUNT &&
                                  index < NonLinearPrefillModelR::CAPACITY_SPLIT_PARAMETER_COUNT;
            if (!std::isfinite(scale) || (inactive ? scale != 0.0 : scale <= 0.0)) {
                return "latency_parameter_learning_rate_scales[" + std::to_string(index) +
                       (inactive ? "] must be zero for inactive cache-source coordinates"
                                 : "] must be finite and positive");
            }
            continue;
        }
        if (!std::isfinite(scale) ||
            (scale <= 0.0 &&
             (scale < 0.0 || index < NonLinearPrefillModelR::
                                         RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX))) {
            return "latency_parameter_learning_rate_scales[" + std::to_string(index) +
                   (index < NonLinearPrefillModelR::
                                RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX
                        ? "] must be finite and positive"
                        : "] must be finite and nonnegative");
        }
    }
    if (!std::isfinite(latencyNonlinearScale) || latencyNonlinearScale <= 0.0) {
        return "latency_nonlinear_scale must be finite and positive";
    }
    if (!std::isfinite(queueLearningRate) || queueLearningRate <= 0.0) {
        return "queue_learning_rate must be finite and positive";
    }
    if (latencyColdStartFitEpochs == 0)
        return "latency_cold_start_fit_epochs must be positive";
    if (latencyHistoryLimit < latencyFitBatchSize) {
        return "latency_history_limit must be >= latency_fit_batch_size";
    }
    if (latencyHistoryBucketCapacity == 0) {
        return "latency_history_bucket_capacity must be positive";
    }
    if (latencyPersistInterval == 0) return "latency_persist_interval must be positive";
    if (latencyPersistIntervalMs <= 0)
        return "latency_persist_interval_ms must be positive";
    if (latencyModelTrustMinRevision <= 1) {
        return "latency_model_trust_min_revision must be > 1";
    }
    if (!std::isfinite(latencyZeroAnchorMaxMs) || latencyZeroAnchorMaxMs <= 0.0) {
        return "latency_zero_anchor_max_ms must be finite and positive";
    }
    if (latencyZeroAnchorMinSamples == 0) {
        return "latency_zero_anchor_min_samples must be positive";
    }
    if (!std::isfinite(residualLearningRate) || residualLearningRate < 0.0 ||
        residualLearningRate > 1.0) {
        return "residual_learning_rate must be finite and within [0, 1]";
    }
    if (!std::isfinite(residualUpdateErrorFloorMs) || residualUpdateErrorFloorMs < 0.0) {
        return "residual_update_error_floor_ms must be finite and nonnegative";
    }
    if (!std::isfinite(residualUpdateErrorRatio) || residualUpdateErrorRatio < 0.0) {
        return "residual_update_error_ratio must be finite and nonnegative";
    }
    if (residualIdleDecayStartMs < 0) {
        return "residual_idle_decay_start_ms must be nonnegative";
    }
    if (residualIdleDecayEndMs <= residualIdleDecayStartMs ||
        residualIdleDecayEndMs > std::numeric_limits<int64_t>::max() / 1000) {
        return "residual_idle_decay_end_ms must exceed start and fit microseconds";
    }
    if (!std::isfinite(timeCompression) || timeCompression <= 0.0) {
        return "time_compression must be finite and positive";
    }
    return {};
}

LatencyResidualModel::RuntimeConfig AgentSchedulerConfig::residualRuntimeConfig() const
{
    return {residualLearningRate, residualUpdateErrorFloorMs, residualUpdateErrorRatio,
            residualIdleDecayStartMs * 1000, residualIdleDecayEndMs * 1000};
}

AgentSchedulerConfig AgentSchedulerConfig::fromServiceConfig(
    const NaviSchedConfig &config, double timeCompression)
{
    AgentSchedulerConfig result;
    result.batchTimeoutMs = config.batchTimeoutMs;
    result.closedBatchKeepMs = config.closedBatchKeepMs;
    result.latencyFitBatchSize = config.latencyFitBatchSize;
    result.latencyFitInterval = config.latencyFitInterval;
    result.latencyLearningRate = config.latencyLearningRate;
    result.latencyOnlineLearningRateScale = config.latencyOnlineLearningRateScale;
    result.latencyCurrentSampleWeight = config.latencyCurrentSampleWeight;
    result.latencyErrorClipMs = config.latencyErrorClipMs;
    result.latencyParameterLearningRateScales = config.latencyParameterLearningRateScales;
    result.latencyNonlinearScale = config.latencyNonlinearScale;
    result.queueLearningRate = config.queueLearningRate;
    result.latencyCheckGradient = config.latencyCheckGradient;
    result.latencyColdStartFitEpochs = config.latencyColdStartFitEpochs;
    result.latencyHistoryLimit = config.latencyHistoryLimit;
    result.latencyHistoryBucketCapacity = config.latencyHistoryBucketCapacity;
    result.latencyPersistInterval = config.latencyPersistInterval;
    result.latencyPersistIntervalMs = config.latencyPersistIntervalMs;
    result.latencyWarmupSampleCount = config.latencyWarmupSampleCount;
    result.latencyModelTrustMinRevision = config.latencyModelTrustMinRevision;
    result.latencyZeroAnchorMaxMs = config.latencyZeroAnchorMaxMs;
    result.latencyZeroAnchorMinSamples = config.latencyZeroAnchorMinSamples;
    result.latencyValidationEnabled = config.latencyValidationEnabled;
    result.residualLearningRate = config.residualLearningRate;
    result.residualUpdateErrorFloorMs = config.residualUpdateErrorFloorMs;
    result.residualUpdateErrorRatio = config.residualUpdateErrorRatio;
    result.residualIdleDecayStartMs = config.residualIdleDecayStartMs;
    result.residualIdleDecayEndMs = config.residualIdleDecayEndMs;
    result.timeCompression = timeCompression;
    return result;
}

void AgentSchedulerConfigR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

bool AgentSchedulerConfigR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "config_json", _initialConfigJson, _initialConfigJson);
    return true;
}

navi::ErrorCode AgentSchedulerConfigR::init(navi::ResourceInitContext &)
{
    if (_initialConfigJson.empty()) {
        _snapshot.configJson = toJson(_snapshot.config);
        _snapshot.configHash = hashJson(_snapshot.configJson);
        _snapshot.generation = 1;
    } else {
        std::string errorMessage;
        if (!parse(_initialConfigJson, _snapshot.config, errorMessage)) {
            NAVI_KERNEL_LOG(ERROR, "invalid initial Agent scheduler config: %s",
                            errorMessage.c_str());
            return navi::EC_INIT_RESOURCE;
        }
        _snapshot.configJson = toJson(_snapshot.config);
        _snapshot.configHash = hashJson(_snapshot.configJson);
        _snapshot.generation = 1;
    }
    NAVI_KERNEL_LOG(INFO,
                    "[navi-scheduler-config-update] role [agent], status [initialized], "
                    "generation [%lu], hash [%s], effectiveConfig [%s]",
                    _snapshot.generation, _snapshot.configHash.c_str(),
                    _snapshot.configJson.c_str());
    return navi::EC_NONE;
}

AgentSchedulerConfigR::Snapshot AgentSchedulerConfigR::get() const
{
    autil::ScopedLock lock(_lock);
    return _snapshot;
}

AgentSchedulerConfigR::ApplyResult AgentSchedulerConfigR::apply(
    uint64_t generation, const std::string &configHash, const std::string &configJson,
    Snapshot &snapshot, std::string &errorMessage)
{
    AgentSchedulerConfig updated;
    if (generation == 0 || configHash.empty() ||
        !parse(configJson, updated, errorMessage)) {
        snapshot = get();
        return ApplyResult::INVALID;
    }
    const auto canonicalJson = toJson(updated);
    if (hashJson(canonicalJson) != configHash) {
        errorMessage = "Agent scheduler config hash mismatch";
        snapshot = get();
        return ApplyResult::INVALID;
    }
    std::vector<navi::AsyncPipePtr> notificationPipes;
    Snapshot previous;
    {
        autil::ScopedLock lock(_lock);
        if (configHash == _snapshot.configHash || generation <= _snapshot.generation) {
            snapshot = _snapshot;
            return ApplyResult::STALE;
        }
        previous = _snapshot;
        _snapshot.config = std::move(updated);
        _snapshot.generation = generation;
        _snapshot.configHash = configHash;
        _snapshot.configJson = std::move(canonicalJson);
        snapshot = _snapshot;
        for (auto it = _subscribers.begin(); it != _subscribers.end();) {
            if (auto pipe = it->second.lock()) {
                notificationPipes.emplace_back(std::move(pipe));
                ++it;
            } else {
                it = _subscribers.erase(it);
            }
        }
    }
    NAVI_KERNEL_LOG(INFO,
                    "[navi-scheduler-config-update] role [agent], status [applied], "
                    "previousGeneration [%lu], generation [%lu], previousHash [%s], "
                    "hash [%s], previousConfig [%s], effectiveConfig [%s]",
                    previous.generation, snapshot.generation, previous.configHash.c_str(),
                    snapshot.configHash.c_str(), previous.configJson.c_str(),
                    snapshot.configJson.c_str());
    for (const auto &pipe : notificationPipes) {
        if (!pipe || navi::EC_NONE != pipe->setDataOverride(nullptr, true, true)) {
            NAVI_KERNEL_LOG(WARN,
                            "notify Agent scheduler config subscriber failed, "
                            "generation [%lu]",
                            generation);
        }
    }
    return ApplyResult::APPLIED;
}

bool AgentSchedulerConfigR::subscribe(const std::string &subscriberId,
                                      const navi::AsyncPipePtr &notificationPipe)
{
    if (subscriberId.empty() || !notificationPipe) {
        return false;
    }
    autil::ScopedLock lock(_lock);
    _subscribers[subscriberId] = notificationPipe;
    return true;
}

void AgentSchedulerConfigR::unsubscribe(const std::string &subscriberId,
                                        const navi::AsyncPipePtr &notificationPipe)
{
    autil::ScopedLock lock(_lock);
    auto it = _subscribers.find(subscriberId);
    if (it == _subscribers.end()) {
        return;
    }
    auto current = it->second.lock();
    if (!notificationPipe || !current || current == notificationPipe) {
        _subscribers.erase(it);
    }
}

bool AgentSchedulerConfigR::parse(const std::string &configJson,
                                  AgentSchedulerConfig &config, std::string &errorMessage)
{
    try {
        AgentSchedulerConfig parsed;
        autil::legacy::FromJsonString(parsed, configJson);
        errorMessage = parsed.validationError();
        if (!errorMessage.empty()) {
            errorMessage = "invalid Agent scheduler config values: " + errorMessage;
            return false;
        }
        config = std::move(parsed);
        return true;
    } catch (const std::exception &error) {
        errorMessage = error.what();
        return false;
    } catch (...) {
        errorMessage = "Agent scheduler config must be a JSON object";
        return false;
    }
}

std::string AgentSchedulerConfigR::toJson(const AgentSchedulerConfig &config)
{
    return autil::legacy::ToJsonString(config, true);
}

std::string AgentSchedulerConfigR::hashConfig(const AgentSchedulerConfig &config)
{
    return hashJson(toJson(config));
}

std::string AgentSchedulerConfigR::hashJson(const std::string &canonicalConfigJson)
{
    const auto value = autil::HashAlgorithm::hashString64(canonicalConfigJson.data(),
                                                          canonicalConfigJson.size());
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
}

REGISTER_RESOURCE(AgentSchedulerConfigR);

}
