#include "aios/navi_sched/adapters/rtp_llm/RtpLLMAdapter.h"

#include <navi_sched/NaviSchedAgent.h>
#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"
#include "autil/legacy/jsonizable.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

namespace navi_sched {

namespace {

thread_local int64_t gRtpLLMCacheReportLatencyUs = 0;

struct AgentConfig : autil::legacy::Jsonizable {
    std::string name;
    std::string bizName;
    std::string logDir;
    std::string latencyModelStateFile;
    std::string latencyModelFingerprint;
    std::string latencyHistoryFingerprint;
    std::string latencyHistoryBinaryFingerprint;
    std::string serverIp;
    std::string latencyPredictionMode = "split";
    size_t latencyFitBatchSize = 4;
    size_t latencyFitInterval = 0;
    double latencyLearningRate = 0.05;
    double latencyOnlineLearningRateScale = 1.0;
    double latencyCurrentSampleWeight = 4.0;
    double latencyNonlinearScale = 1500.0;
    double queueLearningRate = 0.002;
    size_t latencyColdStartFitEpochs = 100;
    size_t latencyHistoryLimit = 2000;
    size_t latencyHistoryBucketCapacity = 64;
    size_t latencyPersistInterval = 256;
    int64_t latencyPersistIntervalMs = 10 * 1000;
    size_t latencyWarmupSampleCount = 10;
    double latencyZeroAnchorMaxMs = 256.0;
    size_t latencyZeroAnchorMinSamples = 8;
    bool latencyValidationEnabled = false;
    double residualLearningRate = 0.8;
    double residualUpdateErrorFloorMs = 100.0;
    double residualUpdateErrorRatio = 0.1;
    int64_t residualIdleDecayStartMs = 2 * 60 * 1000;
    int64_t residualIdleDecayEndMs = 5 * 60 * 1000;
    int64_t batchTimeoutMs = 10;
    int64_t totalBlocks = 0;
    int32_t serverPort = 0;
    int32_t gigPort = 0;

    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("latency_model_state_file", latencyModelStateFile,
                     latencyModelStateFile);
        json.Jsonize("latency_model_fingerprint", latencyModelFingerprint,
                     latencyModelFingerprint);
        json.Jsonize("latency_fit_batch_size", latencyFitBatchSize, latencyFitBatchSize);
        json.Jsonize("latency_fit_interval", latencyFitInterval, latencyFitInterval);
        json.Jsonize("latency_learning_rate", latencyLearningRate, latencyLearningRate);
        json.Jsonize("latency_online_learning_rate_scale", latencyOnlineLearningRateScale,
                     latencyOnlineLearningRateScale);
        json.Jsonize("latency_current_sample_weight", latencyCurrentSampleWeight,
                     latencyCurrentSampleWeight);
        json.Jsonize("latency_nonlinear_scale", latencyNonlinearScale,
                     latencyNonlinearScale);
        json.Jsonize("queue_learning_rate", queueLearningRate, queueLearningRate);
        json.Jsonize("latency_prediction_mode", latencyPredictionMode,
                     latencyPredictionMode);
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
        json.Jsonize("latency_zero_anchor_max_ms", latencyZeroAnchorMaxMs,
                     latencyZeroAnchorMaxMs);
        json.Jsonize("latency_zero_anchor_min_samples", latencyZeroAnchorMinSamples,
                     latencyZeroAnchorMinSamples);
        json.Jsonize("latency_validation", latencyValidationEnabled,
                     latencyValidationEnabled);
        json.Jsonize("residual_learning_rate", residualLearningRate,
                     residualLearningRate);
        json.Jsonize("residual_update_error_floor_ms", residualUpdateErrorFloorMs,
                     residualUpdateErrorFloorMs);
        json.Jsonize("residual_update_error_ratio", residualUpdateErrorRatio,
                     residualUpdateErrorRatio);
        json.Jsonize("residual_idle_decay_start_ms", residualIdleDecayStartMs,
                     residualIdleDecayStartMs);
        json.Jsonize("residual_idle_decay_end_ms", residualIdleDecayEndMs,
                     residualIdleDecayEndMs);
        json.Jsonize("batch_timeout_ms", batchTimeoutMs, batchTimeoutMs);
        json.Jsonize("cache_total_blocks", totalBlocks, totalBlocks);
        json.Jsonize("gig_port", gigPort, gigPort);
        json.Jsonize("biz_name", bizName, bizName);
    }
};

std::string firstNonEmptyEnv(std::initializer_list<const char *> names,
                             const std::string &defaultValue = "")
{
    for (const auto *name : names) {
        auto value = autil::EnvUtil::getEnv(name, std::string());
        if (!value.empty()) {
            return value;
        }
    }
    return defaultValue;
}

std::string makeLatencyModelFingerprint()
{
    autil::legacy::json::JsonMap values;
    values["fingerprint_version"] = int64_t(2);
    values["framework"] = std::string("rtp_llm");
    values["hardware"] =
        firstNonEmptyEnv({"RTP_LLM_NAVI_SCHED_ACCELERATOR_TYPE", "ACCELERATOR_TYPE",
                          "GPU_TYPE", "DEPLOYMENT_NAME", "HIPPO_ROLE"},
                         "unknown");
    values["tensor_parallel_size"] = firstNonEmptyEnv({"TP_SIZE"}, "1");
    values["pipeline_parallel_size"] = firstNonEmptyEnv({"PP_SIZE"}, "1");
    values["data_parallel_size"] = firstNonEmptyEnv({"DP_SIZE"}, "1");
    values["expert_parallel_size"] = firstNonEmptyEnv({"EP_SIZE"}, "1");
    values["sequence_parallel_size"] = firstNonEmptyEnv({"SP_SIZE"}, "1");
    values["model"] =
        firstNonEmptyEnv({"MODEL_TYPE", "MODEL_NAME", "MODEL_ID", "CHECKPOINT_PATH"}, "");
    values["dtype"] = firstNonEmptyEnv({"WEIGHT_TYPE", "DATA_TYPE", "DTYPE"}, "");
    values["quantization"] = firstNonEmptyEnv(
        {"QUANTIZATION", "QUANTIZATION_TYPE", "QUANT_ALGO", "QUANT_METHOD"}, "");
    values["kv_cache_dtype"] = firstNonEmptyEnv({"KV_CACHE_TYPE", "KV_CACHE_DTYPE"}, "");
    values["speculative_method"] =
        firstNonEmptyEnv({"SPECULATIVE_METHOD", "SPECULATIVE_ALGORITHM", "SP_TYPE"}, "");
    values["speculative_model"] = firstNonEmptyEnv(
        {"SPECULATIVE_MODEL", "DRAFT_MODEL", "DRAFT_MODEL_TYPE", "SP_MODEL_TYPE"}, "");
    values["speculative_quantization"] = firstNonEmptyEnv(
        {"SPECULATIVE_QUANTIZATION", "DRAFT_QUANTIZATION", "DRAFT_QUANT_METHOD"}, "");
    values["speculative_tensor_parallel_size"] =
        firstNonEmptyEnv({"SPECULATIVE_TP_SIZE", "DRAFT_TP_SIZE", "SP_TP_SIZE"}, "");
    values["num_speculative_tokens"] = firstNonEmptyEnv(
        {"NUM_SPECULATIVE_TOKENS", "SPECULATIVE_TOKENS", "GEN_NUM_PER_CIRCLE"}, "0");
    return autil::legacy::ToJsonString(values, true);
}

std::string makeLatencyHistoryFingerprint(const std::string &formulaFingerprint,
                                          const std::string &latencyPredictionMode)
{
    autil::legacy::json::JsonMap values;
    values["fingerprint_version"] = int64_t(2);
    values["formula_fingerprint"] = formulaFingerprint;
    values["latency_prediction_mode"] = latencyPredictionMode;
    values["max_context_batch_size"] =
        firstNonEmptyEnv({"MAX_CONTEXT_BATCH_SIZE", "CONCURRENCY_LIMIT"}, "");
    values["max_seq_len"] = firstNonEmptyEnv({"MAX_SEQ_LEN"}, "");
    values["model_type"] = firstNonEmptyEnv({"MODEL_TYPE"}, "");
    values["model_name"] = firstNonEmptyEnv({"MODEL_NAME"}, "");
    values["model_id"] = firstNonEmptyEnv({"MODEL_ID"}, "");
    values["checkpoint_path"] = firstNonEmptyEnv({"CHECKPOINT_PATH"}, "");
    values["tokenizer_path"] = firstNonEmptyEnv({"TOKENIZER_PATH"}, "");
    values["model_revision"] =
        firstNonEmptyEnv({"MODEL_REVISION", "CHECKPOINT_REVISION", "MODEL_VERSION"}, "");
    values["model_code_revision"] =
        firstNonEmptyEnv({"MODEL_CODE_REVISION", "CODE_REVISION"}, "");
    values["speculative_model_revision"] =
        firstNonEmptyEnv({"SPECULATIVE_MODEL_REVISION", "DRAFT_MODEL_REVISION"}, "");
    values["speculative_model_code_revision"] = firstNonEmptyEnv(
        {"SPECULATIVE_MODEL_CODE_REVISION", "DRAFT_MODEL_CODE_REVISION"}, "");
    return autil::legacy::ToJsonString(values, true);
}

std::string makeLatencyHistoryBinaryFingerprint()
{
    autil::legacy::json::JsonMap values;
    values["framework"] = std::string("rtp_llm");
    values["version"] = firstNonEmptyEnv(
        {"RTP_LLM_VERSION", "RTP_LLM_PACKAGE_VERSION", "PACKAGE_VERSION"}, "");
    values["commit"] = firstNonEmptyEnv(
        {"RTP_LLM_COMMIT", "RTP_LLM_GIT_COMMIT", "FRAMEWORK_GIT_COMMIT"}, "");
    return autil::legacy::ToJsonString(values, true);
}

struct RtpLLMNaviSchedConfig : autil::legacy::Jsonizable {
    std::string logPath;
    AgentConfig agent;

    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("log_path", logPath, logPath);
        json.Jsonize("agent", agent, agent);
    }
};

struct RtpLLMEndpointConfig : autil::legacy::Jsonizable {
    std::string address;

    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("address", address, address);
    }
};

struct RtpLLMGroupEndpointConfig : autil::legacy::Jsonizable {
    RtpLLMEndpointConfig prefillEndpoint;

    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("prefill_endpoint", prefillEndpoint, prefillEndpoint);
    }
};

struct RtpLLMServiceRouteConfig : autil::legacy::Jsonizable {
    bool useLocal = false;
    std::vector<RtpLLMGroupEndpointConfig> roleEndpoints;

    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("use_local", useLocal, useLocal);
        json.Jsonize("role_endpoints", roleEndpoints, roleEndpoints);
    }
};

bool deriveAgentBizName(std::string &bizName, std::string &errorMessage)
{
    const auto value = autil::EnvUtil::getEnv("MODEL_SERVICE_CONFIG", std::string());
    if (value.empty()) {
        return false;
    }
    RtpLLMServiceRouteConfig route;
    try {
        autil::legacy::FromJsonString(route, value);
    } catch (const std::exception &error) {
        errorMessage = std::string("invalid MODEL_SERVICE_CONFIG: ") + error.what();
        return false;
    }
    if (route.useLocal) {
        return false;
    }
    std::set<std::string> addresses;
    for (const auto &group : route.roleEndpoints) {
        if (!group.prefillEndpoint.address.empty()) {
            addresses.insert(group.prefillEndpoint.address);
        }
    }
    if (addresses.empty()) {
        return false;
    }
    if (addresses.size() != 1) {
        errorMessage =
            "NaviSched currently requires all prefill endpoints to use the same address";
        return false;
    }
    bizName = *addresses.begin();
    return true;
}

struct RetainedRtpBatch {
    NaviSchedBatchPtr nativeBatch;
    std::vector<uint64_t> engineRequestIds;
    std::unordered_map<uint64_t, uint64_t> nativeRequestIdByEngineId;
};

struct RetainedBatchRegistry {
    std::mutex mutex;
    std::unordered_map<uint64_t, std::shared_ptr<RetainedRtpBatch>> batches;
};

RetainedBatchRegistry &retainedBatchRegistry()
{
    static RetainedBatchRegistry registry;
    return registry;
}

using ReporterRequests =
    std::unordered_map<const NaviSchedBatchFinishReporter *,
                       std::pair<NaviSchedBatchFinishReporterPtr,
                                 std::vector<std::pair<uint64_t, uint64_t>>>>;

ReporterRequests getRetainedReporterRequests(
    const std::vector<NaviSchedLatencyQuery> &queries, bool erase)
{
    ReporterRequests grouped;
    auto &registry = retainedBatchRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (const auto &query : queries) {
        auto it = registry.batches.find(query.requestId);
        if (it == registry.batches.end()) {
            continue;
        }
        const auto retained = it->second;
        auto batch = retained ? retained->nativeBatch : nullptr;
        if (!batch || !batch->finishReporter) {
            registry.batches.erase(it);
            continue;
        }
        auto &entry = grouped[batch->finishReporter.get()];
        entry.first = batch->finishReporter;
        entry.second.emplace_back(
            query.requestId, retained->nativeRequestIdByEngineId.at(query.requestId));
        if (erase) {
            registry.batches.erase(it);
        }
    }
    return grouped;
}

bool applyUnifiedConfig(AgentConfig &config, std::string &errorMessage)
{
    const auto value =
        autil::EnvUtil::getEnv("RTP_LLM_NAVI_SCHED_CONFIG_JSON", std::string());
    if (value.empty()) {
        return true;
    }
    RtpLLMNaviSchedConfig unifiedConfig;
    unifiedConfig.logPath = config.logDir;
    unifiedConfig.agent = config;
    try {
        autil::legacy::FromJsonString(unifiedConfig, value);
    } catch (const std::exception &error) {
        errorMessage =
            std::string("invalid RTP_LLM_NAVI_SCHED_CONFIG_JSON: ") + error.what();
        return false;
    }
    config = std::move(unifiedConfig.agent);
    config.logDir = std::move(unifiedConfig.logPath);
    return true;
}

std::shared_ptr<NaviSchedAgent> createAgent(const AgentConfig &agentConfig,
                                            std::string &errorMessage)
{
    if (agentConfig.bizName.empty()) {
        errorMessage = "NaviSched Agent biz name is empty";
        return nullptr;
    }
    NaviSchedConfig config;
    config.name = agentConfig.name;
    config.enableKmon = true;
    config.kmonServiceName =
        autil::EnvUtil::getEnv("kmonitorServiceName", std::string("navi_sched"));
    if (config.kmonServiceName.empty()) {
        config.kmonServiceName = "navi_sched";
    }
    config.kmonMetricsPath.clear();
    config.kmonTags.emplace("role", "agent");
    if (!agentConfig.serverIp.empty()) {
        config.kmonTags.emplace("agent_ip", agentConfig.serverIp);
    }
    const auto deploymentName = autil::EnvUtil::getEnv("DEPLOYMENT_NAME", std::string());
    if (!deploymentName.empty()) {
        config.kmonTags.emplace("deployment_name", deploymentName);
    }
    config.gig.grpcPort = agentConfig.gigPort;
    config.gig.grpcIoThreadNum = 20;
    config.waitNoQuery = false;
    config.batchTimeoutMs = agentConfig.batchTimeoutMs;
    config.latencyFitBatchSize = agentConfig.latencyFitBatchSize;
    config.latencyFitInterval = agentConfig.latencyFitInterval;
    config.latencyLearningRate = agentConfig.latencyLearningRate;
    config.latencyOnlineLearningRateScale = agentConfig.latencyOnlineLearningRateScale;
    config.latencyCurrentSampleWeight = agentConfig.latencyCurrentSampleWeight;
    config.latencyNonlinearScale = agentConfig.latencyNonlinearScale;
    config.queueLearningRate = agentConfig.queueLearningRate;
    config.latencyColdStartFitEpochs = agentConfig.latencyColdStartFitEpochs;
    config.latencyHistoryLimit = agentConfig.latencyHistoryLimit;
    config.latencyHistoryBucketCapacity = agentConfig.latencyHistoryBucketCapacity;
    config.latencyPersistInterval = agentConfig.latencyPersistInterval;
    config.latencyPersistIntervalMs = agentConfig.latencyPersistIntervalMs;
    config.latencyWarmupSampleCount = agentConfig.latencyWarmupSampleCount;
    config.latencyZeroAnchorMaxMs = agentConfig.latencyZeroAnchorMaxMs;
    config.latencyZeroAnchorMinSamples = agentConfig.latencyZeroAnchorMinSamples;
    config.latencyValidationEnabled = agentConfig.latencyValidationEnabled;
    config.residualLearningRate = agentConfig.residualLearningRate;
    config.residualUpdateErrorFloorMs = agentConfig.residualUpdateErrorFloorMs;
    config.residualUpdateErrorRatio = agentConfig.residualUpdateErrorRatio;
    config.residualIdleDecayStartMs = agentConfig.residualIdleDecayStartMs;
    config.residualIdleDecayEndMs = agentConfig.residualIdleDecayEndMs;
    config.latencyModelStateFile = agentConfig.latencyModelStateFile;
    config.logFileName = "./logs/navi_sched.log";
    config.snapshotDir = "navi_sched";
    config.gig.snapshotLogDir = "navi_sched";
    NaviSchedAgentConfig configValue;
    configValue.service = config;
    configValue.runtime.bizName = agentConfig.bizName;
    configValue.latencyModelFingerprint = agentConfig.latencyModelFingerprint;
    configValue.latencyHistoryFingerprint = agentConfig.latencyHistoryFingerprint;
    configValue.latencyHistoryBinaryFingerprint =
        agentConfig.latencyHistoryBinaryFingerprint;
    configValue.unifiedLatencyPrediction = agentConfig.latencyPredictionMode == "unified";
    auto agent =
        std::make_shared<NaviSchedAgent>(configValue, agentConfig.serverPort,
                                         agentConfig.serverIp, agentConfig.totalBlocks);
    if (!agent->init()) {
        agent->stop();
        errorMessage = "failed to initialize NaviSched Agent";
        return nullptr;
    }
    return agent;
}

}

RtpLLMAgentInitStatus createRtpLLMAgentFromEnvironment(
    bool enableForRole, const std::string &serverIp, int32_t serverPort,
    std::shared_ptr<NaviSchedAgent> &agent, std::string &errorMessage)
{
    agent.reset();
    if (!enableForRole) {
        return RtpLLMAgentInitStatus::OK;
    }
    AgentConfig config;
    config.name = "sched_agent";
    config.bizName = autil::EnvUtil::getEnv("RTP_LLM_NAVI_SCHED_BIZ_NAME", std::string());
    config.logDir = autil::EnvUtil::getEnv("LOG_PATH", std::string("logs"));
    config.latencyModelStateFile = autil::EnvUtil::getEnv(
        "RTP_LLM_NAVI_SCHED_LATENCY_MODEL_STATE_FILE", std::string());
    config.latencyModelFingerprint = autil::EnvUtil::getEnv(
        "RTP_LLM_NAVI_SCHED_LATENCY_MODEL_FINGERPRINT", std::string());
    config.latencyFitBatchSize =
        autil::EnvUtil::getEnv<size_t>("RTP_LLM_NAVI_SCHED_LATENCY_FIT_BATCH_SIZE", 4);
    config.latencyFitInterval =
        autil::EnvUtil::getEnv<size_t>("RTP_LLM_NAVI_SCHED_LATENCY_FIT_INTERVAL", 0);
    config.latencyLearningRate =
        autil::EnvUtil::getEnv<double>("RTP_LLM_NAVI_SCHED_LATENCY_LEARNING_RATE", 0.05);
    config.latencyOnlineLearningRateScale = autil::EnvUtil::getEnv<double>(
        "RTP_LLM_NAVI_SCHED_LATENCY_ONLINE_LEARNING_RATE_SCALE", 1.0);
    config.latencyCurrentSampleWeight = autil::EnvUtil::getEnv<double>(
        "RTP_LLM_NAVI_SCHED_LATENCY_CURRENT_SAMPLE_WEIGHT", 4.0);
    config.latencyNonlinearScale = autil::EnvUtil::getEnv<double>(
        "RTP_LLM_NAVI_SCHED_LATENCY_NONLINEAR_SCALE", 1500.0);
    config.queueLearningRate =
        autil::EnvUtil::getEnv<double>("RTP_LLM_NAVI_SCHED_QUEUE_LEARNING_RATE", 0.002);
    config.latencyPredictionMode = autil::EnvUtil::getEnv(
        "RTP_LLM_NAVI_SCHED_LATENCY_PREDICTION_MODE", std::string("split"));
    config.latencyColdStartFitEpochs = autil::EnvUtil::getEnv<size_t>(
        "RTP_LLM_NAVI_SCHED_LATENCY_COLD_START_FIT_EPOCHS", 100);
    config.latencyHistoryLimit =
        autil::EnvUtil::getEnv<size_t>("RTP_LLM_NAVI_SCHED_LATENCY_HISTORY_LIMIT", 2000);
    config.latencyHistoryBucketCapacity = autil::EnvUtil::getEnv<size_t>(
        "RTP_LLM_NAVI_SCHED_LATENCY_HISTORY_BUCKET_CAPACITY", 64);
    config.latencyPersistInterval = autil::EnvUtil::getEnv<size_t>(
        "RTP_LLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL", 256);
    config.latencyPersistIntervalMs = autil::EnvUtil::getEnv<int64_t>(
        "RTP_LLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL_MS", 10 * 1000);
    config.latencyWarmupSampleCount = autil::EnvUtil::getEnv<size_t>(
        "RTP_LLM_NAVI_SCHED_LATENCY_WARMUP_SAMPLE_COUNT", 10);
    config.latencyZeroAnchorMaxMs = autil::EnvUtil::getEnv<double>(
        "RTP_LLM_NAVI_SCHED_LATENCY_ZERO_ANCHOR_MAX_MS", 256.0);
    config.latencyZeroAnchorMinSamples = autil::EnvUtil::getEnv<size_t>(
        "RTP_LLM_NAVI_SCHED_LATENCY_ZERO_ANCHOR_MIN_SAMPLES", 8);
    config.latencyValidationEnabled =
        autil::EnvUtil::getEnv("RTP_LLM_NAVI_SCHED_LATENCY_VALIDATION", false);
    config.residualLearningRate =
        autil::EnvUtil::getEnv<double>("RTP_LLM_NAVI_SCHED_RESIDUAL_LEARNING_RATE", 0.03);
    config.residualUpdateErrorFloorMs = autil::EnvUtil::getEnv<double>(
        "RTP_LLM_NAVI_SCHED_RESIDUAL_UPDATE_ERROR_FLOOR_MS", 100.0);
    config.residualUpdateErrorRatio = autil::EnvUtil::getEnv<double>(
        "RTP_LLM_NAVI_SCHED_RESIDUAL_UPDATE_ERROR_RATIO", 0.1);
    config.residualIdleDecayStartMs = autil::EnvUtil::getEnv<int64_t>(
        "RTP_LLM_NAVI_SCHED_RESIDUAL_IDLE_DECAY_START_MS", 2 * 60 * 1000);
    config.residualIdleDecayEndMs = autil::EnvUtil::getEnv<int64_t>(
        "RTP_LLM_NAVI_SCHED_RESIDUAL_IDLE_DECAY_END_MS", 5 * 60 * 1000);
    config.batchTimeoutMs =
        autil::EnvUtil::getEnv<int64_t>("RTP_LLM_NAVI_SCHED_BATCH_TIMEOUT_MS", 10);
    config.totalBlocks =
        autil::EnvUtil::getEnv<int64_t>("RTP_LLM_NAVI_SCHED_CACHE_TOTAL_BLOCKS", 0);
    config.serverIp = serverIp;
    config.serverPort = serverPort;
    config.gigPort =
        autil::EnvUtil::getEnv("RTP_LLM_NAVI_SCHED_AGENT_GIG_PORT", serverPort + 2);
    if (!applyUnifiedConfig(config, errorMessage)) {
        return RtpLLMAgentInitStatus::INVALID_CONFIG;
    }
    if (config.latencyModelFingerprint.empty()) {
        config.latencyModelFingerprint = makeLatencyModelFingerprint();
    }
    config.latencyHistoryFingerprint = makeLatencyHistoryFingerprint(
        config.latencyModelFingerprint, config.latencyPredictionMode);
    config.latencyHistoryBinaryFingerprint = makeLatencyHistoryBinaryFingerprint();
    if (config.bizName.empty() && !deriveAgentBizName(config.bizName, errorMessage)) {
        if (!errorMessage.empty()) {
            return RtpLLMAgentInitStatus::INVALID_CONFIG;
        }
        config.bizName = autil::EnvUtil::getEnv("BIZ_NAME", std::string());
    }
    if (config.bizName.empty()) {
        errorMessage = "RTP_LLM_NAVI_SCHED_BIZ_NAME is required";
        return RtpLLMAgentInitStatus::INVALID_CONFIG;
    }
    if (config.gigPort < 0 || config.gigPort > 65535) {
        errorMessage = "RTP_LLM_NAVI_SCHED_AGENT_GIG_PORT must be between 0 and 65535";
        return RtpLLMAgentInitStatus::INVALID_CONFIG;
    }
    if (!std::isfinite(config.latencyLearningRate) || config.latencyLearningRate <= 0.0) {
        errorMessage =
            "RTP_LLM_NAVI_SCHED_LATENCY_LEARNING_RATE must be positive and finite";
        return RtpLLMAgentInitStatus::INVALID_CONFIG;
    }
    if (!std::isfinite(config.latencyNonlinearScale) ||
        config.latencyNonlinearScale <= 0.0) {
        errorMessage =
            "RTP_LLM_NAVI_SCHED_LATENCY_NONLINEAR_SCALE must be positive and finite";
        return RtpLLMAgentInitStatus::INVALID_CONFIG;
    }
    if (!std::isfinite(config.latencyOnlineLearningRateScale) ||
        config.latencyOnlineLearningRateScale <= 0.0 ||
        !std::isfinite(config.latencyCurrentSampleWeight) ||
        config.latencyCurrentSampleWeight <= 0.0 ||
        config.latencyHistoryBucketCapacity == 0 ||
        !std::isfinite(config.latencyZeroAnchorMaxMs) ||
        config.latencyZeroAnchorMaxMs <= 0.0 || config.latencyZeroAnchorMinSamples == 0 ||
        !std::isfinite(config.residualLearningRate) ||
        config.residualLearningRate <= 0.0 || config.residualLearningRate > 1.0 ||
        !std::isfinite(config.residualUpdateErrorFloorMs) ||
        config.residualUpdateErrorFloorMs < 0.0 ||
        !std::isfinite(config.residualUpdateErrorRatio) ||
        config.residualUpdateErrorRatio < 0.0 || config.residualIdleDecayStartMs < 0 ||
        config.residualIdleDecayEndMs <= config.residualIdleDecayStartMs) {
        errorMessage = "invalid RTP-LLM NaviSched Agent runtime config";
        return RtpLLMAgentInitStatus::INVALID_CONFIG;
    }
    if (!std::isfinite(config.queueLearningRate) || config.queueLearningRate <= 0.0) {
        errorMessage =
            "RTP_LLM_NAVI_SCHED_QUEUE_LEARNING_RATE must be positive and finite";
        return RtpLLMAgentInitStatus::INVALID_CONFIG;
    }
    if (config.latencyPredictionMode != "split" &&
        config.latencyPredictionMode != "unified") {
        errorMessage =
            "RTP_LLM_NAVI_SCHED_LATENCY_PREDICTION_MODE must be split or unified";
        return RtpLLMAgentInitStatus::INVALID_CONFIG;
    }
    if (config.batchTimeoutMs <= 0) {
        errorMessage = "RTP_LLM_NAVI_SCHED_BATCH_TIMEOUT_MS must be positive";
        return RtpLLMAgentInitStatus::INVALID_CONFIG;
    }
    if (config.latencyPersistIntervalMs <= 0) {
        errorMessage = "RTP_LLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL_MS must be positive";
        return RtpLLMAgentInitStatus::INVALID_CONFIG;
    }
    autil::legacy::json::JsonMap effectiveConfig;
    effectiveConfig["name"] = config.name;
    effectiveConfig["biz_name"] = config.bizName;
    effectiveConfig["log_dir"] = config.logDir;
    effectiveConfig["latency_model_state_file"] = config.latencyModelStateFile;
    effectiveConfig["latency_model_fingerprint"] = config.latencyModelFingerprint;
    effectiveConfig["server_ip"] = config.serverIp;
    effectiveConfig["server_port"] = config.serverPort;
    effectiveConfig["gig_port"] = config.gigPort;
    effectiveConfig["latency_prediction_mode"] = config.latencyPredictionMode;
    effectiveConfig["latency_fit_batch_size"] = config.latencyFitBatchSize;
    effectiveConfig["latency_fit_interval"] = config.latencyFitInterval;
    effectiveConfig["latency_learning_rate"] = config.latencyLearningRate;
    effectiveConfig["latency_online_learning_rate_scale"] =
        config.latencyOnlineLearningRateScale;
    effectiveConfig["latency_current_sample_weight"] = config.latencyCurrentSampleWeight;
    effectiveConfig["latency_nonlinear_scale"] = config.latencyNonlinearScale;
    effectiveConfig["queue_learning_rate"] = config.queueLearningRate;
    effectiveConfig["latency_cold_start_fit_epochs"] = config.latencyColdStartFitEpochs;
    effectiveConfig["latency_history_limit"] = config.latencyHistoryLimit;
    effectiveConfig["latency_history_bucket_capacity"] =
        config.latencyHistoryBucketCapacity;
    effectiveConfig["latency_persist_interval"] = config.latencyPersistInterval;
    effectiveConfig["latency_persist_interval_ms"] = config.latencyPersistIntervalMs;
    effectiveConfig["latency_warmup_sample_count"] = config.latencyWarmupSampleCount;
    effectiveConfig["latency_zero_anchor_max_ms"] = config.latencyZeroAnchorMaxMs;
    effectiveConfig["latency_zero_anchor_min_samples"] =
        config.latencyZeroAnchorMinSamples;
    effectiveConfig["latency_validation"] = config.latencyValidationEnabled;
    effectiveConfig["residual_learning_rate"] = config.residualLearningRate;
    effectiveConfig["residual_update_error_floor_ms"] = config.residualUpdateErrorFloorMs;
    effectiveConfig["residual_update_error_ratio"] = config.residualUpdateErrorRatio;
    effectiveConfig["residual_idle_decay_start_ms"] = config.residualIdleDecayStartMs;
    effectiveConfig["residual_idle_decay_end_ms"] = config.residualIdleDecayEndMs;
    effectiveConfig["batch_timeout_ms"] = config.batchTimeoutMs;
    effectiveConfig["cache_total_blocks"] = config.totalBlocks;
    const auto effectiveConfigJson = autil::legacy::ToJsonString(effectiveConfig, true);
    agent = createAgent(config, errorMessage);
    if (!agent) {
        return RtpLLMAgentInitStatus::INIT_FAILED;
    }
    std::fprintf(stderr, "navi_sched_effective_rtp_agent_config=%s\n",
                 effectiveConfigJson.c_str());
    return RtpLLMAgentInitStatus::OK;
}

int32_t getRtpLLMAgentGrpcPort(const std::shared_ptr<NaviSchedAgent> &agent)
{
    return agent ? agent->getGrpcPort() : -1;
}

void stopRtpLLMAgent(const std::shared_ptr<NaviSchedAgent> &agent)
{
    if (agent) {
        agent->stop();
    }
}

bool setRtpLLMForwardBatchHandler(const std::shared_ptr<NaviSchedAgent> &agent,
                                  const NaviSchedForwardBatchHandler &handler)
{
    return agent && agent->setForwardBatchHandler(handler, false);
}

bool addRtpLLMBatchRequest(const std::shared_ptr<NaviSchedAgent> &agent,
                           const NaviSchedBatchRequest &request, int64_t batchId,
                           int64_t expectedBatchSize,
                           const NaviSchedReadyBatchCallback &callback)
{
    if (!agent || 0 == request.requestId || !callback || batchId < 0 ||
        expectedBatchSize < 0 || ((0 == batchId) != (0 == expectedBatchSize))) {
        return false;
    }
    return agent->addBatchRequest(request, batchId, expectedBatchSize, callback);
}

bool startRtpLLMBatch(const NaviSchedBatchFinishReporterPtr &finishReporter,
                      const std::vector<uint64_t> &requestIds)
{
    return finishReporter && finishReporter->startBatch(requestIds);
}

bool finishRtpLLMBatch(const NaviSchedBatchFinishReporterPtr &finishReporter,
                       const std::vector<NaviSchedRequestStats> &requests)
{
    return finishReporter && finishReporter->finishBatch(requests);
}

bool retainRtpLLMReadyBatch(const NaviSchedBatchPtr &batch)
{
    return static_cast<bool>(retainRtpLLMReadyBatchWithResult(batch));
}

RtpLLMReadyBatchRetainResult retainRtpLLMReadyBatchWithResult(
    const NaviSchedBatchPtr &batch)
{
    return retainRtpLLMForwardBatchWithResult(
        batch, batch ? batch->requestIds : std::vector<uint64_t>());
}

RtpLLMReadyBatchRetainResult retainRtpLLMForwardBatchWithResult(
    const NaviSchedBatchPtr &batch, const std::vector<uint64_t> &engineRequestIds)
{
    if (!batch || !batch->finishReporter || batch->requestIds.empty() ||
        batch->requestIds.size() != engineRequestIds.size()) {
        return {RtpLLMReadyBatchRetainStatus::INVALID_BATCH, 0, 0};
    }
    auto retained = std::make_shared<RetainedRtpBatch>();
    retained->nativeBatch = batch;
    retained->engineRequestIds = engineRequestIds;
    std::unordered_set<uint64_t> nativeIds;
    for (size_t index = 0; index < engineRequestIds.size(); ++index) {
        const auto engineId = engineRequestIds[index];
        const auto nativeId = batch->requestIds[index];
        if (engineId == 0 || nativeId == 0) {
            return {RtpLLMReadyBatchRetainStatus::ZERO_REQUEST_ID, engineId, 0};
        }
        if (!nativeIds.emplace(nativeId).second ||
            !retained->nativeRequestIdByEngineId.emplace(engineId, nativeId).second) {
            return {RtpLLMReadyBatchRetainStatus::DUPLICATE_REQUEST_ID, engineId, 0};
        }
    }
    auto &registry = retainedBatchRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (auto engineId : engineRequestIds) {
        if (registry.batches.count(engineId) != 0) {
            return {RtpLLMReadyBatchRetainStatus::DUPLICATE_REQUEST_ID, engineId,
                    registry.batches.size()};
        }
    }
    for (auto engineId : engineRequestIds) {
        registry.batches.emplace(engineId, retained);
    }
    return {RtpLLMReadyBatchRetainStatus::OK, 0, registry.batches.size()};
}

RtpLLMRetainedBatchLease::RtpLLMRetainedBatchLease(uint64_t requestId)
    : _requestId(requestId)
{
}

RtpLLMRetainedBatchLease::~RtpLLMRetainedBatchLease() { abort(); }

void RtpLLMRetainedBatchLease::abort()
{
    if (_active.exchange(false, std::memory_order_acq_rel)) {
        abortRetainedRtpLLMBatch(_requestId);
    }
}

void RtpLLMRetainedBatchLease::dismiss()
{
    _active.store(false, std::memory_order_release);
}

RtpLLMRetainedBatchLeasePtr acquireRtpLLMRetainedBatchLease(uint64_t requestId)
{
    auto &registry = retainedBatchRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto it = registry.batches.find(requestId);
    if (registry.batches.end() == it || !it->second) {
        return nullptr;
    }
    return RtpLLMRetainedBatchLeasePtr(new RtpLLMRetainedBatchLease(requestId));
}

bool abortRetainedRtpLLMBatch(uint64_t requestId)
{
    std::shared_ptr<RetainedRtpBatch> retained;
    auto &registry = retainedBatchRegistry();
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        auto it = registry.batches.find(requestId);
        if (registry.batches.end() == it) {
            return false;
        }
        retained = it->second;
        if (!retained || !retained->nativeBatch) {
            registry.batches.erase(it);
            return false;
        }
        for (auto engineId : retained->engineRequestIds) {
            auto current = registry.batches.find(engineId);
            if (registry.batches.end() != current && current->second == retained) {
                registry.batches.erase(current);
            }
        }
    }
    retained->nativeBatch->finishReporter.reset();
    return true;
}

bool startRetainedRtpLLMBatches(const std::vector<NaviSchedLatencyQuery> &queries)
{
    return startRetainedRtpLLMBatches(queries, 0);
}

bool startRetainedRtpLLMBatches(const std::vector<NaviSchedLatencyQuery> &queries,
                                int64_t executionStartTimeUs)
{
    auto grouped = getRetainedReporterRequests(queries, false);
    for (const auto &entry : grouped) {
        std::vector<uint64_t> nativeIds;
        nativeIds.reserve(entry.second.second.size());
        for (const auto &ids : entry.second.second) {
            nativeIds.emplace_back(ids.second);
        }
        if (!entry.second.first->startBatch(nativeIds, executionStartTimeUs)) {
            return false;
        }
    }
    return true;
}

void reportRtpLLMCacheInserted(const std::shared_ptr<NaviSchedAgent> &agent,
                               const NaviSchedBlockHashesConstPtr &addedBlockHashes)
{
    if (!agent) {
        return;
    }
    const auto beginUs = autil::TimeUtility::currentTimeInMicroSeconds();
    agent->reportCacheInserted(addedBlockHashes);
    gRtpLLMCacheReportLatencyUs +=
        autil::TimeUtility::currentTimeInMicroSeconds() - beginUs;
}

void reportRtpLLMCacheRemoved(const std::shared_ptr<NaviSchedAgent> &agent,
                              const NaviSchedBlockHashesConstPtr &removedBlockHashes)
{
    if (!agent) {
        return;
    }
    const auto beginUs = autil::TimeUtility::currentTimeInMicroSeconds();
    agent->reportCacheRemoved(removedBlockHashes);
    gRtpLLMCacheReportLatencyUs +=
        autil::TimeUtility::currentTimeInMicroSeconds() - beginUs;
}

void recordRtpLLMPrefillBatch(
    const std::shared_ptr<NaviSchedAgent> &agent,
    const std::vector<NaviSchedLatencyQuery> &queries, int64_t latencyUs,
    int64_t eventTimeUs,
    const std::vector<NaviSchedBatchFinishReporterPtr> &finishReporters)
{
    if (!agent || queries.empty() || latencyUs < 0) {
        return;
    }
    std::vector<NaviSchedRequestStats> requests;
    requests.reserve(queries.size());
    for (const auto &query : queries) {
        NaviSchedRequestStats request;
        request.requestId = query.requestId;
        request.totalTokenCount = query.totalTokenCount;
        request.cacheHitTokenCount = query.cacheHitTokenCount;
        request.hybridCandidateBlockCount = query.hybridCandidateBlockCount;
        request.hybridFlatMatchedBlockCount = query.hybridFlatMatchedBlockCount;
        request.hybridMinFullReuseBlockCount = query.hybridMinFullReuseBlockCount;
        request.hybridSelectedReuseBlockCount = query.hybridSelectedReuseBlockCount;
        requests.emplace_back(request);
    }
    auto retained = getRetainedReporterRequests(queries, true);
    if (!retained.empty()) {
        std::unordered_map<const NaviSchedBatchFinishReporter *,
                           std::vector<NaviSchedRequestStats>>
            groupedRequests;
        for (auto &request : requests) {
            for (const auto &entry : retained) {
                const auto &requestIds = entry.second.second;
                const auto ids = std::find_if(
                    requestIds.begin(), requestIds.end(),
                    [&request](const auto &value) { return value.first == request.requestId; });
                if (ids != requestIds.end()) {
                    auto nativeRequest = request;
                    nativeRequest.requestId = ids->second;
                    groupedRequests[entry.first].emplace_back(std::move(nativeRequest));
                    break;
                }
            }
        }
        for (const auto &entry : retained) {
            entry.second.first->finishBatch(groupedRequests[entry.first]);
        }
        return;
    }
    if (finishReporters.empty()) {
        uint64_t requestId = 1;
        for (auto &request : requests) {
            request.requestId = requestId++;
        }
        agent->recordObservedBatch(requests, latencyUs / 1000.0, eventTimeUs, {});
        return;
    }
    std::unordered_set<const NaviSchedBatchFinishReporter *> finished;
    for (const auto &reporter : finishReporters) {
        if (reporter && finished.emplace(reporter.get()).second) {
            reporter->finishBatch(requests);
        }
    }
}

void recordRtpLLMObservedPrefillBatch(const std::shared_ptr<NaviSchedAgent> &agent,
                                      const std::vector<NaviSchedLatencyQuery> &queries,
                                      int64_t latencyUs, int64_t eventTimeUs)
{
    if (!agent || queries.empty() || latencyUs < 0) {
        return;
    }
    for (const auto &query : queries) {
        abortRetainedRtpLLMBatch(query.requestId);
    }
    std::vector<NaviSchedRequestStats> requests;
    requests.reserve(queries.size());
    for (const auto &query : queries) {
        NaviSchedRequestStats request;
        request.requestId = query.requestId;
        request.totalTokenCount = query.totalTokenCount;
        request.cacheHitTokenCount = query.cacheHitTokenCount;
        request.hybridCandidateBlockCount = query.hybridCandidateBlockCount;
        request.hybridFlatMatchedBlockCount = query.hybridFlatMatchedBlockCount;
        request.hybridMinFullReuseBlockCount = query.hybridMinFullReuseBlockCount;
        request.hybridSelectedReuseBlockCount = query.hybridSelectedReuseBlockCount;
        requests.emplace_back(request);
    }
    agent->recordObservedBatch(requests, latencyUs / 1000.0, eventTimeUs, {});
}

void reportRtpLLMExecutionLoad(
    const std::shared_ptr<NaviSchedAgent> &agent,
    const std::vector<NaviSchedLatencyQuery> &queuedBatchRequests,
    int32_t runningRequests, int64_t eventTimeUs)
{
    if (agent && queuedBatchRequests.size() <=
                     static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        agent->reportObservedExecutionLoad(
            static_cast<int32_t>(queuedBatchRequests.size()), runningRequests,
            eventTimeUs);
    }
}

void reportRtpLLMCacheStatus(const std::shared_ptr<NaviSchedAgent> &agent,
                             int64_t totalBlocks, int64_t blockSize, int64_t linearStep,
                             bool hybridTailRequired, int64_t eventTimeUs)
{
    if (agent) {
        agent->reportCacheUpdated(totalBlocks, blockSize, linearStep, nullptr, nullptr,
                                  nullptr, nullptr, nullptr, nullptr,
                                  hybridTailRequired ? 1 : 0, eventTimeUs);
    }
}

void reportRtpLLMCacheStateChanged(
    const std::shared_ptr<NaviSchedAgent> &agent,
    const NaviSchedBlockHashesConstPtr &changedBlockHashes,
    const NaviSchedBlockHashesConstPtr &flatPresentBlockHashes,
    const NaviSchedBlockHashesConstPtr &fullReadyBlockHashes,
    const NaviSchedBlockHashesConstPtr &tailReadyBlockHashes, bool, int64_t eventTimeUs,
    int64_t rtpCachedBlockCount)
{
    if (agent) {
        const auto beginUs = autil::TimeUtility::currentTimeInMicroSeconds();
        agent->reportCacheUpdated(-1, -1, -1, flatPresentBlockHashes, changedBlockHashes,
                                  fullReadyBlockHashes, changedBlockHashes,
                                  tailReadyBlockHashes, changedBlockHashes, -1,
                                  eventTimeUs, rtpCachedBlockCount);
        gRtpLLMCacheReportLatencyUs +=
            autil::TimeUtility::currentTimeInMicroSeconds() - beginUs;
    }
}

void resetRtpLLMCacheReportLatency() { gRtpLLMCacheReportLatencyUs = 0; }

int64_t takeRtpLLMCacheReportLatencyUs()
{
    const auto latencyUs = gRtpLLMCacheReportLatencyUs;
    gRtpLLMCacheReportLatencyUs = 0;
    return latencyUs;
}

}
