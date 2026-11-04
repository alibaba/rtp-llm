#include "aios/navi_sched/service/NaviSchedRole.h"
#include "aios/navi_sched/service/NaviSchedRoleRuntimeAccess.h"
#include "aios/navi_sched/service/NaviSchedRoleSummary.h"
#include "aios/network/gig/multi_call/subscribe/VIPSubscribeService.h"
#include "autil/EnvUtil.h"
#include "autil/FileRecorder.h"
#include "autil/StringUtil.h"
#include "autil/TimeUtility.h"
#include "autil/legacy/fast_jsonizable_dec.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/MetricsReporterR.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "navi/config/NaviConfig.h"
#include "navi/engine/Navi.h"
#include "aios/network/http_arpc/HTTPRPCServer.h"
#include <multi_call/config/MultiCallConfig.h>
#include <multi_call/rpc/GigRpcServer.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace navi_sched {

namespace {

struct KmonLifecycleState {
    std::mutex mutex;
    size_t userCount = 0;
    bool startedByNaviSched = false;
    bool exitHandlerRegistered = false;
};

KmonLifecycleState &getKmonLifecycleState()
{
    static auto *state = new KmonLifecycleState();
    return *state;
}

void shutdownKmonAtExit()
{
    multi_call::VIPSubscribeService::destroy();
    auto &state = getKmonLifecycleState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.startedByNaviSched && kmonitor::KMonitorFactory::IsStarted()) {
        kmonitor::KMonitorFactory::Shutdown();
    }
    state.userCount = 0;
    state.startedByNaviSched = false;
}

std::string makeEffectiveServiceConfig(const NaviSchedConfig &config, int32_t grpcPort,
                                       int32_t httpPort)
{
    autil::legacy::json::JsonMap gig;
    gig["heartbeat_thread_num"] = config.gig.heartbeatThreadNum;
    gig["heartbeat_queue_size"] = config.gig.heartbeatQueueSize;
    gig["log_prefix"] = config.gig.logPrefix;
    gig["snapshot_log_dir"] = config.gig.snapshotLogDir;
    gig["configured_http_port"] = config.gig.httpPort;
    gig["configured_arpc_port"] = config.gig.arpcPort;
    gig["configured_grpc_port"] = config.gig.grpcPort;
    gig["effective_grpc_port"] = grpcPort;
    gig["effective_http_port"] = httpPort;
    gig["grpc_io_thread_num"] = config.gig.grpcIoThreadNum;

    autil::legacy::json::JsonMap value;
    autil::legacy::json::JsonMap kmonTags;
    for (const auto &[name, tagValue] : config.kmonTags) {
        kmonTags[name] = tagValue;
    }
    value["name"] = config.name;
    value["log_file_name"] = config.logFileName;
    value["log_level"] = config.logLevel;
    value["enable_kmon"] = config.enableKmon;
    value["kmon_service_name"] = config.kmonServiceName;
    value["kmon_metrics_path"] = config.kmonMetricsPath;
    value["kmon_tags"] = std::move(kmonTags);
    value["kmon_monitor_name_prefix"] = config.kmonMonitorNamePrefix;
    value["kmon_report_interval_ms"] = config.kmonReportIntervalMs;
    value["gig"] = std::move(gig);
    value["wait_no_query"] = config.waitNoQuery;
    value["batch_timeout_ms"] = config.batchTimeoutMs;
    value["closed_batch_keep_ms"] = config.closedBatchKeepMs;
    value["latency_fit_batch_size"] = config.latencyFitBatchSize;
    value["latency_fit_interval"] = config.latencyFitInterval;
    value["latency_learning_rate"] = config.latencyLearningRate;
    value["latency_online_learning_rate_scale"] =
        config.latencyOnlineLearningRateScale;
    value["latency_current_sample_weight"] =
        config.latencyCurrentSampleWeight;
    autil::legacy::json::JsonArray parameterLearningRateScales;
    for (const auto scale : config.latencyParameterLearningRateScales) {
        parameterLearningRateScales.emplace_back(scale);
    }
    value["latency_parameter_learning_rate_scales"] =
        std::move(parameterLearningRateScales);
    value["latency_nonlinear_scale"] = config.latencyNonlinearScale;
    value["queue_learning_rate"] = config.queueLearningRate;
    value["latency_check_gradient"] = config.latencyCheckGradient;
    value["latency_cold_start_fit_epochs"] = config.latencyColdStartFitEpochs;
    value["latency_history_limit"] = config.latencyHistoryLimit;
    value["latency_history_bucket_capacity"] =
        config.latencyHistoryBucketCapacity;
    value["latency_persist_interval"] = config.latencyPersistInterval;
    value["latency_persist_interval_ms"] = config.latencyPersistIntervalMs;
    value["latency_warmup_sample_count"] = config.latencyWarmupSampleCount;
    value["latency_model_trust_min_revision"] = config.latencyModelTrustMinRevision;
    value["latency_zero_anchor_max_ms"] = config.latencyZeroAnchorMaxMs;
    value["latency_zero_anchor_min_samples"] =
        config.latencyZeroAnchorMinSamples;
    value["latency_validation_enabled"] = config.latencyValidationEnabled;
    value["residual_learning_rate"] = config.residualLearningRate;
    value["residual_update_error_floor_ms"] =
        config.residualUpdateErrorFloorMs;
    value["residual_update_error_ratio"] = config.residualUpdateErrorRatio;
    value["residual_idle_decay_start_ms"] =
        config.residualIdleDecayStartMs;
    value["residual_idle_decay_end_ms"] = config.residualIdleDecayEndMs;
    value["latency_model_state_file"] = config.latencyModelStateFile;
    value["enable_snapshot"] = config.enableSnapshot;
    value["snapshot_dir"] = config.snapshotDir;
    value["snapshot_interval_ms"] = config.snapshotIntervalMs;
    value["snapshot_keep_count"] = config.snapshotKeepCount;
    value["enable_vis"] = config.enableVis;
    return autil::legacy::ToJsonString(value, true);
}

void addKmonTags(kmonitor::MetricsConfig &config)
{
    const auto tags = autil::StringUtil::split(
        autil::EnvUtil::getEnv("kmonitorTags", std::string()), "@");
    for (const auto &tag : tags) {
        const auto keyValue = autil::StringUtil::split(tag, "^");
        if (keyValue.size() == 2) {
            config.AddGlobalTag(keyValue[0], keyValue[1]);
        }
    }
    const auto deployment = autil::EnvUtil::getEnv("DEPLOYMENT_NAME", std::string());
    if (!deployment.empty()) {
        config.AddGlobalTag("deployment_name", deployment);
    }
}

bool acquireKmon(const std::string &serviceName)
{
    auto &state = getKmonLifecycleState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!kmonitor::KMonitorFactory::IsStarted()) {
        kmonitor::MetricsConfig config;
        config.set_tenant_name(
            autil::EnvUtil::getEnv("kmonitorTenant", std::string("default")));
        config.set_service_name(serviceName);
        auto sinkAddress = autil::EnvUtil::getEnv(
            "kmonitorSinkAddress",
            autil::EnvUtil::getEnv("HIPPO_SLAVE_IP", std::string("127.0.0.1")));
        const auto sinkPort = autil::EnvUtil::getEnv("kmonitorPort", std::string("4141"));
        if (!sinkPort.empty()) {
            sinkAddress += ":" + sinkPort;
        }
        config.set_sink_address(sinkAddress);
        config.set_inited(true);
        addKmonTags(config);
        if (!kmonitor::KMonitorFactory::Init(config)) {
            kmonitor::KMonitorFactory::Shutdown();
            return false;
        }
        kmonitor::KMonitorFactory::Start();
        if (!kmonitor::KMonitorFactory::IsStarted()) {
            kmonitor::KMonitorFactory::Shutdown();
            return false;
        }
        state.startedByNaviSched = true;
        if (!state.exitHandlerRegistered) {
            if (0 != std::atexit(shutdownKmonAtExit)) {
                kmonitor::KMonitorFactory::Shutdown();
                state.startedByNaviSched = false;
                return false;
            }
            state.exitHandlerRegistered = true;
        }
    }
    ++state.userCount;
    return true;
}

void releaseKmon()
{
    auto &state = getKmonLifecycleState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.userCount == 0) {
        return;
    }
    --state.userCount;
}

}

const std::shared_ptr<navi::Navi> &NaviSchedRoleRuntimeAccess::getNavi(
    const NaviSchedRole &role)
{
    return role.getNavi();
}

const NaviSchedConfig &NaviSchedRoleRuntimeAccess::getConfig(const NaviSchedRole &role)
{
    return role.getConfig();
}

NaviSchedRole::NaviSchedRole(const NaviSchedConfig &config)
    : _config(config), _navi(std::make_shared<navi::Navi>(_config.name))
{
    _navi->setWaitNoQuery(_config.waitNoQuery);
}

NaviSchedRole::~NaviSchedRole() { stop(); }

bool NaviSchedRole::init()
{
    if (_started) {
        return true;
    }
    if (!_navi) {
        _navi = std::make_shared<navi::Navi>(_config.name);
        _navi->setWaitNoQuery(_config.waitNoQuery);
    }
    if (!initKmon() || !initGig()) {
        stop();
        return false;
    }
    if (!_navi->initWithGigRpcServer("", _gigRpcServer)) {
        stop();
        return false;
    }
    const auto effectiveConfig =
        makeEffectiveServiceConfig(_config, getGrpcPort(), getHttpPort());
    std::fprintf(stderr, "navi_sched_effective_service_config=%s\n",
                 effectiveConfig.c_str());
    _started = true;
    return true;
}

void NaviSchedRole::stop()
{
    if (!_started && !_gigRpcServer && !_metricsReporter && !_kmonAcquired) {
        return;
    }
    stopSnapshot();
    _started = false;
    if (_navi) {
        _navi->stop();
    }
    if (_gigRpcServer) {
        _gigRpcServer->release();
    }
    _navi.reset();
    _gigRpcServer.reset();
    _metricsReporter.reset();
    if (_kmonAcquired) {
        releaseKmon();
        _kmonAcquired = false;
    }
}

const NaviSchedConfig &NaviSchedRole::getConfig() const { return _config; }

int32_t NaviSchedRole::getGrpcPort() const
{
    if (!_gigRpcServer || !_gigRpcServer->hasGrpc()) {
        return -1;
    }
    return _gigRpcServer->getGrpcPort();
}

int32_t NaviSchedRole::getHttpPort() const
{
    if (!_gigRpcServer || !_gigRpcServer->getHttpArpcServer()) {
        return -1;
    }
    return _gigRpcServer->getHttpArpcServer()->getPort();
}

void NaviSchedRole::onUpdating() {}

bool NaviSchedRole::onUpdated() { return true; }

std::string NaviSchedRole::makeSnapshot() const
{
    return makeSnapshotSummary()->format();
}

void NaviSchedRole::reportMetrics() const
{
    reportMetric("role.alive", _started ? 1.0 : 0.0);
}

void NaviSchedRole::reportMetric(const std::string &name, double value) const
{
    if (_metricsReporter) {
        _metricsReporter->report(value, name, kmonitor::MetricType::STATUS, nullptr,
                                 kmonitor::MetricLevel::FATAL);
    }
}

void NaviSchedRole::reportMetric(const std::string &name, double value,
                                 const kmonitor::MetricsTags &tags) const
{
    if (_metricsReporter) {
        _metricsReporter->report(value, name, kmonitor::MetricType::STATUS, &tags,
                                 kmonitor::MetricLevel::FATAL);
    }
}

void NaviSchedRole::reportSampleMetric(const std::string &name, double value) const
{
    if (_metricsReporter) {
        _metricsReporter->report(value, name, kmonitor::MetricType::GAUGE, nullptr,
                                 kmonitor::MetricLevel::FATAL);
    }
}

void NaviSchedRole::reportSampleMetric(const std::string &name, double value,
                                       const kmonitor::MetricsTags &tags) const
{
    if (_metricsReporter) {
        _metricsReporter->report(value, name, kmonitor::MetricType::GAUGE, &tags,
                                 kmonitor::MetricLevel::FATAL);
    }
}

uint64_t NaviSchedRole::reportCounterQpsMetric(const std::string &name,
                                               uint64_t total) const
{
    return reportCounterQpsMetricAt(
        name, total, autil::TimeUtility::monotonicTimeUs());
}

uint64_t NaviSchedRole::reportCounterQpsMetricAt(
    const std::string &name, uint64_t total, int64_t reportTimeUs) const
{
    auto [it, inserted] = _lastQpsCounters.emplace(
        name, QpsCounterState{total, reportTimeUs});
    if (inserted) {
        return 0;
    }
    const auto delta = total >= it->second.total
                           ? total - it->second.total
                           : total;
    const auto elapsedUs = reportTimeUs - it->second.reportTimeUs;
    it->second = {total, reportTimeUs};
    if (_metricsReporter && delta > 0 && elapsedUs > 0) {
        const auto qps = static_cast<double>(delta) * 1000000.0 /
                         static_cast<double>(elapsedUs);
        _metricsReporter->report(qps, name, kmonitor::MetricType::GAUGE, nullptr,
                                 kmonitor::MetricLevel::FATAL);
    }
    return delta;
}

bool NaviSchedRole::enableHttpRpc() const { return false; }

bool NaviSchedRole::initKmon()
{
    if (!_config.enableKmon) {
        return true;
    }
    if (!acquireKmon(_config.kmonServiceName)) {
        return false;
    }
    _kmonAcquired = true;
    _metricsReporter = std::make_shared<kmonitor::MetricsReporter>(
        _config.kmonServiceName, _config.kmonMetricsPath,
        kmonitor::MetricsTags(_config.kmonTags), _config.kmonMonitorNamePrefix);
    _navi->initMetrics(*_metricsReporter);
    return true;
}

bool NaviSchedRole::initGig()
{
    const bool publicHttpRpc = enableHttpRpc();
    const size_t attemptCount = publicHttpRpc && 0 == _config.gig.grpcPort ? 16 : 1;
    for (size_t attempt = 0; attempt < attemptCount; ++attempt) {
        auto server = std::make_shared<multi_call::GigRpcServer>();
        multi_call::GigRpcServerConfig serverConfig;
        serverConfig.heartbeatThreadNum = _config.gig.heartbeatThreadNum;
        serverConfig.heartbeatQueueSize = _config.gig.heartbeatQueueSize;
        serverConfig.logPrefix =
            _config.gig.logPrefix.empty() ? _config.name : _config.gig.logPrefix;
        serverConfig.snapshotLogDir = _config.gig.snapshotLogDir;
        serverConfig.enableAgentStat = true;
        if (!server->init(serverConfig)) {
            return false;
        }
        multi_call::GrpcServerDescription grpcDescription;
        grpcDescription.port = std::to_string(_config.gig.grpcPort);
        grpcDescription.ioThreadNum = _config.gig.grpcIoThreadNum;
        if (!server->startGrpcServer(grpcDescription)) {
            server->release();
            continue;
        }
        const auto grpcPort = server->getGrpcPort();
        if (grpcPort <= 0 || grpcPort >= 65535) {
            server->release();
            continue;
        }
        multi_call::ArpcServerDescription arpcDescription;
        arpcDescription.name = _config.name + "Arpc";
        arpcDescription.port = std::to_string(_config.gig.arpcPort);
        if (!server->startArpcServer(arpcDescription)) {
            server->release();
            continue;
        }
        multi_call::HttpArpcServerDescription httpDescription;
        httpDescription.name = _config.name + "HttpArpc";
        const auto httpPort = _config.gig.httpPort > 0 ? _config.gig.httpPort
                              : publicHttpRpc          ? grpcPort + 1
                                                       : 0;
        httpDescription.port = std::to_string(httpPort);
        if (!server->startHttpArpcServer(httpDescription)) {
            server->release();
            continue;
        }
        _gigRpcServer = std::move(server);
        return true;
    }
    return false;
}

bool NaviSchedRole::startSnapshot()
{
    if (!_config.enableSnapshot && !_config.enableKmon) {
        return true;
    }
    if ((_config.enableKmon && _config.kmonReportIntervalMs <= 0) ||
        (_config.enableSnapshot &&
         (_config.snapshotIntervalMs <= 0 || _config.snapshotDir.empty() ||
          _config.snapshotKeepCount == 0))) {
        return false;
    }
    if (_config.enableKmon) {
        _metricsThread = autil::LoopThread::createLoopThread(
            [this]() { reportMetrics(); }, _config.kmonReportIntervalMs * 1000,
            "sched_kmonitor");
        if (!_metricsThread) {
            return false;
        }
    }
    if (_config.enableSnapshot) {
        _snapshotThread = autil::LoopThread::createLoopThread(
            [this]() { recordSnapshot(); }, _config.snapshotIntervalMs * 1000,
            "sched_snapshot");
        if (!_snapshotThread) {
            stopSnapshot();
            return false;
        }
    }
    return true;
}

void NaviSchedRole::stopSnapshot()
{
    if (_metricsThread) {
        _metricsThread->stop();
        _metricsThread.reset();
    }
    if (_snapshotThread) {
        _snapshotThread->stop();
        _snapshotThread.reset();
    }
}

void NaviSchedRole::recordSnapshot() const
{
    autil::FileRecorder::newRecord(
        makeSnapshot(), _config.snapshotDir + "/" + _config.name, "snapshot",
        _snapshotId.fetch_add(1, std::memory_order_relaxed), _config.snapshotKeepCount);
}

bool NaviSchedRole::update(const std::string &configStr,
                           const std::shared_ptr<navi::ResourceMap> &rootResourceMap)
{
    if (!rootResourceMap || (!_started && !init())) {
        return false;
    }
    if (_metricsReporter &&
        !rootResourceMap->hasResource(kmonitor::MetricsReporterR::RESOURCE_ID)) {
        rootResourceMap->addResource(
            std::make_shared<kmonitor::MetricsReporterR>(_metricsReporter));
    }
    stopSnapshot();
    navi::NaviConfig config;
    if (!config.loadConfig(configStr)) {
        startSnapshot();
        return false;
    }
    if (config.logConfig.fileAppenders.empty() && !_config.logFileName.empty()) {
        navi::FileLogConfig appender;
        appender.fileName = _config.logFileName;
        appender.levelStr = _config.logLevel;
        config.logConfig.fileAppenders.emplace_back(std::move(appender));
    }
    onUpdating();
    auto naviSuccess = _navi->update(config, *rootResourceMap);
    auto success = naviSuccess && onUpdated();
    if (success) {
        std::fprintf(stderr, "navi_sched_effective_runtime_config=%s\n",
                     configStr.c_str());
    }
    return startSnapshot() && success;
}

const std::shared_ptr<navi::Navi> &NaviSchedRole::getNavi() const { return _navi; }

std::shared_ptr<NaviSchedRoleSummary> NaviSchedRole::makeSnapshotSummary() const
{
    auto summary = std::make_shared<NaviSchedRoleSummary>();
    summary->name = _config.name;
    summary->started = _started;
    summary->grpcPort = getGrpcPort();
    return summary;
}

}
