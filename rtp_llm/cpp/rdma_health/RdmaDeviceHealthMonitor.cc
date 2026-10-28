#include "rtp_llm/cpp/rdma_health/RdmaDeviceHealthMonitor.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <exception>
#include <utility>

#include "kmonitor/client/KMonitorFactory.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {
namespace rdma_health {
namespace {

using accl::barex::BAREX_ERR_NOT_SUPPORTED;
using accl::barex::BAREX_SUCCESS;
using accl::barex::DeviceFaultReason;
using accl::barex::DeviceHealthState;
using accl::barex::XDeviceHealthReason;
using accl::barex::XDeviceProbeResult;
using accl::barex::XTransportDeviceHealthState;

// UNKNOWN / DEGRADED 分别在第 1、61、121……次出现时输出 WARN，计数不随恢复清零。
constexpr uint64_t kWarnEvery = 60;

const char* stateName(XTransportDeviceHealthState state) {
    switch (state) {
        case XTransportDeviceHealthState::AVAILABLE:
            return "AVAILABLE";
        case XTransportDeviceHealthState::DEGRADED:
            return "DEGRADED";
        case XTransportDeviceHealthState::UNAVAILABLE:
            return "UNAVAILABLE";
        case XTransportDeviceHealthState::UNKNOWN:
            return "UNKNOWN";
    }
    return "INVALID";
}

const char* healthStateName(DeviceHealthState state) {
    switch (state) {
        case DeviceHealthState::HEALTHY:
            return "HEALTHY";
        case DeviceHealthState::FAULT:
            return "FAULT";
        case DeviceHealthState::UNKNOWN:
            return "UNKNOWN";
    }
    return "INVALID";
}

const char* reasonName(XDeviceHealthReason reason) {
    switch (reason) {
        case XDeviceHealthReason::NONE:
            return "NONE";
        case XDeviceHealthReason::DEVICE_REMOVED:
            return "DEVICE_REMOVED";
        case XDeviceHealthReason::QUERY_DEVICE_HARD_ERROR:
            return "QUERY_DEVICE_HARD_ERROR";
        case XDeviceHealthReason::PORT_NOT_ACTIVE:
            return "PORT_NOT_ACTIVE";
        case XDeviceHealthReason::PHYSICAL_LINK_DOWN:
            return "PHYSICAL_LINK_DOWN";
        case XDeviceHealthReason::QUERY_TEMPORARY_ERROR:
            return "QUERY_TEMPORARY_ERROR";
        case XDeviceHealthReason::NO_DEVICE:
            return "NO_DEVICE";
    }
    return "INVALID";
}

const char* faultReasonName(DeviceFaultReason reason) {
    switch (reason) {
        case DeviceFaultReason::NONE:
            return "NONE";
        case DeviceFaultReason::MANAGER_NOT_ACTIVE:
            return "MANAGER_NOT_ACTIVE";
        case DeviceFaultReason::NO_MANAGED_DEVICE:
            return "NO_MANAGED_DEVICE";
        case DeviceFaultReason::BACKEND_NOT_SUPPORTED:
            return "BACKEND_NOT_SUPPORTED";
        case DeviceFaultReason::DEVICE_REMOVED:
            return "DEVICE_REMOVED";
        case DeviceFaultReason::QUERY_DEVICE_FAILED:
            return "QUERY_DEVICE_FAILED";
        case DeviceFaultReason::QUERY_PORT_FAILED:
            return "QUERY_PORT_FAILED";
        case DeviceFaultReason::PORT_NOT_ACTIVE:
            return "PORT_NOT_ACTIVE";
        case DeviceFaultReason::PHYSICAL_LINK_NOT_UP:
            return "PHYSICAL_LINK_NOT_UP";
        case DeviceFaultReason::PORT_STATE_INCONSISTENT:
            return "PORT_STATE_INCONSISTENT";
        case DeviceFaultReason::INVALID_DEVICE_CONTEXT:
            return "INVALID_DEVICE_CONTEXT";
    }
    return "INVALID";
}

const char* faultHandlerName(FaultHandler handler) {
    return handler == FaultHandler::ABORT ? "ABORT" : "LOG";
}

std::string formatSummary(const RdmaDeviceHealthMonitor::HealthResult& r) {
    return fmtstr("query_result %d, state %s, total %u, healthy %u, fault %u, unknown %u",
                  static_cast<int>(r.query_result),
                  stateName(r.state),
                  r.total_count,
                  r.healthy_count,
                  r.fault_count,
                  r.unknown_count);
}

std::string formatDevice(const XDeviceProbeResult& d) {
    const auto& detail = d.detail;
    // 保留探测顺序，输出全部原因（含次要原因）；空列表输出 reasons []，不改变主原因及故障判断。
    std::string reasons;
    for (const auto reason : detail.reasons) {
        if (!reasons.empty()) {
            reasons += ",";
        }
        reasons += faultReasonName(reason);
    }
    return fmtstr("name %s, index %u, port %u, device_state %s, uplink_state %s, reason %s, sys_errno %d, "
                  "message '%s', primary_reason %s, reasons [%s], raw_port_state %u, raw_physical_state %u, "
                  "query_device ret/errno %d/%d, query_port ret/errno %d/%d",
                  d.device_name.c_str(),
                  d.device_index,
                  static_cast<unsigned>(d.port_num),
                  healthStateName(d.device_state),
                  healthStateName(d.uplink_state),
                  reasonName(d.reason),
                  d.sys_errno,
                  d.message.c_str(),
                  faultReasonName(detail.primary_reason),
                  reasons.c_str(),
                  static_cast<unsigned>(detail.raw_port_state),
                  static_cast<unsigned>(detail.raw_physical_state),
                  detail.query_device_ret,
                  detail.query_device_errno,
                  detail.query_port_ret,
                  detail.query_port_errno);
}

void logDevices(uint32_t level, const char* prefix, const RdmaDeviceHealthMonitor::HealthResult& r) {
    for (const auto& d : r.devices) {
        RTP_LLM_LOG(level, "%s device: %s", prefix, formatDevice(d).c_str());
    }
}

std::string formatConfig(const RdmaDeviceHealthConfig& c) {
    return fmtstr("fault_handler %s, probe_interval_ms %" PRId64 ", device_fault_confirm_ms %" PRId64
                  ", fault_window_ms %" PRId64 ", unavailable_threshold_ms %" PRId64 ", flap_warn_threshold %" PRId64,
                  c.fault_handler.c_str(),
                  c.probe_interval_ms,
                  c.device_fault_confirm_ms,
                  c.fault_window_ms,
                  c.unavailable_threshold_ms,
                  c.flap_warn_threshold);
}

void appendError(std::string* errors, const std::string& error) {
    errors->append(errors->empty() ? "" : "; ");
    errors->append(error);
}

void checkRange(const char* field, int64_t value, int64_t min_value, int64_t max_value, std::string* errors) {
    if (value < min_value || value > max_value) {
        appendError(errors,
                    fmtstr("%s=%" PRId64 " not in [%" PRId64 ", %" PRId64 "]", field, value, min_value, max_value));
    }
}

void checkCoversProbeIntervals(const char* field, int64_t value, int64_t probe_interval_ms, std::string* errors) {
    constexpr int64_t kMinIntervals = RdmaDeviceHealthConfig::kMinConfirmProbeIntervals;
    // 等价于 value < N * probe_interval_ms，避免越界输入在乘法中溢出。
    if (value / kMinIntervals < probe_interval_ms) {
        appendError(errors,
                    fmtstr("%s=%" PRId64 " less than %" PRId64 " * probe_interval_ms (%" PRId64 ")",
                           field,
                           value,
                           kMinIntervals,
                           probe_interval_ms));
    }
}

int64_t steadyNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

RdmaDeviceHealthMonitor::HealthResult probeBarex() {
    return accl::barex::XDeviceHealth::ProbeDeviceHealth();
}

}  // namespace

bool validateConfig(const RdmaDeviceHealthConfig& config, FaultHandler* handler, std::string* error) {
    using C = RdmaDeviceHealthConfig;
    std::string  errors;
    FaultHandler parsed = FaultHandler::LOG;
    if (config.fault_handler == "LOG") {
        parsed = FaultHandler::LOG;
    } else if (config.fault_handler == "ABORT") {
        parsed = FaultHandler::ABORT;
    } else {
        appendError(&errors, fmtstr("fault_handler='%s' not in {LOG, ABORT}", config.fault_handler.c_str()));
    }
    checkRange("probe_interval_ms", config.probe_interval_ms, C::kMinProbeIntervalMs, C::kMaxProbeIntervalMs, &errors);
    checkRange("device_fault_confirm_ms",
               config.device_fault_confirm_ms,
               C::kMinDeviceFaultConfirmMs,
               C::kMaxDeviceFaultConfirmMs,
               &errors);
    checkCoversProbeIntervals(
        "device_fault_confirm_ms", config.device_fault_confirm_ms, config.probe_interval_ms, &errors);
    checkRange("fault_window_ms", config.fault_window_ms, C::kMinFaultWindowMs, C::kMaxFaultWindowMs, &errors);
    checkRange("unavailable_threshold_ms",
               config.unavailable_threshold_ms,
               C::kMinUnavailableThresholdMs,
               C::kMaxUnavailableThresholdMs,
               &errors);
    checkCoversProbeIntervals(
        "unavailable_threshold_ms", config.unavailable_threshold_ms, config.probe_interval_ms, &errors);
    if (config.unavailable_threshold_ms > config.fault_window_ms) {
        appendError(&errors,
                    fmtstr("unavailable_threshold_ms=%" PRId64 " greater than fault_window_ms=%" PRId64,
                           config.unavailable_threshold_ms,
                           config.fault_window_ms));
    }
    checkRange(
        "flap_warn_threshold", config.flap_warn_threshold, C::kMinFlapWarnThreshold, C::kMaxFlapWarnThreshold, &errors);
    if (!errors.empty()) {
        if (error != nullptr) {
            *error = std::move(errors);
        }
        return false;
    }
    if (handler != nullptr) {
        *handler = parsed;
    }
    return true;
}

bool ensureMonitorStarted(const RdmaDeviceHealthConfig& config, const kmonitor::MetricsReporterPtr& reporter) {
    return RdmaDeviceHealthMonitor::instance().ensureStarted(config, reporter);
}

RdmaDeviceHealthMonitor& RdmaDeviceHealthMonitor::instance() {
    // stop() 在静态析构时记录日志，因此先构造 logger，使其晚于 monitor 析构。
    Logger::getEngineLogger();
    static RdmaDeviceHealthMonitor monitor(&probeBarex, &steadyNowMs, [] { std::abort(); });
    return monitor;
}

RdmaDeviceHealthMonitor::RdmaDeviceHealthMonitor(ProbeFn probe, NowFn now, AbortFn abort):
    probe_(std::move(probe)), now_(std::move(now)), abort_(std::move(abort)) {
    // 在线程启动前注册；与 factory 关闭竞争时，要么回调 stop/join，要么拒绝本次启动。
    // 注册不访问 BareX 单例；未构造 monitor 的进程关闭时也不会创建它。
    if (!registerKmonitorShutdownHook(this, [this] { stop(); })) {
        stopping_ = true;
    }
}

RdmaDeviceHealthMonitor::~RdmaDeviceHealthMonitor() {
    // 先等待上报结束，再注销，避免注销到 join 之间的 factory 关闭窗口。
    // 注销与关闭回调串行化，返回后不会再有回调访问 this。
    stop();
    unregisterKmonitorShutdownHook(this);
}

bool RdmaDeviceHealthMonitor::ensureStarted(const RdmaDeviceHealthConfig&       config,
                                            const kmonitor::MetricsReporterPtr& reporter) {
    if (!config.enabled) {
        RTP_LLM_LOG_INFO("[RDMA_DEVICE_HEALTH] disabled by config, monitor not started");
        return false;
    }
    FaultHandler handler = FaultHandler::LOG;
    std::string  error;
    if (!validateConfig(config, &handler, &error)) {
        RTP_LLM_LOG_ERROR("[RDMA_DEVICE_HEALTH] invalid config, monitor not started: %s", error.c_str());
        return false;
    }
    std::lock_guard<std::mutex> lock(start_mutex_);
    if (stopping_) {
        RTP_LLM_LOG_WARNING("[RDMA_DEVICE_HEALTH] monitor is stopping or kmonitor has shut down, monitor not started");
        return false;
    }
    if (started_) {
        warnIfConfigDiffers(config);
        return true;
    }
    // 仅在线程启动前写入，启动后只读。
    config_        = config;
    fault_handler_ = handler;
    reporter_      = reporter;
    try {
        thread_ = std::thread(&RdmaDeviceHealthMonitor::run, this);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("[RDMA_DEVICE_HEALTH] monitor thread start failed: %s", e.what());
        return false;
    }
    started_ = true;
    RTP_LLM_LOG_INFO("[RDMA_DEVICE_HEALTH] monitor started, %s, metrics %s",
                     formatConfig(config_).c_str(),
                     reporter ? "enabled" : "disabled");
    return true;
}

void RdmaDeviceHealthMonitor::stop() {
    std::lock_guard<std::mutex> lock(start_mutex_);
    {
        // 在 wait_mutex_ 下设置 stopping_，避免通知落在 run() 的条件检查与等待之间，
        // 导致 stop() 丢失唤醒并等待整个探测间隔。
        std::lock_guard<std::mutex> wait_lock(wait_mutex_);
        stopping_ = true;
    }
    wait_cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
        RTP_LLM_LOG_INFO("[RDMA_DEVICE_HEALTH] monitor stopped");
    }
}

void RdmaDeviceHealthMonitor::run() {
    while (!stopping_) {
        bool keep_running = true;
        try {
            const auto begin  = std::chrono::steady_clock::now();
            const auto result = probe_();
            last_probe_latency_us_ =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
            keep_running = process(result, now_());
        } catch (const std::exception& e) {
            // 异常若逃出线程入口会触发 std::terminate，因此在此记录并继续下一轮。
            RTP_LLM_LOG_ERROR("[RDMA_DEVICE_HEALTH] probe raised exception: %s", e.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("[RDMA_DEVICE_HEALTH] probe raised unknown exception");
        }
        if (!keep_running) {
            return;
        }
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait_for(
            lock, std::chrono::milliseconds(config_.probe_interval_ms), [this] { return stopping_.load(); });
    }
}

bool RdmaDeviceHealthMonitor::process(const HealthResult& r, int64_t now_ms) {
    if (r.query_result == BAREX_ERR_NOT_SUPPORTED) {
        // 后端不支持健康查询时结束 monitor 线程，避免持续输出 UNKNOWN；不退出进程。
        ++log_counts_.not_supported;
        RTP_LLM_LOG_INFO("[RDMA_DEVICE_HEALTH] device health query not supported by the BareX backend, monitor exits");
        return false;
    }
    const EventReason reason = evaluate(r, now_ms);
    if (reason == EventReason::DEVICE_FAULT_CONFIRMED) {
        // 设备确认处理会先尝试上报，再按配置决定是否 abort。
        handleFatal(r, now_ms);
    } else {
        if (reason == EventReason::UNAVAILABLE_THRESHOLD_EXCEEDED) {
            warnUnavailable(r, now_ms);
        }
        reportMetrics(r, now_ms);
    }
    return true;
}

RdmaDeviceHealthMonitor::EventReason RdmaDeviceHealthMonitor::evaluate(const HealthResult& r, int64_t now_ms) {
    // 设备确认去重不能阻止恢复统计、当前状态上报或累计不可用告警。
    pruneFaultPeriods(now_ms);
    if (r.query_result != BAREX_SUCCESS || r.state == State::UNKNOWN) {
        // 总体 UNKNOWN 不等于每个 device 都 UNKNOWN；仅成功查询中的明确 device HEALTHY 证据清除确认计时。
        // 否则保留起点，经过时间继续计入；当轮不确认，也不关闭已有总体故障期。
        if (r.query_result == BAREX_SUCCESS
            && std::any_of(r.devices.begin(), r.devices.end(), [](const XDeviceProbeResult& device) {
                   return device.device_state == DeviceHealthState::HEALTHY;
               })) {
            device_fault_since_ms_ = -1;
        }
        if (++unknown_count_ % kWarnEvery == 1) {
            ++log_counts_.unknown_warn;
            RTP_LLM_LOG_WARNING(
                "[RDMA_DEVICE_HEALTH] unknown, %s, occurrences %" PRIu64, formatSummary(r).c_str(), unknown_count_);
        }
        return EventReason::NONE;
    }
    if (r.state == State::AVAILABLE) {
        device_fault_since_ms_ = -1;
        onRecovered(now_ms);
        return EventReason::NONE;
    }
    if (r.state == State::DEGRADED) {
        // 至少一个 NIC HEALTHY，但并非全部明确 HEALTHY；其余可能为 FAULT 或 UNKNOWN。
        // 保持已有总体故障期，不视为完全恢复；清除设备硬故障确认计时。
        device_fault_since_ms_ = -1;
        if (++degraded_count_ % kWarnEvery == 1) {
            ++log_counts_.degraded_warn;
            RTP_LLM_LOG_WARNING(
                "[RDMA_DEVICE_HEALTH] degraded, %s, occurrences %" PRIu64, formatSummary(r).c_str(), degraded_count_);
            logDevices(alog::LOG_LEVEL_WARN, "[RDMA_DEVICE_HEALTH] degraded", r);
        }
        return EventReason::NONE;
    }
    // 以下处理 UNAVAILABLE；只有此状态会开启总体故障期并评估设备确认或累计阈值。
    if (fault_since_ms_ < 0) {
        fault_since_ms_ = now_ms;
        ++log_counts_.fault_begin;
        RTP_LLM_LOG_WARNING("[RDMA_DEVICE_HEALTH] fault begin, %s, unavailable_ms_in_window %" PRId64
                            ", warn when it reaches %" PRId64 " ms within fault_window_ms %" PRId64,
                            formatSummary(r).c_str(),
                            unavailableMsInWindow(now_ms),
                            config_.unavailable_threshold_ms,
                            config_.fault_window_ms);
        logDevices(alog::LOG_LEVEL_WARN, "[RDMA_DEVICE_HEALTH] fault begin", r);
    }
    if (!isDeviceFault(r)) {
        device_fault_since_ms_ = -1;
    } else if (device_fault_since_ms_ < 0) {
        device_fault_since_ms_ = now_ms;
        ++log_counts_.device_fault_begin;
        RTP_LLM_LOG_WARNING("[RDMA_DEVICE_HEALTH] device fault begin, %s, confirm after %" PRId64 " ms",
                            formatSummary(r).c_str(),
                            config_.device_fault_confirm_ms);
    }
    // LOG 按总体故障期去重，ABORT 按 monitor 生命周期去重；设备确认优先于同轮累计 WARN。
    // 去重不会永久屏蔽累计 WARN，后续 UNAVAILABLE 轮仍按阈值和窗口限频告警。
    if (!fatal_handled_in_current_period_ && (fault_handler_ == FaultHandler::LOG || !ever_confirmed_fault_)
        && device_fault_since_ms_ >= 0 && now_ms - device_fault_since_ms_ >= config_.device_fault_confirm_ms) {
        return EventReason::DEVICE_FAULT_CONFIRMED;
    }
    if (unavailableMsInWindow(now_ms) >= config_.unavailable_threshold_ms) {
        return EventReason::UNAVAILABLE_THRESHOLD_EXCEEDED;
    }
    return EventReason::NONE;
}

void RdmaDeviceHealthMonitor::onRecovered(int64_t now_ms) {
    if (fault_since_ms_ < 0) {
        return;
    }
    fault_periods_.push_back({fault_since_ms_, now_ms});
    fault_since_ms_ = -1;
    fatal_handled_in_current_period_ = false;
    ++log_counts_.recovered;
    const int64_t unavailable_ms = unavailableMsInWindow(now_ms);
    RTP_LLM_LOG_INFO("[RDMA_DEVICE_HEALTH] recovered, fault lasted %" PRId64 " ms, unavailable_ms_in_window %" PRId64,
                     now_ms - fault_periods_.back().begin_ms,
                     unavailable_ms);

    const int64_t recover_count = recoverCountInWindow(now_ms);
    if (recover_count >= config_.flap_warn_threshold && now_ms - last_flap_warn_ms_ >= config_.fault_window_ms) {
        ++log_counts_.flapping;
        RTP_LLM_LOG_WARNING("[RDMA_DEVICE_FLAPPING] %" PRId64 " recoveries within fault_window_ms %" PRId64
                            ", unavailable_ms_in_window %" PRId64 ", unavailable_threshold_ms %" PRId64,
                            recover_count,
                            config_.fault_window_ms,
                            unavailable_ms,
                            config_.unavailable_threshold_ms);
        last_flap_warn_ms_ = now_ms;
    }
}

void RdmaDeviceHealthMonitor::warnUnavailable(const HealthResult& r, int64_t now_ms) {
    // 与抖动告警一样，两次 WARN 至少相隔一个窗口；恢复不重置限频。
    if (now_ms - last_unavailable_warn_ms_ < config_.fault_window_ms) {
        return;
    }
    last_unavailable_warn_ms_ = now_ms;
    ++log_counts_.unavailable_warn;
    RTP_LLM_LOG_WARNING("[RDMA_DEVICE_UNAVAILABLE] reason UNAVAILABLE_THRESHOLD_EXCEEDED, unavailable_ms_in_window %" PRId64
                        ", unavailable_threshold_ms %" PRId64 ", fault_window_ms %" PRId64 ", action WARN_ONLY, %s",
                        unavailableMsInWindow(now_ms),
                        config_.unavailable_threshold_ms,
                        config_.fault_window_ms,
                        formatSummary(r).c_str());
    logDevices(alog::LOG_LEVEL_WARN, "[RDMA_DEVICE_UNAVAILABLE]", r);
}

void RdmaDeviceHealthMonitor::handleFatal(const HealthResult& r, int64_t now_ms) {
    if (fatal_handled_in_current_period_.exchange(true)) {
        return;
    }
    const bool previously_confirmed = ever_confirmed_fault_.exchange(true);
    if (fault_handler_ == FaultHandler::ABORT && previously_confirmed) {
        return;
    }
    ++log_counts_.fatal;
    RTP_LLM_LOG_ERROR("[RDMA_DEVICE_FATAL] reason DEVICE_FAULT_CONFIRMED, fault_duration_ms %" PRId64
                      ", device_fault_duration_ms %" PRId64 ", unavailable_ms_in_window %" PRId64
                      ", fault_window_ms %" PRId64 ", fault_handler %s, %s",
                      now_ms - fault_since_ms_,
                      device_fault_since_ms_ >= 0 ? now_ms - device_fault_since_ms_ : 0,
                      unavailableMsInWindow(now_ms),
                      config_.fault_window_ms,
                      faultHandlerName(fault_handler_),
                      formatSummary(r).c_str());
    logDevices(alog::LOG_LEVEL_ERROR, "[RDMA_DEVICE_FATAL]", r);
    // abort 前尝试上报 confirmed_fault=1，但不保证指标已送达。
    reportMetrics(r, now_ms);
    if (fault_handler_ == FaultHandler::LOG || stopping_) {
        return;
    }
    Logger::getEngineLogger().flush();
    abort_();
}

bool RdmaDeviceHealthMonitor::isDeviceFault(const HealthResult& r) const {
    if (r.devices.empty()) {
        return false;
    }
    for (const auto& d : r.devices) {
        if (d.reason != XDeviceHealthReason::DEVICE_REMOVED
            && d.reason != XDeviceHealthReason::QUERY_DEVICE_HARD_ERROR) {
            return false;
        }
    }
    return true;
}

void RdmaDeviceHealthMonitor::pruneFaultPeriods(int64_t now_ms) {
    // 故障期按结束时间排列；结束时间早于窗口起点的记录不再贡献累计时长或恢复次数。
    while (!fault_periods_.empty() && now_ms - fault_periods_.front().end_ms > config_.fault_window_ms) {
        fault_periods_.pop_front();
    }
}

void RdmaDeviceHealthMonitor::reportMetrics(const HealthResult& r, int64_t now_ms) {
    auto reporter = reporter_.lock();
    // stopKmonitorFactory() 通过注册回调先 stop/join 本线程，再释放 factory 的指标资源。
    // IsStarted() 仅用于跳过未运行的 factory；它本身不是并发关闭屏障。
    if (reporter == nullptr || !kmonitor::KMonitorFactory::IsStarted()) {
        return;
    }
    auto collector = makeMetricsCollector(r, now_ms);
    reporter->report<RdmaDeviceHealthMetrics, RdmaDeviceHealthMetricsCollector>(nullptr, &collector);
}

RdmaDeviceHealthMetricsCollector RdmaDeviceHealthMonitor::makeMetricsCollector(const HealthResult& r,
                                                                               int64_t             now_ms) const {
    RdmaDeviceHealthMetricsCollector c;
    c.query_result             = static_cast<int64_t>(r.query_result);
    c.query_failed             = r.query_result != BAREX_SUCCESS;
    c.state                    = static_cast<int64_t>(r.state);
    c.total_count              = r.total_count;
    c.fault_count              = r.fault_count;
    c.confirmed_fault          = ever_confirmed_fault_ ? 1 : 0;
    c.fault_duration_ms        = fault_since_ms_ >= 0 ? now_ms - fault_since_ms_ : 0;
    c.recover_count_in_window  = recoverCountInWindow(now_ms);
    c.unavailable_ms_in_window = unavailableMsInWindow(now_ms);
    c.probe_latency_us         = last_probe_latency_us_;
    return c;
}

int64_t RdmaDeviceHealthMonitor::recoverCountInWindow(int64_t now_ms) const {
    int64_t count = 0;
    for (const auto& p : fault_periods_) {
        if (now_ms - p.end_ms <= config_.fault_window_ms) {
            ++count;
        }
    }
    return count;
}

int64_t RdmaDeviceHealthMonitor::unavailableMsInWindow(int64_t now_ms) const {
    // 累计总体故障期与窗口重叠的经过时间，包含中间 UNKNOWN/DEGRADED，而非仅 UNAVAILABLE 样本时长。
    // 每个已关闭故障期至少需要一轮 UNAVAILABLE 和一轮 AVAILABLE；pruneFaultPeriods 移除过期记录。
    const int64_t window_begin = now_ms - config_.fault_window_ms;
    int64_t       total        = 0;
    for (const auto& p : fault_periods_) {
        const int64_t begin = std::max(p.begin_ms, window_begin);
        const int64_t end   = std::min(p.end_ms, now_ms);
        if (end > begin) {
            total += end - begin;
        }
    }
    if (fault_since_ms_ >= 0) {
        total += now_ms - std::max(fault_since_ms_, window_begin);
    }
    return total;
}

void RdmaDeviceHealthMonitor::warnIfConfigDiffers(const RdmaDeviceHealthConfig& config) {
    std::string diffs;
    auto        append = [&diffs](const std::string& diff) {
        diffs.append(diffs.empty() ? "" : "; ");
        diffs.append(diff);
    };
    auto compare = [&append](const char* field, int64_t effective, int64_t requested) {
        if (effective != requested) {
            append(fmtstr("%s %" PRId64 " vs %" PRId64, field, effective, requested));
        }
    };
    if (config.fault_handler != config_.fault_handler) {
        append(fmtstr("fault_handler %s vs %s", config_.fault_handler.c_str(), config.fault_handler.c_str()));
    }
    compare("probe_interval_ms", config_.probe_interval_ms, config.probe_interval_ms);
    compare("device_fault_confirm_ms", config_.device_fault_confirm_ms, config.device_fault_confirm_ms);
    compare("fault_window_ms", config_.fault_window_ms, config.fault_window_ms);
    compare("unavailable_threshold_ms", config_.unavailable_threshold_ms, config.unavailable_threshold_ms);
    compare("flap_warn_threshold", config_.flap_warn_threshold, config.flap_warn_threshold);
    if (diffs.empty()) {
        return;
    }
    ++log_counts_.config_differs;
    RTP_LLM_LOG_WARNING("[RDMA_DEVICE_HEALTH] config differs, keep the effective one (effective vs requested): %s",
                        diffs.c_str());
}

}  // namespace rdma_health
}  // namespace rtp_llm
