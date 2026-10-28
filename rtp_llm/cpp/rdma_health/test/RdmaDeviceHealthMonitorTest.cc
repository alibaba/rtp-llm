#include <atomic>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/rdma_health/RdmaDeviceHealthMonitor.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace rdma_health {
namespace {

using accl::barex::BAREX_ERR_NOT_SUPPORTED;
using accl::barex::BAREX_ERR_STAT;
using accl::barex::BAREX_SUCCESS;
using accl::barex::DeviceHealthState;
using accl::barex::XDeviceHealthReason;
using accl::barex::XDeviceProbeResult;
using Result      = RdmaDeviceHealthMonitor::HealthResult;
using State       = RdmaDeviceHealthMonitor::State;
using EventReason = RdmaDeviceHealthMonitor::EventReason;

// 用于检查停止等操作没有等待整个探测间隔或确认时长；留出调度余量，仍短于待检测的长时间阻塞。
// 这只是测试等待上限，不保证机器极端繁忙时也不会超时。
constexpr int64_t kPromptMs       = 10000;
constexpr int64_t kLongIntervalMs = RdmaDeviceHealthConfig::kMaxProbeIntervalMs;

bool isDeviceLevel(XDeviceHealthReason reason) {
    return reason == XDeviceHealthReason::DEVICE_REMOVED || reason == XDeviceHealthReason::QUERY_DEVICE_HARD_ERROR;
}

Result makeResult(State state, std::initializer_list<XDeviceHealthReason> fault_reasons = {}) {
    Result r;
    r.state        = state;
    r.query_result = BAREX_SUCCESS;
    for (const auto reason : fault_reasons) {
        XDeviceProbeResult d;
        d.device_name  = "mlx5_" + std::to_string(r.devices.size());
        d.device_index = static_cast<uint32_t>(r.devices.size());
        d.port_num     = 1;
        d.device_state = isDeviceLevel(reason) ? DeviceHealthState::FAULT : DeviceHealthState::HEALTHY;
        d.uplink_state = DeviceHealthState::FAULT;
        d.reason       = reason;
        r.devices.push_back(std::move(d));
    }
    r.total_count = static_cast<uint32_t>(r.devices.size());
    r.fault_count = static_cast<uint32_t>(r.devices.size());
    return r;
}

// 结果构造函数区分传输可用性、查询执行失败、端口故障和设备故障。
Result availableResult() {
    Result r        = makeResult(State::AVAILABLE);
    r.total_count   = 2;
    r.healthy_count = 2;
    return r;
}

Result degradedResult() {
    Result r        = makeResult(State::DEGRADED, {XDeviceHealthReason::PORT_NOT_ACTIVE});
    r.total_count   = 2;
    r.healthy_count = 1;
    return r;
}

// uplink 为 UNKNOWN；显式 device HEALTHY 表示设备可查询，但端口查询无法确定健康状态。
Result unknownResult(std::initializer_list<DeviceHealthState> device_states = {DeviceHealthState::UNKNOWN,
                                                                               DeviceHealthState::UNKNOWN}) {
    Result r = makeResult(State::UNKNOWN);
    for (const auto state : device_states) {
        XDeviceProbeResult d;
        d.device_name  = "mlx5_" + std::to_string(r.devices.size());
        d.device_index = static_cast<uint32_t>(r.devices.size());
        d.port_num     = 1;
        d.device_state = state;
        d.uplink_state = DeviceHealthState::UNKNOWN;
        d.reason       = XDeviceHealthReason::QUERY_TEMPORARY_ERROR;
        r.devices.push_back(std::move(d));
    }
    r.total_count   = static_cast<uint32_t>(r.devices.size());
    r.unknown_count = r.total_count;
    return r;
}

Result queryFailedResult() {
    Result r       = makeResult(State::UNKNOWN);
    r.query_result = BAREX_ERR_STAT;
    return r;
}

Result unsupportedResult() {
    Result r       = makeResult(State::UNKNOWN);
    r.query_result = BAREX_ERR_NOT_SUPPORTED;
    return r;
}

Result portFaultResult() {
    return makeResult(State::UNAVAILABLE, {XDeviceHealthReason::PORT_NOT_ACTIVE, XDeviceHealthReason::PHYSICAL_LINK_DOWN});
}

Result removedDevicesResult() {
    return makeResult(State::UNAVAILABLE, {XDeviceHealthReason::DEVICE_REMOVED, XDeviceHealthReason::DEVICE_REMOVED});
}

Result hardDeviceErrorResult() {
    return makeResult(State::UNAVAILABLE,
                      {XDeviceHealthReason::QUERY_DEVICE_HARD_ERROR, XDeviceHealthReason::QUERY_DEVICE_HARD_ERROR});
}

Result mixedDeviceFaultResult() {
    return makeResult(State::UNAVAILABLE, {XDeviceHealthReason::DEVICE_REMOVED, XDeviceHealthReason::QUERY_DEVICE_HARD_ERROR});
}

Result mixedRemoved() {
    return makeResult(State::UNAVAILABLE, {XDeviceHealthReason::DEVICE_REMOVED, XDeviceHealthReason::PORT_NOT_ACTIVE});
}

RdmaDeviceHealthConfig enabledConfig(const std::string& fault_handler = "ABORT") {
    RdmaDeviceHealthConfig config;
    config.enabled       = true;
    config.fault_handler = fault_handler;
    return config;
}

int64_t elapsedMs(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
}

// 轮询至 pred() 成立；达到 kPromptMs 后仍不成立则返回 false。
template<typename Pred>
bool waitUntil(Pred pred) {
    const auto begin = std::chrono::steady_clock::now();
    while (!pred()) {
        if (elapsedMs(begin) >= kPromptMs) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// 使用注入时钟直接调用 process(r, now_ms)，默认探测间隔 1s、设备确认 3s、窗口 600s、
// 累计阈值 120s、抖动阈值 10 次；注释中的 @t 表示 now_ms = t * 1000。
class RdmaDeviceHealthMonitorTest: public ::testing::Test {
protected:
    void SetUp() override {
        resetMonitor(FaultHandler::ABORT);
    }

    void resetMonitor(FaultHandler handler) {
        monitor_.reset(new RdmaDeviceHealthMonitor(
            [] { return Result(); }, [this] { return now_ms_; }, [this] { abort_at_ms_.push_back(now_ms_); }));
        monitor_->config_        = enabledConfig(handler == FaultHandler::ABORT ? "ABORT" : "LOG");
        monitor_->fault_handler_ = handler;
        abort_at_ms_.clear();
    }

    void feed(const Result& r, int64_t sec) {
        now_ms_ = sec * 1000;
        monitor_->process(r, now_ms_);
    }

    // 在闭区间 [begin_sec, end_sec] 内每秒输入一次 r。
    void feedRange(const Result& r, int64_t begin_sec, int64_t end_sec) {
        for (int64_t t = begin_sec; t <= end_sec; ++t) {
            feed(r, t);
        }
    }

    // 每次恢复包含一轮端口故障和一轮 AVAILABLE，故障发生一秒后恢复。
    void feedFlaps(int count, int64_t start_sec, int64_t step_sec) {
        for (int i = 0; i < count; ++i) {
            feed(portFaultResult(), start_sec + i * step_sec);
            feed(availableResult(), start_sec + i * step_sec + 1);
        }
    }

    int64_t unavailableMs() const {
        return monitor_->unavailableMsInWindow(now_ms_);
    }

    void expectUnavailableWarning() const {
        EXPECT_TRUE(abort_at_ms_.empty());
        EXPECT_FALSE(monitor_->ever_confirmed_fault_);
        EXPECT_EQ(0u, monitor_->log_counts_.fatal);
        EXPECT_EQ(1u, monitor_->log_counts_.unavailable_warn);
        EXPECT_EQ(now_ms_, monitor_->last_unavailable_warn_ms_);
        EXPECT_EQ(0, monitor_->makeMetricsCollector(portFaultResult(), now_ms_).confirmed_fault);
    }

    std::unique_ptr<RdmaDeviceHealthMonitor> monitor_;
    int64_t                                  now_ms_ = 0;
    std::vector<int64_t>                     abort_at_ms_;
};

// ---------------------------------------------------------------------------------------------------------------------
// 故障期、累计告警与设备确认。

TEST_F(RdmaDeviceHealthMonitorTest, AvailableNeverAborts) {
    feedRange(availableResult(), 0, 100);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(-1, monitor_->fault_since_ms_);
    EXPECT_TRUE(monitor_->fault_periods_.empty());
    EXPECT_EQ(0, unavailableMs());
    EXPECT_EQ(0u, monitor_->log_counts_.recovered);
}

TEST_F(RdmaDeviceHealthMonitorTest, DeviceLogsIncludePrimaryAndAllReasons) {
    using accl::barex::DeviceFaultReason;
    struct Case {
        std::vector<DeviceFaultReason> reasons;
        std::string                    expected_diagnostics;
    };
    const std::vector<Case> cases = {
        {{}, "primary_reason NONE, reasons [], raw_port_state "},
        {{DeviceFaultReason::QUERY_DEVICE_FAILED},
         "primary_reason QUERY_DEVICE_FAILED, reasons [QUERY_DEVICE_FAILED], raw_port_state "},
        {{DeviceFaultReason::QUERY_DEVICE_FAILED, DeviceFaultReason::PORT_NOT_ACTIVE},
         "primary_reason QUERY_DEVICE_FAILED, reasons [QUERY_DEVICE_FAILED,PORT_NOT_ACTIVE], raw_port_state "},
    };
    // 子进程中把日志定向到 stdout；不改变其他用例的进程级 logger 配置，也不启动后台线程。
    ASSERT_EXIT(
        {
            auto& logger = Logger::getEngineLogger();
            logger.logger_->setAppender(alog::ConsoleAppender::getAppender());
            logger.logger_->setLevel(alog::LOG_LEVEL_WARN);
            for (const auto& c : cases) {
                resetMonitor(FaultHandler::LOG);
                auto  r               = hardDeviceErrorResult();
                auto& detail          = r.devices.front().detail;
                detail.reasons        = c.reasons;
                detail.primary_reason = c.reasons.empty() ? DeviceFaultReason::NONE : c.reasons.front();
                ::testing::internal::CaptureStdout();
                feed(r, 0);
                logger.flush();
                const auto log          = ::testing::internal::GetCapturedStdout();
                const auto device_begin = log.find("name mlx5_0,");
                EXPECT_NE(std::string::npos, device_begin) << log;
                if (device_begin != std::string::npos) {
                    const auto device_log = log.substr(device_begin, log.find('\n', device_begin) - device_begin);
                    EXPECT_NE(std::string::npos, device_log.find(c.expected_diagnostics)) << device_log;
                }
                // 完整诊断列表只影响日志，不改变设备硬故障确认的起点或提前触发处置。
                EXPECT_EQ(0, monitor_->device_fault_since_ms_);
                EXPECT_EQ(1u, monitor_->log_counts_.fault_begin);
                EXPECT_EQ(0u, monitor_->log_counts_.fatal);
            }
            monitor_.reset();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RdmaDeviceHealthMonitorTest, UnavailableBelowThresholdDoesNotWarn) {
    feedRange(portFaultResult(), 0, 119);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(0, monitor_->fault_since_ms_);
    EXPECT_EQ(-1, monitor_->device_fault_since_ms_);
    EXPECT_EQ(119000, unavailableMs());
    EXPECT_FALSE(monitor_->ever_confirmed_fault_);
    EXPECT_EQ(1u, monitor_->log_counts_.fault_begin);
    EXPECT_EQ(0u, monitor_->log_counts_.device_fault_begin);
}

TEST_F(RdmaDeviceHealthMonitorTest, UnavailableReachesThresholdOnlyWarns) {
    for (const auto handler : {FaultHandler::ABORT, FaultHandler::LOG}) {
        resetMonitor(handler);
        feedRange(portFaultResult(), 0, 119);
        EXPECT_EQ(0u, monitor_->log_counts_.unavailable_warn);
        feed(portFaultResult(), 120);
        expectUnavailableWarning();
        EXPECT_EQ(120000, monitor_->makeMetricsCollector(portFaultResult(), now_ms_).unavailable_ms_in_window);
    }
}

TEST_F(RdmaDeviceHealthMonitorTest, UnavailableWarningIsRateLimited) {
    feedRange(portFaultResult(), 0, 120);
    expectUnavailableWarning();
    feedRange(portFaultResult(), 121, 719);
    EXPECT_EQ(1u, monitor_->log_counts_.unavailable_warn);
    EXPECT_EQ(120000, monitor_->last_unavailable_warn_ms_);
    feed(portFaultResult(), 720);
    EXPECT_EQ(2u, monitor_->log_counts_.unavailable_warn);
    EXPECT_EQ(720000, monitor_->last_unavailable_warn_ms_);
    EXPECT_EQ(600000, unavailableMs());
    EXPECT_FALSE(monitor_->ever_confirmed_fault_);
    EXPECT_TRUE(abort_at_ms_.empty());
}

TEST_F(RdmaDeviceHealthMonitorTest, RecoveryDoesNotResetWarningRateLimit) {
    feedRange(portFaultResult(), 0, 120);
    expectUnavailableWarning();
    feed(availableResult(), 121);
    feedRange(portFaultResult(), 122, 719);
    EXPECT_EQ(1u, monitor_->log_counts_.recovered);
    EXPECT_EQ(1u, monitor_->log_counts_.unavailable_warn);
    EXPECT_EQ(120000, monitor_->last_unavailable_warn_ms_);
    feed(portFaultResult(), 720);
    EXPECT_EQ(2u, monitor_->log_counts_.unavailable_warn);
    EXPECT_EQ(720000, monitor_->last_unavailable_warn_ms_);
    EXPECT_EQ(599000, unavailableMs());
    EXPECT_FALSE(monitor_->ever_confirmed_fault_);
    EXPECT_TRUE(abort_at_ms_.empty());
}

TEST_F(RdmaDeviceHealthMonitorTest, UnavailableWarningUsesConfiguredWindow) {
    monitor_->config_.fault_window_ms          = 10000;
    monitor_->config_.unavailable_threshold_ms = 2000;
    feedRange(portFaultResult(), 0, 1);
    EXPECT_EQ(0u, monitor_->log_counts_.unavailable_warn);
    feed(portFaultResult(), 2);
    expectUnavailableWarning();
    feedRange(portFaultResult(), 3, 11);
    EXPECT_EQ(1u, monitor_->log_counts_.unavailable_warn);
    feed(portFaultResult(), 12);
    EXPECT_EQ(2u, monitor_->log_counts_.unavailable_warn);
    EXPECT_EQ(12000, monitor_->last_unavailable_warn_ms_);
    EXPECT_TRUE(abort_at_ms_.empty());
}

TEST_F(RdmaDeviceHealthMonitorTest, RecoveryKeepsAccumulatedDuration) {
    feedRange(portFaultResult(), 0, 59);
    feed(availableResult(), 60);
    EXPECT_EQ(-1, monitor_->fault_since_ms_);
    ASSERT_EQ(1u, monitor_->fault_periods_.size());
    EXPECT_EQ(0, monitor_->fault_periods_.front().begin_ms);
    EXPECT_EQ(60000, monitor_->fault_periods_.front().end_ms);
    EXPECT_EQ(60000, unavailableMs());

    // 已关闭故障期 60s，加当前未关闭故障期 59s。
    feedRange(portFaultResult(), 61, 120);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(61000, monitor_->fault_since_ms_);
    EXPECT_EQ(119000, unavailableMs());
    feed(portFaultResult(), 121);
    expectUnavailableWarning();
}

TEST_F(RdmaDeviceHealthMonitorTest, UnknownKeepsFaultPeriod) {
    feedRange(portFaultResult(), 0, 49);
    feedRange(unknownResult(), 50, 69);
    EXPECT_EQ(0, monitor_->fault_since_ms_);
    EXPECT_TRUE(monitor_->fault_periods_.empty());
    feedRange(portFaultResult(), 70, 119);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(119000, unavailableMs());
    feed(portFaultResult(), 120);
    expectUnavailableWarning();
}

TEST_F(RdmaDeviceHealthMonitorTest, QueryFailureKeepsFaultPeriod) {
    feedRange(portFaultResult(), 0, 49);
    feedRange(queryFailedResult(), 50, 69);
    EXPECT_EQ(0, monitor_->fault_since_ms_);
    EXPECT_TRUE(monitor_->fault_periods_.empty());
    feedRange(portFaultResult(), 70, 119);
    EXPECT_TRUE(abort_at_ms_.empty());
    feed(portFaultResult(), 120);
    expectUnavailableWarning();
}

TEST_F(RdmaDeviceHealthMonitorTest, UnknownWarningIsRateLimited) {
    feedRange(unknownResult(), 0, 99);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(100u, monitor_->unknown_count_);
    // 第 1 次和第 61 次出现时告警。
    EXPECT_EQ(2u, monitor_->log_counts_.unknown_warn);
}

TEST_F(RdmaDeviceHealthMonitorTest, UnknownDoesNotStartFaultPeriod) {
    for (const auto& unknown :
         {unknownResult(), unknownResult({DeviceHealthState::HEALTHY, DeviceHealthState::UNKNOWN})}) {
        resetMonitor(FaultHandler::ABORT);
        feedRange(unknown, 0, 200);
        EXPECT_TRUE(abort_at_ms_.empty());
        EXPECT_EQ(-1, monitor_->fault_since_ms_);
        EXPECT_EQ(-1, monitor_->device_fault_since_ms_);
        EXPECT_EQ(0, unavailableMs());
        EXPECT_EQ(0u, monitor_->log_counts_.fault_begin);
        EXPECT_EQ(0u, monitor_->log_counts_.unavailable_warn);
        EXPECT_EQ(0u, monitor_->log_counts_.recovered);
    }
}

TEST_F(RdmaDeviceHealthMonitorTest, UnsupportedBackendStopsMonitorWithoutUnknownWarning) {
    EXPECT_TRUE(monitor_->process(queryFailedResult(), 0));
    EXPECT_EQ(1u, monitor_->unknown_count_);

    EXPECT_FALSE(monitor_->process(unsupportedResult(), 1000));
    EXPECT_EQ(1u, monitor_->log_counts_.not_supported);
    EXPECT_EQ(1u, monitor_->unknown_count_);
    EXPECT_EQ(1u, monitor_->log_counts_.unknown_warn);
    EXPECT_EQ(-1, monitor_->fault_since_ms_);
    EXPECT_TRUE(abort_at_ms_.empty());
}

TEST_F(RdmaDeviceHealthMonitorTest, RemovedDevicesConfirmAfterDeviceConfirmDuration) {
    feedRange(removedDevicesResult(), 0, 2);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(0, monitor_->device_fault_since_ms_);
    feed(removedDevicesResult(), 3);
    ASSERT_EQ(1u, abort_at_ms_.size());
    EXPECT_EQ(3000, abort_at_ms_[0]);
    EXPECT_TRUE(monitor_->ever_confirmed_fault_);
    EXPECT_EQ(1u, monitor_->log_counts_.device_fault_begin);
}

TEST_F(RdmaDeviceHealthMonitorTest, HardErrorDevicesConfirmAfterDeviceConfirmDuration) {
    for (const auto& r : {hardDeviceErrorResult(), mixedDeviceFaultResult()}) {
        resetMonitor(FaultHandler::ABORT);
        feedRange(r, 0, 2);
        EXPECT_TRUE(abort_at_ms_.empty());
        feed(r, 3);
        ASSERT_EQ(1u, abort_at_ms_.size());
        EXPECT_EQ(3000, abort_at_ms_[0]);
        EXPECT_EQ(1u, monitor_->log_counts_.device_fault_begin);
    }
}

TEST_F(RdmaDeviceHealthMonitorTest, UnknownKeepsDeviceFaultTimer) {
    // 设备 UNKNOWN、查询失败和空快照均无可信健康证据，保留原确认起点。
    for (const auto& unknown : {unknownResult(), queryFailedResult(), unknownResult({})}) {
        resetMonitor(FaultHandler::ABORT);
        feedRange(removedDevicesResult(), 0, 1);
        feed(unknown, 2);
        EXPECT_EQ(0, monitor_->device_fault_since_ms_);
        EXPECT_EQ(0, monitor_->fault_since_ms_);
        EXPECT_EQ(2000, unavailableMs());
        EXPECT_TRUE(abort_at_ms_.empty());
        EXPECT_FALSE(monitor_->ever_confirmed_fault_);
        feed(removedDevicesResult(), 3);
        ASSERT_EQ(1u, abort_at_ms_.size());
        EXPECT_EQ(3000, abort_at_ms_[0]);
    }
}

TEST_F(RdmaDeviceHealthMonitorTest, UnknownWithHealthyDeviceRestartsDeviceFaultTimer) {
    for (const auto handler : {FaultHandler::ABORT, FaultHandler::LOG}) {
        for (const auto& device_fault : {removedDevicesResult(), hardDeviceErrorResult(), mixedDeviceFaultResult()}) {
            for (const auto& unknown : {unknownResult({DeviceHealthState::HEALTHY, DeviceHealthState::UNKNOWN}),
                                        unknownResult({DeviceHealthState::UNKNOWN, DeviceHealthState::HEALTHY}),
                                        unknownResult({DeviceHealthState::HEALTHY, DeviceHealthState::HEALTHY})}) {
                resetMonitor(handler);
                feed(device_fault, 0);
                feedRange(unknown, 1, 2);
                EXPECT_EQ(-1, monitor_->device_fault_since_ms_);
                EXPECT_EQ(0, monitor_->fault_since_ms_);
                EXPECT_EQ(2000, unavailableMs());
                EXPECT_EQ(0u, monitor_->log_counts_.recovered);
                EXPECT_TRUE(monitor_->fault_periods_.empty());
                EXPECT_FALSE(monitor_->ever_confirmed_fault_);
                EXPECT_TRUE(abort_at_ms_.empty());

                // 设备确认计时在 3s 重新开始，6s 确认，不复用 0s 的起点。
                feedRange(device_fault, 3, 5);
                EXPECT_EQ(3000, monitor_->device_fault_since_ms_);
                EXPECT_FALSE(monitor_->ever_confirmed_fault_);
                EXPECT_EQ(0u, monitor_->log_counts_.fatal);
                EXPECT_TRUE(abort_at_ms_.empty());
                feed(device_fault, 6);
                EXPECT_TRUE(monitor_->ever_confirmed_fault_);
                EXPECT_EQ(1u, monitor_->log_counts_.fatal);
                EXPECT_EQ(2u, monitor_->log_counts_.device_fault_begin);
                if (handler == FaultHandler::ABORT) {
                    ASSERT_EQ(1u, abort_at_ms_.size());
                    EXPECT_EQ(6000, abort_at_ms_[0]);
                } else {
                    EXPECT_TRUE(abort_at_ms_.empty());
                }
            }
        }
    }
}

TEST_F(RdmaDeviceHealthMonitorTest, UnknownWithHealthyDeviceKeepsUnavailableWindow) {
    for (const auto handler : {FaultHandler::ABORT, FaultHandler::LOG}) {
        resetMonitor(handler);
        feedRange(portFaultResult(), 0, 117);
        feed(removedDevicesResult(), 118);
        EXPECT_EQ(118000, monitor_->device_fault_since_ms_);
        const auto unknown = unknownResult({DeviceHealthState::HEALTHY, DeviceHealthState::UNKNOWN});
        feedRange(unknown, 119, 120);
        EXPECT_EQ(-1, monitor_->device_fault_since_ms_);
        EXPECT_EQ(0, monitor_->fault_since_ms_);
        EXPECT_EQ(120000, unavailableMs());
        EXPECT_EQ(0u, monitor_->log_counts_.recovered);
        EXPECT_TRUE(monitor_->fault_periods_.empty());
        EXPECT_EQ(0u, monitor_->log_counts_.unavailable_warn);
        EXPECT_EQ(0u, monitor_->log_counts_.fatal);
        EXPECT_FALSE(monitor_->ever_confirmed_fault_);
        EXPECT_TRUE(abort_at_ms_.empty());
        const auto metrics = monitor_->makeMetricsCollector(unknown, now_ms_);
        EXPECT_EQ(static_cast<int64_t>(State::UNKNOWN), metrics.state);
        EXPECT_EQ(0, metrics.confirmed_fault);
        EXPECT_EQ(120000, metrics.fault_duration_ms);
        EXPECT_EQ(120000, metrics.unavailable_ms_in_window);
        EXPECT_EQ(0, metrics.recover_count_in_window);

        // UNKNOWN 当轮不触发累计 WARN；下一轮 UNAVAILABLE 使用保留的故障期时长，仅告警。
        feed(portFaultResult(), 121);
        expectUnavailableWarning();
        EXPECT_EQ(121000, unavailableMs());
    }
}

TEST_F(RdmaDeviceHealthMonitorTest, UnknownWithOnlyHealthyUplinksKeepsDeviceFaultTimer) {
    auto unknown = unknownResult();
    for (auto& device : unknown.devices) {
        device.uplink_state = DeviceHealthState::HEALTHY;
    }
    feed(removedDevicesResult(), 0);
    feedRange(unknown, 1, 3);
    EXPECT_EQ(0, monitor_->device_fault_since_ms_);
    EXPECT_FALSE(monitor_->ever_confirmed_fault_);
    EXPECT_TRUE(abort_at_ms_.empty());
    feed(removedDevicesResult(), 4);
    ASSERT_EQ(1u, abort_at_ms_.size());
    EXPECT_EQ(4000, abort_at_ms_[0]);
}

TEST_F(RdmaDeviceHealthMonitorTest, QueryFailureIgnoresHealthyDeviceDetails) {
    auto failed = unknownResult({DeviceHealthState::HEALTHY, DeviceHealthState::UNKNOWN});
    failed.query_result = BAREX_ERR_STAT;
    feedRange(removedDevicesResult(), 0, 1);
    feed(failed, 2);
    EXPECT_EQ(0, monitor_->device_fault_since_ms_);
    EXPECT_FALSE(monitor_->ever_confirmed_fault_);
    EXPECT_TRUE(abort_at_ms_.empty());
    feed(removedDevicesResult(), 3);
    ASSERT_EQ(1u, abort_at_ms_.size());
    EXPECT_EQ(3000, abort_at_ms_[0]);
}

TEST_F(RdmaDeviceHealthMonitorTest, AvailableResetsDeviceFaultTimer) {
    feedRange(removedDevicesResult(), 0, 1);
    feed(availableResult(), 2);
    EXPECT_EQ(-1, monitor_->device_fault_since_ms_);
    feedRange(removedDevicesResult(), 3, 5);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(3000, monitor_->device_fault_since_ms_);
    feed(removedDevicesResult(), 6);
    ASSERT_EQ(1u, abort_at_ms_.size());
    EXPECT_EQ(6000, abort_at_ms_[0]);
    EXPECT_EQ(2u, monitor_->log_counts_.device_fault_begin);
}

TEST_F(RdmaDeviceHealthMonitorTest, DegradedResetsDeviceFaultTimer) {
    feedRange(removedDevicesResult(), 0, 1);
    feed(degradedResult(), 2);
    EXPECT_EQ(-1, monitor_->device_fault_since_ms_);
    // 总体故障期仍未关闭。
    EXPECT_EQ(0, monitor_->fault_since_ms_);
    feedRange(removedDevicesResult(), 3, 5);
    EXPECT_TRUE(abort_at_ms_.empty());
    feed(removedDevicesResult(), 6);
    ASSERT_EQ(1u, abort_at_ms_.size());
    EXPECT_EQ(6000, abort_at_ms_[0]);
}

TEST_F(RdmaDeviceHealthMonitorTest, PortFaultResetsDeviceFaultTimer) {
    feedRange(removedDevicesResult(), 0, 1);
    feed(portFaultResult(), 2);
    EXPECT_EQ(-1, monitor_->device_fault_since_ms_);
    EXPECT_EQ(0, monitor_->fault_since_ms_);
    feedRange(removedDevicesResult(), 3, 5);
    EXPECT_TRUE(abort_at_ms_.empty());
    feed(removedDevicesResult(), 6);
    ASSERT_EQ(1u, abort_at_ms_.size());
    EXPECT_EQ(6000, abort_at_ms_[0]);
}

TEST_F(RdmaDeviceHealthMonitorTest, DeviceFaultTimerStartsAtFirstDeviceRound) {
    feedRange(portFaultResult(), 0, 9);
    feedRange(removedDevicesResult(), 10, 12);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(0, monitor_->fault_since_ms_);
    EXPECT_EQ(10000, monitor_->device_fault_since_ms_);
    feed(removedDevicesResult(), 13);
    ASSERT_EQ(1u, abort_at_ms_.size());
    EXPECT_EQ(13000, abort_at_ms_[0]);
}

TEST_F(RdmaDeviceHealthMonitorTest, PartiallyRemovedUsesUnavailableThreshold) {
    feedRange(mixedRemoved(), 0, 119);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(-1, monitor_->device_fault_since_ms_);
    feed(mixedRemoved(), 120);
    expectUnavailableWarning();
    EXPECT_EQ(0u, monitor_->log_counts_.device_fault_begin);
}

TEST_F(RdmaDeviceHealthMonitorTest, DegradedNeverAborts) {
    feedRange(degradedResult(), 0, 100);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(-1, monitor_->fault_since_ms_);
    EXPECT_EQ(101u, monitor_->degraded_count_);
    // 第 1 次和第 61 次出现时告警。
    EXPECT_EQ(2u, monitor_->log_counts_.degraded_warn);
}

TEST_F(RdmaDeviceHealthMonitorTest, DegradedKeepsFaultPeriod) {
    feedRange(portFaultResult(), 0, 4);
    feedRange(degradedResult(), 5, 119);
    EXPECT_EQ(0, monitor_->fault_since_ms_);
    EXPECT_EQ(119000, unavailableMs());
    EXPECT_TRUE(abort_at_ms_.empty());
    feed(portFaultResult(), 120);
    expectUnavailableWarning();
}

TEST_F(RdmaDeviceHealthMonitorTest, DegradedRoundsDoNotConfirm) {
    feedRange(portFaultResult(), 0, 4);
    feedRange(degradedResult(), 5, 200);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_FALSE(monitor_->ever_confirmed_fault_);
    EXPECT_EQ(200000, unavailableMs());
    // DEGRADED 当轮不触发累计 WARN；下一轮 UNAVAILABLE 使用累计故障期时长告警。
    EXPECT_EQ(0u, monitor_->log_counts_.unavailable_warn);
    feed(portFaultResult(), 201);
    expectUnavailableWarning();
}

TEST_F(RdmaDeviceHealthMonitorTest, LongOutagesWithShortRecoveriesAccumulate) {
    feedRange(portFaultResult(), 0, 49);
    feedRange(availableResult(), 50, 59);
    feedRange(portFaultResult(), 60, 109);
    feedRange(availableResult(), 110, 119);
    EXPECT_EQ(2u, monitor_->fault_periods_.size());
    EXPECT_EQ(100000, unavailableMs());
    feedRange(portFaultResult(), 120, 139);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_EQ(119000, unavailableMs());
    feed(portFaultResult(), 140);
    expectUnavailableWarning();
}

TEST_F(RdmaDeviceHealthMonitorTest, OldFaultSlidesOutOfWindow) {
    feedRange(portFaultResult(), 0, 99);
    feedRange(availableResult(), 100, 650);
    ASSERT_EQ(1u, monitor_->fault_periods_.size());
    // 已关闭故障期 [0s, 100s] 逐渐滑出窗口，当前故障期同步增长，总累计时长保持 49s。
    for (int64_t t = 651; t <= 700; ++t) {
        feed(portFaultResult(), t);
        EXPECT_EQ(49000, unavailableMs()) << t;
    }
    EXPECT_EQ(1u, monitor_->fault_periods_.size());
    feedRange(portFaultResult(), 701, 770);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_TRUE(monitor_->fault_periods_.empty());
    EXPECT_EQ(119000, unavailableMs());
    feed(portFaultResult(), 771);
    expectUnavailableWarning();
}

// ---------------------------------------------------------------------------------------------------------------------
// 抖动告警。

TEST_F(RdmaDeviceHealthMonitorTest, FrequentShortFaultsWarnWithoutAbort) {
    // 每 10s 故障 1s，持续 20 分钟；任一 600s 窗口累计故障时长最多 60s。
    feedFlaps(120, 0, 10);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_FALSE(monitor_->ever_confirmed_fault_);
    // 在 91s（第 10 次恢复）和 691s（相隔一个窗口）告警。
    EXPECT_EQ(2u, monitor_->log_counts_.flapping);
    EXPECT_EQ(120u, monitor_->log_counts_.recovered);
    EXPECT_EQ(60000, unavailableMs());
    EXPECT_EQ(61, monitor_->recoverCountInWindow(now_ms_));
}

TEST_F(RdmaDeviceHealthMonitorTest, TenRecoveriesInWindowWarnFlapping) {
    feedFlaps(10, 0, 2);
    EXPECT_EQ(1u, monitor_->log_counts_.flapping);
    EXPECT_EQ(10u, monitor_->fault_periods_.size());
    EXPECT_EQ(10u, monitor_->log_counts_.recovered);
    EXPECT_TRUE(abort_at_ms_.empty());
}

TEST_F(RdmaDeviceHealthMonitorTest, NineRecoveriesInWindowDoNotWarn) {
    feedFlaps(9, 0, 2);
    EXPECT_EQ(0u, monitor_->log_counts_.flapping);
    EXPECT_EQ(9u, monitor_->fault_periods_.size());
}

TEST_F(RdmaDeviceHealthMonitorTest, FlappingWarnsOncePerWindow) {
    feedFlaps(20, 0, 2);
    EXPECT_EQ(1u, monitor_->log_counts_.flapping);
    EXPECT_EQ(20u, monitor_->fault_periods_.size());
}

TEST_F(RdmaDeviceHealthMonitorTest, RecoveriesSpreadBeyondWindowDoNotWarn) {
    // 在 1s、71s……631s 恢复，任一 600s 窗口内均不足 10 次。
    feedFlaps(10, 0, 70);
    EXPECT_EQ(0u, monitor_->log_counts_.flapping);
    EXPECT_EQ(10u, monitor_->log_counts_.recovered);
    EXPECT_EQ(9, monitor_->recoverCountInWindow(now_ms_));
}

TEST_F(RdmaDeviceHealthMonitorTest, FlappingWarnsAgainInNextWindow) {
    // 先在 1s～19s 恢复，中间无事件，再在 621s～639s 恢复。
    feedFlaps(10, 0, 2);
    EXPECT_EQ(1u, monitor_->log_counts_.flapping);
    feedFlaps(10, 620, 2);
    EXPECT_EQ(2u, monitor_->log_counts_.flapping);
    ASSERT_EQ(10u, monitor_->fault_periods_.size());
    EXPECT_EQ(620000, monitor_->fault_periods_.front().begin_ms);
    EXPECT_EQ(621000, monitor_->fault_periods_.front().end_ms);
}

TEST_F(RdmaDeviceHealthMonitorTest, AvailableWithoutFaultIsNotRecovery) {
    feedRange(availableResult(), 0, 10);
    EXPECT_TRUE(monitor_->fault_periods_.empty());
    EXPECT_EQ(0u, monitor_->log_counts_.recovered);
}

// ---------------------------------------------------------------------------------------------------------------------
// 设备确认处置。

TEST_F(RdmaDeviceHealthMonitorTest, LogModeHandlesContinuousFaultOnceWithoutAbort) {
    resetMonitor(FaultHandler::LOG);
    feedRange(removedDevicesResult(), 0, 130);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_TRUE(monitor_->ever_confirmed_fault_);
    EXPECT_EQ(1u, monitor_->log_counts_.fatal);
    EXPECT_TRUE(monitor_->fatal_handled_in_current_period_);
    EXPECT_EQ(1u, monitor_->log_counts_.unavailable_warn);
    EXPECT_EQ(120000, monitor_->last_unavailable_warn_ms_);

    // 设备确认去重后仍更新当前指标，并允许累计不可用告警。
    const auto c = monitor_->makeMetricsCollector(removedDevicesResult(), now_ms_);
    EXPECT_EQ(1, c.confirmed_fault);
    EXPECT_EQ(130000, c.fault_duration_ms);
    EXPECT_EQ(130000, c.unavailable_ms_in_window);
    EXPECT_EQ(static_cast<int64_t>(State::UNAVAILABLE), c.state);
    EXPECT_EQ(2, c.fault_count);
}

TEST_F(RdmaDeviceHealthMonitorTest, LogModeConfirmsNewFaultAfterRecovery) {
    resetMonitor(FaultHandler::LOG);
    feedRange(removedDevicesResult(), 0, 125);
    ASSERT_TRUE(monitor_->ever_confirmed_fault_);
    feed(availableResult(), 126);
    EXPECT_EQ(-1, monitor_->fault_since_ms_);
    EXPECT_EQ(1u, monitor_->log_counts_.recovered);
    ASSERT_EQ(1u, monitor_->fault_periods_.size());
    EXPECT_EQ(0, monitor_->fault_periods_.front().begin_ms);
    EXPECT_EQ(126000, monitor_->fault_periods_.front().end_ms);

    EXPECT_FALSE(monitor_->fatal_handled_in_current_period_);
    EXPECT_TRUE(monitor_->ever_confirmed_fault_);
    const auto recovered_metrics = monitor_->makeMetricsCollector(availableResult(), now_ms_);
    EXPECT_EQ(1, recovered_metrics.confirmed_fault);
    EXPECT_EQ(0, recovered_metrics.fault_duration_ms);
    EXPECT_EQ(126000, recovered_metrics.unavailable_ms_in_window);

    feedRange(removedDevicesResult(), 127, 129);
    EXPECT_EQ(127000, monitor_->device_fault_since_ms_);
    EXPECT_EQ(1u, monitor_->log_counts_.fatal);
    EXPECT_FALSE(monitor_->fatal_handled_in_current_period_);
    feed(removedDevicesResult(), 130);
    EXPECT_EQ(2u, monitor_->log_counts_.fatal);
    EXPECT_TRUE(monitor_->fatal_handled_in_current_period_);
    feedRange(removedDevicesResult(), 131, 200);
    EXPECT_EQ(2u, monitor_->log_counts_.fault_begin);
    EXPECT_EQ(2u, monitor_->log_counts_.device_fault_begin);
    EXPECT_EQ(2u, monitor_->log_counts_.fatal);
    EXPECT_TRUE(abort_at_ms_.empty());
}

TEST_F(RdmaDeviceHealthMonitorTest, LogModeUnknownAndDegradedDoNotResetHandledFault) {
    for (const auto& intermediate : {unknownResult(),
                                     queryFailedResult(),
                                     degradedResult(),
                                     unknownResult({DeviceHealthState::HEALTHY, DeviceHealthState::UNKNOWN})}) {
        resetMonitor(FaultHandler::LOG);
        feedRange(removedDevicesResult(), 0, 3);
        ASSERT_TRUE(monitor_->fatal_handled_in_current_period_);
        feedRange(intermediate, 4, 10);
        EXPECT_TRUE(monitor_->fatal_handled_in_current_period_);
        EXPECT_EQ(0, monitor_->fault_since_ms_);
        EXPECT_EQ(0u, monitor_->log_counts_.recovered);
        EXPECT_EQ(1, monitor_->makeMetricsCollector(intermediate, now_ms_).confirmed_fault);
        feedRange(removedDevicesResult(), 11, 20);
        EXPECT_EQ(1u, monitor_->log_counts_.fatal);
        EXPECT_TRUE(abort_at_ms_.empty());
    }
}

TEST_F(RdmaDeviceHealthMonitorTest, AbortModeHandlesFaultOnlyOnceAcrossRecovery) {
    feedRange(removedDevicesResult(), 0, 3);
    ASSERT_EQ(1u, abort_at_ms_.size());
    EXPECT_EQ(3000, abort_at_ms_[0]);
    feed(availableResult(), 4);
    EXPECT_FALSE(monitor_->fatal_handled_in_current_period_);
    EXPECT_TRUE(monitor_->ever_confirmed_fault_);
    EXPECT_EQ(1, monitor_->makeMetricsCollector(availableResult(), now_ms_).confirmed_fault);
    feedRange(hardDeviceErrorResult(), 5, 130);
    EXPECT_EQ(2u, monitor_->log_counts_.device_fault_begin);
    EXPECT_EQ(1u, monitor_->log_counts_.fatal);
    ASSERT_EQ(1u, abort_at_ms_.size());
    EXPECT_EQ(3000, abort_at_ms_[0]);
    EXPECT_EQ(1u, monitor_->log_counts_.unavailable_warn);
    EXPECT_EQ(121000, monitor_->last_unavailable_warn_ms_);
}

TEST_F(RdmaDeviceHealthMonitorTest, LogModeDeviceConfirmationDoesNotResetUnavailableWarningRateLimit) {
    resetMonitor(FaultHandler::LOG);
    feedRange(removedDevicesResult(), 0, 120);
    ASSERT_EQ(1u, monitor_->log_counts_.fatal);
    ASSERT_EQ(1u, monitor_->log_counts_.unavailable_warn);
    feed(availableResult(), 121);
    feedRange(hardDeviceErrorResult(), 122, 719);
    EXPECT_EQ(2u, monitor_->log_counts_.fatal);
    EXPECT_EQ(1u, monitor_->log_counts_.unavailable_warn);
    EXPECT_EQ(120000, monitor_->last_unavailable_warn_ms_);
    feed(hardDeviceErrorResult(), 720);
    EXPECT_EQ(2u, monitor_->log_counts_.unavailable_warn);
    EXPECT_EQ(720000, monitor_->last_unavailable_warn_ms_);
    EXPECT_EQ(599000, unavailableMs());
    EXPECT_EQ(1, monitor_->makeMetricsCollector(hardDeviceErrorResult(), now_ms_).confirmed_fault);
    EXPECT_TRUE(abort_at_ms_.empty());
}

TEST_F(RdmaDeviceHealthMonitorTest, DeviceConfirmationPrecedesSameRoundUnavailableWarning) {
    for (const auto handler : {FaultHandler::ABORT, FaultHandler::LOG}) {
        resetMonitor(handler);
        feedRange(portFaultResult(), 0, 116);
        feedRange(removedDevicesResult(), 117, 119);
        EXPECT_EQ(0u, monitor_->log_counts_.fatal);
        EXPECT_EQ(0u, monitor_->log_counts_.unavailable_warn);
        feed(removedDevicesResult(), 120);
        EXPECT_EQ(1u, monitor_->log_counts_.fatal);
        EXPECT_EQ(0u, monitor_->log_counts_.unavailable_warn);
        feed(removedDevicesResult(), 121);
        EXPECT_EQ(1u, monitor_->log_counts_.fatal);
        EXPECT_EQ(1u, monitor_->log_counts_.unavailable_warn);
        EXPECT_EQ(121000, monitor_->last_unavailable_warn_ms_);
        if (handler == FaultHandler::ABORT) {
            ASSERT_EQ(1u, abort_at_ms_.size());
            EXPECT_EQ(120000, abort_at_ms_[0]);
        } else {
            EXPECT_TRUE(abort_at_ms_.empty());
        }
    }
}

TEST_F(RdmaDeviceHealthMonitorTest, PortWarningDoesNotSuppressDeviceFault) {
    for (const auto& device_fault : {removedDevicesResult(), hardDeviceErrorResult(), mixedDeviceFaultResult()}) {
        resetMonitor(FaultHandler::ABORT);
        feedRange(portFaultResult(), 0, 120);
        expectUnavailableWarning();
        feedRange(portFaultResult(), 121, 130);
        feedRange(device_fault, 131, 133);
        EXPECT_TRUE(abort_at_ms_.empty());
        EXPECT_FALSE(monitor_->ever_confirmed_fault_);
        feedRange(device_fault, 134, 140);
        ASSERT_EQ(1u, abort_at_ms_.size());
        EXPECT_EQ(134000, abort_at_ms_[0]);
        EXPECT_TRUE(monitor_->ever_confirmed_fault_);
        EXPECT_EQ(1u, monitor_->log_counts_.fatal);
        EXPECT_EQ(1u, monitor_->log_counts_.unavailable_warn);
        EXPECT_EQ(1u, monitor_->log_counts_.device_fault_begin);
        EXPECT_EQ(1, monitor_->makeMetricsCollector(device_fault, now_ms_).confirmed_fault);
    }
}

TEST_F(RdmaDeviceHealthMonitorTest, StoppingSuppressesAbort) {
    monitor_->stopping_ = true;
    feedRange(removedDevicesResult(), 0, 3);
    EXPECT_TRUE(abort_at_ms_.empty());
    EXPECT_TRUE(monitor_->ever_confirmed_fault_);
    EXPECT_EQ(1u, monitor_->log_counts_.fatal);
}

TEST_F(RdmaDeviceHealthMonitorTest, EvaluateReportsEventReason) {
    feedRange(removedDevicesResult(), 0, 2);
    EXPECT_EQ(EventReason::DEVICE_FAULT_CONFIRMED, monitor_->evaluate(removedDevicesResult(), 3000));

    resetMonitor(FaultHandler::ABORT);
    feedRange(portFaultResult(), 0, 119);
    EXPECT_EQ(EventReason::UNAVAILABLE_THRESHOLD_EXCEEDED, monitor_->evaluate(portFaultResult(), 120000));

    // 设备硬故障仅持续 2s 时，累计阈值可以告警，但不得提前 abort。
    resetMonitor(FaultHandler::ABORT);
    feedRange(portFaultResult(), 0, 117);
    feedRange(removedDevicesResult(), 118, 119);
    EXPECT_EQ(EventReason::UNAVAILABLE_THRESHOLD_EXCEEDED, monitor_->evaluate(removedDevicesResult(), 120000));
    feed(removedDevicesResult(), 120);
    expectUnavailableWarning();
    feed(removedDevicesResult(), 121);
    ASSERT_EQ(1u, abort_at_ms_.size());
    EXPECT_EQ(121000, abort_at_ms_[0]);
    EXPECT_EQ(1u, monitor_->log_counts_.fatal);
}

TEST_F(RdmaDeviceHealthMonitorTest, MetricsStateEncodingIsUnchanged) {
    // 三态编码统一不改变进程四态的指标编码，尤其 UNKNOWN 仍为 3。
    const auto expectEncoding = [this](State state, int64_t encoding) {
        EXPECT_EQ(encoding, static_cast<int64_t>(state));
        EXPECT_EQ(encoding, monitor_->makeMetricsCollector(makeResult(state), now_ms_).state);
    };
    expectEncoding(State::AVAILABLE, 0);
    expectEncoding(State::DEGRADED, 1);
    expectEncoding(State::UNAVAILABLE, 2);
    expectEncoding(State::UNKNOWN, 3);
}

TEST_F(RdmaDeviceHealthMonitorTest, MetricsCollectorReflectsState) {
    feedRange(portFaultResult(), 0, 3);
    auto c = monitor_->makeMetricsCollector(portFaultResult(), now_ms_);
    EXPECT_EQ(static_cast<int64_t>(BAREX_SUCCESS), c.query_result);
    EXPECT_FALSE(c.query_failed);
    EXPECT_EQ(static_cast<int64_t>(State::UNAVAILABLE), c.state);
    EXPECT_EQ(2, c.total_count);
    EXPECT_EQ(2, c.fault_count);
    EXPECT_EQ(0, c.confirmed_fault);
    EXPECT_EQ(3000, c.fault_duration_ms);
    EXPECT_EQ(3000, c.unavailable_ms_in_window);
    EXPECT_EQ(0, c.recover_count_in_window);

    feed(queryFailedResult(), 4);
    c = monitor_->makeMetricsCollector(queryFailedResult(), now_ms_);
    EXPECT_EQ(static_cast<int64_t>(BAREX_ERR_STAT), c.query_result);
    EXPECT_TRUE(c.query_failed);
    EXPECT_EQ(static_cast<int64_t>(State::UNKNOWN), c.state);
    EXPECT_EQ(4000, c.fault_duration_ms);
    EXPECT_EQ(4000, c.unavailable_ms_in_window);

    feed(availableResult(), 5);
    c = monitor_->makeMetricsCollector(availableResult(), now_ms_);
    EXPECT_EQ(0, c.fault_duration_ms);
    EXPECT_EQ(1, c.recover_count_in_window);
    EXPECT_EQ(5000, c.unavailable_ms_in_window);
    // 故障期 [0s, 5s] 在 600s～605s 之间逐渐滑出 600s 窗口。
    EXPECT_EQ(5000, monitor_->makeMetricsCollector(availableResult(), 600000).unavailable_ms_in_window);
    EXPECT_EQ(2000, monitor_->makeMetricsCollector(availableResult(), 603000).unavailable_ms_in_window);
    EXPECT_EQ(0, monitor_->makeMetricsCollector(availableResult(), 605000).unavailable_ms_in_window);
    // 5s 的恢复事件在 605s 边界仍计数，超过该边界后移出窗口。
    EXPECT_EQ(1, monitor_->makeMetricsCollector(availableResult(), 605000).recover_count_in_window);
    EXPECT_EQ(0, monitor_->makeMetricsCollector(availableResult(), 605001).recover_count_in_window);
}

// ---------------------------------------------------------------------------------------------------------------------
// 使用真实 monitor 线程的生命周期测试。

class RdmaDeviceHealthMonitorLifecycleTest: public ::testing::Test {
protected:
    std::unique_ptr<RdmaDeviceHealthMonitor> makeMonitor(Result probe_result) {
        return std::unique_ptr<RdmaDeviceHealthMonitor>(new RdmaDeviceHealthMonitor(
            [this, probe_result] {
                if (++probe_count_ > 1 && stop_on_second_probe_ != nullptr) {
                    stop_on_second_probe_->stopping_ = true;
                }
                return probe_result;
            },
            [] {
                return std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            },
            [this] { ++abort_count_; }));
    }

    std::atomic<int> probe_count_{0};
    std::atomic<int> abort_count_{0};
    // 设置后，第二次探测标记此 monitor 为 stopping，使未按预期退出的线程也能结束。
    // 必须在 ensureStarted() 前设置，通过线程创建将其发布给 monitor 线程。
    RdmaDeviceHealthMonitor* stop_on_second_probe_ = nullptr;
};

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, DisabledDoesNotStart) {
    auto                   monitor = makeMonitor(availableResult());
    RdmaDeviceHealthConfig config;
    ASSERT_FALSE(config.enabled);
    EXPECT_FALSE(monitor->ensureStarted(config, nullptr));
    EXPECT_FALSE(monitor->started_);
    EXPECT_FALSE(monitor->thread_.joinable());
}

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, InvalidConfigDoesNotStart) {
    auto monitor = makeMonitor(availableResult());

    auto short_interval              = enabledConfig();
    short_interval.probe_interval_ms = 99;
    std::string error;
    EXPECT_FALSE(validateConfig(short_interval, nullptr, &error));
    EXPECT_NE(std::string::npos, error.find("probe_interval_ms"));
    EXPECT_FALSE(monitor->ensureStarted(short_interval, nullptr));

    auto short_confirm                    = enabledConfig();
    short_confirm.device_fault_confirm_ms = 1999;
    error.clear();
    EXPECT_FALSE(validateConfig(short_confirm, nullptr, &error));
    EXPECT_NE(std::string::npos, error.find("device_fault_confirm_ms"));
    EXPECT_FALSE(monitor->ensureStarted(short_confirm, nullptr));

    auto threshold_above_window                     = enabledConfig();
    threshold_above_window.unavailable_threshold_ms = 600001;
    error.clear();
    EXPECT_FALSE(validateConfig(threshold_above_window, nullptr, &error));
    EXPECT_NE(std::string::npos, error.find("greater than fault_window_ms"));
    EXPECT_FALSE(monitor->ensureStarted(threshold_above_window, nullptr));

    EXPECT_FALSE(monitor->started_);
    EXPECT_FALSE(monitor->thread_.joinable());
}

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, SecondStartKeepsFirstConfig) {
    auto monitor                  = makeMonitor(availableResult());
    auto first                    = enabledConfig();
    first.probe_interval_ms       = 60000;
    first.device_fault_confirm_ms = 120000;
    ASSERT_TRUE(monitor->ensureStarted(first, nullptr));
    const auto thread_id = monitor->thread_.get_id();

    auto second                    = first;
    second.device_fault_confirm_ms = 180000;
    EXPECT_TRUE(monitor->ensureStarted(second, nullptr));
    EXPECT_EQ(thread_id, monitor->thread_.get_id());
    EXPECT_EQ(1u, monitor->log_counts_.config_differs);
    EXPECT_EQ(120000, monitor->config_.device_fault_confirm_ms);

    // 相同配置不告警。
    EXPECT_TRUE(monitor->ensureStarted(first, nullptr));
    EXPECT_EQ(1u, monitor->log_counts_.config_differs);
    monitor->stop();
}

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, StartAfterStopFails) {
    auto monitor = makeMonitor(availableResult());
    auto config  = enabledConfig();
    ASSERT_TRUE(monitor->ensureStarted(config, nullptr));
    monitor->stop();
    EXPECT_FALSE(monitor->ensureStarted(config, nullptr));
    EXPECT_FALSE(monitor->thread_.joinable());

    auto never_started = makeMonitor(availableResult());
    never_started->stop();
    EXPECT_FALSE(never_started->ensureStarted(config, nullptr));
}

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, StopIsIdempotent) {
    auto never_started = makeMonitor(availableResult());
    never_started->stop();
    never_started->stop();
    EXPECT_FALSE(never_started->thread_.joinable());

    auto started = makeMonitor(availableResult());
    ASSERT_TRUE(started->ensureStarted(enabledConfig(), nullptr));
    started->stop();
    started->stop();
    EXPECT_FALSE(started->thread_.joinable());
}

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, StopWakesUpLongInterval) {
    auto monitor                   = makeMonitor(availableResult());
    auto config                    = enabledConfig();
    config.probe_interval_ms       = kLongIntervalMs;
    config.device_fault_confirm_ms = 2 * kLongIntervalMs;
    ASSERT_TRUE(monitor->ensureStarted(config, nullptr));
    // 首轮探测后调用 stop，此时线程将进入整段探测间隔的等待。
    ASSERT_TRUE(waitUntil([this] { return probe_count_.load() >= 1; }));
    const auto begin = std::chrono::steady_clock::now();
    monitor->stop();
    EXPECT_LT(elapsedMs(begin), kPromptMs);
    EXPECT_FALSE(monitor->thread_.joinable());
    EXPECT_EQ(1, probe_count_.load());
}

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, DeviceFaultAbortsAfterDeviceConfirmAndOnce) {
    auto monitor                   = makeMonitor(removedDevicesResult());
    auto config                    = enabledConfig("ABORT");
    config.probe_interval_ms       = 100;
    config.device_fault_confirm_ms = RdmaDeviceHealthConfig::kMinDeviceFaultConfirmMs;
    // 将累计告警阈值设为一小时，使真实线程测试聚焦设备确认。
    config.fault_window_ms          = RdmaDeviceHealthConfig::kMaxFaultWindowMs;
    config.unavailable_threshold_ms = RdmaDeviceHealthConfig::kMaxUnavailableThresholdMs;
    ASSERT_TRUE(monitor->ensureStarted(config, nullptr));
    ASSERT_TRUE(waitUntil([this] { return abort_count_.load() > 0; }));

    // 后续再探测几轮，不得重复 abort。
    const int probes = probe_count_.load();
    ASSERT_TRUE(waitUntil([this, probes] { return probe_count_.load() >= probes + 3; }));
    // stop() 会 join，最终检查时已开始的探测轮次均已结束。
    monitor->stop();
    EXPECT_EQ(1, abort_count_.load());
}

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, DestructorJoinsPromptly) {
    auto monitor                   = makeMonitor(availableResult());
    auto config                    = enabledConfig();
    config.probe_interval_ms       = kLongIntervalMs;
    config.device_fault_confirm_ms = 2 * kLongIntervalMs;
    ASSERT_TRUE(monitor->ensureStarted(config, nullptr));
    // 首轮探测后析构，此时线程将进入整段探测间隔的等待。
    ASSERT_TRUE(waitUntil([this] { return probe_count_.load() >= 1; }));
    const auto begin = std::chrono::steady_clock::now();
    // 析构仍 joinable 的 std::thread 会触发 std::terminate，正常返回应表明已完成 join。
    monitor.reset();
    EXPECT_LT(elapsedMs(begin), kPromptMs);
    EXPECT_EQ(1, probe_count_.load());
    EXPECT_EQ(0, abort_count_.load());
}

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, UnsupportedBackendThreadExitsAfterFirstProbe) {
    auto monitor                   = makeMonitor(unsupportedResult());
    auto config                    = enabledConfig("ABORT");
    config.probe_interval_ms       = 100;
    config.device_fault_confirm_ms = 1000;
    stop_on_second_probe_          = monitor.get();
    ASSERT_TRUE(monitor->ensureStarted(config, nullptr));
    // 不先 stop，直接 join：等待首轮探测后正常退出，或由第二轮探测标记 stopping_ 后退出。
    // 此检查不依赖固定 sleep 或超时期限。
    monitor->thread_.join();
    EXPECT_EQ(1, probe_count_.load());
    EXPECT_FALSE(monitor->stopping_.load());
    // 日志计数由 monitor 线程写入，join 后可安全读取。
    EXPECT_EQ(1u, monitor->log_counts_.not_supported);
    EXPECT_EQ(0u, monitor->log_counts_.unknown_warn);
    EXPECT_EQ(0, abort_count_.load());
    // 线程已退出后仍可调用 stop()。
    monitor->stop();
    EXPECT_FALSE(monitor->thread_.joinable());
}

// 关闭改变进程级 factory/回调注册状态并停止已有 monitor，因此用子进程隔离，避免影响其它 UT。
// factory 可重新初始化，但已停止的 monitor 不可重启。
TEST_F(RdmaDeviceHealthMonitorLifecycleTest, FactoryShutdownWaitsForInFlightReport) {
    ASSERT_EXIT(
        {
            if (!initKmonitorFactory()) {
                _exit(1);
            }
            // KMonitor 由 factory 持有并在 Shutdown 时释放，验证在途上报不会访问已释放的指标资源。
            // 删除器为空：owning reporter 在 factory 关闭后析构会解引用已释放的 KMonitor，属 kmonitor 既有问题。
            kmonitor::KMonitorPtr monitor_resource(kmonitor::KMonitorFactory::GetKMonitor("rdma_shutdown_test"),
                                                   [](kmonitor::KMonitor*) {});
            auto reporter = std::make_shared<kmonitor::MetricsReporter>(monitor_resource, "", kmonitor::MetricsTags());
            // 阻塞真实 report 的 metrics group 查询，扩大已开始上报与 factory 关闭之间的竞态窗口。
            auto report_lock = std::unique_ptr<autil::ScopedWriteLock>(
                new autil::ScopedWriteLock(reporter->_metricsGroupManager->_groupRwLock));
            auto monitor = makeMonitor(availableResult());
            if (!monitor->ensureStarted(enabledConfig(), reporter)
                || !waitUntil([this] { return probe_count_.load() == 1; })) {
                _exit(2);
            }
            std::atomic<bool> shutdown_finished{false};
            std::thread shutdown([&] {
                stopKmonitorFactory();
                shutdown_finished = true;
            });
            if (!waitUntil([&] { return monitor->stopping_.load() || shutdown_finished.load(); })) {
                _exit(3);
            }
            // stop 已开始但在途 report 被锁阻塞，factory 必须仍存活；不能只检查最终没有崩溃。
            if (!monitor->stopping_.load() || shutdown_finished.load() || !kmonitor::KMonitorFactory::IsStarted()) {
                _exit(4);
            }
            report_lock.reset();
            shutdown.join();
            if (!shutdown_finished.load() || kmonitor::KMonitorFactory::IsStarted()
                || monitor->thread_.joinable() || probe_count_.load() != 1) {
                _exit(5);
            }
            monitor.reset();
            reporter.reset();
            _exit(0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, FactoryShutdownStopsAllMonitorsAndRejectsLateStart) {
    ASSERT_EXIT(
        {
            auto first = makeMonitor(availableResult());
            auto second = makeMonitor(availableResult());
            if (!first->ensureStarted(enabledConfig(), nullptr) || !second->ensureStarted(enabledConfig(), nullptr)) {
                _exit(1);
            }
            // 首次调用停止并 join 所有 monitor；生产关闭流程调用一次即可。
            stopKmonitorFactory();
            // 第二次验证重复关闭安全：存活 monitor 的回调仍会执行，stop() 必须可重复调用。
            // 本用例未初始化实际 factory，重点验证回调和 monitor 的关闭状态。
            stopKmonitorFactory();
            if (!first->stopping_.load() || !second->stopping_.load()
                || first->thread_.joinable() || second->thread_.joinable()
                || first->ensureStarted(enabledConfig(), nullptr)) {
                _exit(2);
            }
            auto late = makeMonitor(availableResult());
            if (late->ensureStarted(enabledConfig(), nullptr) || late->thread_.joinable()) {
                _exit(3);
            }
            late.reset();
            second.reset();
            first.reset();
            _exit(0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RdmaDeviceHealthMonitorLifecycleTest, DestroyedMonitorUnregistersBeforeFactoryShutdown) {
    ASSERT_EXIT(
        {
            {
                auto monitor = makeMonitor(availableResult());
                if (!monitor->ensureStarted(enabledConfig(), nullptr)) {
                    _exit(1);
                }
            }
            // 销毁后关闭不得再调用已释放 monitor 的 stop()。
            stopKmonitorFactory();
            // 第二次验证注销后重复关闭仍安全，不得残留访问已销毁对象的回调；生产只需关闭一次。
            stopKmonitorFactory();
            auto late = makeMonitor(availableResult());
            if (!late->stopping_.load() || late->ensureStarted(enabledConfig(), nullptr)) {
                _exit(2);
            }
            late.reset();
            _exit(0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

// ---------------------------------------------------------------------------------------------------------------------
// 配置校验。

void expectInvalid(const RdmaDeviceHealthConfig& config, const std::string& field) {
    std::string error;
    EXPECT_FALSE(validateConfig(config, nullptr, &error)) << field;
    EXPECT_NE(std::string::npos, error.find(field)) << error;
}

void expectValid(const RdmaDeviceHealthConfig& config) {
    std::string error;
    EXPECT_TRUE(validateConfig(config, nullptr, &error)) << error;
    EXPECT_TRUE(error.empty());
}

TEST(RdmaDeviceHealthValidateConfigTest, Defaults) {
    RdmaDeviceHealthConfig config;
    FaultHandler           handler = FaultHandler::ABORT;
    EXPECT_TRUE(validateConfig(config, &handler, nullptr));
    EXPECT_EQ(FaultHandler::LOG, handler);
    EXPECT_EQ(1000, config.probe_interval_ms);
    EXPECT_EQ(3000, config.device_fault_confirm_ms);
    EXPECT_EQ(600000, config.fault_window_ms);
    EXPECT_EQ(120000, config.unavailable_threshold_ms);
    EXPECT_EQ(10, config.flap_warn_threshold);
}

TEST(RdmaDeviceHealthValidateConfigTest, ProbeIntervalBoundaries) {
    RdmaDeviceHealthConfig config;
    config.probe_interval_ms = 99;
    expectInvalid(config, "probe_interval_ms=99 not in");
    config.probe_interval_ms = 100;
    expectValid(config);
    config.probe_interval_ms        = 60000;
    config.device_fault_confirm_ms  = 120000;
    config.unavailable_threshold_ms = 120000;
    expectValid(config);
    config.probe_interval_ms        = 60001;
    config.device_fault_confirm_ms  = 600000;
    config.unavailable_threshold_ms = 600000;
    expectInvalid(config, "probe_interval_ms=60001 not in");
}

TEST(RdmaDeviceHealthValidateConfigTest, DeviceFaultConfirmBoundaries) {
    RdmaDeviceHealthConfig config;
    config.probe_interval_ms       = 100;
    config.device_fault_confirm_ms = 999;
    expectInvalid(config, "device_fault_confirm_ms=999 not in");
    config.device_fault_confirm_ms = 1000;
    expectValid(config);
    config.device_fault_confirm_ms = 600000;
    expectValid(config);
    config.device_fault_confirm_ms = 600001;
    expectInvalid(config, "device_fault_confirm_ms=600001 not in");

    // device_fault_confirm_ms >= 2 * probe_interval_ms.
    config.probe_interval_ms       = 1000;
    config.device_fault_confirm_ms = 1999;
    expectInvalid(config, "device_fault_confirm_ms=1999 less than 2 * probe_interval_ms");
    config.device_fault_confirm_ms = 2000;
    expectValid(config);
}

TEST(RdmaDeviceHealthValidateConfigTest, FaultWindowBoundaries) {
    RdmaDeviceHealthConfig config;
    config.fault_window_ms = 9999;
    expectInvalid(config, "fault_window_ms=9999 not in");
    config.fault_window_ms          = 10000;
    config.unavailable_threshold_ms = 10000;
    expectValid(config);
    config.fault_window_ms = 3600000;
    expectValid(config);
    config.fault_window_ms = 3600001;
    expectInvalid(config, "fault_window_ms=3600001 not in");
}

TEST(RdmaDeviceHealthValidateConfigTest, UnavailableThresholdBoundaries) {
    RdmaDeviceHealthConfig config;
    config.probe_interval_ms        = 100;
    config.unavailable_threshold_ms = 999;
    expectInvalid(config, "unavailable_threshold_ms=999 not in");
    config.unavailable_threshold_ms = 1000;
    expectValid(config);
    config.fault_window_ms          = 3600000;
    config.unavailable_threshold_ms = 3600000;
    expectValid(config);
    config.unavailable_threshold_ms = 3600001;
    expectInvalid(config, "unavailable_threshold_ms=3600001 not in");

    // unavailable_threshold_ms >= 2 * probe_interval_ms.
    config.probe_interval_ms        = 1000;
    config.unavailable_threshold_ms = 1999;
    expectInvalid(config, "unavailable_threshold_ms=1999 less than 2 * probe_interval_ms");
    config.unavailable_threshold_ms = 2000;
    expectValid(config);
}

TEST(RdmaDeviceHealthValidateConfigTest, UnavailableThresholdNotAboveWindow) {
    RdmaDeviceHealthConfig config;
    config.fault_window_ms          = 60000;
    config.unavailable_threshold_ms = 60001;
    expectInvalid(config, "unavailable_threshold_ms=60001 greater than fault_window_ms=60000");
    config.unavailable_threshold_ms = 60000;
    expectValid(config);
}

TEST(RdmaDeviceHealthValidateConfigTest, FlapThresholdBoundaries) {
    RdmaDeviceHealthConfig config;
    config.flap_warn_threshold = 1;
    expectInvalid(config, "flap_warn_threshold");
    config.flap_warn_threshold = 2;
    expectValid(config);
    config.flap_warn_threshold = 1000;
    expectValid(config);
    config.flap_warn_threshold = 1001;
    expectInvalid(config, "flap_warn_threshold");
}

TEST(RdmaDeviceHealthValidateConfigTest, ReportsAllInvalidFields) {
    RdmaDeviceHealthConfig config;
    config.probe_interval_ms   = 99;
    config.flap_warn_threshold = 1;
    std::string error;
    EXPECT_FALSE(validateConfig(config, nullptr, &error));
    EXPECT_NE(std::string::npos, error.find("probe_interval_ms"));
    EXPECT_NE(std::string::npos, error.find("flap_warn_threshold"));
}

TEST(RdmaDeviceHealthValidateConfigTest, FaultHandlerParsing) {
    RdmaDeviceHealthConfig config;
    FaultHandler           handler = FaultHandler::ABORT;
    config.fault_handler           = "LOG";
    EXPECT_TRUE(validateConfig(config, &handler, nullptr));
    EXPECT_EQ(FaultHandler::LOG, handler);

    config.fault_handler = "ABORT";
    EXPECT_TRUE(validateConfig(config, &handler, nullptr));
    EXPECT_EQ(FaultHandler::ABORT, handler);

    for (const char* invalid : {"", "log", "EXIT"}) {
        config.fault_handler = invalid;
        handler              = FaultHandler::ABORT;
        std::string error;
        EXPECT_FALSE(validateConfig(config, &handler, &error)) << invalid;
        EXPECT_NE(std::string::npos, error.find("fault_handler")) << error;
        // 校验失败时不修改 handler。
        EXPECT_EQ(FaultHandler::ABORT, handler);
    }
}

}  // namespace
}  // namespace rdma_health
}  // namespace rtp_llm
