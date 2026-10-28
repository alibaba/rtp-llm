#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "aios/network/accl-barex/include/accl/barex/xdevice_health.h"
#include "rtp_llm/cpp/rdma_health/RdmaDeviceHealth.h"
#include "rtp_llm/cpp/rdma_health/RdmaDeviceHealthMetrics.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/RdmaDeviceHealthConfig.h"

namespace rtp_llm {
namespace rdma_health {

// CacheStore 和 P2PConnector 共享进程级 monitor，评估状态由 monitor 线程持有。
// 全部设备硬故障确认后使用 fault_handler；累计故障期时长只用于限频 WARN，不触发退出。
class RdmaDeviceHealthMonitor {
public:
    using HealthResult = accl::barex::XTransportDeviceHealthResult;
    using State        = accl::barex::XTransportDeviceHealthState;
    using ProbeFn      = std::function<HealthResult()>;
    using NowFn        = std::function<int64_t()>;
    using AbortFn      = std::function<void()>;

    static RdmaDeviceHealthMonitor& instance();
    ~RdmaDeviceHealthMonitor();

    RdmaDeviceHealthMonitor(const RdmaDeviceHealthMonitor&)            = delete;
    RdmaDeviceHealthMonitor& operator=(const RdmaDeviceHealthMonitor&) = delete;

    bool ensureStarted(const RdmaDeviceHealthConfig& config, const kmonitor::MetricsReporterPtr& reporter);
    // 可重复调用；本对象一旦 stop() 就不能重启，即使 factory 随后重新初始化。
    void stop();

private:
    enum class EventReason {
        NONE,
        // 全设备硬故障的 UNAVAILABLE 轮，保留的确认计时达到 device_fault_confirm_ms。
        // 无健康设备证据的 UNKNOWN 轮保留起点且经过时间继续计入，但当轮不确认。
        DEVICE_FAULT_CONFIRMED,
        // UNAVAILABLE 轮累计故障期时长达到阈值，且当轮未选择设备确认；不确认设备故障、不退出。
        UNAVAILABLE_THRESHOLD_EXCEEDED,
    };

    // 总体故障期从首次 UNAVAILABLE 开始，到下次 AVAILABLE 结束；中间 DEGRADED/UNKNOWN 不关闭故障期。
    struct FaultPeriod {
        int64_t begin_ms;
        int64_t end_ms;
    };

    // 日志计数用于 UT 验证限频；config_differs 在 start_mutex_ 下更新，其余由 monitor 线程更新。
    struct LogCounts {
        uint64_t unknown_warn       = 0;
        uint64_t degraded_warn      = 0;
        uint64_t fault_begin        = 0;
        uint64_t device_fault_begin = 0;
        uint64_t recovered          = 0;
        uint64_t flapping           = 0;
        uint64_t unavailable_warn   = 0;
        uint64_t fatal              = 0;
        uint64_t not_supported      = 0;
        uint64_t config_differs     = 0;
    };

    RdmaDeviceHealthMonitor(ProbeFn probe, NowFn now, AbortFn abort);

    void run();
    // BareX 后端不支持健康查询时返回 false，通知 monitor 线程结束；不退出进程。
    bool process(const HealthResult& r, int64_t now_ms);
    // 更新故障期及抖动统计并选择事件；设备确认处置和累计不可用 WARN 由 process() 分派执行。
    EventReason                      evaluate(const HealthResult& r, int64_t now_ms);
    void                             onRecovered(int64_t now_ms);
    void                             warnUnavailable(const HealthResult& r, int64_t now_ms);
    void                             handleFatal(const HealthResult& r, int64_t now_ms);
    bool                             isDeviceFault(const HealthResult& r) const;
    void                             pruneFaultPeriods(int64_t now_ms);
    void                             reportMetrics(const HealthResult& r, int64_t now_ms);
    RdmaDeviceHealthMetricsCollector makeMetricsCollector(const HealthResult& r, int64_t now_ms) const;
    int64_t                          recoverCountInWindow(int64_t now_ms) const;
    int64_t                          unavailableMsInWindow(int64_t now_ms) const;
    void                             warnIfConfigDiffers(const RdmaDeviceHealthConfig& config);

private:
    ProbeFn probe_;
    NowFn   now_;
    AbortFn abort_;

    RdmaDeviceHealthConfig config_;
    FaultHandler           fault_handler_ = FaultHandler::LOG;
    // monitor 不跨轮持有 reporter 强引用；上报中成功取得的强引用仍可能延长 reporter 生命周期。
    // weak_ptr 不保护 factory 内部资源；stopKmonitorFactory() 通过停止回调先 stop/join，再释放指标资源。
    std::weak_ptr<kmonitor::MetricsReporter> reporter_;

    // 当前总体故障期起点；没有未关闭的故障期时为 -1。
    int64_t fault_since_ms_ = -1;
    // 设备硬故障确认起点，无计时时为 -1；UNKNOWN 无可信设备健康证据时保留起点，经过时间继续计入。
    int64_t device_fault_since_ms_    = -1;
    // 上次抖动/累计不可用告警时间；初值表示从未告警，使首次满足条件即可告警。
    // 取 INT64_MIN / 2 而非 INT64_MIN，保证 now_ms - last 不溢出。
    int64_t last_flap_warn_ms_        = INT64_MIN / 2;
    int64_t last_unavailable_warn_ms_ = INT64_MIN / 2;
    // 结束时间仍在 fault_window_ms 内的已关闭故障期，按时间从早到晚排列。
    std::deque<FaultPeriod> fault_periods_;
    uint64_t                unknown_count_         = 0;
    uint64_t                degraded_count_        = 0;
    int64_t                 last_probe_latency_us_ = 0;
    LogCounts               log_counts_;

    std::atomic<bool> started_{false};
    std::atomic<bool> stopping_{false};
    // 仅在 AVAILABLE 关闭当前总体故障期时复位，用于同故障期内的确认去重。
    std::atomic<bool> fatal_handled_in_current_period_{false};
    // monitor 生命周期内保持粘性；用于 confirmed_fault 指标，并防止 ABORT 重复处置。
    std::atomic<bool> ever_confirmed_fault_{false};

    std::mutex              start_mutex_;
    std::mutex              wait_mutex_;
    std::condition_variable wait_cv_;
    std::thread             thread_;
};

}  // namespace rdma_health
}  // namespace rtp_llm
