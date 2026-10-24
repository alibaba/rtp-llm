#pragma once

#include <cstdint>
#include <string>

namespace rtp_llm {

// RDMA 健康 monitor 配置；以下取值范围同时用于服务参数校验和 RDMA 实现的配置校验。
struct RdmaDeviceHealthConfig {
    static constexpr int64_t kMinProbeIntervalMs        = 100;
    static constexpr int64_t kMaxProbeIntervalMs        = 60000;
    static constexpr int64_t kMinDeviceFaultConfirmMs   = 1000;
    static constexpr int64_t kMaxDeviceFaultConfirmMs   = 600000;
    static constexpr int64_t kMinFaultWindowMs          = 10000;
    static constexpr int64_t kMaxFaultWindowMs          = 3600000;
    static constexpr int64_t kMinUnavailableThresholdMs = 1000;
    static constexpr int64_t kMaxUnavailableThresholdMs = 3600000;
    static constexpr int64_t kMinFlapWarnThreshold      = 2;
    static constexpr int64_t kMaxFlapWarnThreshold      = 1000;
    // device_fault_confirm_ms 和 unavailable_threshold_ms 均须至少覆盖此数量的探测间隔。
    static constexpr int64_t kMinConfirmProbeIntervals = 2;

    bool enabled = false;
    // LOG 按总体故障期处理一次设备确认；ABORT 在 monitor 生命周期内只处理首次确认。
    // 累计故障期 WARN 不受此设置控制；同轮设备确认优先，后续 UNAVAILABLE 轮仍可告警。
    std::string fault_handler     = "LOG";
    int64_t     probe_interval_ms = 1000;
    // 全设备移除或硬查询错误的 UNAVAILABLE 轮启动确认计时；再次出现该状态且计时达到阈值才确认。
    // UNKNOWN 仅在查询成功且有明确 device HEALTHY 证据时清除计时；否则保留起点并计入经过时间。
    // UNKNOWN 当轮不确认设备故障。
    // AVAILABLE、DEGRADED 或非全设备硬故障的 UNAVAILABLE 也清除确认计时。
    int64_t device_fault_confirm_ms = 3000;
    // 累计总体故障期时长与抖动告警共用的滑动窗口；故障期从首次 UNAVAILABLE 到下次 AVAILABLE。
    int64_t fault_window_ms = 600000;
    // 窗口内总体故障期累计阈值，包含中间 UNKNOWN/DEGRADED；仅 UNAVAILABLE 轮评估，只 WARN、不退出。
    // 设备确认优先于同轮 WARN；告警至少相隔 fault_window_ms，恢复不重置限频，阈值不得超过窗口。
    int64_t unavailable_threshold_ms = 120000;
    // AVAILABLE 关闭故障期时，窗口内恢复次数达到此阈值且距上次抖动 WARN 至少一个窗口才告警。
    int64_t flap_warn_threshold = 10;
};

}  // namespace rtp_llm
