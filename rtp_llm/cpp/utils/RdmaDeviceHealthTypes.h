#pragma once

#include <stdint.h>

// RDMA 设备健康探测的跨层类型契约：枚举、边界/默认常量与纯数据结构。
// 探测实现位于仓外的 RDMA messager，本仓只负责把配置从 CLI/env 透传到
// MessagerInitParams；配置的最终合法性由消费方（messager 侧 monitor 启动时）校验。
// 该头是 struct-only leaf，配置层与传输层引用同一份定义，避免跨层复制产生漂移。
namespace rtp_llm {

enum class RdmaDeviceHealthFaultHandler : uint8_t {
    LOG   = 0,
    ABORT = 1,
};

inline const char* rdmaDeviceHealthFaultHandlerName(RdmaDeviceHealthFaultHandler handler) {
    switch (handler) {
        case RdmaDeviceHealthFaultHandler::LOG:
            return "LOG";
        case RdmaDeviceHealthFaultHandler::ABORT:
            return "ABORT";
    }
    return "UNKNOWN";
}

constexpr uint32_t kMinRdmaDeviceHealthProbeIntervalMs     = 100;
constexpr uint32_t kMaxRdmaDeviceHealthProbeIntervalMs     = 60000;
constexpr uint32_t kDefaultRdmaDeviceHealthProbeIntervalMs = 1000;

constexpr uint32_t kMinRdmaDeviceHealthFaultConfirmCount     = 1;
constexpr uint32_t kMaxRdmaDeviceHealthFaultConfirmCount     = 100;
constexpr uint32_t kDefaultRdmaDeviceHealthFaultConfirmCount = 3;

struct RdmaDeviceHealthMonitorConfig {
    bool                         enabled{false};
    RdmaDeviceHealthFaultHandler fault_handler{RdmaDeviceHealthFaultHandler::LOG};
    uint32_t                     probe_interval_ms{kDefaultRdmaDeviceHealthProbeIntervalMs};
    uint32_t                     fault_confirm_count{kDefaultRdmaDeviceHealthFaultConfirmCount};
};

}  // namespace rtp_llm
