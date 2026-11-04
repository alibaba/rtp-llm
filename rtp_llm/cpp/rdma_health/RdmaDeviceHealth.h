#pragma once

#include <string>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/RdmaDeviceHealthConfig.h"

namespace rtp_llm {
namespace rdma_health {

// 仅控制已确认的全设备硬故障处理；累计故障期阈值只产生限频 WARN，不触发退出。
// 累计 WARN 仅在 UNAVAILABLE 轮评估，设备确认优先于同轮 WARN。
enum class FaultHandler : int {
    LOG   = 0,
    ABORT = 1,
};

// 校验取值范围并解析 fault_handler；失败返回 false，并通过 error 列出无效字段。
// handler 和 error 可为 nullptr；失败时不修改 handler。
bool validateConfig(const RdmaDeviceHealthConfig& config, FaultHandler* handler, std::string* error);

// 尝试启动进程级 RDMA 健康 monitor，或复用此前的成功启动。
// 禁用、配置无效、已停止、关闭期间注册被拒绝或线程创建失败时返回 false。
// 返回 true 不保证线程仍运行：BareX 后端不支持健康查询时，首轮探测后会退出 monitor 线程。
// 调用方不得因返回 false 改变已成功的 RDMA 初始化结果；此接口不承诺所有异常均被兜底。
//
// 仅在 BareX RDMA 初始化成功后调用，使 monitor 单例晚于设备 manager 构造、早于其析构。
bool ensureMonitorStarted(const RdmaDeviceHealthConfig& config, const kmonitor::MetricsReporterPtr& reporter);

}  // namespace rdma_health
}  // namespace rtp_llm
