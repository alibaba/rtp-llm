#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"
#include "autil/Log.h"
#include "autil/legacy/jsonizable.h"

ARPC_BEGIN_NAMESPACE(arpc)

class RdmaMempoolConfig: public autil::legacy::Jsonizable {
public:
    RdmaMempoolConfig();
    ~RdmaMempoolConfig() = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

private:
    std::map<uint64_t, int> ConvertToReserveMap(const std::string& str);

public:
    bool                    enableAutoShrink{true};
    float                   shrinkFreeRatio{0.05};
    float                   shrinkFreeToBusy{1};
    uint64_t                shrinkIntervalSec{5 * 60};
    uint64_t                shrinkStartWaitTimeSec{10 * 60};
    std::map<uint64_t, int> cpuReserves;
    std::map<uint64_t, int> gpuReserves;
    uint64_t                cpuMaxTotalBytes{0};
    uint64_t                gpuMaxTotalBytes{8589934592};  // 默认设置limit gpu 8GB
    bool                    enableAsyncAlloc{false};       // 启用异步申请
    uint32_t                asyncAllocThreadCount{1};      // 异步申请线程池中的线程个数
    uint32_t                asyncAllocQueueDepth{10};      // 异步申请线程池的队列深度

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc)