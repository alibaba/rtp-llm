#include "aios/network/rdma/accl/RdmaMempoolConfig.h"

#include "autil/EnvUtil.h"

ARPC_BEGIN_NAMESPACE(arpc)

const std::string kEnableAutoShrink          = "ARPC_RDMA_MEMPOOL_ENABLE_AUTO_SHRINK";
const std::string kEnvShrinkFreeRatio        = "ARPC_RDMA_SHRINK_RATIO";
const std::string kEnvShrinkFreeToBusyRatio  = "ARPC_RDMA_SHRINK_FREE_TO_BUSY_RATIO";
const std::string kEnvShrinkIntervalSec      = "ARPC_RDMA_SHRINK_INTERVAL_SEC";
const std::string kEnvShrinkStartWaitTimeSec = "ARPC_RDMA_SHRINK_START_WAIT_TIME_SEC";
const std::string kEnvInitCPUMem             = "ARPC_RDMA_RESERVE_CPU_MEM";
const std::string kEnvInitGPUMem             = "ARPC_RDMA_RESERVE_GPU_MEM";
const std::string kEnvCPUMaxTotalBytes       = "ARPC_RDMA_MEMPOOL_CPU_MAX_TOTAL_BYTES";
const std::string kEnvGPUMaxTotalBytes       = "ARPC_RDMA_MEMPOOL_GPU_MAX_TOTAL_BYTES";
const std::string kEnvEnableAsyncAlloc       = "ARPC_RDMA_MEMPOOL_ENABLE_ASYNC_ALLOC";
const std::string kEnvAsyncAllocThreadCount  = "ARPC_RDMA_MEMPOOL_ASYNC_ALLOC_THREAD_COUNT";
const std::string kEnvAsyncAllocQueueDepth   = "ARPC_RDMA_MEMPOOL_ASYNC_ALLOC_QUEUE_DEPTH";

AUTIL_LOG_SETUP(arpc, RdmaMempoolConfig);

RdmaMempoolConfig::RdmaMempoolConfig() {
    enableAutoShrink       = autil::EnvUtil::getEnv(kEnableAutoShrink, enableAutoShrink);
    shrinkFreeRatio        = autil::EnvUtil::getEnv(kEnvShrinkFreeRatio, shrinkFreeRatio);
    shrinkFreeToBusy       = autil::EnvUtil::getEnv(kEnvShrinkFreeToBusyRatio, shrinkFreeToBusy);
    shrinkIntervalSec      = autil::EnvUtil::getEnv(kEnvShrinkIntervalSec, shrinkIntervalSec);
    shrinkStartWaitTimeSec = autil::EnvUtil::getEnv(kEnvShrinkStartWaitTimeSec, shrinkStartWaitTimeSec);

    std::string strCPUReserves = autil::EnvUtil::getEnv(kEnvInitCPUMem);
    cpuReserves                = ConvertToReserveMap(strCPUReserves);

    std::string strGPUReserves = autil::EnvUtil::getEnv(kEnvInitGPUMem);
    gpuReserves                = ConvertToReserveMap(strGPUReserves);

    cpuMaxTotalBytes = autil::EnvUtil::getEnv(kEnvCPUMaxTotalBytes, cpuMaxTotalBytes);
    gpuMaxTotalBytes = autil::EnvUtil::getEnv(kEnvGPUMaxTotalBytes, gpuMaxTotalBytes);

    enableAsyncAlloc      = autil::EnvUtil::getEnv(kEnvEnableAsyncAlloc, enableAsyncAlloc);
    asyncAllocThreadCount = autil::EnvUtil::getEnv(kEnvAsyncAllocThreadCount, asyncAllocThreadCount);
    asyncAllocQueueDepth  = autil::EnvUtil::getEnv(kEnvAsyncAllocQueueDepth, asyncAllocQueueDepth);
}

void RdmaMempoolConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) {
    json.Jsonize("enable_auto_shrink", enableAutoShrink);
    json.Jsonize("shrink_free_ratio", shrinkFreeRatio);
    json.Jsonize("shrink_free_to_busy_ratio", shrinkFreeToBusy);
    json.Jsonize("shrink_interval_sec", shrinkIntervalSec);
    json.Jsonize("shrink_start_wait_time_sec", shrinkStartWaitTimeSec);
    json.Jsonize("reserve_cpu_mem", cpuReserves);
    json.Jsonize("reserve_gpu_mem", gpuReserves);
    json.Jsonize("cpu_max_total_bytes", cpuMaxTotalBytes);
    json.Jsonize("gpu_max_total_bytes", gpuMaxTotalBytes);
    json.Jsonize("enable_async_alloc", enableAsyncAlloc);
    json.Jsonize("async_alloc_thread_count", asyncAllocThreadCount);
    json.Jsonize("async_alloc_queue_depth", asyncAllocQueueDepth);
}

std::map<uint64_t, int> RdmaMempoolConfig::ConvertToReserveMap(const std::string& strValue) {
    std::map<uint64_t, int> reserves;
    if (strValue.empty()) {
        return reserves;
    }

    auto kvs = autil::StringUtil::split(strValue, ";");
    for (auto& kv : kvs) {
        auto pos = kv.find(":");
        if (pos == std::string::npos) {
            continue;
        }

        auto key   = kv.substr(0, pos);
        auto value = kv.substr(pos + 1);
        if (key.empty() || value.empty()) {
            continue;
        }

        uint64_t size;
        int64_t  count;
        if (!autil::StringUtil::strToUInt64(key.c_str(), size)
            || !autil::StringUtil::strToInt64(value.c_str(), count)) {
            AUTIL_LOG(WARN, "parse reserve config failed, kv %s", kv.c_str());
            continue;
        }
        reserves[size] = count;
    }
    return reserves;
}

ARPC_END_NAMESPACE(arpc)