#include "aios/network/rdma/accl/RdmaMempoolConfig.h"

#include "autil/EnvUtil.h"
#include "unittest/unittest.h"

using namespace ::testing;

ARPC_BEGIN_NAMESPACE(arpc)

TEST(RdmaMempoolConfigTest, testDefault) {
    RdmaMempoolConfig config;
    EXPECT_TRUE(config.enableAutoShrink);
    EXPECT_FLOAT_EQ(float(0.05), config.shrinkFreeRatio);
    EXPECT_FLOAT_EQ(float(1), config.shrinkFreeToBusy);
    EXPECT_EQ(5 * 60, config.shrinkIntervalSec);
    EXPECT_EQ(10 * 60, config.shrinkStartWaitTimeSec);
    EXPECT_TRUE(config.cpuReserves.empty());
    EXPECT_TRUE(config.gpuReserves.empty());
    EXPECT_EQ(0, config.cpuMaxTotalBytes);
    EXPECT_EQ(8589934592, config.gpuMaxTotalBytes);
    EXPECT_FALSE(config.enableAsyncAlloc);
    EXPECT_EQ(1, config.asyncAllocThreadCount);
    EXPECT_EQ(10, config.asyncAllocQueueDepth);

    EXPECT_EQ("{\"async_alloc_queue_depth\":10,\"async_alloc_thread_count\":1,\"cpu_max_total_bytes\":0,"
              "\"enable_async_alloc\":false,\"enable_auto_shrink\":true,\"gpu_max_total_bytes\":8589934592,"
              "\"reserve_cpu_mem\":[],\"reserve_gpu_mem\":[],\"shrink_free_ratio\":0.05,"
              "\"shrink_free_to_busy_ratio\":1,\"shrink_interval_sec\":300,\"shrink_start_wait_time_sec\":600}",
              autil::legacy::ToJsonString(config, true));
}

TEST(RdmaMempoolConfigTest, testFromEnv) {
    {
        autil::EnvGuard   guard("ARPC_RDMA_MEMPOOL_ENABLE_AUTO_SHRINK", "0");
        RdmaMempoolConfig config;
        EXPECT_FALSE(config.enableAutoShrink);
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_MEMPOOL_ENABLE_AUTO_SHRINK", "1");
        RdmaMempoolConfig config;
        EXPECT_TRUE(config.enableAutoShrink);
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_SHRINK_RATIO", "0.1");
        RdmaMempoolConfig config;
        EXPECT_FLOAT_EQ(float(0.1), config.shrinkFreeRatio);
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_SHRINK_FREE_TO_BUSY_RATIO", "1.5");
        RdmaMempoolConfig config;
        EXPECT_FLOAT_EQ(float(1.5), config.shrinkFreeToBusy);
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_SHRINK_INTERVAL_SEC", "1024");
        RdmaMempoolConfig config;
        EXPECT_EQ(1024, config.shrinkIntervalSec);
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_SHRINK_START_WAIT_TIME_SEC", "2048");
        RdmaMempoolConfig config;
        EXPECT_EQ(2048, config.shrinkStartWaitTimeSec);
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_RESERVE_CPU_MEM", "1:2;3:4;5;:6;;");
        RdmaMempoolConfig config;
        EXPECT_EQ(2, config.cpuReserves.size());
        EXPECT_EQ(2, config.cpuReserves.at(1));
        EXPECT_EQ(4, config.cpuReserves.at(3));
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_RESERVE_GPU_MEM", "1:2;3:4;5;:6;;");
        RdmaMempoolConfig config;
        EXPECT_EQ(2, config.gpuReserves.size());
        EXPECT_EQ(2, config.gpuReserves.at(1));
        EXPECT_EQ(4, config.gpuReserves.at(3));
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_MEMPOOL_CPU_MAX_TOTAL_BYTES", "1023");
        RdmaMempoolConfig config;
        EXPECT_EQ(1023, config.cpuMaxTotalBytes);
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_MEMPOOL_GPU_MAX_TOTAL_BYTES", "1023");
        RdmaMempoolConfig config;
        EXPECT_EQ(1023, config.gpuMaxTotalBytes);
    }

    {
        autil::EnvGuard   guard("ARPC_RDMA_MEMPOOL_ENABLE_ASYNC_ALLOC", "0");
        RdmaMempoolConfig config;
        EXPECT_FALSE(config.enableAsyncAlloc);
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_MEMPOOL_ENABLE_ASYNC_ALLOC", "1");
        RdmaMempoolConfig config;
        EXPECT_TRUE(config.enableAsyncAlloc);
    }

    {
        autil::EnvGuard   guard("ARPC_RDMA_MEMPOOL_ASYNC_ALLOC_THREAD_COUNT", "1234");
        RdmaMempoolConfig config;
        EXPECT_EQ(1234, config.asyncAllocThreadCount);
    }
    {
        autil::EnvGuard   guard("ARPC_RDMA_MEMPOOL_ASYNC_ALLOC_QUEUE_DEPTH", "5678");
        RdmaMempoolConfig config;
        EXPECT_EQ(5678, config.asyncAllocQueueDepth);
    }
}

ARPC_END_NAMESPACE(arpc)