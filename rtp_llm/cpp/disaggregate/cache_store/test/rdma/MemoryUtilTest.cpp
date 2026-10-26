#include "gtest/gtest.h"

#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "aios/network/accl-barex/include/accl/barex/barex_types.h"
#include "aios/network/rdma/accl/unittest/mock/MockRdmaMempool.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaMemoryUtilImpl.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace rtp_llm {

class MemoryUtilTest: public CacheStoreTestBase {};

TEST_F(MemoryUtilTest, testMemoryOps) {
    uint64_t buffer_len = 1024;

    auto cpu_buffer = device_util_->mallocCPU(buffer_len);
    ASSERT_TRUE(cpu_buffer != nullptr);

    auto gpu_buffer = device_util_->mallocGPU(buffer_len);
    ASSERT_TRUE(gpu_buffer != nullptr);

    ASSERT_FALSE(memory_util_->isMemoryMr(cpu_buffer, buffer_len, false, false));
    ASSERT_FALSE(memory_util_->isMemoryMr(gpu_buffer, buffer_len, true, false));

    // test memset cpu
    device_util_->memsetCPU(cpu_buffer, '0', buffer_len);
    ASSERT_EQ('0', ((char*)(cpu_buffer))[0]);

    // test memset gpu and memcopy from gpu to cpu
    ASSERT_TRUE(device_util_->memsetGPU(gpu_buffer, '1', buffer_len));
    ASSERT_TRUE(device_util_->memcopy(cpu_buffer, false, gpu_buffer, true, buffer_len));
    ASSERT_EQ('1', ((char*)(cpu_buffer))[0]);

    // test memsetGPU with wrong ptr
    // ASSERT_FALSE(memory_util_->memsetGPU(cpu_buffer, 'a', buffer_len));

    // test memcopy from cpu to gpu
    device_util_->memsetCPU(cpu_buffer, '2', buffer_len);
    ASSERT_TRUE(device_util_->memcopy(gpu_buffer, true, cpu_buffer, false, buffer_len));
    auto cpu_buffer2 = device_util_->mallocCPU(buffer_len);
    ASSERT_TRUE(device_util_->memcopy(cpu_buffer2, false, gpu_buffer, true, buffer_len));
    ASSERT_EQ('2', ((char*)(cpu_buffer))[0]);

    // test memcopy wrong buffer
    // ASSERT_FALSE(device_util_->memcopy(cpu_buffer, true, gpu_buffer, true, buffer_len));

    device_util_->freeCPU(cpu_buffer);
    device_util_->freeCPU(cpu_buffer2);
    device_util_->freeGPU(gpu_buffer);
}

TEST_F(MemoryUtilTest, testMemoryMr_tcpMode) {
    if (memory_util_->isRdmaMode()) {
        return;
    }

    uint64_t buffer_len = 1024;

    auto cpu_buffer = device_util_->mallocCPU(buffer_len);
    ASSERT_TRUE(cpu_buffer != nullptr);

    auto gpu_buffer = device_util_->mallocGPU(buffer_len);
    ASSERT_TRUE(gpu_buffer != nullptr);

    ASSERT_FALSE(memory_util_->isMemoryMr(cpu_buffer, buffer_len, false, true));
    ASSERT_FALSE(memory_util_->isMemoryMr(gpu_buffer, buffer_len, true, true));

    ASSERT_TRUE(memory_util_->regUserMr(cpu_buffer, buffer_len, false));
    ASSERT_TRUE(memory_util_->regUserMr(gpu_buffer, buffer_len, true));

    ASSERT_FALSE(memory_util_->isMemoryMr(cpu_buffer, buffer_len, false, true));
    ASSERT_FALSE(memory_util_->isMemoryMr(gpu_buffer, buffer_len, true, true));

    ::accl::barex::memp_t mem;
    ASSERT_FALSE(memory_util_->findMemoryMr(&mem, cpu_buffer, buffer_len, false, true));
    ASSERT_FALSE(memory_util_->findMemoryMr(&mem, gpu_buffer, buffer_len, true, true));

    ASSERT_TRUE(memory_util_->deregUserMr(cpu_buffer, false));
    ASSERT_TRUE(memory_util_->deregUserMr(gpu_buffer, true));
}

TEST_F(MemoryUtilTest, testMemoryMr_rdmaMode) {
    if (!memory_util_->isRdmaMode()) {
        return;
    }

    uint64_t buffer_len = 1024;

    auto cpu_buffer = device_util_->mallocCPU(buffer_len);
    ASSERT_TRUE(cpu_buffer != nullptr);

    auto gpu_buffer = device_util_->mallocGPU(buffer_len);
    ASSERT_TRUE(gpu_buffer != nullptr);

    ASSERT_FALSE(memory_util_->isMemoryMr(cpu_buffer, buffer_len, false, true));
    ASSERT_FALSE(memory_util_->isMemoryMr(gpu_buffer, buffer_len, true, true));

    ASSERT_TRUE(memory_util_->regUserMr(cpu_buffer, buffer_len, false));
    ASSERT_TRUE(memory_util_->regUserMr(gpu_buffer, buffer_len, true));

    ASSERT_TRUE(memory_util_->isMemoryMr(cpu_buffer, buffer_len, false, true));
    ASSERT_TRUE(memory_util_->isMemoryMr(gpu_buffer, buffer_len, true, true));

    ::accl::barex::memp_t mem;
    ASSERT_TRUE(memory_util_->findMemoryMr(&mem, cpu_buffer, buffer_len, false, true));
    ASSERT_TRUE(memory_util_->findMemoryMr(&mem, gpu_buffer, buffer_len, true, true));

    ASSERT_TRUE(memory_util_->deregUserMr(cpu_buffer, false));
    ASSERT_TRUE(memory_util_->deregUserMr(gpu_buffer, true));

    ASSERT_FALSE(memory_util_->isMemoryMr(cpu_buffer, buffer_len, false, true));
    ASSERT_FALSE(memory_util_->isMemoryMr(gpu_buffer, buffer_len, true, true));
}

TEST_F(MemoryUtilTest, testMemoryMr_Partial) {
    if (!memory_util_->isRdmaMode()) {
        return;
    }
    autil::EnvGuard env_guard("CACHE_STORE_MAX_REG_MEM_SIZE", "2048");

    auto memory_util = createMemoryUtilImpl(autil::EnvUtil::getEnv(kEnvRdmaMode, false));

    uint64_t buffer_len = 1 * 1024 * 1024;  // 1MB

    auto gpu_buffer1 = device_util_->mallocGPU(buffer_len);
    ASSERT_TRUE(gpu_buffer1 != nullptr);

    auto gpu_buffer2 = device_util_->mallocGPU(buffer_len);
    ASSERT_TRUE(gpu_buffer2 != nullptr);

    ASSERT_FALSE(memory_util->regUserMr(gpu_buffer1, buffer_len, true, 333));
    ASSERT_FALSE(memory_util->regUserMr(gpu_buffer1, buffer_len, true, 4096));
    ASSERT_FALSE(memory_util->regUserMr(gpu_buffer1, 123, true, 1024));

    ASSERT_TRUE(memory_util->regUserMr(gpu_buffer2, buffer_len, true, 1024));
    ASSERT_TRUE(memory_util->regUserMr(gpu_buffer1, buffer_len, true, 2048));

    auto convert_void = [](void* base, int64_t len) -> void* { return (void*)((int64_t)(base) + len); };

    for (int64_t i = 0; i < 1024; i++) {
        ASSERT_TRUE(memory_util->isMemoryMr(convert_void(gpu_buffer1, i * 1024), 1024, true, true));
        ASSERT_TRUE(memory_util->isMemoryMr(convert_void(gpu_buffer2, i * 1024), 1024, true, true));
    }

    ASSERT_TRUE(memory_util->deregUserMr(gpu_buffer2, true));

    ASSERT_FALSE(memory_util->isMemoryMr(convert_void(gpu_buffer1, 1024 * 1024), 1024, true, true));
    ASSERT_FALSE(memory_util->isMemoryMr(convert_void(gpu_buffer1, -1), 1024, true, true));
    ASSERT_FALSE(memory_util->isMemoryMr(gpu_buffer2, 1024, true, true));

    ASSERT_TRUE(memory_util->deregUserMr(gpu_buffer1, true));
}

TEST_F(MemoryUtilTest, testMemoryMr_aligned) {
    auto                               mock_rdma_mempool = new arpc::MockRdmaMempool();
    std::shared_ptr<arpc::RdmaMempool> rdma_mempool(mock_rdma_mempool);
    auto                               memory_util = createMemoryUtilImpl(autil::EnvUtil::getEnv(kEnvRdmaMode, false));
    auto                               rdma_memory_util = std::dynamic_pointer_cast<RdmaMemoryUtilImpl>(memory_util);
    rdma_memory_util->rdma_mempool_                     = rdma_mempool;

    uint64_t nic_allowed_size = 64L * 1024 * 1024;
    ON_CALL(*mock_rdma_mempool, RegUserMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(
            Invoke([nic_allowed_size](::accl::barex::memp_t& mem, void* buf, uint64_t size, bool gpu) -> bool {
                if (size <= nic_allowed_size) {
                    return true;
                }
                return false;
            }));
    EXPECT_CALL(*mock_rdma_mempool, RegUserMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(::testing::AnyNumber());

    uint64_t buffer_len   = 14085881856L;
    uint64_t aligned_size = 294912;
    void*    buf          = (void*)0x43f5260cc0;

    ASSERT_TRUE(memory_util->regUserMr(buf, buffer_len, true, aligned_size));
}

}  // namespace rtp_llm
