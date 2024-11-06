#include "gtest/gtest.h"

#include "autil/EnvUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/CommonDefine.h"
#include "maga_transformer/cpp/disaggregate/cache_store/Interface.h"
#include "aios/network/accl-barex/include/accl/barex/barex_types.h"

namespace rtp_llm {

class MemoryUtilTest: public ::testing::Test {
public:
    void SetUp() override;

protected:
    std::unique_ptr<MemoryUtil> memory_util_;
};

void MemoryUtilTest::SetUp() {
    memory_util_.reset(new MemoryUtil(createMemoryUtilImpl(autil::EnvUtil::getEnv(kEnvRdmaMode, false))));
}

TEST_F(MemoryUtilTest, testMemoryOps) {
    uint64_t buffer_len = 1024;

    auto cpu_buffer = memory_util_->mallocCPU(buffer_len);
    ASSERT_TRUE(cpu_buffer != nullptr);

    auto gpu_buffer = memory_util_->mallocGPU(buffer_len);
    ASSERT_TRUE(gpu_buffer != nullptr);

    ASSERT_FALSE(memory_util_->isMemoryMr(cpu_buffer, buffer_len, false, false));
    ASSERT_FALSE(memory_util_->isMemoryMr(gpu_buffer, buffer_len, true, false));

    // test memset cpu
    memory_util_->memsetCPU(cpu_buffer, '0', buffer_len);
    ASSERT_EQ('0', ((char*)(cpu_buffer))[0]);

    // test memset gpu and memcopy from gpu to cpu
    ASSERT_TRUE(memory_util_->memsetGPU(gpu_buffer, '1', buffer_len));
    ASSERT_TRUE(memory_util_->memcopy(cpu_buffer, false, gpu_buffer, true, buffer_len));
    ASSERT_EQ('1', ((char*)(cpu_buffer))[0]);

    // test memsetGPU with wrong ptr
    //ASSERT_FALSE(memory_util_->memsetGPU(cpu_buffer, 'a', buffer_len));

    // test memcopy from cpu to gpu
    memory_util_->memsetCPU(cpu_buffer, '2', buffer_len);
    ASSERT_TRUE(memory_util_->memcopy(gpu_buffer, true, cpu_buffer, false, buffer_len));
    auto cpu_buffer2 = memory_util_->mallocCPU(buffer_len);
    ASSERT_TRUE(memory_util_->memcopy(cpu_buffer2, false, gpu_buffer, true, buffer_len));
    ASSERT_EQ('2', ((char*)(cpu_buffer))[0]);

    // test memcopy wrong buffer
    //ASSERT_FALSE(memory_util_->memcopy(cpu_buffer, true, gpu_buffer, true, buffer_len));

    memory_util_->freeCPU(cpu_buffer);
    memory_util_->freeCPU(cpu_buffer2);
    memory_util_->freeGPU(gpu_buffer);
}

TEST_F(MemoryUtilTest, testMemoryMr_tcpMode) {
    if (memory_util_->rdmaMode()) {
        return;
    }

    uint64_t buffer_len = 1024;

    auto cpu_buffer = memory_util_->mallocCPU(buffer_len);
    ASSERT_TRUE(cpu_buffer != nullptr);

    auto gpu_buffer = memory_util_->mallocGPU(buffer_len);
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
    if (!memory_util_->rdmaMode()) {
        return;
    }

    uint64_t buffer_len = 1024;

    auto cpu_buffer = memory_util_->mallocCPU(buffer_len);
    ASSERT_TRUE(cpu_buffer != nullptr);

    auto gpu_buffer = memory_util_->mallocGPU(buffer_len);
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

}  // namespace rtp_llm