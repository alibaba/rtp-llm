#include "gtest/gtest.h"

#include "autil/EnvUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/test/DeviceUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/CommonDefine.h"
#include "maga_transformer/cpp/disaggregate/cache_store/Interface.h"
#include "aios/network/accl-barex/include/accl/barex/barex_types.h"

namespace rtp_llm {

class MemoryUtilTest: public ::testing::Test {
public:
    void SetUp() override;

protected:
    std::unique_ptr<MemoryUtil> memory_util_;
    std::unique_ptr<DeviceUtil> device_util_;
};

void MemoryUtilTest::SetUp() {
    memory_util_ = createMemoryUtilImpl(autil::EnvUtil::getEnv(kEnvRdmaMode, false));
    device_util_ = std::make_unique<DeviceUtil>();
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

}  // namespace rtp_llm