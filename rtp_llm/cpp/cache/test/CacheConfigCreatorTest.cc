
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class CacheConfigCreatorTest: public DeviceTestBase {
    void SetUp() override {
        device_reserve_memory_size_ = 2092L * 1024 * 1024;
        DeviceTestBase::SetUp();
    }

    void TearDown() override {
        DeviceTestBase::TearDown();
    }
};

TEST_F(CacheConfigCreatorTest, testGetKVCacheMemorySize) {
    GptInitParameter param;
    param.reserve_runtime_mem_mb_ = 2;
    param.kv_cache_mem_mb_        = 10;
    CacheConfigCreator creator;
    auto               result1 = creator.getKVCacheMemorySize(param);
    ASSERT_EQ(10 * 1024 * 1024, result1);

    param.kv_cache_mem_mb_ = 0;
    auto result2           = creator.getKVCacheMemorySize(param);
    ASSERT_TRUE(result2 > 0);

    param.reserve_runtime_mem_mb_ = 200000;
    std::string exception         = "";
    try {
        creator.getKVCacheMemorySize(param);
    } catch (const std::exception& e) {
        exception = e.what();
        printf("exception: %s", e.what());
    }
    ASSERT_STREQ("[ERROR] device reserved memory", exception.substr(0, 30).c_str());
}

TEST_F(CacheConfigCreatorTest, testCreateConfig) {
    GptInitParameter param;
    param.block_nums_         = 200;
    param.num_layers_         = 1;
    param.head_num_kv_        = 4;
    param.size_per_head_      = 128;
    param.seq_size_per_block_ = 8;
    CacheConfigCreator creator;
    auto               result1 = creator.createConfig(param);
    ASSERT_EQ(result1.block_nums, 200);
    ASSERT_EQ(result1.local_head_num_kv, 4);

    param.block_nums_ = 0;
    auto result2      = creator.createConfig(param);
    ASSERT_TRUE(result2.block_nums > 0);
    ASSERT_EQ(result2.local_head_num_kv, 4);

    param.kv_cache_mem_mb_ = 32;
    param.head_num_kv_     = 1024;
    auto result3           = creator.createConfig(param);
    ASSERT_EQ(result3.block_nums, 8);
}

}  // namespace rtp_llm
