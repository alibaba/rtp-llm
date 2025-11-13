
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

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
    RuntimeConfig runtime_config;
    runtime_config.reserve_runtime_mem_mb = 2;
    KVCacheConfig kv_cache_config;
    kv_cache_config.kv_cache_mem_mb = 10;
    ModelConfig model_config;
    ParallelismConfig parallelism_config;

    CacheConfigCreator creator;
    auto               result1 = creator.getKVCacheMemorySize(runtime_config, kv_cache_config, model_config, parallelism_config);
    ASSERT_EQ(10 * 1024 * 1024, result1);

    kv_cache_config.kv_cache_mem_mb = 0;
    auto result2           = creator.getKVCacheMemorySize(runtime_config, kv_cache_config, model_config, parallelism_config);
    ASSERT_TRUE(result2 > 0);

    runtime_config.reserve_runtime_mem_mb = 200000;
    std::string exception         = "";
    try {
        creator.getKVCacheMemorySize(runtime_config, kv_cache_config, model_config, parallelism_config);
    } catch (const std::exception& e) {
        exception = e.what();
        printf("exception: %s", e.what());
    }
    ASSERT_STREQ("[ERROR] device reserved memory", exception.substr(0, 30).c_str());
}

TEST_F(CacheConfigCreatorTest, testCreateConfig) {
    ModelConfig model_config;
    model_config.num_layers         = 1;
    model_config.attn_config.kv_head_num = 4;
    model_config.attn_config.size_per_head = 128;
    model_config.attn_config.tokens_per_block = 8;
    ParallelismConfig parallelism_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    kv_cache_config.kv_cache_mem_mb = 0; // Use default calculation
    CacheConfigCreator creator;
    auto               result1 = creator.createConfig(model_config, parallelism_config, runtime_config, kv_cache_config);
    ASSERT_TRUE(result1.block_nums > 0);
    ASSERT_EQ(result1.local_head_num_kv, 4);

    kv_cache_config.kv_cache_mem_mb = 32;
    model_config.attn_config.kv_head_num = 1024;
    auto result3           = creator.createConfig(model_config, parallelism_config, runtime_config, kv_cache_config);
    ASSERT_TRUE(result3.block_nums > 0);
}

}  // namespace rtp_llm
