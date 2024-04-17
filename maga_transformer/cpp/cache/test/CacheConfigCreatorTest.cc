
#include "gtest/gtest.h"

#define private public
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

class CacheConfigCreatorTest: public DeviceTestBase {
protected:
};

TEST_F(CacheConfigCreatorTest, testGetKVCacheMemorySize) {
    GptInitParameter param;
    param.reserve_runtime_mem_mb_ = 20;
    param.kv_cache_mem_mb_ = 10;
    CacheConfigCreator creator;
    auto [success, cache_size] = creator.getKVCacheMemorySize(param);
    ASSERT_TRUE(success);
    ASSERT_EQ(10 * 1024 * 1024, cache_size);

    param.kv_cache_mem_mb_ = 0;
    auto [success2, cache_size2] = creator.getKVCacheMemorySize(param);
    ASSERT_TRUE(success2);
    ASSERT_TRUE(cache_size2 > 0);

    param.reserve_runtime_mem_mb_ = 200000;
    auto [success3, cache_size3] = creator.getKVCacheMemorySize(param);
    ASSERT_FALSE(success3);
}

TEST_F(CacheConfigCreatorTest, testCreateConfig) {
    setenv("TEST_BLOCK_NUM", "200", 1);
    GptInitParameter param;
    param.num_layers_ = 1;
    param.head_num_kv_ = 4;
    param.size_per_head_ = 128;
    param.seq_size_per_block_ = 8;
    CacheConfigCreator creator;
    auto [success, cache_config] = creator.createConfig(param);
    ASSERT_TRUE(success);
    ASSERT_EQ(cache_config.block_nums, 200);
    ASSERT_EQ(cache_config.local_head_num_kv, 4);

    setenv("TEST_BLOCK_NUM", "abc200", 1);
    auto [success2, cache_config2] = creator.createConfig(param);
    ASSERT_FALSE(success2);

    unsetenv("TEST_BLOCK_NUM");
    auto [success3, cache_config3] = creator.createConfig(param);
    ASSERT_TRUE(success3);
    ASSERT_TRUE(cache_config.block_nums > 0);
    ASSERT_EQ(cache_config.local_head_num_kv, 4);

    param.kv_cache_mem_mb_ = 1;
    param.head_num_kv_ = 1024 * 1024;
    auto [success4, cache_config4] = creator.createConfig(param);
    ASSERT_FALSE(success4);
}

}  // namespace rtp_llm
