
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
};

TEST_F(CacheConfigCreatorTest, testGetKVCacheMemorySize) {
    GptInitParameter param;
    param.reserve_runtime_mem_mb_ = 20;
    param.kv_cache_mem_mb_ = 10;
    CacheConfigCreator creator;
    auto result1 = creator.getKVCacheMemorySize(param);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(10 * 1024 * 1024, result1.value());

    param.kv_cache_mem_mb_ = 0;
    auto result2 = creator.getKVCacheMemorySize(param);
    ASSERT_TRUE(result2.ok());
    ASSERT_TRUE(result2.value() > 0);

    param.reserve_runtime_mem_mb_ = 200000;
    auto result3 = creator.getKVCacheMemorySize(param);
    ASSERT_FALSE(result3.ok());
}

TEST_F(CacheConfigCreatorTest, testCreateConfig) {
    GptInitParameter param;
    param.block_nums_ = 200;
    param.num_layers_ = 1;
    param.head_num_kv_ = 4;
    param.size_per_head_ = 128;
    param.seq_size_per_block_ = 8;
    CacheConfigCreator creator;
    auto result1 = creator.createConfig(param);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().block_nums, 200);
    ASSERT_EQ(result1.value().local_head_num_kv, 4);

    param.block_nums_ = 0;
    auto result2 = creator.createConfig(param);
    ASSERT_TRUE(result2.ok());
    ASSERT_TRUE(result2.value().block_nums > 0);
    ASSERT_EQ(result2.value().local_head_num_kv, 4);

    param.kv_cache_mem_mb_ = 1;
    param.head_num_kv_ = 1024 * 1024;
    auto result3 = creator.createConfig(param);
    ASSERT_FALSE(result3.ok());
}

}  // namespace rtp_llm
