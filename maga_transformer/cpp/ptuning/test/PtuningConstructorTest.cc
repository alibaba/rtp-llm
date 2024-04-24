
#include "gtest/gtest.h"

#define private public
#include "maga_transformer/cpp/ptuning/Ptuning.h"
#include "maga_transformer/cpp/ptuning/PtuningConstructor.h"
#include "maga_transformer/cpp/utils/TimeUtility.h"
#include "maga_transformer/cpp/normal_engine/test/MockEngine.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/core/Tensor.h"
#include <cuda_runtime.h>

#include <cstdlib>
#include <memory>
#include <thread>
#include <chrono>

using namespace std;

namespace rtp_llm {

class PtuningConstructorTest : public DeviceTestBase {
};

TEST_F(PtuningConstructorTest, testMultiTaskPromptConstruct) {
    PtuningConstructor constructor;
    GptInitParameter params;
    vector<int> prompt_1 = {1, 2, 3};
    vector<int> prompt_2 = {4, 5, 6, 7};
    params.multi_task_prompt_tokens = {{1, prompt_1}, {2, prompt_2}};
    CustomConfig config;
    auto engine = createMockEngine(device_, config);
    ASSERT_EQ(engine->cache_manager_->freeBlockNums(), 99);
    auto result = constructor.construct(params, engine.get(), engine->cache_manager_.get());
    ASSERT_EQ(result.size(), 2);
    ASSERT_EQ(engine->cache_manager_->freeBlockNums(), 95);

    const auto& item1 = result[1];
    ASSERT_EQ(item1.prefix_type, PrefixType::PromptTuning);
    ASSERT_EQ(item1.prefix_length, 3);
    ASSERT_TRUE(!item1.block_cache.empty());
    ASSERT_EQ(item1.prefix_tensor, std::nullopt);
    ASSERT_EQ(item1.prefix_prompt, prompt_1);

    const auto& item2 = result[2];
    ASSERT_EQ(item2.prefix_type, PrefixType::PromptTuning);
    ASSERT_EQ(item2.prefix_length, 4);
    ASSERT_TRUE(!item2.block_cache.empty());
    ASSERT_EQ(item2.prefix_tensor, std::nullopt);
    ASSERT_EQ(item2.prefix_prompt, prompt_2);
}

TEST_F(PtuningConstructorTest, testPtuningConstruct) {
    // shrink default memory pool because the above case malloc/free gpu from/to default pool, it leads the result of cudaMemGetInfo inaccurate
    cudaDeviceSynchronize();
    cudaMemPool_t defaultPool;
    cudaDeviceGetDefaultMemPool(&defaultPool, 0);
    cudaMemPoolTrimTo(defaultPool, 0);

    CustomConfig config;
    auto engine = createMockEngine(device_, config); 
    auto cache_config = engine->cacheManager()->cacheConfig();
    int64_t pre_seq_len = 3;
    auto size = cache_config.layer_num * 2 * cache_config.local_head_num_kv * pre_seq_len * cache_config.size_per_head;
    auto prefix_prompt = torch::arange(0, size).reshape({(int64_t)cache_config.layer_num * 2,
            (int64_t)cache_config.local_head_num_kv, (int64_t)pre_seq_len, (int64_t)cache_config.size_per_head});
    const auto& params = engine->magaInitParams();
    PtuningConstructor constructor;
    auto result = constructor.createPtuningV2(*params.gpt_init_parameter, engine->cacheManager().get(), prefix_prompt);
    ASSERT_EQ(result.prefix_type, PrefixType::PTuningV2);
    ASSERT_EQ(result.prefix_length, 3);
    ASSERT_TRUE(!result.block_cache.empty());
    ASSERT_FALSE(result.prefix_tensor == std::nullopt);
    ASSERT_TRUE(result.prefix_prompt.empty());
}

}
