
#include "gtest/gtest.h"

#define private public
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
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

class SystemPromptConstructorTest : public DeviceTestBase {
};

TEST_F(SystemPromptConstructorTest, testMultiTaskPromptConstruct) {
    SystemPromptConstructor constructor;
    GptInitParameter params;
    vector<int> prompt_1 = {1, 2, 3};
    vector<int> prompt_2 = {4, 5, 6, 7};
    params.multi_task_prompt_tokens_ = {{1, prompt_1}, {2, prompt_2}};
    CustomConfig config;
    auto gpt_init_params = GptInitParameter();
    auto engine = createMockEngine(device_, config, gpt_init_params);
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlockNums(), 99);
    auto result = constructor.construct(params, engine.get(), engine->resourceContext().cache_manager.get());
    ASSERT_EQ(result.size(), 2);
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlockNums(), 97);

    const auto& item1 = result[1];
    ASSERT_EQ(item1.prompt_length, 3);
    ASSERT_TRUE(!item1.block_cache.empty());
    ASSERT_EQ(item1.prompt_token, prompt_1);

    const auto& item2 = result[2];
    ASSERT_EQ(item2.prompt_length, 4);
    ASSERT_TRUE(!item2.block_cache.empty());
    ASSERT_EQ(item2.prompt_token, prompt_2);
}

}
