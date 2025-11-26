
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <cuda_runtime.h>

#include <cstdlib>
#include <memory>
#include <thread>
#include <chrono>

using namespace std;

namespace rtp_llm {

class SystemPromptConstructorTest: public DeviceTestBase {};

TEST_F(SystemPromptConstructorTest, testMultiTaskPromptConstruct) {
    SystemPromptConstructor constructor;
    GptInitParameter        params;
    vector<int>             prompt_1 = {1, 2, 3};
    vector<int>             prompt_2 = {4, 5, 6, 7};
    params.multi_task_prompt_tokens_ = {{"1", prompt_1}, {"2", prompt_2}};
    CustomConfig config;
    auto         gpt_init_params = GptInitParameter();
    auto         engine          = createMockEngine(device_, config, gpt_init_params);
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlocksNum(), 99);
    const_cast<ResourceContext*>(&engine->resourceContext())->reuse_cache = true;
    auto result_status =
        constructor.construct(params, engine.get(), engine->resourceContext().cache_manager.get(), true);
    ASSERT_EQ(result_status.ok(), true);
    auto result = result_status.value();
    ASSERT_EQ(result.size(), 2);
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlocksNum(), 96);  // 99 - (2 + 1)

    const auto& item1 = result["1"];
    ASSERT_EQ(item1.prompt_tokens.size(), 3);
    ASSERT_TRUE(!item1.block_ids.empty());
    ASSERT_EQ(item1.prompt_tokens, prompt_1);

    const auto& item2 = result["2"];
    ASSERT_EQ(item2.prompt_tokens.size(), 4);
    ASSERT_TRUE(!item2.block_ids.empty());
    ASSERT_EQ(item2.prompt_tokens, prompt_2);
}

}  // namespace rtp_llm
