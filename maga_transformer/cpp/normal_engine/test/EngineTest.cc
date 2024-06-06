#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"
#include <cstdlib>

#define private public

#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/models/W.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "maga_transformer/cpp/normal_engine/test/MockEngine.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include <memory>

using namespace std;
namespace W  = ft::W;
namespace ft = fastertransformer;
namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:

};

TEST_F(NormalEngineTest, testInt8KVCache) {
    CustomConfig config;
    config.int8_kv_cache = true;
    auto gpt_init_params = GptInitParameter();
    auto engine = createMockEngine(device_, config, gpt_init_params);

    std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
    query->input_ids                       = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 5;
    query->generate_config->is_streaming   = false;

    shared_ptr<GenerateStream> stream      = engine->enqueue(query);

    ASSERT_TRUE(stream != nullptr);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.output_len, 5);
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.input_len, 7);
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.iter_count, 5);

    ASSERT_TRUE(stream->finished());
    auto output2 = stream->nextOutput();
    ASSERT_TRUE(!output2.ok());
}

TEST_F(NormalEngineTest, testSimple) {
    CustomConfig config;
    auto gpt_init_params = GptInitParameter();
    auto engine = createMockEngine(device_, config, gpt_init_params);

    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_FALSE(engine->resourceContext().system_prompt);
    ASSERT_FALSE(engine->resourceContext().reuse_cache);
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlockNums(), 99);

    // test streaming query
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 3;
        query->generate_config->is_streaming   = true;

        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.iter_count, 1);

        auto output2 = stream->nextOutput();
        ASSERT_TRUE(output2.ok());
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.output_len, 2);
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.iter_count, 2);

        auto output3 = stream->nextOutput();
        ASSERT_TRUE(output3.ok());
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.output_len, 3);
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.iter_count, 3);

        ASSERT_TRUE(stream->finished());
        auto output4 = stream->nextOutput();
        ASSERT_TRUE(!output4.ok());
    }

    // test non-streaming query
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 5;
        query->generate_config->is_streaming   = false;

        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.output_len, 5);
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.iter_count, 5);

        ASSERT_TRUE(stream->finished());
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testNewDevice) {
    setenv("USE_NEW_DEVICE_IMPL", "1", 1);
    CustomConfig config;
    // TODO(xinfei.sxf) split case
    config.int8_kv_cache = true;
    auto gpt_init_params = GptInitParameter();
    auto engine = createMockEngine(device_, config, gpt_init_params);

    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_FALSE(engine->resourceContext().system_prompt);
    ASSERT_FALSE(engine->resourceContext().reuse_cache);
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlockNums(), 99);

    std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
    query->input_ids                       = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 1;
    // query->generate_config->is_streaming   = true;

    shared_ptr<GenerateStream> stream = engine->enqueue(query);

    ASSERT_TRUE(stream != nullptr);
    auto output3 = stream->nextOutput();
    ASSERT_TRUE(output3.ok());
    ASSERT_EQ(output3.value().generate_outputs[0].aux_info.output_len, 1);
    ASSERT_EQ(output3.value().generate_outputs[0].aux_info.input_len, 7);
    ASSERT_EQ(output3.value().generate_outputs[0].aux_info.iter_count, 1);

    ASSERT_TRUE(stream->finished());
    auto output4 = stream->nextOutput();
    ASSERT_TRUE(!output4.ok());
    unsetenv("USE_NEW_DEVICE_IMPL");
}


TEST_F(NormalEngineTest, testSystemPrompt) {
    CustomConfig config;
    vector<int> prompt_1 = {1, 2, 3};
    vector<int> prompt_2 = {4, 5, 6, 7, 8, 9};
    config.multi_task_prompt_tokens = {{1, prompt_1}, {2, prompt_2}};
    auto gpt_init_params = GptInitParameter();
    auto engine = createMockEngine(device_, config, gpt_init_params);
    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_TRUE(engine->resourceContext().system_prompt);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlockNums(), 96);

    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 2);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->finished());
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
    {
        std::shared_ptr<GenerateInput> query    = make_shared<GenerateInput>();
        query->input_ids                        = createBuffer<int32_t>({7}, {10, 20, 30, 40, 50, 60, 70}, AllocationType::HOST);
        query->generate_config                  = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens  = 1;
        shared_ptr<GenerateStream> stream       = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->finished());
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
    {
        std::shared_ptr<GenerateInput> query    = make_shared<GenerateInput>();
        query->input_ids                        = createBuffer<int32_t>({7}, {10, 20, 30, 40, 50, 60, 70}, AllocationType::HOST);
        query->generate_config                  = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens  = 1;
        query->generate_config->task_id         = 2;
        shared_ptr<GenerateStream> stream       = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 6);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 4);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->finished());
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testReuseCacheOption) {
    CustomConfig config;
    config.reuse_cache = true;
    auto gpt_init_params = GptInitParameter();
    auto engine = createMockEngine(device_, config, gpt_init_params);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlockNums(), 99);

    config.reuse_cache = false;
    auto gpt_init_params2 = GptInitParameter();
    auto engine2 = createMockEngine(device_, config, gpt_init_params2);
    ASSERT_FALSE(engine2->resourceContext().reuse_cache);
}

TEST_F(NormalEngineTest, testReuseCache) {
    setenv("REUSE_CACHE", "1", 1);
    setenv("ENABLE_PAGED_TRT_FMHA", "OFF", 1);
    CustomConfig config;
    config.reuse_cache = true;
    auto gpt_init_params = GptInitParameter();
    auto engine = createMockEngine(device_, config, gpt_init_params);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);
    ASSERT_EQ(engine->resourceContext().cache_manager->freeBlockNums(), 99);
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->finished());
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = createBuffer<int32_t>({7}, {1, 2, 3, 4, 50, 60, 70}, AllocationType::HOST);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 4);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->finished());
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

}  // namespace rtp_llm
