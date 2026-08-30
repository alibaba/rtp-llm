#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"
#include <cstdlib>

#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include <memory>
#include <stdexcept>

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:
};

TEST_F(NormalEngineTest, testDecodeWarmUpUsesHybridCacheTagsAndGeometry) {
    CustomConfig config;
    config.warm_up          = true;
    config.decode_role      = true;
    config.hybrid_attention = true;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);

    struct WarmUpObservation {
        std::vector<std::string> tags;
        std::vector<int64_t>     block_shape;
        std::vector<int64_t>     kernel_block_shape;
        std::vector<int32_t>     group_types;
        std::vector<int32_t>     full_kernel_row;
        std::vector<int32_t>     linear_kernel_row;
        CacheGroupType           full_type                    = CacheGroupType::SWA;
        size_t                   full_seq_size_per_block      = 0;
        size_t                   full_kernel_size_per_block   = 0;
        CacheGroupType           linear_type                  = CacheGroupType::SWA;
        size_t                   linear_seq_size_per_block    = 0;
        size_t                   linear_kernel_size_per_block = 0;
    } observation;

    NormalExecutor::test_model_factory = [&](const GptModelInitParams& init_params) {
        const auto& cache_config                 = init_params.cache_manager->cacheConfig();
        const auto& full_group                   = cache_config.group("full");
        observation.full_type                    = full_group.policy.group_type;
        observation.full_seq_size_per_block      = full_group.seqSizePerBlock();
        observation.full_kernel_size_per_block   = full_group.kernelSeqSizePerBlock();
        const auto& linear_group                 = cache_config.group("linear");
        observation.linear_type                  = linear_group.policy.group_type;
        observation.linear_seq_size_per_block    = linear_group.seqSizePerBlock();
        observation.linear_kernel_size_per_block = linear_group.kernelSeqSizePerBlock();
        return std::make_unique<MockModel>(model_config.vocab_size, [&](const GptModelInputs& inputs) {
            observation.tags = inputs.kv_cache_group_tags;
            observation.block_shape.assign(inputs.kv_cache_block_id.sizes().begin(),
                                           inputs.kv_cache_block_id.sizes().end());
            observation.kernel_block_shape.assign(inputs.kv_cache_kernel_block_id.sizes().begin(),
                                                  inputs.kv_cache_kernel_block_id.sizes().end());
            const auto group_types = inputs.kv_cache_group_types.cpu().contiguous();
            observation.group_types.assign(group_types.data_ptr<int32_t>(),
                                           group_types.data_ptr<int32_t>() + group_types.numel());
            const auto  kernel_blocks = inputs.kv_cache_kernel_block_id.cpu().contiguous();
            const auto  batch_size    = static_cast<size_t>(kernel_blocks.size(1));
            const auto  row_width     = static_cast<size_t>(kernel_blocks.size(2));
            const auto* rows          = kernel_blocks.data_ptr<int32_t>();
            observation.full_kernel_row.assign(rows, rows + row_width);
            observation.linear_kernel_row.assign(rows + batch_size * row_width,
                                                 rows + batch_size * row_width + row_width);
        });
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    auto engine = std::make_shared<NormalEngine>(params, nullptr);

    EXPECT_EQ(observation.tags, (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(observation.block_shape, (std::vector<int64_t>{2, runtime_config.max_generate_batch_size, 10}));
    EXPECT_EQ(observation.kernel_block_shape, (std::vector<int64_t>{2, runtime_config.max_generate_batch_size, 10}));
    EXPECT_EQ(observation.group_types,
              (std::vector<int32_t>{static_cast<int32_t>(CacheGroupType::FULL),
                                    static_cast<int32_t>(CacheGroupType::LINEAR)}));
    EXPECT_EQ(observation.full_kernel_row, std::vector<int32_t>(10, 0));
    EXPECT_EQ(observation.linear_kernel_row, std::vector<int32_t>(10, 0));
    EXPECT_EQ(observation.full_type, CacheGroupType::FULL);
    EXPECT_EQ(observation.full_seq_size_per_block, 2u);
    EXPECT_EQ(observation.full_kernel_size_per_block, 2u);
    EXPECT_EQ(observation.linear_type, CacheGroupType::LINEAR);
    EXPECT_EQ(observation.linear_seq_size_per_block, 2u);
    EXPECT_EQ(observation.linear_kernel_size_per_block, 2u);
}

TEST_F(NormalEngineTest, testDecodeWarmUpRestoresPriorCacheManagerOnSuccessAndException) {
    CustomConfig config;
    config.decode_role      = true;
    config.hybrid_attention = true;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);

    NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
        return std::make_unique<MockModel>(model_config.vocab_size);
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;
    auto engine = std::make_shared<NormalEngine>(params, nullptr);
    ASSERT_TRUE(engine->stop().ok());

    const auto prior_cache_manager = engine->resourceContext().cache_manager;
    ASSERT_NE(prior_cache_manager, nullptr);

    EXPECT_NO_THROW((void)engine->decodeWarmUp(params));
    EXPECT_EQ(engine->resourceContext().cache_manager, prior_cache_manager);

    NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
        return std::make_unique<MockModel>(model_config.vocab_size, [](const GptModelInputs&) {
            throw std::runtime_error("injected warmup forward failure");
        });
    };
    EXPECT_THROW((void)engine->decodeWarmUp(params), std::runtime_error);
    EXPECT_EQ(engine->resourceContext().cache_manager, prior_cache_manager);

    // decodeWarmUp's existing cleanup is only reached on success. Avoid leaking
    // the injected failure's trace/executor state into other tests.
    setTraceMemory(false);
    engine->executor_.reset(nullptr);
}

TEST_F(NormalEngineTest, testFp8KVCache) {
    CustomConfig config;
    config.kv_cache_data_type = DataType::TYPE_FP8_E4M3;
    auto engine               = createMockEngine(config);

    std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 5;
    query->generate_config->is_streaming   = false;

    shared_ptr<GenerateStream> stream = engine->enqueue(query);

    ASSERT_TRUE(stream != nullptr);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.output_len, 5);
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.input_len, 7);
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.iter_count, 5);

    ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    auto output2 = stream->nextOutput();
    ASSERT_TRUE(!output2.ok());
}

TEST_F(NormalEngineTest, testSimple) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_FALSE(engine->resourceContext().system_prompt);
    ASSERT_FALSE(engine->resourceContext().reuse_cache);

    // test streaming query
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 3;
        query->generate_config->is_streaming   = true;
        query->generate_config->gen_timeline   = true;
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

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output4 = stream->nextOutput();
        ASSERT_TRUE(!output4.ok());
    }

    // test non-streaming query
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 5;
        query->generate_config->is_streaming   = false;

        shared_ptr<GenerateStream> stream = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.output_len, 5);
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.iter_count, 5);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testSystemPrompt) {
    CustomConfig config;
    vector<int>  prompt_1           = {1, 2, 3};
    vector<int>  prompt_2           = {4, 5, 6, 7, 8, 9};
    config.multi_task_prompt_tokens = {{"1", prompt_1}, {"2", prompt_2}};
    auto engine                     = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_TRUE(engine->resourceContext().system_prompt);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);

    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
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

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({10, 20, 30, 40, 50, 60, 70}, torch::kInt32);
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

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({10, 20, 30, 40, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->task_id        = "2";
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 6);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 6);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testReuseCacheOption) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);

    config.reuse_cache = false;
    auto engine2       = createMockEngine(config);
    ASSERT_FALSE(engine2->resourceContext().reuse_cache);
}

TEST_F(NormalEngineTest, testReuseCache) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
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

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
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

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testQueryReuseCacheWhenSwitchIsOn) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);

    // First query with reuse_cache = true
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Second query with reuse_cache = false (should not reuse cache)
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = false;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Third query with reuse_cache = true (should reuse cache)
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 4);  // Should be 4 because reuse_cache = true
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testQueryReuseCacheWhenSwitchIsOff) {
    // Test with engine-level reuse_cache = false (master switch off)
    CustomConfig config;
    config.reuse_cache = false;
    auto engine        = createMockEngine(config);
    ASSERT_FALSE(engine->resourceContext().reuse_cache);

    // Query with reuse_cache = true, but should be ignored because engine-level is false
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;  // This should be ignored
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because engine-level reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Query with reuse_cache = false, should also result in no cache reuse
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = false;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because engine-level reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithPrefillCP) {
    CustomConfig config;
    config.output_vocab_ids   = {0, 2, 7};
    config.prefill_cp_enabled = true;
    EXPECT_THROW(createMockEngine(config), std::exception);
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithDeviceInput) {
    CustomConfig config;
    config.output_vocab_ids = {0, 2, 7};
    setenv("RTP_LLM_DEVICE_INPUT", "1", 1);
    EXPECT_THROW(createMockEngine(config), std::exception);
    unsetenv("RTP_LLM_DEVICE_INPUT");
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithSpeculative) {
    CustomConfig config;
    config.output_vocab_ids    = {0, 2, 7};
    config.speculative_enabled = true;
    EXPECT_THROW(createMockEngine(config), std::exception);
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithWarmUpWithLoss) {
    CustomConfig config;
    config.output_vocab_ids  = {0, 2, 7};
    config.warm_up_with_loss = true;
    EXPECT_THROW(createMockEngine(config), std::exception);
}

TEST_F(NormalEngineTest, testRejectInvalidOutputVocabIds) {
    CustomConfig unsorted;
    unsorted.output_vocab_ids = {7, 2, 0};
    EXPECT_THROW(createMockEngine(unsorted), std::exception);

    CustomConfig duplicated;
    duplicated.output_vocab_ids = {0, 2, 2, 7};
    EXPECT_THROW(createMockEngine(duplicated), std::exception);

    CustomConfig out_of_range;
    out_of_range.output_vocab_ids = {0, 2, 100};  // vocab_size is 100
    EXPECT_THROW(createMockEngine(out_of_range), std::exception);
}

TEST_F(NormalEngineTest, testAllowsUnsupportedCombosWithoutOutputVocab) {
    CustomConfig config;
    config.prefill_cp_enabled  = true;
    config.speculative_enabled = true;
    config.warm_up_with_loss   = true;
    EXPECT_NO_THROW(createMockEngine(config));
}

}  // namespace rtp_llm
