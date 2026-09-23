#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"
#include <cstdlib>

#define private public
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#undef private

#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/models/logits_processor/CodebookLogitsProcessor.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include <memory>

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:
};

TEST_F(NormalEngineTest, testDecodeWarmupReserveTokensAreConvertedToBlocksAfterAddition) {
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/7, /*reserve_tokens=*/1, /*tokens_per_block=*/8), 1u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/8, /*reserve_tokens=*/1, /*tokens_per_block=*/8), 2u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/7, /*reserve_tokens=*/9, /*tokens_per_block=*/8), 2u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/9, /*reserve_tokens=*/8, /*tokens_per_block=*/8), 3u);
    EXPECT_ANY_THROW(
        NormalEngine::warmUpReservedBlockCount(/*seq_len=*/1, /*reserve_tokens=*/1, /*tokens_per_block=*/0));
}

TEST_F(NormalEngineTest, testRejectGenerationPrefillWithSpeculativeBeforeRunnerCreation) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.hw_kernel_config.enable_cuda_graph                        = true;
    params.hw_kernel_config.generation_prefill_capture_token_buckets = {8, 16};
    params.parallelism_config.role_type                              = RoleType::PDFUSION;
    params.pd_sep_config.role_type                                   = RoleType::PDFUSION;

    // Exercise the real constructor, not just the wrapper ownership predicate.
    // Both the configured execution mode and a supplied propose model must
    // reject the combination before warmup, KV allocation or runner creation.
    for (const auto speculative_type : {SP_TYPE_MTP, SP_TYPE_DSPARK, SP_TYPE_NONE}) {
        for (const bool has_propose_model : {false, true}) {
            if (speculative_type == SP_TYPE_NONE && !has_propose_model) {
                continue;
            }
            SCOPED_TRACE(::testing::Message() << "speculative_type=" << static_cast<int>(speculative_type)
                                              << " has_propose_model=" << has_propose_model);
            params.sp_config.type = speculative_type;
            auto propose_params =
                has_propose_model ? std::make_unique<ProposeModelEngineInitParams>(SP_TYPE_MTP, 2) : nullptr;
            try {
                NormalEngine engine(params, std::move(propose_params));
                FAIL() << "explicit generation-prefill/speculative combination must fail initialization";
            } catch (const std::exception& error) {
                EXPECT_NE(std::string(error.what())
                              .find("GENERATION_PREFILL_CAPTURE_CONFIG does not support speculative execution"),
                          std::string::npos)
                    << error.what();
            }
        }
    }
}

TEST_F(NormalEngineTest, testPdRolesIgnoreGenerationPrefillWithOrWithoutSpeculativeConfig) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.hw_kernel_config.enable_cuda_graph                        = true;
    params.hw_kernel_config.generation_prefill_capture_token_buckets = {8, 16};
    params.runtime_config.warm_up                                    = false;

    // Keep the real NormalEngine constructor and its configuration checks.
    // Only model execution is mocked; no draft model is needed to exercise the
    // retained speculative configuration that previously failed at startup.
    NormalExecutor::test_model_factory = [vocab = model_config.vocab_size](const GptModelInitParams&) {
        return std::make_unique<MockModel>(vocab);
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    for (const auto role_type : {RoleType::PREFILL, RoleType::DECODE}) {
        for (const auto speculative_type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_DSPARK}) {
            SCOPED_TRACE(::testing::Message() << "role_type=" << static_cast<int>(role_type)
                                              << " speculative_type=" << static_cast<int>(speculative_type));
            params.parallelism_config.role_type = role_type;
            params.pd_sep_config.role_type      = role_type;
            params.sp_config.type               = speculative_type;
            EXPECT_NO_THROW({ NormalEngine engine(params, nullptr); });
        }
    }
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

TEST_F(NormalEngineTest, testParallelDispatchMultipleRequests) {
    CustomConfig config;
    config.engine_async_worker_count                     = 2;
    auto                                     engine      = createMockEngine(config);
    std::weak_ptr<autil::LockFreeThreadPool> thread_pool = engine->thread_pool_;
    ASSERT_FALSE(thread_pool.expired());

    auto make_query = [](int input_token) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({input_token}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 2;
        query->generate_config->is_streaming   = false;
        return query;
    };

    auto [accepted, streams] = engine->enqueueMultiple({make_query(1), make_query(2)});
    ASSERT_EQ(accepted, std::vector<bool>({true, true}));
    ASSERT_EQ(streams.size(), 2u);
    for (auto& stream : streams) {
        ASSERT_NE(stream, nullptr);
        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs.size(), 1u);
        EXPECT_EQ(output.value().generate_outputs[0].aux_info.input_len, 1);
        EXPECT_EQ(output.value().generate_outputs[0].aux_info.output_len, 2);
        EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    }

    engine.reset();
    EXPECT_TRUE(thread_pool.expired());
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

TEST_F(NormalEngineTest, testCodebookRejectsInvalidModelAtInitialization) {
    CustomConfig config;
    config.output_vocab_ids = {0, 10, 11, 20, 21};
    for (const auto& groups : std::vector<std::vector<std::vector<int64_t>>>{
             {{2, 1}, {3, 4}}, {{1, 1}, {3, 4}}, {{1, 2}, {5}}, {{}, {3, 4}}, {{0, 1}, {3, 4}}}) {
        config.output_vocab_groups = groups;
        try {
            createMockEngine(config);
            FAIL() << "invalid codebook configuration was accepted";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("codebook"), std::string::npos);
        }
    }
}

TEST_F(NormalEngineTest, testCodebookDynamicBeamEndToEndAndFakeStream) {
    CustomConfig config;
    config.output_vocab_ids    = {0, 10, 11, 12, 20, 21, 22};
    config.output_vocab_groups = {{1, 2, 3}, {4, 5, 6}};
    auto engine                = createMockEngine(config);
    ASSERT_NE(engine, nullptr);
    ASSERT_TRUE(engine->resource_context_.codebook_masks.is_cuda());
    auto initial_masks = engine->resource_context_.codebook_masks.clone();

    // Scheduler padding streams are internal work, not user codebook sequences.
    auto fake = engine->createMinFakeStream(1);
    ASSERT_FALSE(fake->hasError());

    auto query                                 = std::make_shared<GenerateInput>();
    query->input_ids                           = torch::tensor({8, 9}, torch::kInt32);
    query->begin_time_us                       = autil::TimeUtility::currentTimeInMicroSeconds();
    query->generate_config                     = std::make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens     = 2;
    query->generate_config->variable_num_beams = {2, 4};
    query->generate_config->is_streaming       = false;
    auto stream                                = engine->enqueue(query);
    ASSERT_FALSE(stream->hasError());
    auto result = stream->nextOutput();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().generate_outputs.size(), 4);
    for (const auto& output : result.value().generate_outputs) {
        auto ids = output.output_ids.reshape({-1});
        ASSERT_EQ(ids.numel(), 2);
        EXPECT_GE(ids[0].item<int>(), 10);
        EXPECT_LE(ids[0].item<int>(), 12);
        EXPECT_GE(ids[1].item<int>(), 20);
        EXPECT_LE(ids[1].item<int>(), 22);
    }
    EXPECT_TRUE(stream->isFinished());
    auto sibling = engine->makeStream(query);
    ASSERT_FALSE(sibling->hasError());
    for (const auto& request : {stream, sibling}) {
        auto processor = std::dynamic_pointer_cast<CodebookLogitsProcessor>(request->getAllLogitsProcessorPtr().back());
        ASSERT_NE(processor, nullptr);
        EXPECT_EQ(processor->masks_.data_ptr(), engine->resource_context_.codebook_masks.data_ptr());
    }
    EXPECT_TRUE(torch::equal(initial_masks, engine->resource_context_.codebook_masks));
}

TEST_F(NormalEngineTest, testCodebookKeepsBeamWidthAfterEarlyStop) {
    CustomConfig config;
    config.output_vocab_ids                            = {0, 10, 11, 20, 21};
    config.output_vocab_groups                         = {{1, 2}, {3, 4}};
    auto engine                                        = createMockEngine(config);
    auto query                                         = std::make_shared<GenerateInput>();
    query->input_ids                                   = torch::tensor({8, 9}, torch::kInt32);
    query->begin_time_us                               = autil::TimeUtility::currentTimeInMicroSeconds();
    query->generate_config                             = std::make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens             = 2;
    query->generate_config->variable_num_beams         = {2, 4};
    query->generate_config->stop_words_list            = {{10}};
    query->generate_config->is_streaming               = false;
    auto healthy_query                                 = std::make_shared<GenerateInput>(*query);
    healthy_query->generate_config                     = std::make_shared<GenerateConfig>(*query->generate_config);
    healthy_query->generate_config->variable_num_beams = {2, 3};
    auto [accepted, streams]                           = engine->enqueueMultiple({query, healthy_query});
    ASSERT_EQ(accepted, std::vector<bool>({true, true}));
    ASSERT_EQ(streams.size(), 2);
    auto result = streams[0]->nextOutput();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().generate_outputs.size(), 4);
    EXPECT_FALSE(streams[0]->hasError());
    EXPECT_EQ(streams[0]->outputTokenLen(), 2);
    // Only three candidates are legal. Keep the requested beam width and do not
    // constrain the extra top-k entry, which may come from a masked position.

    auto healthy_result = streams[1]->nextOutput();
    ASSERT_TRUE(healthy_result.ok()) << healthy_result.status().ToString();
    ASSERT_EQ(healthy_result.value().generate_outputs.size(), 3);
    for (const auto& output : healthy_result.value().generate_outputs) {
        auto ids = output.output_ids.reshape({-1});
        ASSERT_EQ(ids.numel(), 2);
        if (ids[0].item<int>() == 10) {
            EXPECT_EQ(ids[1].item<int>(), 0);  // Finished beam keeps its EOS continuation.
        } else {
            EXPECT_EQ(ids[0].item<int>(), 11);
            EXPECT_TRUE(ids[1].item<int>() == 20 || ids[1].item<int>() == 21);
        }
    }
}

}  // namespace rtp_llm
