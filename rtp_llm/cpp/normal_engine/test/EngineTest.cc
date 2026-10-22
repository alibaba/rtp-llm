#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"
#include <cstdlib>

#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include "autil/EnvUtil.h"
#include <memory>

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:
};

TEST_F(NormalEngineTest, NormalExecutorTerminalInputDoesNotRunModel) {
    auto engine = createMockEngine(CustomConfig{});
    ASSERT_TRUE(engine->stop().ok());
    auto& executor = *engine->executor_;
    ASSERT_TRUE(executor.supportsTerminalStop());
    EXPECT_FALSE(executor.terminalStopObserved());
    ASSERT_TRUE(executor.process({}).ok());
    EXPECT_FALSE(executor.terminalStopObserved());
    executor.requestTerminalStop();
    ASSERT_TRUE(executor.process({}).ok());
    EXPECT_TRUE(executor.terminalStopObserved());
}

TEST_F(NormalEngineTest, ScrTemplatePrefillRunsBeforeLoopAndReleasesCache) {
    autil::EnvGuard shapes("RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS", "2,8");
    autil::EnvGuard pool_limit("TORCH_CUDA_POOL_PREALLOC_MB", "1024");
    ModelConfig     model;
    RuntimeConfig   runtime;
    KVCacheConfig   cache;
    CustomConfig    config;
    config.reuse_cache             = true;
    auto params                    = createEngineInitParams(config, model, runtime, cache);
    params.pd_sep_config.role_type = RoleType::PREFILL;
    int forwards                   = 0;
    class CountingModel: public MockModel {
    public:
        CountingModel(size_t vocab, int& forwards): MockModel(vocab), forwards_(forwards) {}
        GptModelOutputs forward(const GptModelInputs& inputs) override {
            ++forwards_;
            return MockModel::forward(inputs);
        }

    private:
        int& forwards_;
    };
    NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
        return std::make_unique<CountingModel>(model.vocab_size, forwards);
    };
    std::shared_ptr<NormalEngine> engine;
    try {
        engine = std::make_shared<NormalEngine>(params, nullptr, true);
    } catch (...) {
        NormalExecutor::test_model_factory = nullptr;
        throw;
    }
    NormalExecutor::test_model_factory = nullptr;
    EXPECT_EQ(forwards, 2);
    EXPECT_FALSE(engine->running_);
    EXPECT_FALSE(engine->loop_thread_);
    EXPECT_FALSE(engine->resourceContext().cache_manager->coordinator_);
    EXPECT_EQ(engine->resourceContext().cache_manager->freeBlocksNum(),
              engine->resourceContext().cache_manager->availableBlocksNum());
    EXPECT_EQ(engine->resourceContext().cache_manager->freeBlocksNum(), cache.test_block_num - 1);
    ASSERT_TRUE(engine->startLoop().ok());
    EXPECT_TRUE(engine->running_);
    EXPECT_TRUE(engine->resourceContext().cache_manager->coordinator_);
    auto first_thread = engine->loop_thread_;
    ASSERT_TRUE(engine->startLoop().ok());
    EXPECT_EQ(engine->loop_thread_, first_thread);
    auto input                             = std::make_shared<GenerateInput>();
    input->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
    input->generate_config                 = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 1;
    auto output                            = engine->enqueue(input)->nextOutput();
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    EXPECT_GT(forwards, 2);
}

TEST_F(NormalEngineTest, testInt8KVCache) {
    CustomConfig config;
    config.kv_cache_data_type = DataType::TYPE_INT8;
    auto engine               = createMockEngine(config);

    std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 5;
    query->generate_config->is_streaming   = false;

    try {
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
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
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

    // test non-streaming query with AuxInfo-only frontend metric progress
    {
        std::shared_ptr<GenerateInput> query              = make_shared<GenerateInput>();
        query->input_ids                                  = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                            = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens            = 5;
        query->generate_config->is_streaming              = false;
        query->generate_config->frontend_metric_streaming = true;

        shared_ptr<GenerateStream> stream = engine->enqueue(query);
        ASSERT_TRUE(stream != nullptr);

        for (int output_len = 1; output_len < 5; ++output_len) {
            auto metric_output = stream->nextOutput();
            ASSERT_TRUE(metric_output.ok());
            ASSERT_TRUE(metric_output.value().frontend_metric_only);
            const auto& generated = metric_output.value().generate_outputs[0];
            ASSERT_FALSE(generated.output_ids.defined());
            ASSERT_EQ(generated.aux_info.output_len, output_len);
            ASSERT_EQ(generated.aux_info.step_output_len, 1);
            ASSERT_TRUE(metric_output.value().frontend_context_token_num.has_value());
            ASSERT_EQ(metric_output.value().frontend_context_token_num.value(), 7);
            ASSERT_TRUE(metric_output.value().frontend_context_token_num_with_cache.has_value());
            ASSERT_EQ(metric_output.value().frontend_context_token_num_with_cache.value(), 7);
            ASSERT_TRUE(metric_output.value().frontend_context_execute_time_us.has_value());
            ASSERT_TRUE(metric_output.value().frontend_context_execute_time_with_cache_us.has_value());
            ASSERT_TRUE(metric_output.value().frontend_generate_token_num.has_value());
            ASSERT_EQ(metric_output.value().frontend_generate_token_num.value(), output_len - 1);
            ASSERT_TRUE(metric_output.value().frontend_generate_execute_time_us.has_value());
            if (output_len > 1) {
                ASSERT_GT(metric_output.value().frontend_generate_execute_time_us.value(), 0);
            }
        }

        auto final_output = stream->nextOutput();
        ASSERT_TRUE(final_output.ok());
        ASSERT_FALSE(final_output.value().frontend_metric_only);
        const auto& generated = final_output.value().generate_outputs[0];
        ASSERT_TRUE(generated.output_ids.defined());
        ASSERT_EQ(generated.output_ids.numel(), 5);
        ASSERT_EQ(generated.aux_info.output_len, 5);
        ASSERT_TRUE(final_output.value().frontend_generate_token_num.has_value());
        ASSERT_EQ(final_output.value().frontend_generate_token_num.value(), 4);
        ASSERT_TRUE(final_output.value().frontend_generate_execute_time_us.has_value());
        ASSERT_GT(final_output.value().frontend_generate_execute_time_us.value(), 0);

        auto exhausted = stream->nextOutput();
        ASSERT_FALSE(exhausted.ok());
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
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 13);  // Includes six system-prompt tokens.

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

TEST_F(NormalEngineTest, CaptureAdmissionUsesNormalizedInputBeforeCacheAndCancelsSafely) {
    class CaptureExecutor: public Executor {
    public:
        absl::Status process(const std::list<GenerateStreamPtr>&, int64_t = 0) override {
            return absl::OkStatus();
        }
        bool tryReserveHiddenStateCapture(int64_t request_id, int64_t valid_tokens) override {
            reservations.emplace_back(request_id, valid_tokens);
            if (fail_reserve) {
                throw std::runtime_error("capture reservation unavailable");
            }
            return grant;
        }
        void cancelHiddenStateCapture(int64_t request_id) override {
            cancellations.push_back(request_id);
        }
        bool                                     grant        = true;
        bool                                     fail_reserve = false;
        std::vector<std::pair<int64_t, int64_t>> reservations;
        std::vector<int64_t>                     cancellations;
    };
    CustomConfig config;
    config.reuse_cache              = true;
    config.multi_task_prompt_tokens = {{"capture", {4, 5, 6}}};
    auto engine                     = createMockEngine(config);
    ASSERT_TRUE(engine->stop().ok());
    auto owner             = std::make_shared<CaptureExecutor>();
    engine->executor_      = owner;
    const auto free_blocks = engine->resourceContext().cache_manager->freeBlocksNum();
    auto       make_query  = [](int64_t id, bool capture, bool reuse, int tokens) {
        auto query                                    = std::make_shared<GenerateInput>();
        query->request_id                             = id;
        query->input_ids                              = torch::tensor({1, 2}, torch::kInt32);
        query->generate_config                        = std::make_shared<GenerateConfig>();
        query->generate_config->task_id               = "capture";
        query->generate_config->capture_hidden_states = capture;
        query->generate_config->reuse_cache           = reuse;
        query->generate_config->max_new_tokens        = tokens;
        return query;
    };
    auto captured = engine->makeStream(make_query(100, true, false, 4));
    ASSERT_EQ(owner->reservations, (std::vector<std::pair<int64_t, int64_t>>{{100, 5}}));
    EXPECT_EQ(captured->inputLength(), 5);
    EXPECT_EQ(captured->curBlocksNum(), 0);
    EXPECT_TRUE(captured->hasHiddenStateCaptureReservation());
    EXPECT_TRUE(engine->resourceContext().reuse_cache);
    EXPECT_EQ(engine->resourceContext().cache_manager->freeBlocksNum(), free_blocks);
    captured->releaseResource();
    captured->cancelHiddenStateCapture();
    EXPECT_EQ(owner->cancellations, (std::vector<int64_t>{100}));

    for (const auto& query :
         {make_query(101, false, false, 4), make_query(102, false, true, 4), make_query(103, true, true, 4)}) {
        auto stream = engine->makeStream(query);
        EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
        EXPECT_FALSE(stream->hasError());
        EXPECT_EQ(stream->generateConfig()->reuse_cache, query->generate_config->reuse_cache);
    }
    EXPECT_EQ(owner->reservations.size(), 1u);
    auto legacy = engine->makeStream(make_query(104, false, true, 0));
    EXPECT_TRUE(legacy->hasHiddenStateCaptureReservation());
    legacy.reset();
    EXPECT_EQ(owner->cancellations, (std::vector<int64_t>{100, 104}));
    auto rejected_group = engine->enqueueMultiple({make_query(111, true, false, 4), make_query(112, false, true, 0)});
    EXPECT_EQ(rejected_group.first, (std::vector<bool>{false, false}));
    for (const auto& stream : rejected_group.second) {
        EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    }
    EXPECT_EQ(owner->cancellations, (std::vector<int64_t>{100, 104, 111, 112}));
    owner->grant = false;
    auto denied  = engine->makeStream(make_query(105, true, false, 4));
    EXPECT_FALSE(denied->hasError());
    EXPECT_FALSE(denied->hasHiddenStateCaptureReservation());
    EXPECT_FALSE(denied->generateConfig()->reuse_cache);
    EXPECT_EQ(owner->reservations.back(), (std::pair<int64_t, int64_t>{105, 5}));
    owner->grant        = true;
    owner->fail_reserve = true;
    auto reserve_error  = engine->makeStream(make_query(110, true, false, 4));
    EXPECT_FALSE(reserve_error->hasError());
    EXPECT_FALSE(reserve_error->hasHiddenStateCaptureReservation());
    owner->fail_reserve                    = false;
    auto beam_query                        = make_query(106, true, false, 4);
    beam_query->generate_config->num_beams = 2;
    auto beam                              = engine->makeStream(beam_query);
    EXPECT_FALSE(beam->hasError());
    EXPECT_FALSE(beam->hasHiddenStateCaptureReservation());
    const auto attempts                = owner->reservations.size();
    engine->parallelism_config.tp_rank = 1;
    auto peer                          = engine->makeStream(make_query(107, true, false, 4));
    EXPECT_FALSE(peer->hasHiddenStateCaptureReservation());
    engine->parallelism_config.tp_rank  = 0;
    engine->pd_sep_config.role_type     = RoleType::DECODE;
    engine->resource_context_.role_type = RoleType::DECODE;
    auto decode                         = engine->makeStream(make_query(108, true, false, 4));
    EXPECT_FALSE(decode->hasHiddenStateCaptureReservation());
    EXPECT_FALSE(decode->requiresFullContextForHiddenStateCapture());
    EXPECT_FALSE(decode->hasError());
    EXPECT_EQ(owner->reservations.size(), attempts);
    engine->pd_sep_config.role_type     = RoleType::PDFUSION;
    engine->resource_context_.role_type = RoleType::PDFUSION;
    auto unexecuted                     = engine->makeStream(make_query(109, true, false, 4));
    EXPECT_TRUE(unexecuted->hasHiddenStateCaptureReservation());
    const std::weak_ptr<Executor> weak_owner = owner;
    owner.reset();
    engine->executor_.reset();
    EXPECT_TRUE(weak_owner.expired());
    EXPECT_NO_THROW(unexecuted->cancelHiddenStateCapture());
    EXPECT_FALSE(unexecuted->hasHiddenStateCaptureReservation());
}

TEST_F(NormalEngineTest, testPrefillOnlyReturnsOneEmptyOutput) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    auto query                             = make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 0;

    auto       stream       = engine->enqueue(query);
    const auto input_length = stream->seqLength();
    auto       output       = stream->nextOutput();

    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 1);
    EXPECT_TRUE(output.value().generate_outputs[0].finished);
    EXPECT_EQ(output.value().generate_outputs[0].output_ids.sizes(), (torch::IntArrayRef{1, 0}));
    EXPECT_EQ(stream->seqLength(), input_length);
    EXPECT_EQ(stream->outputTokenLen(), 0);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
}

TEST_F(NormalEngineTest, testDecodeRoleRejectsPrefillOnlyBeforeScheduling) {
    CustomConfig config;
    config.role_type = RoleType::DECODE;
    auto engine      = createMockEngine(config);

    auto zero_query                             = make_shared<GenerateInput>();
    zero_query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    zero_query->generate_config                 = make_shared<GenerateConfig>();
    zero_query->generate_config->max_new_tokens = 0;

    auto zero_stream = engine->enqueue(zero_query);
    ASSERT_NE(zero_stream, nullptr);
    EXPECT_TRUE(zero_stream->hasError());
    EXPECT_EQ(zero_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(zero_stream->curBlocksNum(), 0);
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);

    auto positive_query                             = make_shared<GenerateInput>();
    positive_query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    positive_query->generate_config                 = make_shared<GenerateConfig>();
    positive_query->generate_config->max_new_tokens = 1;

    auto positive_stream = engine->makeStream(positive_query);
    ASSERT_NE(positive_stream, nullptr);
    EXPECT_FALSE(positive_stream->hasError());
    EXPECT_EQ(positive_stream->curBlocksNum(), 0);
}

TEST_F(NormalEngineTest, testDirectEngineEntriesConvertPrefillValidationErrorsToStreamErrors) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    auto make_invalid_query = [] {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 0;
        query->generate_config->min_new_tokens = 1;
        return query;
    };

    std::shared_ptr<GenerateStream> stream;
    EXPECT_NO_THROW(stream = engine->enqueue(make_invalid_query()));
    ASSERT_NE(stream, nullptr);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);

    auto make_stream_query = make_invalid_query();
    EXPECT_NO_THROW(stream = engine->makeStream(make_stream_query));
    ASSERT_NE(stream, nullptr);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);

    std::vector<std::shared_ptr<GenerateInput>>                  inputs{make_invalid_query(), make_invalid_query()};
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>> result;
    EXPECT_NO_THROW(result = engine->enqueueMultiple(inputs));
    EXPECT_EQ(result.first, std::vector<bool>({false, false}));
    ASSERT_EQ(result.second.size(), 2);
    for (const auto& invalid_stream : result.second) {
        ASSERT_NE(invalid_stream, nullptr);
        EXPECT_TRUE(invalid_stream->hasError());
        EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    }
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
}

TEST_F(NormalEngineTest, testEnqueueMultipleRejectsMixedModesBeforeRoleFiltering) {
    CustomConfig config;
    config.role_type = RoleType::DECODE;
    auto engine      = createMockEngine(config);

    auto make_query = [](int64_t request_id, int64_t group_id, int max_new_tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->request_id                      = request_id;
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        query->group_id                        = group_id;
        query->group_size                      = 2;
        return query;
    };

    for (bool generation_first : {false, true}) {
        const int64_t group_id = 100 + generation_first;
        auto          prefill  = make_query(10 + generation_first, group_id, 0);
        auto          generate = make_query(20 + generation_first, group_id, 1);
        auto          inputs   = generation_first ? std::vector<std::shared_ptr<GenerateInput>>{generate, prefill} :
                                                    std::vector<std::shared_ptr<GenerateInput>>{prefill, generate};

        auto [enqueue_successes, streams] = engine->enqueueMultiple(inputs);

        EXPECT_EQ(enqueue_successes, std::vector<bool>({false, false}));
        ASSERT_EQ(streams.size(), 2);
        for (const auto& stream : streams) {
            EXPECT_TRUE(stream->hasError());
            EXPECT_FALSE(stream->hasEvent(StreamEvents::CanRun));
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(stream->stopReason(), kMixedForceBatchGroupError);
        }
        EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
    }
}

TEST_F(NormalEngineTest, testEnqueueMultipleIgnoresInvalidStreamForModeCheck) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    auto make_query = [](int max_new_tokens, int min_new_tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        query->generate_config->min_new_tokens = min_new_tokens;
        return query;
    };

    auto [enqueue_successes, streams] = engine->enqueueMultiple({make_query(0, 1), make_query(1, 0)});

    ASSERT_EQ(streams.size(), 2);
    EXPECT_EQ(enqueue_successes, std::vector<bool>({false, true}));
    EXPECT_TRUE(streams[0]->hasError());
    EXPECT_EQ(streams[0]->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_FALSE(streams[1]->hasError());
    auto output = streams[1]->nextOutput();
    ASSERT_TRUE(output.ok());
    EXPECT_TRUE(output.value().generate_outputs[0].finished);
}

TEST(NormalEnginePolicyTest, testPrefillOnlyStreamDoesNotReserveSpeculativeSteps) {
    EXPECT_EQ(NormalEngine::reserveStepForStream(/*configured_reserve_step=*/4, /*is_prefill_only=*/true), 0);
    EXPECT_EQ(NormalEngine::reserveStepForStream(/*configured_reserve_step=*/4, /*is_prefill_only=*/false), 4);
    EXPECT_EQ(NormalEngine::reserveStepForStream(/*configured_reserve_step=*/0, /*is_prefill_only=*/true), 0);
}

TEST(NormalEnginePolicyTest, testEmptyBatchEarlyReturnPreservesRequiredAlignment) {
    EXPECT_TRUE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/true, /*tp_size=*/1, /*enable_ffn_disaggregate=*/false));
    EXPECT_FALSE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/true, /*tp_size=*/1, /*enable_ffn_disaggregate=*/true));
    EXPECT_FALSE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/true, /*tp_size=*/2, /*enable_ffn_disaggregate=*/false));
    EXPECT_FALSE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/false, /*tp_size=*/1, /*enable_ffn_disaggregate=*/false));
}

TEST(NormalEnginePolicyTest, testBatchDecodeSchedulerCannotSuppressMultiDpMtpFakePrefill) {
    EXPECT_FALSE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/true, /*use_batch_decode_scheduler=*/true, /*dp_size=*/2));
    EXPECT_FALSE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/false, /*use_batch_decode_scheduler=*/true, /*dp_size=*/1));
    EXPECT_TRUE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/false, /*use_batch_decode_scheduler=*/true, /*dp_size=*/2));
    EXPECT_TRUE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/false, /*use_batch_decode_scheduler=*/false, /*dp_size=*/1));
}

}  // namespace rtp_llm
