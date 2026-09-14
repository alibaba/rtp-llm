#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"
#include <cstdlib>
#include <algorithm>

#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <stdexcept>
#include "rtp_llm/cpp/cache/KVCachePhysicalMemoryController.h"

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:
};

class NormalWarmupConfigTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        previous_factory_ = std::move(NormalExecutor::test_model_factory);
    }

    void TearDown() override {
        NormalExecutor::test_model_factory = std::move(previous_factory_);
        DeviceTestBase::TearDown();
    }

private:
    NormalExecutor::ModelFactory previous_factory_;
};

TEST_F(NormalWarmupConfigTest, testExecutorGraphConfigIsLocalToWarmup) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto mutable_params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    auto cache_manager =
        std::make_shared<KVCacheManager>(makeMhaCacheConfig(2, 5, 2, 64, 2, DataType::TYPE_FP16), true);
    ASSERT_TRUE(cache_manager->init());
    const bool vmm_available = VmmBackend().isAvailable();
    for (bool graph_enabled : {false, true}) {
        mutable_params.hw_kernel_config.enable_cuda_graph = graph_enabled;
        const EngineInitParams params                     = mutable_params;
        for (bool warm_up : {false, true}) {
            for (bool has_cache : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "vmm=" << vmm_available << " graph=" << graph_enabled
                                                  << " warm_up=" << warm_up << " cache=" << has_cache);
                int factory_calls                  = 0;
                NormalExecutor::test_model_factory = [&](const GptModelInitParams& model_params) {
                    ++factory_calls;
                    EXPECT_EQ(params.hw_kernel_config.enable_cuda_graph, graph_enabled);
                    EXPECT_NE(&model_params.hw_kernel_config, &params.hw_kernel_config);
                    EXPECT_EQ(model_params.hw_kernel_config.enable_cuda_graph,
                              graph_enabled && !(warm_up && has_cache && vmm_available));
                    return std::make_unique<MockModel>(model_config.vocab_size);
                };
                { NormalExecutor executor(params, has_cache ? cache_manager : nullptr, warm_up); }
                EXPECT_EQ(factory_calls, 1);
                EXPECT_EQ(params.hw_kernel_config.enable_cuda_graph, graph_enabled);
                NormalExecutor::test_model_factory = nullptr;
            }
        }
    }
}

TEST_F(NormalWarmupConfigTest, testDecodeWarmupFailurePreservesConstCallerConfig) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto mutable_params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    mutable_params.runtime_config.warm_up  = true;
    mutable_params.pd_sep_config.role_type = RoleType::DECODE;
    const bool vmm_available               = VmmBackend().isAvailable();
    for (bool graph_enabled : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "vmm=" << vmm_available << " graph=" << graph_enabled);
        mutable_params.hw_kernel_config.enable_cuda_graph = graph_enabled;
        const EngineInitParams params                     = mutable_params;
        int                    factory_calls              = 0;
        NormalExecutor::test_model_factory = [&](const GptModelInitParams& model_params) -> std::unique_ptr<ModelBase> {
            ++factory_calls;
            // Check during construction as well as after failure: restoring
            // a mutated caller config later would still violate this contract.
            EXPECT_EQ(params.hw_kernel_config.enable_cuda_graph, graph_enabled);
            EXPECT_EQ(model_params.hw_kernel_config.enable_cuda_graph, graph_enabled && !vmm_available);
            throw std::runtime_error("injected warmup model construction failure");
        };
        try {
            NormalEngine engine(params, nullptr);
            FAIL() << "warmup model construction should fail";
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "injected warmup model construction failure");
        }
        EXPECT_EQ(factory_calls, 1);
        EXPECT_EQ(params.hw_kernel_config.enable_cuda_graph, graph_enabled);
        NormalExecutor::test_model_factory = nullptr;
        setTraceMemory(false);
    }
}

TEST_F(NormalEngineTest, testEmptyTpSleepRoundKeepsSkipInputWithoutFakeForward) {
    class RecordingExecutor: public Executor {
    public:
        absl::Status process(const std::list<GenerateStreamPtr>& streams, int64_t) override {
            ++calls;
            stream_count = streams.size();
            all_fake = std::all_of(streams.begin(), streams.end(), [](const auto& stream) {
                return stream->isFakeStream();
            });
            return absl::OkStatus();
        }
        int calls = 0;
        size_t stream_count = 0;
        bool all_fake = false;
    };

    auto engine = createMockEngine(CustomConfig{});
    // Join the real single-GPU loop before changing its test-only topology.
    // Do not stop the round fence: step() below must exercise ON admission.
    engine->running_ = false;
    ASSERT_TRUE(engine->scheduler_->stop().ok());
    engine->loop_thread_->join();
    auto recorder = std::make_unique<RecordingExecutor>();
    auto* observed = recorder.get();
    engine->executor_ = std::move(recorder);
    engine->running_ = true;

    for (bool sleep_enabled : {false, true}) {
        engine->sleep_controller_.setEnabled(sleep_enabled);
        for (int dp_size : {1, 2}) {
            for (int ep_size : {1, 2}) {
                SCOPED_TRACE(::testing::Message() << "sleep=" << sleep_enabled
                             << " dp=" << dp_size << " ep=" << ep_size);
                engine->parallelism_config.dp_size = dp_size;
                engine->parallelism_config.tp_size = 2 / dp_size;
                engine->parallelism_config.ep_size = ep_size;
                engine->parallelism_config.world_size = 2;
                engine->parallelism_config.tp_rank = 0;
                const int previous_calls = observed->calls;
                ASSERT_TRUE(engine->step().ok());
                // TP-only must still call process() to broadcast skip_run.
                // Cross-DP peers must keep their pre-existing fake forward.
                EXPECT_EQ(observed->calls, previous_calls + 1);
                EXPECT_EQ(observed->stream_count, dp_size > 1 ? 1u : 0u);
                if (dp_size > 1) {
                    EXPECT_TRUE(observed->all_fake);
                }
            }
        }
    }
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

// Regression guard for the pause/quiesce acknowledgement ordering. pauseAndWaitQuiesced()
// bumps pause_epoch_ and blocks until the loop thread's enterPausedState() records a quiesce
// for that epoch. If the acknowledgement could be lost (a stale-epoch ack, or an ack clobbered
// by the pause publish) the coordinator would block until the deadline and return
// DeadlineExceeded. Driving many cycles against the live loop thread exercises that interleaving.
TEST_F(NormalEngineTest, testPauseQuiesceAckNoLostNotification) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    constexpr int kCycles = 300;
    for (int i = 0; i < kCycles; ++i) {
        auto status = engine->pauseAndWaitQuiesced(5000);
        ASSERT_TRUE(status.ok()) << "cycle " << i << ": " << status.ToString();
        // The quiesce acknowledgement reached at least the epoch this pause published.
        ASSERT_GE(engine->quiesced_pause_epoch_, engine->pause_epoch_.load());
        engine->restart();
    }
}

// Multiple coordinators race to pause the same engine. Only one wins the CAS and bumps the
// epoch, but every waiter must observe the single quiesce acknowledgement for that epoch and
// return OK -- none may be stranded by the pause/ack interleaving.
TEST_F(NormalEngineTest, testConcurrentPauseWaitersAllQuiesce) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    constexpr int kThreads = 8;
    constexpr int kRounds  = 50;
    for (int round = 0; round < kRounds; ++round) {
        std::atomic<int>         ok_count{0};
        std::vector<std::thread> waiters;
        waiters.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            waiters.emplace_back([&] {
                if (engine->pauseAndWaitQuiesced(5000).ok()) {
                    ok_count.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& w : waiters) {
            w.join();
        }
        ASSERT_EQ(ok_count.load(), kThreads) << "round " << round;
        engine->restart();
    }
}

}  // namespace rtp_llm
