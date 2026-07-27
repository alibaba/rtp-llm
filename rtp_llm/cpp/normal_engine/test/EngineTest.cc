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
#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:
};

class PauseDrainTrackingExecutor: public Executor {
public:
    absl::Status process(const std::list<GenerateStreamPtr>&, int64_t) override {
        events.push_back("process");
        return absl::OkStatus();
    }

    absl::Status processForPause() override {
        events.push_back("pause_wave");
        return process_for_pause_status;
    }

    absl::Status drainAsyncRunners() override {
        events.push_back("drain");
        if (on_drain) {
            on_drain();
        }
        return drain_status;
    }

    std::vector<std::string> events;
    std::function<void()>    on_drain;
    absl::Status             process_for_pause_status = absl::OkStatus();
    absl::Status             drain_status             = absl::OkStatus();
};

TEST(CollectiveSleepQuiesceAlignmentTest, CompensatesRankSkewBeforeReadyBarrier) {
    using Alignment = NormalEngine::CollectiveQuiesceAlignment;
    Alignment rank0;
    Alignment rank1;

    for (int i = 0; i < 4; ++i) {
        rank0.noteForwardEnqueued();
        rank1.noteForwardEnqueued();
    }

    // rank0 observes the ready verdict first and stops normal scheduling. rank1 has already
    // entered one more peer-symmetric forward by the time it observes the same verdict.
    rank0.beginTargetExchange();
    rank1.noteForwardEnqueued();
    rank1.beginTargetExchange();

    const auto target = std::max(rank0.forward_sequence, rank1.forward_sequence);
    ASSERT_TRUE(rank0.acceptTargetSequence(target));
    ASSERT_TRUE(rank1.acceptTargetSequence(target));
    EXPECT_TRUE(rank0.needsCompensationForward());
    EXPECT_FALSE(rank0.readyForBarrier());
    EXPECT_FALSE(rank1.needsCompensationForward());
    EXPECT_TRUE(rank1.readyForBarrier());

    rank0.noteForwardEnqueued();
    EXPECT_FALSE(rank0.needsCompensationForward());
    EXPECT_TRUE(rank0.readyForBarrier());
    EXPECT_EQ(rank0.forward_sequence, rank1.forward_sequence);

    rank0.beginReadyBarrier();
    rank1.beginReadyBarrier();
    EXPECT_EQ(rank0.phase, Alignment::Phase::READY_BARRIER);
    EXPECT_EQ(rank1.phase, Alignment::Phase::READY_BARRIER);
}

TEST(CollectiveSleepQuiesceAlignmentTest, ResetsForTwoConsecutiveSleepsWithOppositeSkew) {
    using Alignment = NormalEngine::CollectiveQuiesceAlignment;
    Alignment rank0;
    Alignment rank1;

    auto align_cycle = [](Alignment& early, Alignment& late, int late_extra_forwards) {
        early.beginTargetExchange();
        for (int i = 0; i < late_extra_forwards; ++i) {
            late.noteForwardEnqueued();
        }
        late.beginTargetExchange();

        const auto target = std::max(early.forward_sequence, late.forward_sequence);
        EXPECT_TRUE(early.acceptTargetSequence(target));
        EXPECT_TRUE(late.acceptTargetSequence(target));
        while (early.needsCompensationForward()) {
            early.noteForwardEnqueued();
        }
        while (late.needsCompensationForward()) {
            late.noteForwardEnqueued();
        }
        EXPECT_TRUE(early.readyForBarrier());
        EXPECT_TRUE(late.readyForBarrier());
        EXPECT_EQ(early.forward_sequence, late.forward_sequence);
        early.beginReadyBarrier();
        late.beginReadyBarrier();
        early.resetToConsensus();
        late.resetToConsensus();
    };

    for (int i = 0; i < 3; ++i) {
        rank0.noteForwardEnqueued();
        rank1.noteForwardEnqueued();
    }
    align_cycle(rank0, rank1, 2);
    const auto first_cycle_sequence = rank0.forward_sequence;
    EXPECT_EQ(rank0.phase, Alignment::Phase::CONSENSUS);
    EXPECT_EQ(rank1.phase, Alignment::Phase::CONSENSUS);

    rank0.noteForwardEnqueued();
    rank1.noteForwardEnqueued();
    align_cycle(rank1, rank0, 1);
    EXPECT_GT(rank0.forward_sequence, first_cycle_sequence);
    EXPECT_EQ(rank0.forward_sequence, rank1.forward_sequence);
    EXPECT_EQ(rank0.phase, Alignment::Phase::CONSENSUS);
    EXPECT_EQ(rank1.phase, Alignment::Phase::CONSENSUS);
}

class CudaGraphLifecycleMockModel: public ModelBase {
public:
    GptModelOutputs forward(const GptModelInputs&) override {
        return {};
    }

    void invalidateCudaGraphs() override {
        ++invalidate_count;
    }

    void recaptureCudaGraphs() override {
        ++recapture_count;
    }

    int invalidate_count{0};
    int recapture_count{0};
};

TEST_F(NormalEngineTest, testCudaGraphLifecyclePropagatesForTwoCycles) {
    CustomConfig config;
    auto         engine = createMockEngine(config);
    ASSERT_TRUE(engine->stop().ok());

    auto* executor = dynamic_cast<NormalExecutor*>(engine->executor_.get());
    ASSERT_NE(executor, nullptr);
    auto  model      = std::make_unique<CudaGraphLifecycleMockModel>();
    auto* model_view = model.get();
    executor->setModel(std::move(model));

    for (int cycle = 1; cycle <= 2; ++cycle) {
        engine->invalidateCudaGraphs();
        EXPECT_EQ(model_view->invalidate_count, cycle);
        engine->recaptureCudaGraphs();
        EXPECT_EQ(model_view->recapture_count, cycle);
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

TEST_F(NormalEngineTest, testTp1PauseQuiesceDrainsBeforeAck) {
    CustomConfig config;
    auto         engine = createMockEngine(config);
    ASSERT_TRUE(engine->stop().ok());

    auto  executor                        = std::make_unique<PauseDrainTrackingExecutor>();
    auto* view                            = executor.get();
    engine->executor_                     = std::move(executor);
    engine->parallelism_config.tp_size    = 1;
    engine->parallelism_config.tp_rank    = 0;
    engine->parallelism_config.dp_size    = 1;
    engine->parallelism_config.ep_size    = 1;
    engine->parallelism_config.world_size = 1;

    engine->pause();
    const auto epoch = engine->pause_epoch_.load();
    view->on_drain   = [&] { EXPECT_LT(engine->quiesced_pause_epoch_, epoch); };

    const auto status = engine->quiesceExecutorForPause(epoch);
    ASSERT_TRUE(status.ok()) << status.ToString();
    engine->completePauseQuiesce(epoch, status);

    EXPECT_EQ(view->events, std::vector<std::string>({"drain"}));
    EXPECT_EQ(engine->quiesced_pause_epoch_, epoch);
}

TEST_F(NormalEngineTest, testPureTpPauseWavePrecedesRunnerDrain) {
    CustomConfig config;
    auto         engine = createMockEngine(config);
    ASSERT_TRUE(engine->stop().ok());

    auto  executor                        = std::make_unique<PauseDrainTrackingExecutor>();
    auto* view                            = executor.get();
    engine->executor_                     = std::move(executor);
    engine->parallelism_config.tp_size    = 2;
    engine->parallelism_config.tp_rank    = 0;
    engine->parallelism_config.dp_size    = 1;
    engine->parallelism_config.ep_size    = 1;
    engine->parallelism_config.world_size = 2;

    engine->pause();
    const auto status = engine->quiesceExecutorForPause(engine->pause_epoch_.load());

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(view->events, std::vector<std::string>({"pause_wave", "drain"}));
}

TEST_F(NormalEngineTest, testRunnerDrainFailureFencesPauseEpoch) {
    CustomConfig config;
    auto         engine = createMockEngine(config);
    ASSERT_TRUE(engine->stop().ok());

    auto  executor                        = std::make_unique<PauseDrainTrackingExecutor>();
    auto* view                            = executor.get();
    view->drain_status                    = absl::InternalError("injected runner failure");
    engine->executor_                     = std::move(executor);
    engine->parallelism_config.tp_size    = 1;
    engine->parallelism_config.tp_rank    = 0;
    engine->parallelism_config.world_size = 1;

    engine->pause();
    const auto epoch  = engine->pause_epoch_.load();
    const auto status = engine->quiesceExecutorForPause(epoch);
    engine->completePauseQuiesce(epoch, status);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(engine->failed_pause_epoch_, epoch);
    EXPECT_LT(engine->quiesced_pause_epoch_, epoch);

    // A later successful retry must not overwrite the first failure for this epoch.
    engine->completePauseQuiesce(epoch, absl::OkStatus());
    EXPECT_EQ(engine->failed_pause_epoch_, epoch);
    EXPECT_LT(engine->quiesced_pause_epoch_, epoch);
}

TEST_F(NormalEngineTest, testMissingExecutorFailsQuiesceWithoutDereference) {
    CustomConfig config;
    auto         engine = createMockEngine(config);
    ASSERT_TRUE(engine->stop().ok());

    engine->executor_.reset();
    engine->pause();
    const auto status = engine->quiesceExecutorForPause(engine->pause_epoch_.load());

    EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
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
