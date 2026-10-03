#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "torch/all.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {

// Exercise the admission initializer directly: these regressions do not need
// cache allocation, a model, or CUDA tensors. The target enables private access
// only to install the already-selected stream cohort.
class BatchDecodeSchedulerRealOutputTest: public ::testing::Test {
protected:
    void SetUp() override {
        model_config_.max_seq_len                                                       = 128;
        model_config_.vocab_size                                                        = 256;
        model_config_.attn_config.tokens_per_block                                      = 128;
        runtime_config_.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 1;
    }

    std::shared_ptr<NormalGenerateStream> makeStream() {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
        input->need_release_resource           = false;
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 16;
        input->generate_config->ignore_eos     = true;
        input->generate_config->is_streaming   = true;
        auto stream =
            std::make_shared<NormalGenerateStream>(input, model_config_, runtime_config_, ResourceContext{}, nullptr);
        auto speculative    = std::make_shared<SpeculativeExecutorStreamOutput>();
        speculative->tokens = torch::full({1, 2}, -1, torch::kInt32);
        stream->setSPOutputBuffer(speculative);
        return stream;
    }

    void initialize(BatchDecodeScheduler& scheduler, const GenerateStreamPtr& stream) {
        scheduler.running_streams_ = {stream};
        scheduler.initRunningStreams();
    }

    void
    commitAndCheck(const std::shared_ptr<NormalGenerateStream>& stream, int sampled_token, int expected_output_token) {
        auto                       tokens     = torch::tensor({sampled_token}, torch::kInt32).reshape({1, 1});
        const auto                 old_length = stream->seqLength();
        const StreamSpecUpdateInfo update{tokens, 1, -1, torch::Tensor(), torch::Tensor()};
        stream->specUpdate(update);

        ASSERT_FALSE(stream->hasError());
        ASSERT_EQ(stream->seqLength(), old_length + 1);
        EXPECT_EQ(stream->completeTokenIds().index({0, old_length}).item<int32_t>(), expected_output_token);
        // Even the legacy zero-output control must not overwrite the recurrent
        // target anchor or the sampler-owned input tensor.
        EXPECT_EQ(stream->getSPOutputBuffer()->tokens.index({0, 0}).item<int32_t>(), sampled_token);
        EXPECT_EQ(tokens.item<int32_t>(), sampled_token);
        ASSERT_TRUE(stream->hasOutput());
        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs.size(), 1);
        const auto& ids = output.value().generate_outputs[0].output_ids;
        ASSERT_EQ(ids.numel(), 1);
        EXPECT_EQ(ids.item<int32_t>(), expected_output_token);
    }

    ModelConfig   model_config_;
    RuntimeConfig runtime_config_;
};

TEST_F(BatchDecodeSchedulerRealOutputTest, LegacyConstructorDefaultsToPerfAndDecode) {
    BatchDecodeScheduler scheduler(runtime_config_, nullptr, nullptr);
    auto                 stream = makeStream();
    stream->setPerfTest(false);
    ASSERT_TRUE(stream->isContextStream());
    // A WAITING stream with no CanRun event keeps moveToNext() cache-free.
    initialize(scheduler, stream);
    EXPECT_TRUE(stream->isPerfTest());
    EXPECT_FALSE(stream->isContextStream());
    EXPECT_FALSE(stream->hasError());
}

TEST_F(BatchDecodeSchedulerRealOutputTest, CoordinatedSchedulerDoesNotAdmitIncompleteBatch) {
    BatchDecodeScheduler scheduler(runtime_config_, nullptr, nullptr, 0, true);
    scheduler.updateSchedulerInfo(R"({"batch_size":2,"mode":"prefill","real_output":true})");
    auto       stream     = makeStream();
    const auto old_length = stream->seqLength();
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    auto scheduled = scheduler.schedule();
    ASSERT_TRUE(scheduled.ok());
    EXPECT_TRUE(scheduled.value().empty());
    EXPECT_EQ(scheduler.onflightStreams(), 1);
    EXPECT_EQ(stream->seqLength(), old_length);
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
    EXPECT_TRUE(stream->isContextStream());
}

TEST_F(BatchDecodeSchedulerRealOutputTest, PrefillRealOutputClearsPerfWithoutSkippingPrefill) {
    BatchDecodeScheduler scheduler(runtime_config_, nullptr, nullptr);
    scheduler.updateSchedulerInfo(R"({"batch_size":1,"mode":"prefill","real_output":true})");
    auto stream = makeStream();
    stream->setPerfTest(true);  // Admission must clear even an inherited perf flag.
    initialize(scheduler, stream);
    EXPECT_FALSE(stream->isPerfTest());
    EXPECT_TRUE(stream->isContextStream());
    EXPECT_FALSE(stream->hasError());
}

TEST_F(BatchDecodeSchedulerRealOutputTest, OmittedOrFalseRealOutputRestoresLegacyOnNextCohort) {
    BatchDecodeScheduler scheduler(runtime_config_, nullptr, nullptr);
    for (const auto* legacy_json :
         {R"({"batch_size":1,"mode":"prefill"})", R"({"batch_size":1,"mode":"prefill","real_output":false})"}) {
        scheduler.updateSchedulerInfo(R"({"batch_size":1,"mode":"prefill","real_output":true})");
        auto real_stream = makeStream();
        initialize(scheduler, real_stream);
        ASSERT_FALSE(real_stream->isPerfTest());

        scheduler.updateSchedulerInfo(legacy_json);
        auto legacy_stream = makeStream();
        legacy_stream->setPerfTest(false);
        initialize(scheduler, legacy_stream);
        EXPECT_TRUE(legacy_stream->isPerfTest());
        EXPECT_TRUE(legacy_stream->isContextStream());
    }
}

TEST_F(BatchDecodeSchedulerRealOutputTest, RealOutputSpecUpdatePreservesTokensAndNextAnchor) {
    BatchDecodeScheduler scheduler(runtime_config_, nullptr, nullptr);
    scheduler.updateSchedulerInfo(R"({"batch_size":1,"mode":"prefill","real_output":true})");
    auto stream = makeStream();
    initialize(scheduler, stream);
    ASSERT_TRUE(stream->isContextStream());
    ASSERT_FALSE(stream->isPerfTest());
    stream->generate_status_->status = StreamState::RUNNING;

    commitAndCheck(stream, 7, 7);  // Real target prefill anchor.
    ASSERT_FALSE(stream->isContextStream());
    commitAndCheck(stream, 11, 11);  // Next speculative accepted token.
    EXPECT_EQ(stream->getCompleteTokenIds()->completeTokenIdsVec(0), (std::vector<int>{1, 2, 3, 7, 11}));
}

TEST_F(BatchDecodeSchedulerRealOutputTest, LegacySpecUpdateStillZerosOutputButKeepsRealAnchor) {
    BatchDecodeScheduler scheduler(runtime_config_, nullptr, nullptr);
    scheduler.updateSchedulerInfo(R"({"batch_size":1,"mode":"prefill"})");
    auto stream = makeStream();
    initialize(scheduler, stream);
    ASSERT_TRUE(stream->isPerfTest());
    stream->generate_status_->status = StreamState::RUNNING;

    commitAndCheck(stream, 7, 0);
    commitAndCheck(stream, 11, 0);
    EXPECT_EQ(stream->getCompleteTokenIds()->completeTokenIdsVec(0), (std::vector<int>{1, 2, 3, 0, 0}));
}

}  // namespace rtp_llm
