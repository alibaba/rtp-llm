#include <gtest/gtest.h>
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {
class MtpStreamCommitTest : public DeviceTestBase {
protected:
    std::shared_ptr<NormalGenerateStream> makeStream(int max_new_tokens) {
        ModelConfig model;
        model.max_seq_len = 32;
        model.vocab_size = 128;
        model.special_tokens.eos_token_id = 99;
        RuntimeConfig runtime;
        auto cache = std::make_shared<KVCacheManager>(
            test::makeSimpleMhaCacheConfig(1, 9, 2, DataType::TYPE_INT8));
        cache->init();
        ResourceContext resource;
        resource.cache_manager = cache;
        auto input = std::make_shared<GenerateInput>();
        input->input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = max_new_tokens;
        input->generate_config->is_streaming = true;
        input->generate_config->aux_info = true;
        input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = std::make_shared<NormalGenerateStream>(input, model, runtime, resource, nullptr);
        auto speculative = std::make_shared<SpeculativeExecutorStreamOutput>();
        speculative->tokens = torch::zeros({1, 4}, torch::kInt32);
        stream->setSPOutputBuffer(speculative);
        return stream;
    }

    void update(const std::shared_ptr<NormalGenerateStream>& stream,
                std::vector<int32_t> tokens, int accepted, int valid, bool forced = false) {
        const int before = stream->seqLength();
        auto tensor = torch::tensor(tokens, torch::kInt32).reshape({1, static_cast<int64_t>(tokens.size())});
        StreamSpecUpdateInfo info{tensor, static_cast<int>(tokens.size()), -1, {}, {}};
        info.speculative_propose_step = 3;
        info.accepted_draft_tokens = accepted;
        info.speculative_round = SpeculativeRoundMetrics{1, before, 3, valid, accepted, forced};
        stream->specUpdate(info);
    }
};

TEST_F(MtpStreamCommitTest, EveryAcceptanceLengthExcludesBonusAndPreservesLegacyFields) {
    for (int accepted = 0; accepted <= 3; ++accepted) {
        auto stream = makeStream(16);
        std::vector<int32_t> tokens;
        for (int i = 0; i <= accepted; ++i) tokens.push_back(10 + i);
        update(stream, tokens, accepted, 3);
        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        const auto& aux = output.value().generate_outputs.at(0).aux_info;
        EXPECT_EQ(aux.speculative_metrics.at("accepted_draft_tokens"), accepted);
        EXPECT_EQ(aux.speculative_metrics.at("verified_candidates"), 3);
        EXPECT_EQ(aux.speculative_metrics.at("generated_tokens"), accepted + 1);
        EXPECT_EQ(aux.speculative_metrics.at("accept_length_" + std::to_string(accepted)), 1);
        EXPECT_EQ(aux.speculative_draft_rounds, 1);
    }
}

TEST_F(MtpStreamCommitTest, AcceptedEosDropsUncommittedSuffix) {
    auto stream = makeStream(10);
    update(stream, {4, 99, 5, 6}, 3, 2);
    EXPECT_EQ(stream->seqLength(), 5);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    const auto& aux = output.value().generate_outputs.at(0).aux_info;
    EXPECT_EQ(aux.speculative_metrics.at("accepted_draft_tokens"), 2);
    EXPECT_EQ(aux.speculative_metrics.at("generated_tokens"), 2);
    EXPECT_EQ(aux.speculative_draft_rounds, 1);
    EXPECT_EQ(stream->getSPOutputBuffer()->tokens[0][0].item<int32_t>(), 99);
    // Legacy positional counters retain their raw-verifier contract.
    EXPECT_EQ(aux.speculative_accepted_tokens_per_pos, (std::vector<int32_t>{1, 1, 1}));
}

TEST_F(MtpStreamCommitTest, MaxLengthRetainsAcceptedDraftInsteadOfSubtractingBonus) {
    auto stream = makeStream(1);
    update(stream, {4, 5, 6, 7}, 3, 1);
    EXPECT_EQ(stream->seqLength(), 4);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    const auto& aux = output.value().generate_outputs.at(0).aux_info;
    EXPECT_EQ(aux.speculative_metrics.at("accepted_draft_tokens"), 1);
    EXPECT_EQ(aux.speculative_metrics.at("generated_tokens"), 1);
    EXPECT_EQ(stream->getSPOutputBuffer()->tokens[0][0].item<int32_t>(), 4);
}

TEST_F(MtpStreamCommitTest, EosWinsWhenTheSameUpdateAlsoReachesMaxLength) {
    auto stream = makeStream(4);
    update(stream, {4, 99, 5, 6}, 3, 2);
    EXPECT_EQ(stream->seqLength(), 5);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    const auto& aux = output.value().generate_outputs.at(0).aux_info;
    EXPECT_EQ(aux.speculative_metrics.at("generated_tokens"), 2);
}

TEST_F(MtpStreamCommitTest, LateVerifyCannotAppendAfterPendingGenerateDone) {
    auto stream = makeStream(10);
    stream->generate_status_->status.store(StreamState::RUNNING);
    stream->generateConfig()->stop_words_list = {{4, 5}};

    update(stream, {4, 5}, 1, 3);
    ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    ASSERT_EQ(stream->getStatus(), StreamState::RUNNING);
    auto terminal = stream->nextOutput(10);
    ASSERT_TRUE(terminal.ok());
    ASSERT_TRUE(terminal.value().generate_outputs.at(0).finished);
    ASSERT_EQ(terminal.value().generate_outputs.at(0).aux_info.output_len, 2);

    update(stream, {6}, 0, 3);
    EXPECT_EQ(stream->seqLength(), 5);
    EXPECT_EQ(stream->getSPOutputBuffer()->tokens[0][0].item<int32_t>(), 5);
    auto next = stream->nextOutput(10);
    ASSERT_FALSE(next.ok());
    EXPECT_EQ(next.status().code(), ErrorCode::FINISHED);
}

TEST_F(MtpStreamCommitTest, ForcedFixtureNeverEntersNormalAcceptanceTotals) {
    auto stream = makeStream(4);
    update(stream, {4, 5, 6, 7}, 3, 3, true);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    const auto& metrics = output.value().generate_outputs.at(0).aux_info.speculative_metrics;
    EXPECT_EQ(metrics.at("excluded_forced_rounds"), 1);
    EXPECT_EQ(metrics.at("verify_rounds"), 0);
    EXPECT_EQ(metrics.at("verified_candidates"), 0);
}
}  // namespace rtp_llm
