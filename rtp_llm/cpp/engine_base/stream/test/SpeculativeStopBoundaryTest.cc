#include "gtest/gtest.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {
namespace {
std::shared_ptr<NormalGenerateStream>
makeStopStream(int maximum, int minimum = 0, bool ignore_eos = false, std::vector<std::vector<int>> stops = {}) {
    auto input                              = std::make_shared<GenerateInput>();
    input->input_ids                        = torch::tensor(std::vector<int32_t>{1, 2}, torch::kInt32);
    input->generate_config                  = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens  = maximum;
    input->generate_config->min_new_tokens  = minimum;
    input->generate_config->ignore_eos      = ignore_eos;
    input->generate_config->stop_words_list = std::move(stops);
    ModelConfig model;
    model.max_seq_len                 = 64;
    model.vocab_size                  = 128;
    model.special_tokens.eos_token_id = 99;
    return std::make_shared<NormalGenerateStream>(input, model, RuntimeConfig{}, ResourceContext{}, nullptr);
}

void appendAccepted(const std::shared_ptr<NormalGenerateStream>& stream, const std::vector<int32_t>& tokens) {
    // Real token history/update and real finish logic, CPU tensors only.
    auto accepted    = torch::tensor(tokens, torch::kInt32).reshape({1, static_cast<int64_t>(tokens.size())});
    int  error_token = -1;
    ASSERT_TRUE(stream->complete_token_ids_->update(
        accepted, 0, tokens.size(), stream->inputLength(), stream->maxTokenNum(), 128, false, 0, error_token));
}
}  // namespace

TEST(SpeculativeStopBoundary, EosBeforeCapTrimsAcceptedBatch) {
    auto stream = makeStopStream(4);
    appendAccepted(stream, {5, 6, 99, 7});
    ASSERT_TRUE(stream->needFinish());
    EXPECT_EQ(stream->seqLength(), stream->inputLength() + 3);
}

TEST(SpeculativeStopBoundary, EosBeforeCapTrimsClippedAcceptedBatch) {
    auto stream = makeStopStream(3);
    appendAccepted(stream, {5, 99, 6, 7});
    ASSERT_TRUE(stream->needFinish());
    EXPECT_EQ(stream->seqLength(), stream->inputLength() + 2);
}

TEST(SpeculativeStopBoundary, StopWordBeforeCapTrimsAcceptedBatch) {
    auto stream = makeStopStream(4, 0, false, {{5, 6}});
    appendAccepted(stream, {5, 6, 7, 8});
    ASSERT_TRUE(stream->needFinish());
    EXPECT_EQ(stream->seqLength(), stream->inputLength() + 2);
}

TEST(SpeculativeStopBoundary, MinimumExcludesEarlierEosInSameBatch) {
    auto stream = makeStopStream(4, 4);
    appendAccepted(stream, {5, 99, 6, 7});
    ASSERT_TRUE(stream->needFinish());
    EXPECT_EQ(stream->seqLength(), stream->inputLength() + 4);
}

TEST(SpeculativeStopBoundary, MinimumExcludesEarlierStopWordInSameBatch) {
    auto stream = makeStopStream(4, 4, false, {{5, 6}});
    appendAccepted(stream, {5, 6, 7, 8});
    ASSERT_TRUE(stream->needFinish());
    EXPECT_EQ(stream->seqLength(), stream->inputLength() + 4);
}

TEST(SpeculativeStopBoundary, IgnoreEosKeepsLengthLimit) {
    auto stream = makeStopStream(4, 0, true, {{99}});
    appendAccepted(stream, {5, 99, 6, 7});
    ASSERT_TRUE(stream->needFinish());
    EXPECT_EQ(stream->seqLength(), stream->inputLength() + 4);
}

TEST(SpeculativeStopBoundary, EosAtMinimumIsAccepted) {
    auto stream = makeStopStream(4, 2);
    appendAccepted(stream, {5, 99, 6, 7});
    ASSERT_TRUE(stream->needFinish());
    EXPECT_EQ(stream->seqLength(), stream->inputLength() + 2);
}

TEST(SpeculativeStopBoundary, NoStopKeepsLengthLimit) {
    auto stream = makeStopStream(4);
    appendAccepted(stream, {5, 6, 7, 8});
    ASSERT_TRUE(stream->needFinish());
    EXPECT_EQ(stream->seqLength(), stream->inputLength() + 4);
}

TEST(SpeculativeStopBoundary, NonterminalBatchDoesNotFinish) {
    auto stream = makeStopStream(8);
    appendAccepted(stream, {5, 6, 7, 8});
    EXPECT_FALSE(stream->needFinish());
    EXPECT_EQ(stream->seqLength(), stream->inputLength() + 4);
}
}  // namespace rtp_llm
