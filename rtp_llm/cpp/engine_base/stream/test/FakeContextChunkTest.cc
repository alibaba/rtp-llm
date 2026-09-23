#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "gtest/gtest.h"

#include <exception>
#include <memory>
#include <vector>

namespace rtp_llm {
namespace {

ModelConfig fakeModelConfig() {
    ModelConfig model;
    model.max_seq_len                         = 512;
    model.vocab_size                          = 128;
    model.hidden_size                         = 8;
    model.attn_config.tokens_per_block        = 256;
    model.attn_config.kernel_tokens_per_block = 256;
    model.special_tokens.eos_token_id         = -1;
    return model;
}

std::shared_ptr<NormalGenerateStream> makeContext(const std::vector<int32_t>& tokens, bool fast_gen, bool fake) {
    auto input                             = std::make_shared<GenerateInput>();
    input->input_ids                       = torch::tensor(tokens, torch::kInt32);
    input->generate_config                 = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 8;
    input->generate_config->top_k          = 1;
    input->fake_query                      = fake;
    RuntimeConfig runtime;
    runtime.fifo_scheduler_config.enable_fast_gen          = fast_gen;
    runtime.fifo_scheduler_config.fast_gen_max_context_len = 4;
    auto stream = std::make_shared<NormalGenerateStream>(input, fakeModelConfig(), runtime, ResourceContext{}, nullptr);
    stream->setIsFakeStream(fake);
    stream->fakeInitKVBlock();
    return stream;
}

TEST(FakeContextChunkTest, InitializesExecutableContextWithoutChangingTokens) {
    for (const bool fast_gen : {false, true}) {
        for (const auto& tokens : {std::vector<int32_t>{7}, std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7}}) {
            auto stream = makeContext(tokens, fast_gen, true);
            EXPECT_EQ(stream->contextLength(), fast_gen ? 0 : tokens.size());
            stream->initFakeContextChunk();
            EXPECT_TRUE(stream->isFakeStream());
            EXPECT_TRUE(stream->isContextStream());
            EXPECT_EQ(stream->enable_fast_gen_, fast_gen);
            EXPECT_FALSE(stream->isChunkStream());
            EXPECT_EQ(stream->prefixLength(), 0);
            EXPECT_EQ(stream->contextLength(), tokens.size());
            EXPECT_EQ(stream->seqLength(), tokens.size());
            EXPECT_EQ(stream->currentExecuteTokens(0), std::vector<int>(tokens.begin(), tokens.end()));
        }
    }
}

TEST(FakeContextChunkTest, MtpProductionPrefillFactoryInitializesFastGen) {
    for (const bool fast_gen : {false, true}) {
        RuntimeConfig runtime;
        runtime.fifo_scheduler_config.enable_fast_gen          = fast_gen;
        runtime.fifo_scheduler_config.fast_gen_max_context_len = 4;
        auto stream = MtpExecutor::createMinFakePrefillStream(3, fakeModelConfig(), runtime, ResourceContext{});
        EXPECT_TRUE(stream->isFakeStream());
        EXPECT_TRUE(stream->isContextStream());
        EXPECT_EQ(stream->enable_fast_gen_, fast_gen);
        EXPECT_FALSE(stream->isChunkStream());
        EXPECT_EQ(stream->inputLength(), 1);
        EXPECT_EQ(stream->contextLength(), 1);
        EXPECT_EQ(stream->prefixLength(), 0);
        EXPECT_EQ(stream->seqLength(), 1);
        EXPECT_EQ(stream->currentExecuteTokens(0), std::vector<int>({0}));
    }
}

TEST(FakeContextChunkTest, ReinitializationDoesNotAdvanceOrSkipInput) {
    auto stream = makeContext({1, 2, 3}, true, true);
    stream->initFakeContextChunk();
    stream->initFakeContextChunk();
    EXPECT_EQ(stream->prefixLength(), 0);
    EXPECT_EQ(stream->contextLength(), 3);
    EXPECT_EQ(stream->seqLength(), 3);
    EXPECT_EQ(stream->currentExecuteTokens(0), std::vector<int>({1, 2, 3}));
}

TEST(FakeContextChunkTest, RejectsRealRequestWithoutChangingItsChunkCursor) {
    auto stream = makeContext({1, 2, 3, 4, 5}, true, false);
    EXPECT_THROW(stream->initFakeContextChunk(), std::exception);
    EXPECT_EQ(stream->currentChunkLen(), 0);
    EXPECT_EQ(stream->prefixLength(), 0);
    auto capacity = stream->acquireCapacity(3);
    ASSERT_TRUE(capacity.ok());
    EXPECT_EQ(capacity.value(), 3);
    EXPECT_EQ(stream->currentExecuteTokens(0), std::vector<int>({1, 2, 3}));
}

TEST(FakeContextChunkTest, RejectsDecodePlaceholderWithoutChangingPhase) {
    auto stream = makeContext({7}, true, true);
    stream->setIsContextStream(false);
    EXPECT_THROW(stream->initFakeContextChunk(), std::exception);
    EXPECT_FALSE(stream->isContextStream());
    EXPECT_EQ(stream->currentChunkLen(), 0);
    EXPECT_EQ(stream->seqLength(), 1);
}

TEST(FakeContextChunkTest, RejectsPartiallyScheduledFakeContext) {
    auto stream = makeContext({1, 2, 3, 4, 5}, true, true);
    ASSERT_TRUE(stream->acquireCapacity(2).ok());
    EXPECT_THROW(stream->initFakeContextChunk(), std::exception);
    EXPECT_EQ(stream->currentChunkLen(), 2);
    EXPECT_EQ(stream->contextLength(), 2);
    EXPECT_EQ(stream->currentExecuteTokens(0), std::vector<int>({1, 2}));
}

TEST(FakeContextChunkTest, RejectsAlreadyGeneratedFakeContext) {
    auto stream = makeContext({7}, true, true);
    stream->setSeqLength(2);
    EXPECT_THROW(stream->initFakeContextChunk(), std::exception);
    EXPECT_EQ(stream->currentChunkLen(), 0);
}

TEST(FullContextInitializationTest, RealRequestsExposeTheWholePrompt) {
    for (const bool fast_gen : {false, true}) {
        for (const auto& tokens : {std::vector<int32_t>{7}, std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7}}) {
            auto stream = makeContext(tokens, fast_gen, false);
            stream->initNonChunkedContext();
            EXPECT_FALSE(stream->isFakeStream());
            EXPECT_TRUE(stream->isContextStream());
            EXPECT_EQ(stream->enable_fast_gen_, fast_gen);
            EXPECT_FALSE(stream->isChunkStream());
            EXPECT_EQ(stream->prefixLength(), 0);
            EXPECT_EQ(stream->contextLength(), tokens.size());
            EXPECT_EQ(stream->currentExecuteTokens(), std::vector<int>(tokens.begin(), tokens.end()));
        }
    }
}

TEST(FullContextInitializationTest, ReinitializationDoesNotAdvanceThePrompt) {
    auto stream = makeContext({1, 2, 3, 4, 5, 6, 7}, true, false);
    stream->initNonChunkedContext();
    stream->initNonChunkedContext();
    EXPECT_EQ(stream->contextLength(), 7);
    EXPECT_EQ(stream->prefixLength(), 0);
    EXPECT_EQ(stream->seqLength(), 7);
}

TEST(FullContextInitializationTest, PreservesAReusedPrefix) {
    auto stream = makeContext({1, 2, 3, 4, 5, 6, 7}, true, false);
    stream->setReuseLength(3);
    stream->initNonChunkedContext();
    stream->initNonChunkedContext();
    EXPECT_EQ(stream->reuseLength(), 3);
    EXPECT_EQ(stream->prefixLength(), 3);
    EXPECT_EQ(stream->contextLength(), 4);
    EXPECT_EQ(stream->currentExecuteTokens(), std::vector<int>({4, 5, 6, 7}));
}

TEST(FullContextInitializationTest, RejectsPartiallyScheduledContext) {
    auto stream = makeContext({1, 2, 3, 4, 5, 6, 7}, true, false);
    ASSERT_TRUE(stream->acquireCapacity(3).ok());
    EXPECT_THROW(stream->initNonChunkedContext(), std::exception);
    EXPECT_EQ(stream->currentChunkLen(), 3);
    EXPECT_EQ(stream->contextLength(), 3);
}

TEST(FullContextInitializationTest, RejectsDecodeAndConsumedContext) {
    auto decode = makeContext({1, 2, 3}, true, false);
    decode->setIsContextStream(false);
    EXPECT_THROW(decode->initNonChunkedContext(), std::exception);
    auto consumed = makeContext({1, 2, 3}, true, false);
    consumed->setSeqLength(4);
    EXPECT_THROW(consumed->initNonChunkedContext(), std::exception);
}

TEST(FullContextInitializationTest, DoesNotActivateOrAdvancePipelineChunksImplicitly) {
    auto stream = makeContext({1, 2, 3, 4, 5, 6, 7}, true, false);
    EXPECT_EQ(stream->contextLength(), 0);
    ASSERT_TRUE(stream->acquireNextChunk().ok());
    EXPECT_EQ(stream->contextLength(), 4);
    EXPECT_EQ(stream->currentExecuteTokens(), std::vector<int>({1, 2, 3, 4}));
}

}  // namespace
}  // namespace rtp_llm
