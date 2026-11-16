#include <gtest/gtest.h>

#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
namespace {

GenerateStreamPtr
makeStream(int length, int reuse, int chunk, RoleType role = RoleType::PREFILL, int return_sequences = 1) {
    auto input                                        = std::make_shared<GenerateInput>();
    input->input_ids                                  = torch::arange(length, torch::kInt32);
    input->generate_config                            = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens            = 8;
    input->generate_config->num_return_sequences      = return_sequences;
    input->generate_config->ignore_eos                = true;
    input->generate_config->frontend_metric_streaming = true;
    ModelConfig model;
    model.max_seq_len = 128;
    model.vocab_size  = 128;
    RuntimeConfig runtime;
    runtime.fifo_scheduler_config.prefill_chunk_size    = chunk;
    runtime.fifo_scheduler_config.max_batch_tokens_size = 64;
    ResourceContext resource;
    resource.role_type = role;
    auto stream        = std::make_shared<NormalGenerateStream>(input, model, runtime, resource, nullptr);
    stream->setReuseLength(reuse);
    stream->setInitialReuseLength(reuse);
    stream->generate_status_->status = StreamState::RUNNING;
    return stream;
}

void finishStep(const GenerateStreamPtr& stream, bool speculative) {
    auto token = torch::tensor({{101}}, torch::kInt32);
    if (speculative) {
        auto buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
        buffer->tokens = torch::zeros({2}, torch::kInt32);
        stream->setSPOutputBuffer(buffer);
        stream->specUpdate({token, 1, 42, torch::Tensor(), torch::Tensor()});
    } else {
        stream->update({token, 1});
    }
}

TEST(ChunkPrefillMetricsTest, CountsPromptOnceAcrossChunksAndPrefixHits) {
    for (auto role : {RoleType::PREFILL, RoleType::PDFUSION}) {
        for (bool speculative : {false, true}) {
            for (int reuse : {0, 3, 31}) {
                for (int chunk : {4, 8, 64}) {
                    SCOPED_TRACE(::testing::Message()
                                 << int(role) << "/" << speculative << "/" << reuse << "/" << chunk);
                    auto                          stream = makeStream(32, reuse, chunk, role);
                    RtpLLMTokenPSMetricsCollector collector;
                    int                           offset = reuse;
                    int                           rounds = 0;
                    while (stream->isContextStream()) {
                        ASSERT_LT(rounds, 10);
                        const int    count    = std::min(chunk, 32 - offset);
                        const int    credited = count + (rounds == 0 ? reuse : 0);
                        StreamGroups groups({stream});
                        EXPECT_EQ(stream->reuseLength(), offset);  // Attention still sees all prior chunks.
                        EXPECT_EQ(stream->currentExecuteTokens().front(), offset);
                        EXPECT_EQ(groups.contextExecuteTokenSize(), count);
                        EXPECT_EQ(groups.contextKvTokenSize(), offset + count);
                        EXPECT_EQ(groups.contextExecuteTokenSizeWithCache(), credited);
                        const auto by_priority = groups.tokenCountsByPriority().at(stream->priority());
                        EXPECT_EQ(by_priority.context, count);
                        EXPECT_EQ(by_priority.context_with_cache, credited);
                        EXPECT_EQ(by_priority.total, count);
                        groups.addContextExecuteTimeUs(1000000);
                        groups.addFrontendContextExecuteMetrics(1000000);
                        collector.addTokenSize(count, groups.contextExecuteTokenSizeWithCache(), 0, count, 1000000);
                        // Reading/constructing accounting views must not consume cache credit.
                        EXPECT_EQ(StreamGroups({stream}).contextExecuteTokenSizeWithCache(), credited);
                        finishStep(stream, speculative);
                        EXPECT_EQ(groups.contextExecuteTokenSizeWithCache(),
                                  credited);  // dispatch cannot alter snapshot
                        offset += count;
                        ++rounds;
                    }
                    EXPECT_EQ(collector.contextTokens(), 32 - reuse);
                    EXPECT_EQ(collector.contextTokensWithCache(), 32);
                    EXPECT_DOUBLE_EQ(collector.contextTPSWithCache(), 32.0 / rounds);
                    collector.setReportWindowUs(2000000);
                    EXPECT_DOUBLE_EQ(collector.contextWallTPSWithCache(), 16.0);
                    EXPECT_EQ(stream->frontend_context_token_num_, 32 - reuse);
                    EXPECT_EQ(stream->frontend_context_token_num_with_cache_, 32);
                    EXPECT_EQ(stream->contextExecuteTimeWithCacheUs(), rounds * 1000000);
                    EXPECT_EQ(stream->initialReuseLength(), reuse);
                    EXPECT_EQ(stream->seqLength(), 33);
                }
            }
        }
    }
}

TEST(ChunkPrefillMetricsTest, NonChunkedAndDecodeAccountingUnchanged) {
    auto         stream = makeStream(32, 8, 0);
    StreamGroups groups({stream});
    EXPECT_EQ(groups.contextExecuteTokenSize(), 24);
    EXPECT_EQ(groups.contextExecuteTokenSizeWithCache(), 32);
    // A non-chunked resumed/recomputed pass retains the existing accounting.
    stream->setReuseLength(16);
    EXPECT_EQ(StreamGroups({stream}).contextExecuteTokenSizeWithCache(), 32);
    finishStep(stream, false);
    StreamGroups decode({stream});
    EXPECT_EQ(decode.contextExecuteTokenSizeWithCache(), 0);
    EXPECT_EQ(decode.totalDecodeBatchSize(), 1);
    EXPECT_EQ(decode.tokenCountsByPriority().at(stream->priority()).generate, 1);
}

TEST(ChunkPrefillMetricsTest, DynamicChunkAndResumeDoNotRecreditHistory) {
    auto stream = makeStream(32, 4, 8);
    EXPECT_EQ(StreamGroups({stream}).contextExecuteTokenSizeWithCache(), 12);
    finishStep(stream, false);
    stream->setChunkSize(4);
    EXPECT_EQ(StreamGroups({stream}).contextExecuteTokenSizeWithCache(), 4);
    // Simulate eviction/re-admission requiring some actual recomputation.
    stream->setReuseLength(8);
    stream->setInitialReuseLength(8);
    EXPECT_EQ(stream->currentExecuteTokens().front(), 8);
    EXPECT_EQ(StreamGroups({stream}).contextExecuteTokenSize(), 4);
    EXPECT_EQ(StreamGroups({stream}).contextExecuteTokenSizeWithCache(), 4);
    // Disabling chunking for the remaining tail must not recredit the prefix.
    stream->setChunkSize(0);
    EXPECT_EQ(StreamGroups({stream}).contextExecuteTokenSizeWithCache(), 24);
}

TEST(ChunkPrefillMetricsTest, MixedBatchPriorityAndTimeAllocationUseSameCredit) {
    auto continued = makeStream(32, 4, 8);
    finishStep(continued, false);
    auto fresh = makeStream(32, 8, 8);
    EXPECT_EQ(StreamGroups({continued, fresh}).tokenCountsByPriority().at(continued->priority()).context_with_cache,
              24);
    continued->generate_input_->priority = 1;
    fresh->generate_input_->priority     = 2;
    StreamGroups groups({continued, fresh});
    EXPECT_EQ(groups.contextExecuteTokenSize(), 16);
    EXPECT_EQ(groups.contextExecuteTokenSizeWithCache(), 24);
    auto counts = groups.tokenCountsByPriority();
    EXPECT_EQ(counts.at(1).context_with_cache, 8);
    EXPECT_EQ(counts.at(2).context_with_cache, 16);
    groups.addContextExecuteTimeUs(240);
    EXPECT_EQ(continued->contextExecuteTimeUs(), 120);
    EXPECT_EQ(fresh->contextExecuteTimeUs(), 120);
    EXPECT_EQ(continued->contextExecuteTimeWithCacheUs(), 80);
    EXPECT_EQ(fresh->contextExecuteTimeWithCacheUs(), 160);
    groups.addFrontendContextExecuteMetrics(240);
    EXPECT_EQ(continued->frontend_context_token_num_with_cache_, 24);
    EXPECT_EQ(fresh->frontend_context_token_num_with_cache_, 0);
}

TEST(ChunkPrefillMetricsTest, MultipleReturnSequencesKeepBatchMultiplier) {
    for (int chunk : {0, 8}) {
        auto stream = makeStream(32, 4, chunk, RoleType::PREFILL, 3);
        ASSERT_FALSE(stream->hasError());
        EXPECT_EQ(stream->currentBatchSize(), 3);
        StreamGroups first({stream});
        EXPECT_EQ(first.contextExecuteTokenSize(), (chunk ? 8 : 28) * 3);
        EXPECT_EQ(first.contextExecuteTokenSizeWithCache(), (chunk ? 12 : 32) * 3);
        if (chunk) {
            // A middle chunk advances without sampling or tiling outputs.
            finishStep(stream, false);
            StreamGroups second({stream});
            EXPECT_EQ(second.contextExecuteTokenSizeWithCache(), 8 * 3);
            EXPECT_EQ(second.contextKvTokenSize(), 20 * 3);
            EXPECT_EQ(second.tokenCountsByPriority().at(stream->priority()).context_with_cache, 8 * 3);
        }
    }
}

TEST(ChunkPrefillMetricsTest, CancellationDoesNotAdvanceOrConsumeCacheCredit) {
    for (bool speculative : {false, true}) {
        auto stream = makeStream(32, 4, 8);
        stream->reportError(ErrorCode::CANCELLED, "cancel before chunk completion");
        finishStep(stream, speculative);
        EXPECT_EQ(stream->reuseLength(), 4);
        EXPECT_EQ(stream->seqLength(), 32);
        EXPECT_EQ(StreamGroups({stream}).contextExecuteTokenSizeWithCache(), 12);
    }
}

TEST(ChunkPrefillMetricsTest, ReportingWindowsDoNotResetPromptCredit) {
    auto    stream   = makeStream(32, 4, 8);
    int64_t reported = 0;
    int     round    = 0;
    while (stream->isContextStream()) {
        ASSERT_LT(round, 4);
        // Each window gets a fresh collector, but accounting belongs to the stream.
        RtpLLMTokenPSMetricsCollector window;
        StreamGroups                  groups({stream});
        window.addTokenSize(groups.contextExecuteTokenSize(),
                            groups.contextExecuteTokenSizeWithCache(),
                            0,
                            groups.modelExecuteTokenSize(),
                            1000000);
        const int expected = round == 0 ? 12 : (round == 3 ? 4 : 8);
        EXPECT_EQ(window.contextTokensWithCache(), expected);
        reported += window.contextTokensWithCache();
        finishStep(stream, false);
        ++round;
    }
    EXPECT_EQ(reported, 32);
    EXPECT_EQ(round, 4);
}

TEST(ChunkPrefillMetricsTest, MixedDecodeAndEmptyBatchDoNotReceivePrefixCredit) {
    auto prefill = makeStream(32, 4, 8);
    finishStep(prefill, false);
    auto decode = makeStream(32, 4, 0);
    finishStep(decode, false);
    StreamGroups groups({prefill, decode});
    EXPECT_EQ(groups.contextExecuteTokenSize(), 8);
    EXPECT_EQ(groups.contextExecuteTokenSizeWithCache(), 8);
    EXPECT_EQ(groups.contextKvTokenSize(), 20);
    EXPECT_EQ(groups.totalDecodeBatchSize(), 1);
    auto counts = groups.tokenCountsByPriority().at(prefill->priority());
    EXPECT_EQ(counts.context, 8);
    EXPECT_EQ(counts.context_with_cache, 8);
    EXPECT_EQ(counts.generate, 1);
    EXPECT_EQ(counts.total, 9);
    groups.addContextExecuteTimeUs(100);
    EXPECT_EQ(prefill->contextExecuteTimeWithCacheUs(), 100);
    EXPECT_EQ(decode->contextExecuteTimeWithCacheUs(), 0);
    StreamGroups empty({});
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.contextExecuteTokenSizeWithCache(), 0);
    EXPECT_EQ(empty.contextKvTokenSize(), 0);
    EXPECT_TRUE(empty.tokenCountsByPriority().empty());
    empty.addContextExecuteTimeUs(100);
    empty.addFrontendContextExecuteMetrics(100);
}

}  // namespace
}  // namespace rtp_llm
