#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/embedding_engine/EmbeddingScheduler.h"

namespace rtp_llm {
namespace {

EmbeddingStreamPtr makeStream(int64_t request_id, const std::vector<int32_t>& lengths) {
    int32_t total = 0;
    for (int32_t len : lengths) {
        total += len;
    }
    std::vector<int32_t> token_ids(static_cast<size_t>(total), 101);
    std::vector<int32_t> token_type_ids(static_cast<size_t>(total), 0);
    auto                 input = std::make_shared<EmbeddingInput>(token_ids, token_type_ids, lengths, request_id);
    return std::make_shared<EmbeddingStream>(input);
}

ModelConfig makeEmbeddingModelConfig(DataType data_type, int64_t max_seq_len = 8194, size_t head_num = 16) {
    ModelConfig model_config;
    model_config.data_type                = data_type;
    model_config.max_seq_len              = max_seq_len;
    model_config.attn_config.head_num     = head_num;
    model_config.attn_config.kv_head_num  = head_num;
    model_config.attn_config.size_per_head = 64;
    return model_config;
}

RuntimeConfig makeRuntimeConfig(int64_t max_context_batch_size = 1, int64_t max_generate_batch_size = 128) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = max_generate_batch_size;
    runtime_config.fifo_scheduler_config.max_context_batch_size = max_context_batch_size;
    return runtime_config;
}

FMHAConfig makeDisabledDenseFmhaConfig() {
    FMHAConfig fmha_config;
    fmha_config.enable_flashinfer_trtllm_gen        = false;
    fmha_config.enable_flashinfer_trt_fmha_v2       = false;
    fmha_config.enable_paged_flashinfer_trt_fmha_v2 = false;
    fmha_config.enable_open_source_fmha             = false;
    fmha_config.disable_flashinfer_native           = true;
    return fmha_config;
}

std::vector<int64_t> collectStreamIds(const std::list<EmbeddingStreamPtr>& batch) {
    std::vector<int64_t> ids;
    ids.reserve(batch.size());
    for (const auto& stream : batch) {
        ids.push_back(stream->streamId());
    }
    return ids;
}

TEST(EmbeddingSchedulerBatchGuardTest, SplitsProductionFp32SkewedBatchBefore19GbQkOutput) {
    const auto         model_config   = makeEmbeddingModelConfig(DataType::TYPE_FP32, /*max_seq_len=*/8194, /*head_num=*/16);
    const auto         runtime_config = makeRuntimeConfig(/*max_context_batch_size=*/1, /*max_generate_batch_size=*/128);
    ConcurrencyConfig  concurrency_config;
    FMHAConfig         fmha_config;
    EmbeddingScheduler scheduler(model_config, concurrency_config, runtime_config, nullptr, fmha_config);

    // Real skewed length sequence reproduced from the production FP32 BGE-M3 OOM batch:
    // total length of items 0..10 is 6060 <= 8194, yet co-batching item 10 (5317) yields
    // [11, 16, 5317, 5317] FP32 qk_output = 19.90 GB (> 17.5 GB free VRAM).
    const std::vector<int32_t> lengths = {64, 193, 103, 75, 71, 71, 71, 69, 66, 58, 5317, 93};
    for (size_t i = 0; i < lengths.size(); ++i) {
        ASSERT_TRUE(scheduler.enqueue(makeStream(static_cast<int64_t>(i), {lengths[i]})).ok());
    }

    auto batch1 = scheduler.scheduleNew();
    ASSERT_TRUE(batch1.ok());
    EXPECT_EQ(collectStreamIds(*batch1), (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

    auto batch2 = scheduler.scheduleNew();
    ASSERT_TRUE(batch2.ok());
    EXPECT_EQ(collectStreamIds(*batch2), (std::vector<int64_t>{10}));

    auto batch3 = scheduler.scheduleNew();
    ASSERT_TRUE(batch3.ok());
    EXPECT_EQ(collectStreamIds(*batch3), (std::vector<int64_t>{11}));
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST(EmbeddingSchedulerBatchGuardTest, AllowsStandaloneLongFp32StreamWithinMaxSeqLen) {
    const auto         model_config   = makeEmbeddingModelConfig(DataType::TYPE_FP32, /*max_seq_len=*/8194, /*head_num=*/16);
    const auto         runtime_config = makeRuntimeConfig(/*max_context_batch_size=*/4, /*max_generate_batch_size=*/128);
    ConcurrencyConfig  concurrency_config;
    FMHAConfig         fmha_config;
    EmbeddingScheduler scheduler(model_config, concurrency_config, runtime_config, nullptr, fmha_config);

    ASSERT_TRUE(scheduler.enqueue(makeStream(100, {5317})).ok());
    ASSERT_TRUE(scheduler.enqueue(makeStream(101, {64})).ok());

    auto batch1 = scheduler.scheduleNew();
    ASSERT_TRUE(batch1.ok());
    EXPECT_EQ(collectStreamIds(*batch1), (std::vector<int64_t>{100}));

    auto batch2 = scheduler.scheduleNew();
    ASSERT_TRUE(batch2.ok());
    EXPECT_EQ(collectStreamIds(*batch2), (std::vector<int64_t>{101}));
}

TEST(EmbeddingSchedulerBatchGuardTest, PreservesBatchingForShortFp32Streams) {
    const auto         model_config   = makeEmbeddingModelConfig(DataType::TYPE_FP32, /*max_seq_len=*/8194, /*head_num=*/16);
    const auto         runtime_config = makeRuntimeConfig(/*max_context_batch_size=*/1, /*max_generate_batch_size=*/128);
    ConcurrencyConfig  concurrency_config;
    FMHAConfig         fmha_config;
    EmbeddingScheduler scheduler(model_config, concurrency_config, runtime_config, nullptr, fmha_config);

    for (int i = 0; i < 20; ++i) {
        ASSERT_TRUE(scheduler.enqueue(makeStream(i, {100})).ok());
    }

    auto batch = scheduler.scheduleNew();
    ASSERT_TRUE(batch.ok());
    EXPECT_EQ(batch->size(), 20U);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST(EmbeddingSchedulerBatchGuardTest, EnforcesPaddedTokenBudgetInFp16FmhaMode) {
    const auto         model_config   = makeEmbeddingModelConfig(DataType::TYPE_FP16, /*max_seq_len=*/8194, /*head_num=*/16);
    const auto         runtime_config = makeRuntimeConfig(/*max_context_batch_size=*/1, /*max_generate_batch_size=*/128);
    ConcurrencyConfig  concurrency_config;
    FMHAConfig         fmha_config;
    EmbeddingScheduler scheduler(model_config, concurrency_config, runtime_config, nullptr, fmha_config);

    // Under FP16 + default FMHA, 3 x 2000 fits padded token budget (6000 <= 8194) and does not hit unfused limit.
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(scheduler.enqueue(makeStream(i, {2000})).ok());
    }
    auto medium_batch = scheduler.scheduleNew();
    ASSERT_TRUE(medium_batch.ok());
    EXPECT_EQ(collectStreamIds(*medium_batch), (std::vector<int64_t>{0, 1, 2}));

    // Co-batching 64 and 5317 has padded tokens 2 * 5317 = 10634 > 8194, so scheduler splits them.
    ASSERT_TRUE(scheduler.enqueue(makeStream(10, {64})).ok());
    ASSERT_TRUE(scheduler.enqueue(makeStream(11, {5317})).ok());

    auto batch1 = scheduler.scheduleNew();
    ASSERT_TRUE(batch1.ok());
    EXPECT_EQ(collectStreamIds(*batch1), (std::vector<int64_t>{10}));

    auto batch2 = scheduler.scheduleNew();
    ASSERT_TRUE(batch2.ok());
    EXPECT_EQ(collectStreamIds(*batch2), (std::vector<int64_t>{11}));
}

TEST(EmbeddingSchedulerBatchGuardTest, ActivatesUnfusedGuardWhenAllDenseFmhaBackendsDisabled) {
    const auto        model_config   = makeEmbeddingModelConfig(DataType::TYPE_FP16, /*max_seq_len=*/8194, /*head_num=*/16);
    const auto        runtime_config = makeRuntimeConfig(/*max_context_batch_size=*/4, /*max_generate_batch_size=*/128);
    ConcurrencyConfig concurrency_config;
    const auto        fmha_config = makeDisabledDenseFmhaConfig();
    EmbeddingScheduler scheduler(model_config, concurrency_config, runtime_config, nullptr, fmha_config);

    // 2 x 2500 padded tokens = 5000 <= 4 * 8194, so padded budget alone would allow co-batching,
    // but unfused score buffer = 2 * 16 * 2500^2 * 4 = 800 MB > 512 MB activates the guard.
    ASSERT_TRUE(scheduler.enqueue(makeStream(0, {2500})).ok());
    ASSERT_TRUE(scheduler.enqueue(makeStream(1, {2500})).ok());

    auto batch1 = scheduler.scheduleNew();
    ASSERT_TRUE(batch1.ok());
    EXPECT_EQ(collectStreamIds(*batch1), (std::vector<int64_t>{0}));

    auto batch2 = scheduler.scheduleNew();
    ASSERT_TRUE(batch2.ok());
    EXPECT_EQ(collectStreamIds(*batch2), (std::vector<int64_t>{1}));
}

TEST(EmbeddingSchedulerBatchGuardTest, OversizedFirstStreamIsRejectedAndWakesWaitFinish) {
    const auto         model_config   = makeEmbeddingModelConfig(DataType::TYPE_FP32, /*max_seq_len=*/8194, /*head_num=*/16);
    const auto         runtime_config = makeRuntimeConfig(/*max_context_batch_size=*/1, /*max_generate_batch_size=*/128);
    ConcurrencyConfig  concurrency_config;
    FMHAConfig         fmha_config;
    EmbeddingScheduler scheduler(model_config, concurrency_config, runtime_config, nullptr, fmha_config);

    auto oversized = makeStream(200, {9000});
    auto waiter    = std::async(std::launch::async, [oversized]() { oversized->waitFinish(); });

    ASSERT_TRUE(scheduler.enqueue(oversized).ok());
    auto batch = scheduler.scheduleNew();
    ASSERT_TRUE(batch.ok());
    EXPECT_TRUE(batch->empty());
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);

    try {
        waiter.get();
        FAIL() << "Expected std::runtime_error from waitFinish()";
    } catch (const std::runtime_error& err) {
        EXPECT_NE(std::string(err.what()).find("long prompt error"), std::string::npos);
    }
}

}  // namespace
}  // namespace rtp_llm
