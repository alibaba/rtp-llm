#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"

namespace rtp_llm {
namespace {

TEST(TransferStageStateTest, ImmediateCompletionWaitsForSubmitterToken) {
    size_t callback_count = 0;
    auto state = std::make_shared<TransferStageState>([&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        ++callback_count;
    });

    state->addBatch();
    state->completeBatch(ErrorInfo::OkStatus());
    EXPECT_EQ(callback_count, 0u);
    state->finishSubmitting();
    EXPECT_EQ(callback_count, 1u);
}

TEST(TransferStageStateTest, EmptyStageCompletesExactlyOnce) {
    size_t callback_count = 0;
    TransferStageState state([&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        ++callback_count;
    });

    state.finishSubmitting();
    state.finishSubmitting();
    EXPECT_EQ(callback_count, 1u);
}

TEST(TransferStageStateTest, ConcurrentBatchesCompleteExactlyOnce) {
    constexpr size_t kBatchCount = 32;
    std::atomic<size_t> callback_count{0};
    auto state = std::make_shared<TransferStageState>([&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        ++callback_count;
    });
    for (size_t index = 0; index < kBatchCount; ++index) {
        state->addBatch();
    }
    state->finishSubmitting();

    std::vector<std::thread> threads;
    threads.reserve(kBatchCount);
    for (size_t index = 0; index < kBatchCount; ++index) {
        threads.emplace_back([state] { state->completeBatch(ErrorInfo::OkStatus()); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(callback_count.load(), 1u);
}

TEST(TransferStageStateTest, PreservesFirstFailure) {
    ErrorCode result = ErrorCode::NONE_ERROR;
    TransferStageState state([&](ErrorInfo error) { result = error.code(); });
    state.addBatch();
    state.addBatch();
    state.completeBatch(ErrorInfo(ErrorCode::INVALID_PARAMS, "first"));
    state.completeBatch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "second"));
    state.finishSubmitting();

    EXPECT_EQ(result, ErrorCode::INVALID_PARAMS);
}

}  // namespace
}  // namespace rtp_llm
