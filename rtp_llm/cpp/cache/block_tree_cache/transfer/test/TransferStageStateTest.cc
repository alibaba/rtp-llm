#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"

namespace rtp_llm {
namespace {

TEST(TransferStageStateTest, ImmediateCompletionWaitsForSubmitterToken) {
    size_t callback_count = 0;
    auto   state          = std::make_shared<TransferStageState>([&](TransferResult error) {
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
    size_t             callback_count = 0;
    TransferStageState state([&](TransferResult error) {
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
    auto                state = std::make_shared<TransferStageState>([&](TransferResult error) {
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

TEST(TransferStageStateTest, PreservesAllFailureMessagesAtEqualPriority) {
    TransferResult     result;
    TransferStageState state([&](TransferResult completed) { result = std::move(completed); });
    state.addBatch();
    state.addBatch();
    state.completeBatch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "first"));
    state.completeBatch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "second"));
    state.finishSubmitting();

    EXPECT_EQ(result.copy_error, CacheCopyError::COPY_FAILED);
    EXPECT_EQ(result.status.ToString(), "first; second");
}

TEST(TransferStageStateTest, FailureStillDrainsAllBatches) {
    size_t             calls = 0;
    TransferStageState state([&](TransferResult error) {
        ++calls;
        EXPECT_EQ(error.status.code(), ErrorCode::EXECUTION_EXCEPTION);
    });
    state.addBatch();
    state.addBatch();
    state.addBatch();
    state.finishSubmitting();
    state.completeBatch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "ordinary failure"));
    state.completeBatch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "CRC mismatch"));
    EXPECT_EQ(calls, 0u);
    state.completeBatch(ErrorInfo::OkStatus());
    EXPECT_EQ(calls, 1u);
}

TEST(TransferStageStateTest, HighestPriorityWinsInEitherCompletionOrderAndStillDrains) {
    for (bool crc_first : {false, true}) {
        size_t             calls = 0;
        TransferResult     result;
        TransferStageState state([&](TransferResult completed) {
            result = std::move(completed);
            ++calls;
        });
        state.addBatch();
        state.addBatch();
        state.addBatch();
        state.finishSubmitting();
        TransferResult crc(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "CRC mismatch"), CacheCopyError::CRC_MISMATCH);
        TransferResult rpc(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RPC failed"), CacheCopyError::RPC_FAILED);
        state.completeBatch(crc_first ? crc : rpc);
        state.completeBatch(crc_first ? rpc : crc);
        EXPECT_EQ(calls, 0u);
        state.completeBatch(ErrorInfo::OkStatus());
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(result.copy_error, CacheCopyError::CRC_MISMATCH);
        EXPECT_EQ(result.status.ToString(), crc_first ? "CRC mismatch; RPC failed" : "RPC failed; CRC mismatch");
    }
}

TEST(TransferStageStateTest, ConcurrentFailuresPreserveHighestPriority) {
    const std::vector<CacheCopyError> errors{CacheCopyError::COPY_FAILED,
                                             CacheCopyError::INVALID_REQUEST,
                                             CacheCopyError::IO_FAILED,
                                             CacheCopyError::RPC_FAILED,
                                             CacheCopyError::CRC_COMPUTE_FAILED,
                                             CacheCopyError::CRC_MISMATCH};
    std::atomic<size_t>               calls{0};
    TransferResult                    result;
    TransferStageState                state(
        [&](TransferResult completed) {
            result = std::move(completed);
            ++calls;
        },
        errors.size());
    for (size_t i = 0; i < errors.size(); ++i) {
        state.addBatch();
    }
    state.finishSubmitting();
    std::vector<std::thread> workers;
    for (size_t index = 0; index < errors.size(); ++index) {
        workers.emplace_back([&state, &errors, index] {
            const auto error = errors[index];
            state.completeBatch({ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, cacheCopyErrorName(error)), error}, {index});
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    EXPECT_EQ(calls.load(), 1u);
    EXPECT_EQ(result.block_errors, errors);
    EXPECT_EQ(result.copy_error, CacheCopyError::CRC_MISMATCH);
    for (const auto error : errors) {
        EXPECT_NE(result.status.ToString().find(cacheCopyErrorName(error)), std::string::npos);
    }
    EXPECT_EQ(std::count(result.status.ToString().begin(), result.status.ToString().end(), ';'), errors.size() - 1);
}

}  // namespace
}  // namespace rtp_llm
