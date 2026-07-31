#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

namespace rtp_llm {
namespace {

std::shared_ptr<AsyncContext> okContext() {
    return std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
}

std::shared_ptr<AsyncContext> failedContext() {
    return std::make_shared<CompletedAsyncContext>(
        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "scripted transfer failure"));
}

class ScriptedPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit ScriptedPerRankEngine(std::deque<std::shared_ptr<AsyncContext>> contexts = {}):
        PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}), contexts_(std::move(contexts)) {}

    std::shared_ptr<AsyncContext> submit(const TransferDescriptor&) override {
        ++submit_count_;
        if (contexts_.empty()) {
            return okContext();
        }
        std::shared_ptr<AsyncContext> context = contexts_.front();
        contexts_.pop_front();
        return context;
    }

    size_t submitCount() const {
        return submit_count_;
    }

private:
    std::deque<std::shared_ptr<AsyncContext>> contexts_;
    size_t                                    submit_count_{0};
};

// Stays pending until complete(); records entry into waitDone() so tests can synchronize on it.
class ControllablePendingContext final: public AsyncContext {
public:
    void waitDone() override {
        std::unique_lock<std::mutex> lock(mutex_);
        wait_entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [this] { return done_; });
    }

    bool done() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_;
    }

    bool success() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_ && success_;
    }

    ErrorInfo errorInfo() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (success_) {
            return ErrorInfo::OkStatus();
        }
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "pending block transfer failed");
    }

    bool waitUntilWaitEnteredFor(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return wait_entered_; });
    }

    void complete(bool success) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            success_ = success;
            done_    = true;
        }
        cv_.notify_all();
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    bool                    wait_entered_{false};
    bool                    done_{false};
    bool                    success_{false};
};

TransferDescriptor descriptor(size_t group_id) {
    return TransferDescriptor::hostToDisk(group_id, 1, 1);
}

void runPendingCase(bool succeed) {
    auto pending = std::make_shared<ControllablePendingContext>();
    auto engine  = std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<AsyncContext>>{pending});
    BlockTransferDispatcher dispatcher(engine);

    std::atomic<bool> result{!succeed};
    std::thread       worker(
        [&] { result.store(dispatcher.executeMultiRank({descriptor(0)}, 100), std::memory_order_release); });

    const bool wait_entered = pending->waitUntilWaitEnteredFor(std::chrono::seconds(5));
    if (wait_entered) {
        EXPECT_FALSE(pending->done());
    }
    // Always complete and join before asserting so a timeout cannot leave a joinable thread.
    pending->complete(succeed);
    worker.join();
    ASSERT_TRUE(wait_entered) << "dispatcher returned without entering waitDone()";
    EXPECT_EQ(result.load(std::memory_order_acquire), succeed);
}

TEST(BlockTransferDispatcherTest, TransferDescriptorUsesPerRankEntry) {
    std::shared_ptr<ScriptedPerRankEngine> engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher                dispatcher(engine);

    EXPECT_TRUE(dispatcher.executePerRank(TransferDescriptor::hostToDisk(0, 1, 1)));
    EXPECT_EQ(engine->submitCount(), 1u);
}

TEST(BlockTransferDispatcherTest, EmptyBatchSucceedsWithoutAnEngine) {
    BlockTransferDispatcher dispatcher(nullptr);
    EXPECT_TRUE(dispatcher.executeMultiRank({}, 0));
}

TEST(BlockTransferDispatcherTest, PerRankBatchStopsAtFirstFailure) {
    auto engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<AsyncContext>>{okContext(), failedContext(), okContext()});
    BlockTransferDispatcher dispatcher(engine);

    EXPECT_FALSE(dispatcher.executeMultiRank({descriptor(0), descriptor(1), descriptor(2)}, 100));
    EXPECT_EQ(engine->submitCount(), 2u);
}

TEST(BlockTransferDispatcherTest, MultiRankFailureDoesNotFallbackToPerRank) {
    auto per_rank_engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<AsyncContext>>{okContext()});
    auto multi_rank_engine = std::make_shared<MultiRankBlockTransferEngine>(std::vector<GroupSetPtr>{}, nullptr);
    BlockTransferDispatcher dispatcher(per_rank_engine, multi_rank_engine);

    EXPECT_FALSE(dispatcher.executeMultiRank({descriptor(0)}, 100));
    EXPECT_EQ(per_rank_engine->submitCount(), 0u);
}

TEST(BlockTransferDispatcherTest, PerRankWaitsForPendingContextThenSucceeds) {
    runPendingCase(true);
}

TEST(BlockTransferDispatcherTest, PerRankWaitsForPendingContextThenFails) {
    runPendingCase(false);
}

}  // namespace
}  // namespace rtp_llm
