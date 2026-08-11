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
        last_batch_size_ = 1;
        return nextContext();
    }

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        ++submit_count_;
        last_batch_size_ = descriptors.size();
        return nextContext();
    }

    size_t submitCount() const {
        return submit_count_;
    }

    size_t lastBatchSize() const {
        return last_batch_size_;
    }

private:
    std::shared_ptr<AsyncContext> nextContext() {
        if (contexts_.empty()) {
            return okContext();
        }
        std::shared_ptr<AsyncContext> context = contexts_.front();
        contexts_.pop_front();
        return context;
    }

    std::deque<std::shared_ptr<AsyncContext>> contexts_;
    size_t                                    submit_count_{0};
    size_t                                    last_batch_size_{0};
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

    const auto context = dispatcher.executeMultiRank({descriptor(0)}, 100);
    ASSERT_EQ(context, pending);
    EXPECT_FALSE(context->done());
    EXPECT_EQ(engine->submitCount(), 1u);

    pending->complete(succeed);
    context->waitDone();
    EXPECT_EQ(context->success(), succeed);
}

TEST(BlockTransferDispatcherTest, TransferDescriptorUsesPerRankEntry) {
    std::shared_ptr<ScriptedPerRankEngine> engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher                dispatcher(engine);

    EXPECT_TRUE(dispatcher.executePerRank(TransferDescriptor::hostToDisk(0, 1, 1)));
    EXPECT_EQ(engine->submitCount(), 1u);
}

TEST(BlockTransferDispatcherTest, EmptyBatchSucceedsWithoutAnEngine) {
    BlockTransferDispatcher dispatcher(nullptr);
    const auto              context = dispatcher.executeMultiRank({}, 0);
    ASSERT_NE(context, nullptr);
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
}

TEST(BlockTransferDispatcherTest, PerRankFallbackSubmitsWholeBatchOnce) {
    auto engine = std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<AsyncContext>>{failedContext()});
    BlockTransferDispatcher dispatcher(engine);

    const auto context = dispatcher.executeMultiRank({descriptor(0), descriptor(1), descriptor(2)}, 100);
    ASSERT_NE(context, nullptr);
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(engine->submitCount(), 1u);
    EXPECT_EQ(engine->lastBatchSize(), 3u);
}

TEST(BlockTransferDispatcherTest, PerRankWaitsForPendingContextThenSucceeds) {
    runPendingCase(true);
}

TEST(BlockTransferDispatcherTest, PerRankWaitsForPendingContextThenFails) {
    runPendingCase(false);
}

}  // namespace
}  // namespace rtp_llm
