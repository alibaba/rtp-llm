#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

namespace rtp_llm {

class LoadAsyncContextTest: public ::testing::Test {
protected:
    class CallbackBarrier {
    public:
        void enterAndWait() {
            std::unique_lock<std::mutex> lock(mutex_);
            entered_ = true;
            cv_.notify_all();
            cv_.wait(lock, [this] { return released_; });
        }

        void waitUntilEntered() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return entered_; });
        }

        void release() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                released_ = true;
            }
            cv_.notify_all();
        }

    private:
        std::mutex              mutex_;
        std::condition_variable cv_;
        bool                    entered_{false};
        bool                    released_{false};
    };

    class ThreadEvent {
    public:
        void notify() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                notified_ = true;
            }
            cv_.notify_all();
        }

        void wait() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return notified_; });
        }

    private:
        std::mutex              mutex_;
        std::condition_variable cv_;
        bool                    notified_{false};
    };

    void SetUp() override {
        coordinator_ = makeCoordinator();
    }

    void TearDown() override {
        coordinator_->shutdown();
    }

    std::shared_ptr<LoadContextCoordinator> makeCoordinator() {
        return std::make_shared<LoadContextCoordinator>(
            [this](const std::shared_ptr<LoadAsyncContext>& context) {
                ++commit_count_;
                return context != nullptr;
            },
            [this](LoadAsyncContext& /*context*/) { ++abort_count_; });
    }

    void resetCoordinator(LoadContextCoordinator::CommitCallback commit_callback,
                          LoadContextCoordinator::AbortCallback  abort_callback) {
        coordinator_->shutdown();
        coordinator_ = std::make_shared<LoadContextCoordinator>(std::move(commit_callback), std::move(abort_callback));
    }

    TransferDescriptor makePendingHostDescriptor() {
        TransferDescriptor desc;
        desc.source_tier   = Tier::HOST;
        desc.source_blocks = {11};
        return desc;
    }

    std::shared_ptr<LoadAsyncContext> createRegisteredContext(std::vector<TransferDescriptor> load_descs,
                                                              std::vector<bool>               joined_load,
                                                              size_t                          matched_blocks) {
        const std::shared_ptr<LoadAsyncContext> context =
            coordinator_->create(std::move(load_descs), std::move(joined_load), matched_blocks);
        if (!coordinator_->registerContext(context)) {
            return nullptr;
        }
        return context;
    }

    std::shared_ptr<LoadAsyncContext> makeContext(size_t transfer_count) {
        std::vector<TransferDescriptor> load_descs;
        if (transfer_count == 0) {
            TransferDescriptor desc;
            desc.source_tier = Tier::DEVICE;
            load_descs.push_back(std::move(desc));
        } else {
            load_descs.assign(transfer_count, makePendingHostDescriptor());
        }
        std::vector<bool> joined_load(load_descs.size(), false);
        return createRegisteredContext(std::move(load_descs), std::move(joined_load), 1);
    }

    std::shared_ptr<LoadAsyncContext> makeCommittedContext(size_t transfer_count) {
        const std::shared_ptr<LoadAsyncContext> context = makeContext(transfer_count);
        if (context == nullptr || !context->commit()) {
            return nullptr;
        }
        return context;
    }

    size_t                                  commit_count_{0};
    size_t                                  abort_count_{0};
    std::shared_ptr<LoadContextCoordinator> coordinator_;
};

TEST_F(LoadAsyncContextTest, SetsTargetBlocks) {
    TransferDescriptor first;
    first.group_set_id  = 3;
    first.path_index    = 5;
    first.source_tier   = Tier::HOST;
    first.source_blocks = {11};

    TransferDescriptor joined;
    joined.group_set_id         = 4;
    joined.path_index           = 6;
    joined.source_tier          = Tier::DISK;
    joined.source_blocks        = {12};

    const std::shared_ptr<LoadAsyncContext> context = createRegisteredContext({first, joined}, {false, true}, 7);
    ASSERT_NE(context, nullptr);

    context->setTargetBlocks(1, {21});
    EXPECT_EQ(context->load_descs_.size(), 2u);
    EXPECT_FALSE(context->joined_load_[0]);
    EXPECT_TRUE(context->joined_load_[1]);
    EXPECT_EQ(context->load_descs_[0].group_set_id, 3u);
    EXPECT_EQ(context->load_descs_[1].path_index, 6u);
    EXPECT_EQ(context->load_descs_[1].target_blocks, (std::vector<BlockIdxType>{21}));
    context->setTargetBlocks(0, {20});
    EXPECT_EQ(context->load_descs_[0].target_blocks, (std::vector<BlockIdxType>{20}));
    context->abort();
    EXPECT_EQ(abort_count_, 1u);
}

TEST_F(LoadAsyncContextTest, CountsStrongestTierOncePerPath) {
    TransferDescriptor device;
    device.path_index  = 1;
    device.source_tier = Tier::DEVICE;

    TransferDescriptor host = device;
    host.source_tier                       = Tier::HOST;

    TransferDescriptor disk = device;
    disk.path_index                        = 2;
    disk.source_tier                       = Tier::DISK;

    const std::shared_ptr<LoadAsyncContext> context =
        createRegisteredContext({device, host, disk}, {false, false, false}, 9);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->matchedBlocks(), 9u);
    EXPECT_EQ(context->matchedBlocks(Tier::DEVICE), 0u);
    EXPECT_EQ(context->matchedBlocks(Tier::HOST), 1u);
    EXPECT_EQ(context->matchedBlocks(Tier::DISK), 1u);
    EXPECT_EQ(context->matchedBlocks(Tier::NONE), 0u);
    context->abort();
}

TEST_F(LoadAsyncContextTest, DestructionAbortsPendingContext) {
    std::weak_ptr<LoadAsyncContext> weak_context;
    {
        const std::shared_ptr<LoadAsyncContext> context = makeContext(1);
        ASSERT_NE(context, nullptr);
        weak_context = context;
    }

    EXPECT_TRUE(weak_context.expired());
    EXPECT_EQ(commit_count_, 0u);
    EXPECT_EQ(abort_count_, 1u);
}

TEST_F(LoadAsyncContextTest, ContextIdsAreStableAndUnique) {
    const std::shared_ptr<LoadAsyncContext> first_context  = makeContext(1);
    const std::shared_ptr<LoadAsyncContext> second_context = makeContext(1);
    ASSERT_NE(first_context, nullptr);
    ASSERT_NE(second_context, nullptr);

    const uint64_t first_context_id = first_context->contextId();
    EXPECT_NE(first_context_id, 0u);
    EXPECT_NE(first_context_id, second_context->contextId());
    EXPECT_EQ(first_context->contextId(), first_context_id);
    EXPECT_FALSE(coordinator_->registerContext(first_context));
    first_context->abort();
    second_context->abort();
}

TEST_F(LoadAsyncContextTest, CommitConsumesPendingRegistration) {
    std::shared_ptr<LoadAsyncContext> context = makeContext(1);
    ASSERT_NE(context, nullptr);

    ASSERT_TRUE(context->commit());
    EXPECT_EQ(commit_count_, 1u);
    EXPECT_FALSE(context->commit());
    context.reset();
    EXPECT_EQ(abort_count_, 0u);
}

TEST_F(LoadAsyncContextTest, CommitFailureTerminalizesContext) {
    resetCoordinator(
        [this](const std::shared_ptr<LoadAsyncContext>& /*context*/) {
            ++commit_count_;
            return false;
        },
        [this](LoadAsyncContext& /*context*/) { ++abort_count_; });
    std::shared_ptr<LoadAsyncContext> context = makeContext(1);
    ASSERT_NE(context, nullptr);

    EXPECT_FALSE(context->commit());
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->commit());
    context.reset();
    EXPECT_EQ(commit_count_, 1u);
    EXPECT_EQ(abort_count_, 0u);
}

TEST_F(LoadAsyncContextTest, AbortDoesNotOverwriteConcurrentCompletion) {
    CallbackBarrier abort_callback;
    resetCoordinator([](const std::shared_ptr<LoadAsyncContext>& context) { return context != nullptr; },
                     [&abort_callback](LoadAsyncContext& /*context*/) { abort_callback.enterAndWait(); });
    const std::shared_ptr<LoadAsyncContext> context = makeContext(1);
    ASSERT_NE(context, nullptr);

    std::thread abort_thread([&context] { context->abort(); });
    abort_callback.waitUntilEntered();
    EXPECT_TRUE(context->completeOne(true));
    EXPECT_TRUE(context->success());

    abort_callback.release();
    abort_thread.join();
    EXPECT_TRUE(context->success());
}

TEST_F(LoadAsyncContextTest, ShutdownAbortsLivePendingContextOnce) {
    std::shared_ptr<LoadAsyncContext> context = makeContext(1);
    ASSERT_NE(context, nullptr);

    coordinator_->shutdown();
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->isRequestCanceled());
    EXPECT_EQ(abort_count_, 1u);

    context.reset();
    EXPECT_EQ(abort_count_, 1u);
}

TEST_F(LoadAsyncContextTest, ShutdownWaitsForCommitCallback) {
    CallbackBarrier   commit_callback;
    ThreadEvent       shutdown_waiting;
    std::atomic<bool> shutdown_finished{false};
    bool              commit_result = false;
    resetCoordinator(
        [&commit_callback](const std::shared_ptr<LoadAsyncContext>& context) {
            commit_callback.enterAndWait();
            return context != nullptr;
        },
        [](LoadAsyncContext& /*context*/) {});
    coordinator_->shutdown_wait_observer_for_test_  = [&shutdown_waiting] { shutdown_waiting.notify(); };
    const std::shared_ptr<LoadAsyncContext> context = makeContext(1);
    ASSERT_NE(context, nullptr);

    std::thread commit_thread([&context, &commit_result] { commit_result = context->commit(); });
    commit_callback.waitUntilEntered();
    std::thread shutdown_thread([this, &shutdown_finished] {
        coordinator_->shutdown();
        shutdown_finished.store(true);
    });
    shutdown_waiting.wait();
    EXPECT_FALSE(shutdown_finished.load());

    commit_callback.release();
    commit_thread.join();
    shutdown_thread.join();

    EXPECT_TRUE(commit_result);
    EXPECT_TRUE(shutdown_finished.load());
    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->commit());
}

TEST_F(LoadAsyncContextTest, ShutdownWaitsForAbortCallback) {
    CallbackBarrier   abort_callback;
    ThreadEvent       shutdown_waiting;
    std::atomic<bool> shutdown_finished{false};
    resetCoordinator([](const std::shared_ptr<LoadAsyncContext>& context) { return context != nullptr; },
                     [this, &abort_callback](LoadAsyncContext& /*context*/) {
                         ++abort_count_;
                         abort_callback.enterAndWait();
                     });
    coordinator_->shutdown_wait_observer_for_test_  = [&shutdown_waiting] { shutdown_waiting.notify(); };
    const std::shared_ptr<LoadAsyncContext> context = makeContext(1);
    ASSERT_NE(context, nullptr);

    std::thread abort_thread([&context] { context->abort(); });
    abort_callback.waitUntilEntered();
    std::thread shutdown_thread([this, &shutdown_finished] {
        coordinator_->shutdown();
        shutdown_finished.store(true);
    });
    shutdown_waiting.wait();
    EXPECT_FALSE(shutdown_finished.load());

    abort_callback.release();
    abort_thread.join();
    shutdown_thread.join();

    EXPECT_TRUE(shutdown_finished.load());
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->isRequestCanceled());
    EXPECT_EQ(abort_count_, 1u);
    context->abort();
    EXPECT_EQ(abort_count_, 1u);
}

TEST_F(LoadAsyncContextTest, ConcurrentShutdownCallersWaitForSameAbortCallback) {
    CallbackBarrier   abort_callback;
    ThreadEvent       second_shutdown_waiting;
    std::atomic<bool> first_shutdown_finished{false};
    std::atomic<bool> second_shutdown_finished{false};
    resetCoordinator([](const std::shared_ptr<LoadAsyncContext>& context) { return context != nullptr; },
                     [this, &abort_callback](LoadAsyncContext& /*context*/) {
                         ++abort_count_;
                         abort_callback.enterAndWait();
                     });
    coordinator_->shutdown_wait_observer_for_test_  = [&second_shutdown_waiting] { second_shutdown_waiting.notify(); };
    const std::shared_ptr<LoadAsyncContext> context = makeContext(1);
    ASSERT_NE(context, nullptr);

    std::thread first_shutdown_thread([this, &first_shutdown_finished] {
        coordinator_->shutdown();
        first_shutdown_finished.store(true);
    });
    abort_callback.waitUntilEntered();

    std::thread second_shutdown_thread([this, &second_shutdown_finished] {
        coordinator_->shutdown();
        second_shutdown_finished.store(true);
    });
    second_shutdown_waiting.wait();
    EXPECT_FALSE(first_shutdown_finished.load());
    EXPECT_FALSE(second_shutdown_finished.load());
    EXPECT_EQ(abort_count_, 1u);

    abort_callback.release();
    first_shutdown_thread.join();
    second_shutdown_thread.join();

    EXPECT_TRUE(first_shutdown_finished.load());
    EXPECT_TRUE(second_shutdown_finished.load());
    EXPECT_EQ(abort_count_, 1u);
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->isRequestCanceled());
}

TEST_F(LoadAsyncContextTest, ShutdownRejectsNewContext) {
    coordinator_->shutdown();
    const std::shared_ptr<LoadAsyncContext> context = makeContext(1);

    EXPECT_EQ(context, nullptr);
    EXPECT_EQ(commit_count_, 0u);
    EXPECT_EQ(abort_count_, 0u);
}

TEST_F(LoadAsyncContextTest, CompleteSingleWork) {
    const std::shared_ptr<LoadAsyncContext> context = makeCommittedContext(1);
    ASSERT_NE(context, nullptr);
    EXPECT_FALSE(context->done());

    EXPECT_TRUE(context->completeOne(true));

    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    context->waitDone();
}

TEST_F(LoadAsyncContextTest, CompleteAfterEveryTransfer) {
    const std::shared_ptr<LoadAsyncContext> context = makeCommittedContext(3);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeOne(false));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeOne(true));

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
}

TEST_F(LoadAsyncContextTest, CancellationWaitsForEveryTransfer) {
    const std::shared_ptr<LoadAsyncContext> context = makeCommittedContext(2);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->requestCancel());
    EXPECT_TRUE(context->isRequestCanceled());
    EXPECT_TRUE(context->requestCancel());
    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeOne(true));

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
}

TEST_F(LoadAsyncContextTest, RejectCompletionAfterTerminalState) {
    const std::shared_ptr<LoadAsyncContext> context = makeCommittedContext(1);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->completeOne(true));
}

TEST_F(LoadAsyncContextTest, TaskFailureCompletesAllPendingTransfers) {
    const std::shared_ptr<LoadAsyncContext> context = makeCommittedContext(3);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->onTaskFail());

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->completeOne(true));
    EXPECT_FALSE(context->onTaskFail());
}

TEST_F(LoadAsyncContextTest, TaskFailureCompletesCanceledContext) {
    const std::shared_ptr<LoadAsyncContext> context = makeCommittedContext(2);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->requestCancel());
    EXPECT_TRUE(context->isRequestCanceled());
    EXPECT_TRUE(context->onTaskFail());

    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->isRequestCanceled());
    EXPECT_FALSE(context->success());
}

TEST_F(LoadAsyncContextTest, ZeroTransferContextIsImmediatelySuccessful) {
    const std::shared_ptr<LoadAsyncContext> context = makeCommittedContext(0);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(context->completeOne(true));
}

}  // namespace rtp_llm
