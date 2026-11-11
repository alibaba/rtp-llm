#include <gtest/gtest.h>
#include <algorithm>

#include <atomic>
#include <chrono>
#include <deque>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::BoundedThread;

class TestSignal {
public:
    void signal() {
        if (!signaled_.exchange(true)) {
            promise_.set_value();
        }
    }

    void wait() const {
        ready_.wait();
    }

    std::future_status waitFor(std::chrono::milliseconds timeout = std::chrono::seconds(5)) const {
        return ready_.wait_for(timeout);
    }

private:
    std::promise<void>       promise_;
    std::shared_future<void> ready_{promise_.get_future().share()};
    std::atomic<bool>        signaled_{false};
};

std::shared_ptr<TransferAsyncContext> okContext() {
    return std::make_shared<CompletedTransferAsyncContext>(ErrorInfo::OkStatus());
}

class ScriptedPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit ScriptedPerRankEngine(std::deque<std::shared_ptr<TransferAsyncContext>> contexts  = {},
                                   size_t                                            max_batch = 8,
                                   bool                                              async     = true):
        PerRankBlockTransferEngine({}, false, {}, 128, max_batch, 4, 10000, nullptr, 0, 4, async),
        contexts_(std::move(contexts)) {}

    std::shared_ptr<TransferAsyncContext> execute(TransferTask task) override {
        const auto& descriptors = task.descriptors();
        ++submit_count_;
        submitted_batches_.push_back(descriptors);
        submitted_deadlines_.push_back(task.deadline());
        submitted_trace_ids_.push_back(task.traceId());
        if (contexts_.empty()) {
            return okContext();
        }
        std::shared_ptr<TransferAsyncContext> context = contexts_.front();
        contexts_.pop_front();
        return context;
    }

    size_t submittedBatchCount() const {
        return submit_count_;
    }

    const std::vector<std::vector<TransferDescriptor>>& submittedBatches() const {
        return submitted_batches_;
    }

private:
    std::deque<std::shared_ptr<TransferAsyncContext>> contexts_;
    size_t                                            submit_count_{0};
    std::vector<std::vector<TransferDescriptor>>      submitted_batches_;
    std::vector<TransferTask::Clock::time_point>      submitted_deadlines_;
    std::vector<uint64_t>                             submitted_trace_ids_;
};

TransferDescriptor descriptor(size_t group_id) {
    return TransferDescriptor::hostToDisk(group_id, 1, 1);
}

TransferDescriptor deviceHostDescriptor(size_t group_id) {
    return TransferDescriptor::hostToDevice(group_id, 1, {1});
}

TransferDescriptor hostDiskDescriptor(size_t group_id) {
    return TransferDescriptor::hostToDisk(group_id, 1, 1);
}

TEST(BlockTransferDispatcherTest, DescriptorVectorUsesPerRankEntry) {
    std::shared_ptr<ScriptedPerRankEngine> engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher                dispatcher(engine);

    auto context =
        dispatcher.executePerRank(TransferTask({TransferDescriptor::hostToDisk(0, 1, 1)}, std::chrono::seconds(30)));
    context->waitDone();
    EXPECT_TRUE(context->success());
    EXPECT_EQ(engine->submittedBatchCount(), 1u);
}

TEST(BlockTransferDispatcherTest, ExpiredTaskFailsBeforePerRankExecution) {
    auto                    engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine);
    TransferTask            task({descriptor(0), descriptor(0)}, std::chrono::seconds(1));
    task.deadline_ = TransferTask::Clock::now() - std::chrono::milliseconds(1);

    auto context = dispatcher.executePerRank(std::move(task));
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_TRUE(context->result().buffer_safe);
    EXPECT_EQ(context->result().copy_error, CacheCopyError::NOT_EXECUTED);
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{CacheCopyError::NOT_EXECUTED, CacheCopyError::NOT_EXECUTED}));
    EXPECT_EQ(engine->submittedBatchCount(), 0u);
}

TEST(BlockTransferDispatcherTest, MissingTraceFailsSafelyWithoutExecutingOrRegistering) {
    auto                    engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine);
    TransferTask            task({descriptor(0), descriptor(0)}, std::chrono::seconds(30));
    task.trace_id_ = 0;

    const auto context = dispatcher.executePerRank(std::move(task));

    ASSERT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_TRUE(context->result().buffer_safe);
    EXPECT_EQ(context->result().copy_error, CacheCopyError::NOT_EXECUTED);
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{CacheCopyError::NOT_EXECUTED, CacheCopyError::NOT_EXECUTED}));
    EXPECT_EQ(engine->submittedBatchCount(), 0u);
    EXPECT_EQ(dispatcher.taskRegistry().size(), 0u);
    EXPECT_EQ(dispatcher.taskRegistry().queryLocal(std::vector<uint64_t>{0}).front(),
              TransferTaskRegistry::State::UNKNOWN);
    EXPECT_TRUE(dispatcher.frozenTasks().empty());
}

class DispatcherBackendTest: public ::testing::TestWithParam<bool> {};

TEST_P(DispatcherBackendTest, LocalRequestSplitsBeforeEngineAndPreservesDeadlineAndIdentity) {
    auto first  = std::make_shared<TransferBatchAsyncContext>();
    auto second = std::make_shared<TransferBatchAsyncContext>();
    auto third  = std::make_shared<TransferBatchAsyncContext>();
    auto engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<TransferAsyncContext>>{first, second, third}, 2, GetParam());
    ASSERT_EQ(engine->asyncEnabled(), GetParam());
    BlockTransferDispatcher dispatcher(engine, nullptr, 16);  // The engine remains bounded by M=2.
    TransferTask task({descriptor(0), descriptor(1), descriptor(0), descriptor(0)}, std::chrono::seconds(30));
    const auto   deadline = task.deadline();
    const auto   trace_id = task.traceId();
    auto         context  = dispatcher.executePerRank(std::move(task));
    ASSERT_EQ(engine->submittedBatches().size(), 3u);
    EXPECT_EQ(engine->submittedBatches()[0].size(), 2u);
    EXPECT_EQ(engine->submittedBatches()[1].size(), 1u);
    EXPECT_EQ(engine->submittedBatches()[2].size(), 1u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(engine->submitted_deadlines_[i], deadline);
        EXPECT_EQ(engine->submitted_trace_ids_[i], trace_id);
    }
    first->complete(TransferStatus::OK);
    third->complete(TransferStatus::OK);
    EXPECT_FALSE(context->done());
    second->complete(TransferStatus::DISK_IO_ERROR);
    ASSERT_TRUE(context->done());
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{
                  CacheCopyError::NONE, CacheCopyError::NONE, CacheCopyError::NONE, CacheCopyError::IO_FAILED}));
    dispatcher.waitForTransfers();
}

INSTANTIATE_TEST_SUITE_P(ExecutionBackends,
                         DispatcherBackendTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "WorkLoop" : "Legacy"; });

TEST(BlockTransferDispatcherTest, EmptyTaskFailsBeforeDispatch) {
    BlockTransferDispatcher dispatcher(nullptr);

    size_t    callback_count = 0;
    ErrorInfo final_error    = ErrorInfo::OkStatus();
    dispatcher.runTransfer(TransferTask({}, std::chrono::milliseconds(100)), [&](TransferResult error) {
        EXPECT_TRUE(error.buffer_safe);
        ++callback_count;
        final_error = std::move(error.status);
    });

    EXPECT_EQ(callback_count, 1u);
    EXPECT_EQ(final_error.code(), ErrorCode::INVALID_PARAMS);
}

TEST(BlockTransferDispatcherTest, PerRankBatchUsesOneSubmit) {
    auto                    engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine);

    size_t    callback_count = 0;
    ErrorInfo final_error    = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "callback not invoked");
    dispatcher.runTransfer(TransferTask({descriptor(0), descriptor(0), descriptor(0)}, std::chrono::milliseconds(100)),
                           [&](TransferResult error) {
                               ++callback_count;
                               final_error = std::move(error.status);
                           });

    EXPECT_EQ(callback_count, 1u);
    EXPECT_TRUE(final_error.ok());
    EXPECT_EQ(engine->submittedBatchCount(), 1u);
}

TEST(BlockTransferDispatcherTest, MultiRankFailureDoesNotFallbackToPerRank) {
    auto per_rank_engine =
        std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{okContext()});
    auto group_set = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, nullptr, nullptr);
    auto multi_rank_engine =
        std::make_shared<MultiRankBlockTransferEngine>(std::vector<GroupSetPtr>{group_set}, nullptr);
    BlockTransferDispatcher dispatcher(per_rank_engine, multi_rank_engine);

    size_t                   callback_count = 0;
    ErrorInfo                final_error    = ErrorInfo::OkStatus();
    const TransferDescriptor unsupported;
    dispatcher.runTransfer(TransferTask({unsupported}, std::chrono::milliseconds(100)), [&](TransferResult error) {
        ++callback_count;
        final_error = std::move(error.status);
    });

    EXPECT_EQ(callback_count, 1u);
    EXPECT_FALSE(final_error.ok());
    EXPECT_EQ(per_rank_engine->submittedBatchCount(), 0u);
}

TEST(BlockTransferDispatcherTest, RunTransferDoesNotWaitForPendingPerRankContext) {
    auto first  = std::make_shared<TransferBatchAsyncContext>();
    auto second = std::make_shared<TransferBatchAsyncContext>();
    auto engine =
        std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{first, second});
    auto   dispatcher     = std::make_shared<BlockTransferDispatcher>(engine);
    size_t callback_count = 0;
    for (const auto& pending : {first, second}) {
        const size_t previous = callback_count;
        dispatcher->runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(30)), [&](TransferResult error) {
            EXPECT_TRUE(error.ok());
            ++callback_count;
        });
        EXPECT_FALSE(pending->done());
        EXPECT_EQ(callback_count, previous);
        pending->complete(ErrorInfo::OkStatus());
        BoundedThread<void> drain([dispatcher] { dispatcher->waitForTransfers(); });
        ASSERT_EQ(drain.waitFor(std::chrono::seconds(5)), std::future_status::ready);
        drain.get();
        EXPECT_EQ(callback_count, previous + 1);
    }
    EXPECT_EQ(engine->submittedBatchCount(), 2u);
}

TEST(BlockTransferDispatcherTest, AsynchronousRunTransferGroupsAndWaitsForEveryBatch) {
    for (bool reject_second_batch : {false, true}) {
        SCOPED_TRACE(reject_second_batch);
        auto first  = std::make_shared<TransferBatchAsyncContext>();
        auto second = std::make_shared<TransferBatchAsyncContext>();
        auto third  = std::make_shared<TransferBatchAsyncContext>();
        auto engine = std::make_shared<ScriptedPerRankEngine>(
            std::deque<std::shared_ptr<TransferAsyncContext>>{first, reject_second_batch ? nullptr : second, third});
        BlockTransferDispatcher dispatcher(engine, nullptr, 2);

        size_t    callback_count = 0;
        ErrorInfo final_error    = ErrorInfo::OkStatus();
        dispatcher.runTransfer(
            TransferTask({descriptor(0), descriptor(1), descriptor(0), descriptor(0)}, std::chrono::seconds(30)),
            [&](TransferResult error) {
                ++callback_count;
                final_error = std::move(error.status);
            });

        ASSERT_EQ(engine->submittedBatches().size(), 3u);
        EXPECT_EQ(engine->submittedBatches()[0].size(), 2u);
        EXPECT_EQ(engine->submittedBatches()[0][0].group_set_id, 0u);
        EXPECT_EQ(engine->submittedBatches()[0][1].group_set_id, 0u);
        EXPECT_EQ(engine->submittedBatches()[1].size(), 1u);
        EXPECT_EQ(engine->submittedBatches()[1][0].group_set_id, 0u);
        EXPECT_EQ(engine->submittedBatches()[2].size(), 1u);
        EXPECT_EQ(engine->submittedBatches()[2][0].group_set_id, 1u);
        if (!reject_second_batch) {
            second->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "second failed"));
        }
        first->complete(ErrorInfo::OkStatus());
        EXPECT_EQ(callback_count, 0u);
        third->complete(ErrorInfo::OkStatus());
        EXPECT_EQ(callback_count, 1u);
        EXPECT_FALSE(final_error.ok());
        if (reject_second_batch) {
            EXPECT_NE(final_error.ToString().find("null context"), std::string::npos);
        }
    }
}

TEST(BlockTransferDispatcherTest, ReorderedBatchesReturnBlockResultsInOriginalTaskOrder) {
    auto first  = std::make_shared<TransferBatchAsyncContext>();
    auto second = std::make_shared<TransferBatchAsyncContext>();
    auto third  = std::make_shared<TransferBatchAsyncContext>();
    auto engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<TransferAsyncContext>>{first, second, third});
    BlockTransferDispatcher dispatcher(engine, nullptr, 2);
    size_t                  callback_count = 0;
    TransferResult          result;
    dispatcher.runTransfer(
        TransferTask({descriptor(0), descriptor(1), descriptor(0), descriptor(0)}, std::chrono::seconds(5)),
        [&](TransferResult completed) {
            ++callback_count;
            result = std::move(completed);
        });
    TransferResult failed(TransferStatus::CRC_MISMATCH);
    failed.block_errors = {CacheCopyError::NOT_EXECUTED, CacheCopyError::CRC_MISMATCH};
    third->complete(TransferStatus::OK);
    first->complete(std::move(failed));
    EXPECT_EQ(callback_count, 0u);
    second->complete(TransferStatus::OK);
    EXPECT_EQ(callback_count, 1u);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.copy_error, CacheCopyError::CRC_MISMATCH);
    EXPECT_EQ(
        result.block_errors,
        (std::vector<CacheCopyError>{
            CacheCopyError::NOT_EXECUTED, CacheCopyError::NONE, CacheCopyError::CRC_MISMATCH, CacheCopyError::NONE}));
}

TEST(BlockTransferDispatcherTest, DrainAndShutdownWaitForCallbackReturnNotOnlyContextReadiness) {
    for (bool shutdown : {false, true}) {
        for (bool complete_before_wait : {false, true}) {
            SCOPED_TRACE(shutdown);
            SCOPED_TRACE(complete_before_wait);
            auto pending = std::make_shared<TransferBatchAsyncContext>();
            auto engine =
                std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{pending});
            auto dispatcher       = std::make_shared<BlockTransferDispatcher>(engine);
            auto entered          = std::make_shared<TestSignal>();
            auto release          = std::make_shared<TestSignal>();
            auto observed         = std::make_shared<TestSignal>();
            auto release_observer = std::make_shared<TestSignal>();
            dispatcher->runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(30)),
                                    [entered, release](TransferResult error) {
                                        EXPECT_TRUE(error.ok());
                                        entered->signal();
                                        release->wait();
                                    });
            dispatcher->drain_observer_for_test_ = [observed, release_observer, complete_before_wait] {
                observed->signal();
                if (complete_before_wait) {
                    release_observer->wait();
                }
            };
            block_tree_cache_detail::ScopeRollback cleanup([release, release_observer] {
                release->signal();
                release_observer->signal();
            });
            BoundedThread<void>                    complete([pending] { pending->complete(ErrorInfo::OkStatus()); });
            ASSERT_EQ(entered->waitFor(), std::future_status::ready);
            EXPECT_TRUE(pending->done());
            BoundedThread<void> drain([dispatcher, shutdown] {
                if (shutdown)
                    dispatcher->shutdown();
                else
                    dispatcher->waitForTransfers();
            });
            ASSERT_EQ(observed->waitFor(), std::future_status::ready);
            EXPECT_EQ(drain.waitFor(std::chrono::milliseconds(0)), std::future_status::timeout);
            release->signal();
            ASSERT_EQ(complete.waitFor(std::chrono::seconds(5)), std::future_status::ready);
            complete.get();
            // Also retire the last ticket before the drain enters its CV wait.
            release_observer->signal();
            ASSERT_EQ(drain.waitFor(std::chrono::seconds(5)), std::future_status::ready);
            drain.get();
        }
    }
}

TEST(BlockTransferDispatcherTest, DrainIncludesNextStageSubmittedByCallback) {
    for (bool synchronous_child : {false, true}) {
        for (bool throwing_parent : {false, true}) {
            SCOPED_TRACE(synchronous_child);
            SCOPED_TRACE(throwing_parent);
            auto first  = std::make_shared<TransferBatchAsyncContext>();
            auto second = std::make_shared<TransferBatchAsyncContext>();
            auto engine = std::make_shared<ScriptedPerRankEngine>(
                std::deque<std::shared_ptr<TransferAsyncContext>>{first, synchronous_child ? okContext() : second});
            auto dispatcher = std::make_shared<BlockTransferDispatcher>(engine);
            auto calls      = std::make_shared<std::atomic<size_t>>(0);
            auto observed   = std::make_shared<TestSignal>();
            dispatcher->runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(30)),
                                    [dispatcher, calls, throwing_parent](TransferResult result) {
                                        EXPECT_TRUE(result.ok());
                                        ++*calls;
                                        dispatcher->runTransfer(TransferTask({descriptor(1)}, std::chrono::seconds(30)),
                                                                [calls](TransferResult child) {
                                                                    EXPECT_TRUE(child.ok());
                                                                    ++*calls;
                                                                });
                                        if (throwing_parent) {
                                            throw std::runtime_error("parent callback failed after child submission");
                                        }
                                    });
            dispatcher->drain_observer_for_test_ = [observed] { observed->signal(); };
            BoundedThread<void> drain([dispatcher] { dispatcher->waitForTransfers(); });
            ASSERT_EQ(observed->waitFor(), std::future_status::ready);
            first->complete(ErrorInfo::OkStatus());
            if (!synchronous_child) {
                EXPECT_EQ(calls->load(), 1u);
                EXPECT_EQ(drain.waitFor(std::chrono::milliseconds(0)), std::future_status::timeout);
                second->complete(ErrorInfo::OkStatus());
            }
            ASSERT_EQ(drain.waitFor(std::chrono::seconds(5)), std::future_status::ready);
            drain.get();
            EXPECT_EQ(calls->load(), 2u);
        }
    }
}

TEST(BlockTransferDispatcherTest, ThrowingCallbackDoesNotLeaveDrainPending) {
    for (bool asynchronous : {false, true}) {
        SCOPED_TRACE(asynchronous);
        auto pending = std::make_shared<TransferBatchAsyncContext>();
        auto engine  = std::make_shared<ScriptedPerRankEngine>(
            std::deque<std::shared_ptr<TransferAsyncContext>>{asynchronous ? pending : okContext()});
        auto   dispatcher = std::make_shared<BlockTransferDispatcher>(engine);
        size_t calls      = 0;
        auto   submit     = [&] {
            dispatcher->runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(30)), [&](TransferResult) {
                ++calls;
                throw std::runtime_error("callback");
            });
        };
        if (asynchronous) {
            submit();
            EXPECT_EQ(calls, 0u);
            EXPECT_NO_THROW(pending->complete(ErrorInfo::OkStatus()));
        } else {
            EXPECT_THROW(submit(), std::runtime_error);
        }
        EXPECT_EQ(calls, 1u);
        BoundedThread<void> drain([dispatcher] { dispatcher->waitForTransfers(); });
        ASSERT_EQ(drain.waitFor(std::chrono::seconds(5)), std::future_status::ready);
        drain.get();
    }
}

TEST(BlockTransferDispatcherTest, RetainedBatchCallbackDoesNotBlockDrain) {
    auto   pending = std::make_shared<TransferBatchAsyncContext>();
    auto   engine = std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{pending});
    auto   dispatcher = std::make_shared<BlockTransferDispatcher>(engine);
    size_t calls      = 0;
    dispatcher->runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(30)), [&](TransferResult) { ++calls; });
    ASSERT_EQ(pending->callbacks_.size(), 1u);
    // Keep StageState alive after completion without retaining its final callback.
    const auto retained = pending->callbacks_.front();
    pending->complete(TransferStatus::OK);
    EXPECT_EQ(calls, 1u);
    BoundedThread<void> drain([dispatcher] { dispatcher->waitForTransfers(); });
    ASSERT_EQ(drain.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    drain.get();
    retained(TransferStatus::OK);
    EXPECT_EQ(calls, 1u);
}

TEST(BlockTransferDispatcherTest, ExpiredTaskFailsWithoutSubmittingABatch) {
    auto                    engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine);
    TransferTask            task({descriptor(0), descriptor(0)}, std::chrono::seconds(1));
    task.deadline_ = TransferTask::Clock::now() - std::chrono::milliseconds(1);

    size_t    callback_count = 0;
    ErrorInfo final_error    = ErrorInfo::OkStatus();
    dispatcher.runTransfer(std::move(task), [&](TransferResult error) {
        ++callback_count;
        EXPECT_EQ(error.copy_error, CacheCopyError::NOT_EXECUTED);
        EXPECT_TRUE(error.buffer_safe);
        EXPECT_EQ(error.block_errors,
                  (std::vector<CacheCopyError>{CacheCopyError::NOT_EXECUTED, CacheCopyError::NOT_EXECUTED}));
        final_error = std::move(error.status);
    });

    EXPECT_EQ(callback_count, 1u);
    EXPECT_EQ(final_error.code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(engine->submittedBatchCount(), 0u);
}

TEST(BlockTransferDispatcherTest, AllDirectionsRespectSelectedBackendBatchLimits) {
    const std::vector<TransferDescriptor> directions{TransferDescriptor::hostToDevice(0, 1, {1}),
                                                     TransferDescriptor::deviceToHost(0, {1}, 1),
                                                     TransferDescriptor::hostToDisk(0, 1, 1),
                                                     TransferDescriptor::diskToHost(0, 1, 1),
                                                     TransferDescriptor::diskToDevice(0, 1, {1}),
                                                     TransferDescriptor::deviceToDisk(0, {1}, 1)};
    for (const bool async : {false, true}) {
        SCOPED_TRACE(async);
        for (const auto& direction : directions) {
            for (const size_t limit : {2u, 8u, 16u}) {
                auto engine = std::make_shared<ScriptedPerRankEngine>(
                    std::deque<std::shared_ptr<TransferAsyncContext>>{}, limit, async);
                BlockTransferDispatcher dispatcher(engine, nullptr, limit);
                size_t                  callback_count = 0;
                dispatcher.runTransfer(
                    TransferTask(std::vector<TransferDescriptor>(100, direction), std::chrono::seconds(30)),
                    [&](TransferResult error) {
                        EXPECT_TRUE(error.ok());
                        ++callback_count;
                    });
                const auto&  batches        = engine->submittedBatches();
                const size_t expected_count = (100 + limit - 1) / limit;
                ASSERT_EQ(batches.size(), expected_count);
                size_t total = 0;
                for (size_t index = 0; index < batches.size(); ++index) {
                    EXPECT_EQ(batches[index].size(), std::min(limit, 100 - total));
                    total += batches[index].size();
                }
                EXPECT_EQ(total, 100u);
                EXPECT_EQ(callback_count, 1u);
            }
        }
    }
}

TEST(BlockTransferDispatcherTest, DefaultBatchLimitIsSharedByDirections) {
    auto device_host_engine =
        std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{}, 64);
    BlockTransferDispatcher         device_host_dispatcher(device_host_engine);
    std::vector<TransferDescriptor> device_host_descriptors;
    for (size_t index = 0; index < 33; ++index) {
        device_host_descriptors.push_back(deviceHostDescriptor(0));
    }
    size_t device_host_callback_count = 0;
    device_host_dispatcher.runTransfer(TransferTask(device_host_descriptors, std::chrono::milliseconds(100)),
                                       [&](TransferResult error) {
                                           EXPECT_TRUE(error.ok());
                                           ++device_host_callback_count;
                                       });
    ASSERT_EQ(device_host_engine->submittedBatches().size(), 2u);
    EXPECT_EQ(device_host_engine->submittedBatches()[0].size(), 32u);
    EXPECT_EQ(device_host_engine->submittedBatches()[1].size(), 1u);
    EXPECT_EQ(device_host_callback_count, 1u);

    auto host_disk_engine =
        std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{}, 64);
    BlockTransferDispatcher         host_disk_dispatcher(host_disk_engine);
    std::vector<TransferDescriptor> host_disk_descriptors;
    for (size_t index = 0; index < 33; ++index) {
        host_disk_descriptors.push_back(hostDiskDescriptor(0));
    }
    size_t host_disk_callback_count = 0;
    host_disk_dispatcher.runTransfer(TransferTask(host_disk_descriptors, std::chrono::milliseconds(100)),
                                     [&](TransferResult error) {
                                         EXPECT_TRUE(error.ok());
                                         ++host_disk_callback_count;
                                     });
    ASSERT_EQ(host_disk_engine->submittedBatches().size(), 2u);
    EXPECT_EQ(host_disk_engine->submittedBatches()[0].size(), 32u);
    EXPECT_EQ(host_disk_engine->submittedBatches()[1].size(), 1u);
    EXPECT_EQ(host_disk_callback_count, 1u);
}

TEST(BlockTransferDispatcherTest, OverlappingTasksKeepIndependentResults) {
    auto failed  = std::make_shared<TransferBatchAsyncContext>();
    auto healthy = std::make_shared<TransferBatchAsyncContext>();
    auto engine =
        std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{failed, healthy});
    BlockTransferDispatcher dispatcher(engine);
    TransferResult          failed_result;
    TransferResult          healthy_result;
    size_t                  failed_calls  = 0;
    size_t                  healthy_calls = 0;
    dispatcher.runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(30)), [&](TransferResult result) {
        failed_result = std::move(result);
        ++failed_calls;
    });
    dispatcher.runTransfer(TransferTask({descriptor(1)}, std::chrono::seconds(30)), [&](TransferResult result) {
        healthy_result = std::move(result);
        ++healthy_calls;
    });
    EXPECT_EQ(engine->submittedBatchCount(), 2u);
    failed->complete({ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "CRC mismatch"), CacheCopyError::CRC_MISMATCH});
    EXPECT_EQ(failed_calls, 1u);
    EXPECT_EQ(healthy_calls, 0u);
    healthy->complete(ErrorInfo::OkStatus());
    EXPECT_EQ(healthy_calls, 1u);
    EXPECT_EQ(failed_result.copy_error, CacheCopyError::CRC_MISMATCH);
    EXPECT_TRUE(healthy_result.ok());
    EXPECT_EQ(healthy_result.copy_error, CacheCopyError::NONE);
}

}  // namespace
}  // namespace rtp_llm
