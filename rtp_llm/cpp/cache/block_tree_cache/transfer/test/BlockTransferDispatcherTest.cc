#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeReclaimer.h"
#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>
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
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::BoundedThread;
using block_transfer_engine_test::bufferSafeResult;

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
    return std::make_shared<CompletedTransferAsyncContext>(bufferSafeResult());
}

class ScriptedPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit ScriptedPerRankEngine(std::deque<std::shared_ptr<TransferAsyncContext>> contexts  = {},
                                   size_t                                            max_batch = 8,
                                   std::vector<GroupSetPtr>                          groups    = {}):
        PerRankBlockTransferEngine(std::move(groups), false, 128, max_batch, 4, 10000, nullptr, 0, 4),
        contexts_(std::move(contexts)) {}

    std::shared_ptr<TransferAsyncContext> executeBatch(TransferTask task) override {
        const auto& descriptors = task.descriptors();
        ++submit_count_;
        submitted_batches_.emplace_back(descriptors.begin(), descriptors.end());
        submitted_deadlines_.push_back(task.deadline());
        submitted_trace_ids_.push_back(task.traceId());
        submitted_proofs_.push_back(task.completionProof());
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
    std::deque<std::shared_ptr<TransferAsyncContext>>  contexts_;
    size_t                                             submit_count_{0};
    std::vector<std::vector<TransferDescriptor>>       submitted_batches_;
    std::vector<TransferTask::Clock::time_point>       submitted_deadlines_;
    std::vector<uint64_t>                              submitted_trace_ids_;
    std::vector<std::shared_ptr<TransferBufferSafety>> submitted_proofs_;
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
    BlockTransferDispatcher                dispatcher(engine, std::make_shared<BlockTreeReclaimer>());

    auto context =
        dispatcher.executePerRank(TransferTask({TransferDescriptor::hostToDisk(0, 1, 1)}, std::chrono::seconds(30)));
    context->waitDone();
    EXPECT_TRUE(context->success());
    EXPECT_EQ(engine->submittedBatchCount(), 1u);
}

TEST(BlockTransferDispatcherTest, ExpiredTaskFailsBeforePerRankExecution) {
    auto                    engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
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


TEST(BlockTransferDispatcherTest, LocalRequestSplitsBeforeEngineAndPreservesDeadlineAndIdentity) {
    auto first  = std::make_shared<TransferBatchAsyncContext>();
    auto second = std::make_shared<TransferBatchAsyncContext>();
    auto third  = std::make_shared<TransferBatchAsyncContext>();
    auto engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<TransferAsyncContext>>{first, second, third}, 2);

    BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    TransferTask task({descriptor(0), descriptor(0), descriptor(0), descriptor(1)}, std::chrono::seconds(30));
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
    first->complete(bufferSafeResult());
    third->complete(bufferSafeResult());
    EXPECT_FALSE(context->done());
    second->complete(TransferStatus::DISK_IO_ERROR);
    ASSERT_TRUE(context->done());
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{
                  CacheCopyError::NONE, CacheCopyError::NONE, CacheCopyError::IO_FAILED, CacheCopyError::NONE}));
    dispatcher.waitForTransfers();
}

TEST(BlockTransferDispatcherTest, EmptyTaskFailsBeforeDispatch) {
    BlockTransferDispatcher dispatcher(nullptr, std::make_shared<BlockTreeReclaimer>());

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
    BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());

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
    BlockTransferDispatcher dispatcher(per_rank_engine, std::make_shared<BlockTreeReclaimer>(), multi_rank_engine);

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
    auto   dispatcher     = std::make_shared<BlockTransferDispatcher>(engine, std::make_shared<BlockTreeReclaimer>());
    size_t callback_count = 0;
    for (const auto& pending : {first, second}) {
        const size_t previous = callback_count;
        dispatcher->runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(30)), [&](TransferResult error) {
            EXPECT_TRUE(error.ok());
            ++callback_count;
        });
        EXPECT_FALSE(pending->done());
        EXPECT_EQ(callback_count, previous);
        pending->complete(bufferSafeResult());
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
        auto host = block_transfer_engine_test::makeHostPool(16, 8);
        const auto target = host->malloc();
        ASSERT_TRUE(target.has_value());
        host->incTreeRef(*target, BlockTreeRefType::LOAD);
        auto group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, host, nullptr);
        auto engine = std::make_shared<ScriptedPerRankEngine>(
            std::deque<std::shared_ptr<TransferAsyncContext>>{first, reject_second_batch ? nullptr : second, third},
            2, std::vector<GroupSetPtr>{group, group});
        BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());

        size_t    callback_count = 0;
        ErrorInfo final_error    = ErrorInfo::OkStatus();
        dispatcher.runTransfer(
            TransferTask({TransferDescriptor::deviceToHost(0, {1}, *target),
                          TransferDescriptor::deviceToHost(0, {2}, *target),
                          TransferDescriptor::deviceToHost(0, {3}, *target),
                          TransferDescriptor::deviceToHost(1, {4}, *target)},
                         std::chrono::seconds(30)),
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
            second->complete(bufferSafeResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "second failed")));
        }
        first->complete(bufferSafeResult());
        EXPECT_EQ(callback_count, 0u);
        third->complete(bufferSafeResult());
        EXPECT_EQ(callback_count, 1u);
        EXPECT_FALSE(final_error.ok());
        EXPECT_EQ(host->isFrozen(*target), reject_second_batch);
        if (reject_second_batch) {
            auto& reclaimer = *dispatcher.reclaimer_;
            ASSERT_EQ(reclaimer.frozen_.size(), 1u);
            EXPECT_TRUE(reclaimer.markWorkerSafe(reclaimer.frozen_.begin()->first, 0));
            EXPECT_NE(final_error.ToString().find("null context"), std::string::npos);
        }
        host->decTreeRef(*target, BlockTreeRefType::LOAD);
    }
}

TEST(BlockTransferDispatcherTest, OutOfOrderCompletionsPreserveBlockResults) {
    auto first  = std::make_shared<TransferBatchAsyncContext>();
    auto second = std::make_shared<TransferBatchAsyncContext>();
    auto third  = std::make_shared<TransferBatchAsyncContext>();
    auto engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<TransferAsyncContext>>{first, second, third}, 2);
    BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    size_t                  callback_count = 0;
    TransferResult          result;
    dispatcher.runTransfer(
        TransferTask({descriptor(0), descriptor(0), descriptor(0), descriptor(1)}, std::chrono::seconds(5)),
        [&](TransferResult completed) {
            ++callback_count;
            result = std::move(completed);
        });
    TransferResult failed(TransferStatus::CRC_MISMATCH);
    failed.block_errors = {CacheCopyError::NOT_EXECUTED, CacheCopyError::CRC_MISMATCH};
    third->complete(bufferSafeResult());
    first->complete(bufferSafeResult(std::move(failed)));
    EXPECT_EQ(callback_count, 0u);
    second->complete(bufferSafeResult());
    EXPECT_EQ(callback_count, 1u);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.copy_error, CacheCopyError::CRC_MISMATCH);
    EXPECT_EQ(
        result.block_errors,
        (std::vector<CacheCopyError>{
            CacheCopyError::NOT_EXECUTED, CacheCopyError::CRC_MISMATCH, CacheCopyError::NONE, CacheCopyError::NONE}));
}

TEST(BlockTransferDispatcherTest, ExecutePerRankDrainWaitsForConsumerCallbackReturn) {
    auto pending = std::make_shared<TransferBatchAsyncContext>();
    auto engine  = std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{pending});
    auto dispatcher = std::make_shared<BlockTransferDispatcher>(engine, std::make_shared<BlockTreeReclaimer>());
    auto context    = dispatcher->executePerRank(TransferTask({descriptor(0)}, std::chrono::seconds(30)));
    auto entered    = std::make_shared<TestSignal>();
    auto release    = std::make_shared<TestSignal>();
    auto observed   = std::make_shared<TestSignal>();
    context->onDone([entered, release](TransferResult result) {
        EXPECT_TRUE(result.ok());
        entered->signal();
        release->wait();
    });
    dispatcher->drain_observer_for_test_ = [observed] { observed->signal(); };
    block_tree_cache_detail::ScopeRollback cleanup([release] { release->signal(); });
    BoundedThread<void>                    complete([pending] { pending->complete(bufferSafeResult()); });
    ASSERT_EQ(entered->waitFor(), std::future_status::ready);
    EXPECT_TRUE(context->done());
    BoundedThread<void> drain([dispatcher] { dispatcher->waitForTransfers(); });
    ASSERT_EQ(observed->waitFor(), std::future_status::ready);
    EXPECT_EQ(drain.waitFor(std::chrono::milliseconds(0)), std::future_status::timeout);
    release->signal();
    ASSERT_EQ(complete.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    complete.get();
    ASSERT_EQ(drain.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    drain.get();
}

TEST(BlockTransferDispatcherTest, DrainAndShutdownWaitForCallbackReturnNotOnlyContextReadiness) {
    for (bool shutdown : {false, true}) {
        for (bool complete_before_wait : {false, true}) {
            SCOPED_TRACE(shutdown);
            SCOPED_TRACE(complete_before_wait);
            auto pending = std::make_shared<TransferBatchAsyncContext>();
            auto engine =
                std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{pending});
            auto dispatcher = std::make_shared<BlockTransferDispatcher>(engine, std::make_shared<BlockTreeReclaimer>());
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
            BoundedThread<void>                    complete([pending] { pending->complete(bufferSafeResult()); });
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
            auto dispatcher = std::make_shared<BlockTransferDispatcher>(engine, std::make_shared<BlockTreeReclaimer>());
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
            first->complete(bufferSafeResult());
            if (!synchronous_child) {
                EXPECT_EQ(calls->load(), 1u);
                EXPECT_EQ(drain.waitFor(std::chrono::milliseconds(0)), std::future_status::timeout);
                second->complete(bufferSafeResult());
            }
            ASSERT_EQ(drain.waitFor(std::chrono::seconds(5)), std::future_status::ready);
            drain.get();
            EXPECT_EQ(calls->load(), 2u);
        }
    }
}

class ControlledProducer final: public PerRankBlockTransferEngine {
public:
    ControlledProducer(): PerRankBlockTransferEngine({}, false, 128, 1, 1, 8, nullptr, 0, 4) {}
    std::shared_ptr<TransferAsyncContext> executeBatch(TransferTask batch) override {
        auto context = std::make_shared<TransferBatchAsyncContext>(batch.descriptors().size(), nullptr, batch.completionProof());
        proofs.push_back(batch.completionProof());
        contexts.push_back(context);
        on_submit(batch.traceId());
        return context;
    }
    std::vector<std::shared_ptr<TransferBufferSafety>> proofs;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts;
    std::function<void(uint64_t)> on_submit;
};

TEST(BlockTransferDispatcherTest, ThrowingCallbackDoesNotLeaveDrainPending) {
    for (bool asynchronous : {false, true}) {
        for (bool failed_result : {false, true}) {
            SCOPED_TRACE(asynchronous);
            SCOPED_TRACE(failed_result);
            const auto result = bufferSafeResult(failed_result ? TransferStatus::DISK_IO_ERROR : TransferStatus::OK);
            auto engine = std::make_shared<ControlledProducer>();
            engine->on_submit = [&](uint64_t) {
                if (!asynchronous) {
                    engine->contexts.back()->complete(result);
                }
            };
            auto dispatcher = std::make_shared<BlockTransferDispatcher>(engine, std::make_shared<BlockTreeReclaimer>());
            size_t calls      = 0;
            auto   submit     = [&] {
                try {
                    dispatcher->runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(30)),
                                            [&](TransferResult delivered) {
                                                EXPECT_EQ(delivered.ok(), !failed_result);
                                                EXPECT_EQ(delivered.status.code(), result.status.code());
                                                EXPECT_EQ(delivered.copy_error, result.copy_error);
                                                ++calls;
                                                throw std::runtime_error("consumer callback failure");
                                            });
                } catch (const std::runtime_error& error) {
                    EXPECT_STREQ(error.what(), "consumer callback failure");
                    throw;
                }
            };
            if (asynchronous) {
                submit();
                EXPECT_EQ(calls, 0u);
                EXPECT_NO_THROW(engine->contexts.back()->complete(result));
            } else {
                EXPECT_THROW(submit(), std::runtime_error);
            }
            EXPECT_EQ(calls, 1u);
            BoundedThread<void> drain([dispatcher] { dispatcher->waitForTransfers(); });
            ASSERT_EQ(drain.waitFor(std::chrono::seconds(5)), std::future_status::ready);
            drain.get();
        }
    }
}

TEST(BlockTransferDispatcherTest, RetainedBatchCallbackDoesNotBlockDrain) {
    auto   pending = std::make_shared<TransferBatchAsyncContext>();
    auto   engine = std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{pending});
    auto   dispatcher = std::make_shared<BlockTransferDispatcher>(engine, std::make_shared<BlockTreeReclaimer>());
    size_t calls      = 0;
    dispatcher->runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(30)), [&](TransferResult) { ++calls; });
    ASSERT_EQ(pending->callbacks_.size(), 1u);
    // Keep StageState alive after completion without retaining its final callback.
    const auto retained = pending->callbacks_.front();
    pending->complete(bufferSafeResult());
    EXPECT_EQ(calls, 1u);
    BoundedThread<void> drain([dispatcher] { dispatcher->waitForTransfers(); });
    ASSERT_EQ(drain.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    drain.get();
    retained(TransferStatus::OK);
    EXPECT_EQ(calls, 1u);
}

TEST(BlockTransferDispatcherTest, ExpiredTaskFailsWithoutSubmittingABatch) {
    auto                    engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
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

TEST(BlockTransferDispatcherTest, AllDirectionsRespectBatchLimit) {
    const std::vector<TransferDescriptor> directions{TransferDescriptor::hostToDevice(0, 1, {1}),
                                                     TransferDescriptor::deviceToHost(0, {1}, 1),
                                                     TransferDescriptor::hostToDisk(0, 1, 1),
                                                     TransferDescriptor::diskToDevice(0, 1, {1})};
    for (const auto& direction : directions) {
        for (const size_t limit : {2u, 8u, 16u}) {
            auto engine = std::make_shared<ScriptedPerRankEngine>(
                std::deque<std::shared_ptr<TransferAsyncContext>>{}, limit);
            BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
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

TEST(BlockTransferDispatcherTest, EngineBatchLimitIsSharedByDirections) {
    auto device_host_engine =
        std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{}, 64);
    BlockTransferDispatcher         device_host_dispatcher(device_host_engine, std::make_shared<BlockTreeReclaimer>());
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
    ASSERT_EQ(device_host_engine->submittedBatches().size(), 1u);
    EXPECT_EQ(device_host_engine->submittedBatches()[0].size(), 33u);
    EXPECT_EQ(device_host_callback_count, 1u);

    auto host_disk_engine =
        std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{}, 64);
    BlockTransferDispatcher         host_disk_dispatcher(host_disk_engine, std::make_shared<BlockTreeReclaimer>());
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
    ASSERT_EQ(host_disk_engine->submittedBatches().size(), 1u);
    EXPECT_EQ(host_disk_engine->submittedBatches()[0].size(), 33u);
    EXPECT_EQ(host_disk_callback_count, 1u);
}

TEST(BlockTransferDispatcherTest, OverlappingTasksKeepIndependentResults) {
    auto failed  = std::make_shared<TransferBatchAsyncContext>();
    auto healthy = std::make_shared<TransferBatchAsyncContext>();
    auto engine =
        std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<TransferAsyncContext>>{failed, healthy});
    BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
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
    failed->complete(
        bufferSafeResult({ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "CRC mismatch"), CacheCopyError::CRC_MISMATCH}));
    EXPECT_EQ(failed_calls, 1u);
    EXPECT_EQ(healthy_calls, 0u);
    healthy->complete(bufferSafeResult());
    EXPECT_EQ(healthy_calls, 1u);
    EXPECT_EQ(failed_result.copy_error, CacheCopyError::CRC_MISMATCH);
    EXPECT_TRUE(healthy_result.ok());
    EXPECT_EQ(healthy_result.copy_error, CacheCopyError::NONE);
}

TEST(BlockTransferDispatcherTest, DelayedFreezePreservesOriginalBlocksAcrossCallerReplacementAndWorkerBatching) {
    using namespace block_transfer_engine_test;
    auto host = makeHostPool(16, 8);
    auto groups =
        std::vector<GroupSetPtr>{makeTestGroupSet(0, makeTestTopology({makeTestGroupBase()}), {"group0"}, {}, host)};
    const auto first       = host->malloc();
    const auto second      = host->malloc();
    const auto replacement = host->malloc();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    ASSERT_TRUE(replacement.has_value());
    for (const auto block : {*first, *second, *replacement})
        host->incTreeRef(block, BlockTreeRefType::LOAD);
    auto first_pending  = std::make_shared<TransferBatchAsyncContext>();
    auto second_pending = std::make_shared<TransferBatchAsyncContext>();
    auto engine         = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<TransferAsyncContext>>{first_pending, second_pending}, 1, groups);
    BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    auto& reclaimer = *dispatcher.reclaimer_;
    size_t   settlements  = 0;
    uint64_t parent_trace = 0;
    {
        TransferTask task(
            {TransferDescriptor::deviceToHost(0, {7}, *first), TransferDescriptor::deviceToHost(0, {8}, *second)},
            std::chrono::seconds(30),
            91);
        task.setWorkflow(73, 61);
        parent_trace = task.traceId();
        dispatcher.runTransfer(
            task,
            [&](TransferResult result) {
                ++settlements;
                EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
                EXPECT_TRUE(host->isFrozen(*first));
                EXPECT_TRUE(host->isFrozen(*second));
                EXPECT_FALSE(host->isFrozen(*replacement));
                host->decTreeRef(*first, BlockTreeRefType::LOAD);
                host->decTreeRef(*second, BlockTreeRefType::LOAD);
            },
            "load");
        task = TransferTask({TransferDescriptor::deviceToHost(0, {9}, *replacement)}, std::chrono::seconds(30));
    }
    ASSERT_EQ(engine->submittedBatchCount(), 2u);
    first_pending->complete(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "physical completion unknown"));
    EXPECT_EQ(settlements, 0u);
    second_pending->complete(bufferSafeResult());
    ASSERT_EQ(settlements, 1u);
    ASSERT_EQ(reclaimer.frozen_.size(), 1u);
    const auto& execution = reclaimer.frozen_.begin()->second;
    EXPECT_EQ(reclaimer.frozen_.begin()->first, std::to_string(parent_trace));
    EXPECT_EQ(reclaimer.frozen_.begin()->first, std::to_string(engine->submitted_trace_ids_.front()));
    ASSERT_EQ(execution.targets.size(), 2u);
    for (size_t index = 0; index < execution.targets.size(); ++index) {
        EXPECT_EQ(execution.targets[index].block, index == 0 ? *first : *second);
        EXPECT_EQ(execution.targets[index].pool.get(), host.get());
    }
    for (const auto& proof : engine->submitted_proofs_)
        proof->report(true);
    EXPECT_EQ(reclaimer.forget_queue_.size(), 0u);
    EXPECT_TRUE(reclaimer.markWorkerSafe(parent_trace, 0));
    EXPECT_FALSE(host->isAllocated(*first));
    EXPECT_FALSE(host->isAllocated(*second));
    EXPECT_TRUE(host->isAllocated(*replacement));
    host->decTreeRef(*replacement, BlockTreeRefType::LOAD);
    reclaimer.drainForget();
    EXPECT_TRUE(reclaimer.records_.empty());
    dispatcher.waitForTransfers();
}

TEST(BlockTransferDispatcherTest, OneTraceBecomesSafeOnlyAfterSubmissionAndEveryBatchProof) {
    auto engine = std::make_shared<ControlledProducer>();
    BlockTransferDispatcher dispatcher(engine, std::make_shared<BlockTreeReclaimer>());
    using State = BlockTreeReclaimer::State;
    TransferTask execution({descriptor(0), descriptor(0), descriptor(0)}, std::chrono::seconds(30));
    const auto trace = execution.traceId();
    engine->on_submit = [&](uint64_t id) {
        EXPECT_EQ(id, trace);
        if (engine->contexts.size() == 1) {
            engine->contexts.front()->complete(bufferSafeResult());
        }
        EXPECT_EQ(dispatcher.reclaimer_->localState(std::to_string(trace)), State::RUNNING);
    };
    TransferTask late({descriptor(0)}, std::chrono::seconds(30));
    auto worker = dispatcher.reclaimer_;
    ASSERT_EQ(worker->queryIfIoFinishedInOneWorkerWithTraceIDs({std::to_string(late.traceId())}).front(), State::REJECTED);
    ASSERT_EQ(worker->forgetTaskInfoOnWorker({std::to_string(late.traceId())}).front(), BlockTreeReclaimer::ForgetResult::DONE);
    auto rejected = dispatcher.executePerRank(std::move(late));
    EXPECT_EQ(rejected->result().copy_error, CacheCopyError::NOT_EXECUTED);
    EXPECT_TRUE(engine->contexts.empty());
    auto original = dispatcher.executePerRank(execution);
    ASSERT_EQ(engine->contexts.size(), 3u);
    execution.deadline_ = TransferTask::Clock::now() - std::chrono::seconds(1);
    auto check_duplicate = [&] {
        auto duplicate = dispatcher.executePerRank(execution);
        EXPECT_EQ(duplicate->errorInfo().code(), ErrorCode::INVALID_PARAMS);
        EXPECT_EQ(duplicate->result().copy_error, CacheCopyError::NOT_EXECUTED);
        EXPECT_EQ(engine->contexts.size(), 3u);
    };
    check_duplicate();
    engine->contexts[1]->complete(bufferSafeResult());
    engine->contexts[2]->complete(TransferStatus::DISK_IO_ERROR);
    ASSERT_TRUE(original->done());
    EXPECT_FALSE(original->success());
    engine->proofs[0]->report(true);
    EXPECT_EQ(dispatcher.reclaimer_->localState(std::to_string(trace)), State::RUNNING);
    engine->proofs[2]->report(true);
    check_duplicate();
    EXPECT_EQ(dispatcher.reclaimer_->localState(std::to_string(trace)), State::BUFFER_SAFE);
}

}  // namespace
}  // namespace rtp_llm
