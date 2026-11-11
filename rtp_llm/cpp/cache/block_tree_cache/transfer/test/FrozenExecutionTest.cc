#include <gtest/gtest.h>

#include <chrono>
#include <atomic>
#include <deque>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using namespace block_transfer_engine_test;
using block_tree_cache_test::BoundedThread;

TransferResult unknownExecution() {
    return TransferResult(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "execution completion unknown"),
                          CacheCopyError::RPC_FAILED);
}

// Deliberately permits repeated/late delivery and registration failure so the
// coordinator's ownership does not depend on a well-behaved mock context.
class ObservableContext final: public TransferAsyncContext {
public:
    explicit ObservableContext(bool throw_on_registration = false,
                               std::optional<TransferResult> inline_result = std::nullopt):
        throw_on_registration_(throw_on_registration), inline_result_(std::move(inline_result)) {}

    void waitDone() override {}
    void onDone(DoneCallback callback) override {
        callback_ = std::move(callback);
        if (inline_result_) {
            deliver(*inline_result_);
        }
        if (throw_on_registration_) {
            throw std::runtime_error("callback registration failed");
        }
    }
    bool done() const override {
        return done_;
    }
    bool success() const override {
        return done_ && result_.ok();
    }
    TransferResult result() const override {
        return result_;
    }
    void deliver(TransferResult result) {
        done_ = true;
        result_ = std::move(result);
        if (callback_) {
            callback_(result_);
        }
    }

private:
    bool                          throw_on_registration_;
    std::optional<TransferResult> inline_result_;
    bool                          done_{false};
    TransferResult                result_;
    DoneCallback                  callback_;
};

class FrozenExecutionEngine final: public PerRankBlockTransferEngine {
public:
    FrozenExecutionEngine(std::vector<GroupSetPtr> groups,
                           std::deque<std::shared_ptr<TransferAsyncContext>> contexts,
                           size_t max_batch = 8):
        PerRankBlockTransferEngine(std::move(groups), false, {}, 128, max_batch, 1, 32, nullptr, 0, 4, false),
        contexts_(std::move(contexts)) {}

    std::shared_ptr<TransferAsyncContext> executeBatch(TransferTask task) override {
        submitted_traces.push_back(task.traceId());
        submitted_descriptors.push_back(task.descriptors());
        submitted_proofs.push_back(task.completionProof());
        if (contexts_.empty()) {
            throw std::runtime_error("unexpected extra execution");
        }
        auto context = contexts_.front();
        contexts_.pop_front();
        return context;
    }

    std::vector<uint64_t>                        submitted_traces;
    std::vector<std::vector<TransferDescriptor>> submitted_descriptors;
    std::vector<std::shared_ptr<TransferBufferSafety>> submitted_proofs;

private:
    std::deque<std::shared_ptr<TransferAsyncContext>> contexts_;
};

class FrozenExecutionTest: public ::testing::Test {
protected:
    void SetUp() override {
        host_ = makeHostPool(/*payload_bytes=*/16, /*usable_count=*/8);
        auto topology = makeTestTopology({makeTestGroupBase()});
        groups_.push_back(makeTestGroupSet(0, std::move(topology), {"group0"}, {}, host_));
    }

    BlockIdxType allocateTarget() {
        const auto block = host_->malloc();
        EXPECT_TRUE(block.has_value());
        if (!block)
            return NULL_BLOCK_IDX;
        host_->incTreeRef(*block, BlockTreeRefType::LOAD);
        return *block;
    }

    TransferTask parent(std::vector<BlockIdxType> targets) const {
        std::vector<TransferDescriptor> descriptors;
        for (const auto block : targets) {
            // The executor is scripted; only the real HOST target is allocated.
            auto desc = TransferDescriptor::deviceToHost(0, {7}, block);
            desc.path_index = descriptors.size();
            descriptors.push_back(std::move(desc));
        }
        TransferTask task(std::move(descriptors), std::chrono::seconds(30), 73);
        task.setWorkflow(900, 901);
        return task;
    }

    std::shared_ptr<FrozenExecutionEngine>
    engine(std::deque<std::shared_ptr<TransferAsyncContext>> contexts, size_t max_batch = 8) const {
        return std::make_shared<FrozenExecutionEngine>(groups_, std::move(contexts), max_batch);
    }

    std::shared_ptr<HostBlockPool> host_;
    std::vector<GroupSetPtr>       groups_;
};

TEST_F(FrozenExecutionTest, ActualExecutionTraceOwnsDeduplicatedTargetsBeforeBusinessRelease) {
    const auto target = allocateTarget();
    auto first  = std::make_shared<ObservableContext>();
    auto second = std::make_shared<ObservableContext>();
    auto executor = engine({first, second}, 1);
    BlockTransferDispatcher dispatcher(executor);
    auto task = parent({target, target});
    const auto parent_trace = task.traceId();
    size_t completions = 0;
    dispatcher.runTransfer(task, [&](TransferResult result) {
        ++completions;
        EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        EXPECT_TRUE(host_->isFrozen(target));
        EXPECT_EQ(host_->treeRefCount(target), 2u);
        EXPECT_EQ(dispatcher.frozenTasks().size(), 1u);
        host_->decTreeRef(target, BlockTreeRefType::LOAD);
    }, "load");
    ASSERT_EQ(executor->submitted_traces.size(), 2u);
    EXPECT_NE(executor->submitted_traces[0], parent_trace);
    EXPECT_EQ(executor->submitted_traces[0], executor->submitted_traces[1]);

    first->deliver(unknownExecution());
    EXPECT_EQ(completions, 0u);
    EXPECT_TRUE(dispatcher.frozenTasks().empty());
    second->deliver(TransferResult(TransferStatus::OK));
    EXPECT_EQ(completions, 1u);
    const auto frozen = dispatcher.frozenTasks();
    ASSERT_EQ(frozen.size(), 1u);
    EXPECT_EQ(std::get<uint64_t>(frozen[0]->execution().trace_id), executor->submitted_traces[0]);
    EXPECT_EQ(frozen[0]->execution().parent_trace_id, parent_trace);
    EXPECT_EQ(frozen[0]->execution().workflow_id, 900u);
    EXPECT_EQ(frozen[0]->execution().request_id, 73);
    EXPECT_EQ(frozen[0]->execution().operation, "load");
    EXPECT_FALSE(frozen[0]->execution().remote);
    EXPECT_EQ(frozen[0]->execution().expected_rank_count, 1u);
    EXPECT_EQ(frozen[0]->execution().descriptors.size(), 2u);
    EXPECT_EQ(frozen[0]->snapshot().target_block_count, 1u);
    ASSERT_NE(frozen[0]->failure().execution_diagnostics, nullptr);
    EXPECT_EQ(std::get<uint64_t>(frozen[0]->failure().execution_diagnostics->trace_id), executor->submitted_traces[0]);
    EXPECT_TRUE(host_->isAllocated(target));
    EXPECT_EQ(host_->treeRefCount(target), 1u);
    frozen[0]->unfreeze();
    EXPECT_FALSE(host_->isAllocated(target));
    EXPECT_TRUE(dispatcher.frozenTasks().empty());
    EXPECT_TRUE(dispatcher.frozenTasks().empty());
    EXPECT_EQ(dispatcher.taskRegistry().size(), 1u);
    dispatcher.waitForTransfers();
}

TEST_F(FrozenExecutionTest, SeparateExecutionTracesSharingTargetReleaseOnlyTheirOwnHold) {
    const auto target = allocateTarget();
    auto first  = std::make_shared<ObservableContext>();
    auto second = std::make_shared<ObservableContext>();
    auto executor = engine({first, second});
    BlockTransferDispatcher dispatcher(executor);
    auto task = parent({target});
    size_t completions = 0;
    const auto complete = [&](TransferResult result) {
        EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        ++completions;
    };
    dispatcher.runTransfer(task, complete, "load");
    dispatcher.runTransfer(task, complete, "load");
    ASSERT_EQ(executor->submitted_traces.size(), 2u);
    EXPECT_NE(executor->submitted_traces[0], executor->submitted_traces[1]);
    first->deliver(unknownExecution());
    second->deliver(unknownExecution());
    EXPECT_EQ(completions, 2u);
    const auto frozen = dispatcher.frozenTasks();
    ASSERT_EQ(frozen.size(), 2u);
    EXPECT_EQ(host_->treeRefCount(target), 3u);
    EXPECT_EQ(host_->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);

    host_->decTreeRef(target, BlockTreeRefType::LOAD);
    frozen[0]->unfreeze();
    frozen[0]->unfreeze();
    EXPECT_TRUE(host_->isFrozen(target));
    EXPECT_EQ(host_->treeRefCount(target), 1u);
    const auto remaining = dispatcher.frozenTasks();
    ASSERT_EQ(remaining.size(), 1u);
    EXPECT_EQ(remaining.front(), frozen[1]);
    frozen[1]->unfreeze();
    EXPECT_FALSE(host_->isAllocated(target));
    EXPECT_EQ(host_->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    EXPECT_TRUE(dispatcher.frozenTasks().empty());
    EXPECT_TRUE(dispatcher.frozenTasks().empty());
    EXPECT_EQ(dispatcher.taskRegistry().size(), 2u);
    dispatcher.waitForTransfers();
}

TEST_F(FrozenExecutionTest, MixedSafeAndBannedReleaseForgetsEveryRankButPreservesBan) {
    const auto target = allocateTarget();
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    TransferTaskRegistry completed, banned;
    const uint64_t trace = 417;
    ASSERT_TRUE(completed.begin(trace));
    ASSERT_TRUE(completed.finish(trace, true));
    ASSERT_EQ(banned.queryAndBlockMissing(std::vector<uint64_t>{trace}).front(),
              TransferTaskRegistry::State::REJECTED);
    FrozenTask::ExecutionInfo execution;
    execution.trace_id = trace;
    execution.operation = "load";
    execution.remote = true;
    execution.expected_rank_count = 2;
    execution.expected_peers = {{0, "worker-0"}, {1, "worker-1"}};
    execution.descriptors.push_back({0, 0, Tier::DISK, Tier::HOST, {}, {{host_, target, 0}}});
    auto frozen = std::make_shared<FrozenTask>(
        FrozenTask::prepare(std::move(execution)), unknownExecution(),
        cleanup->prepare({true, {{0, "worker-0"}, {1, "worker-1"}}}, trace));
    frozen->freeze();
    host_->decTreeRef(target, BlockTreeRefType::LOAD);
    EXPECT_TRUE(host_->isFrozen(target));
    EXPECT_EQ(cleanup->size(), 0u);
    // The cleanup route includes every participant, independent of diagnostics.
    EXPECT_EQ(frozen->recoverySnapshot().last_query, nullptr);
    EXPECT_TRUE(frozen->unfreeze());
    EXPECT_FALSE(host_->isAllocated(target));
    EXPECT_FALSE(frozen->unfreeze());
    EXPECT_EQ(cleanup->size(), 1u);
    frozen.reset();
    size_t forget_calls = 0;
    const auto summary = cleanup->drain([&](const auto& route, const auto& traces, int) {
        EXPECT_TRUE(route.remote);
        EXPECT_EQ(route.peers.size(), 2u);
        EXPECT_EQ(traces, (std::vector<TransferExecutionTrace>{trace}));
        TransferCompletionForgetEntryResult entry;
        entry.trace_id = trace;
        entry.all_done = true;
        for (const auto& peer : route.peers) {
            ++forget_calls;
            auto& registry = peer.rank == 0 ? completed : banned;
            const auto result = registry.forget(trace);
            EXPECT_EQ(result, TransferTaskRegistry::ForgetResult::DONE);
            entry.all_done = entry.all_done && result == TransferTaskRegistry::ForgetResult::DONE;
        }
        TransferCompletionForgetResult result;
        result.entries.push_back(std::move(entry));
        return result;
    }, 100);
    EXPECT_EQ(forget_calls, 2u);
    EXPECT_EQ(summary.attempted, 1u);
    EXPECT_EQ(summary.forgotten, 1u);
    EXPECT_EQ(summary.pending, 0u);
    EXPECT_FALSE(summary.failed);
    EXPECT_EQ(completed.queryLocal(std::vector<uint64_t>{trace}).front(), TransferTaskRegistry::State::UNKNOWN);
    EXPECT_EQ(banned.queryLocal(std::vector<uint64_t>{trace}).front(), TransferTaskRegistry::State::REJECTED);
    EXPECT_FALSE(banned.begin(trace));
}

TEST_F(FrozenExecutionTest, OrdinarySafeReleaseStillQueuesForgetExactlyOnce) {
    const auto target = allocateTarget();
    auto completion = std::make_shared<ObservableContext>();
    auto executor = engine({completion});
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    BlockTransferDispatcher dispatcher(executor);
    dispatcher.setCompletionCleanup(cleanup);
    dispatcher.runTransfer(parent({target}), [&](TransferResult) {
        host_->decTreeRef(target, BlockTreeRefType::LOAD);
    }, "load");
    completion->deliver(unknownExecution());
    const auto frozen = dispatcher.frozenTasks();
    ASSERT_EQ(frozen.size(), 1u);
    EXPECT_EQ(cleanup->size(), 0u);
    EXPECT_TRUE(frozen.front()->unfreeze());
    EXPECT_FALSE(frozen.front()->unfreeze());
    EXPECT_FALSE(host_->isAllocated(target));
    EXPECT_EQ(cleanup->size(), 1u);
    const auto summary = cleanup->drain([&](const auto&, const auto& traces, int) {
        EXPECT_EQ(traces, (std::vector<TransferExecutionTrace>{frozen.front()->execution().trace_id}));
        TransferCompletionForgetResult result;
        TransferCompletionForgetEntryResult entry;
        entry.trace_id = traces.front();
        entry.all_done = true;
        result.entries.push_back(std::move(entry));
        return result;
    }, 100);
    EXPECT_EQ(summary.attempted, 1u);
    EXPECT_EQ(summary.forgotten, 1u);
    EXPECT_EQ(summary.pending, 0u);
    dispatcher.waitForTransfers();
}

TEST_F(FrozenExecutionTest, KnownBusinessOutcomesWithoutSafetyProofDoNotFreeze) {
    const auto target = allocateTarget();
    for (const auto status : {TransferStatus::OK, TransferStatus::DISK_IO_ERROR, TransferStatus::CRC_MISMATCH}) {
        SCOPED_TRACE(static_cast<int>(status));
        auto context = std::make_shared<ObservableContext>();
        auto executor = engine({context});
        BlockTransferDispatcher dispatcher(executor);
        size_t completions = 0;
        dispatcher.runTransfer(parent({target}), [&](TransferResult result) {
            ++completions;
            EXPECT_EQ(result.ok(), status == TransferStatus::OK);
            EXPECT_EQ(result.copy_error, cacheCopyErrorFor(status));
            EXPECT_FALSE(result.buffer_safe);
        });
        context->deliver(TransferResult(status));
        EXPECT_EQ(completions, 1u);
        EXPECT_TRUE(dispatcher.frozenTasks().empty());
        EXPECT_FALSE(host_->isFrozen(target));
        EXPECT_EQ(host_->treeRefCount(target), 1u);
        dispatcher.waitForTransfers();
    }
    host_->decTreeRef(target, BlockTreeRefType::LOAD);
}

TEST_F(FrozenExecutionTest, EvictUnknownFreezesHostAndDiskTargets) {
    TempDirGuard dir("evict_frozen_targets");
    auto disk = makeDiskPool(16, 8, dir.path, std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK));
    groups_ = {makeTestGroupSet(0, makeTestTopology({makeTestGroupBase()}), {"group0"}, {}, host_, disk)};
    for (const auto tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(tierName(tier));
        const auto target = groups_[0]->allocateSingleBlock(tier, BlockTreeRefType::EVICTION);
        ASSERT_NE(target, NULL_BLOCK_IDX);
        std::shared_ptr<IBlockPool> pool = tier == Tier::HOST ?
            std::static_pointer_cast<IBlockPool>(host_) : std::static_pointer_cast<IBlockPool>(disk);
        auto context = std::make_shared<ObservableContext>();
        auto executor = engine({context});
        BlockTransferDispatcher dispatcher(executor);
        auto descriptor = tier == Tier::HOST ? TransferDescriptor::deviceToHost(0, {1}, target) :
                                              TransferDescriptor::hostToDisk(0, 1, target);
        size_t completions = 0;
        dispatcher.runTransfer(makeTransferTask({descriptor}), [&](TransferResult result) {
            ++completions;
            EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
            EXPECT_TRUE(pool->isFrozen(target));
            pool->decTreeRef(target, BlockTreeRefType::EVICTION);
        }, "evict");
        context->deliver(unknownExecution());
        context->deliver(unknownExecution());
        EXPECT_EQ(completions, 1u);
        EXPECT_TRUE(pool->isAllocated(target));
        const auto frozen = dispatcher.frozenTasks();
        ASSERT_EQ(frozen.size(), 1u);
        frozen[0]->unfreeze();
        EXPECT_FALSE(pool->isAllocated(target));
        dispatcher.waitForTransfers();
    }
}

TEST_F(FrozenExecutionTest, RepeatedChildCompletionCannotFinishBeforeOtherInternalBatch) {
    const auto first_target = allocateTarget();
    const auto second_target = allocateTarget();
    auto first  = std::make_shared<ObservableContext>();
    auto second = std::make_shared<ObservableContext>();
    auto executor = engine({first, second}, 1);
    BlockTransferDispatcher dispatcher(executor);
    size_t completions = 0;
    dispatcher.runTransfer(parent({first_target, second_target}), [&](TransferResult result) {
        EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        ++completions;
    });
    first->deliver(unknownExecution());
    first->deliver(unknownExecution());
    EXPECT_EQ(completions, 0u);
    EXPECT_TRUE(dispatcher.frozenTasks().empty());
    second->deliver(TransferResult(TransferStatus::OK));
    first->deliver(unknownExecution());
    second->deliver(unknownExecution());
    EXPECT_EQ(completions, 1u);
    const auto frozen = dispatcher.frozenTasks();
    ASSERT_EQ(frozen.size(), 1u);
    EXPECT_EQ(frozen[0]->snapshot().target_block_count, 2u);
    EXPECT_EQ(host_->treeRefCount(first_target), 2u);
    EXPECT_EQ(host_->treeRefCount(second_target), 2u);
    frozen[0]->unfreeze();
    host_->decTreeRef({first_target, second_target}, BlockTreeRefType::LOAD);
    dispatcher.waitForTransfers();
}

TEST_F(FrozenExecutionTest, RegistrationFailureFreezesBeforeRollbackAndLateDeliveryCannotRefreeze) {
    const auto target = allocateTarget();
    auto context = std::make_shared<ObservableContext>(true);
    auto executor = engine({context});
    BlockTransferDispatcher dispatcher(executor);
    size_t completions = 0;
    dispatcher.runTransfer(parent({target}), [&](TransferResult result) {
        ++completions;
        EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        EXPECT_TRUE(host_->isFrozen(target));
        EXPECT_EQ(dispatcher.frozenTasks().size(), 1u);
        host_->decTreeRef(target, BlockTreeRefType::LOAD);
    });
    EXPECT_EQ(completions, 1u);
    const auto frozen = dispatcher.frozenTasks();
    ASSERT_EQ(frozen.size(), 1u);
    context->deliver(TransferResult(TransferStatus::OK));
    context->deliver(unknownExecution());
    EXPECT_EQ(completions, 1u);
    EXPECT_EQ(dispatcher.frozenTasks().size(), 1u);
    EXPECT_EQ(host_->treeRefCount(target), 1u);
    frozen[0]->unfreeze();
    ASSERT_FALSE(host_->isAllocated(target));
    EXPECT_TRUE(dispatcher.frozenTasks().empty());
    context->deliver(unknownExecution());
    EXPECT_EQ(completions, 1u);
    EXPECT_FALSE(host_->isAllocated(target));
    EXPECT_EQ(frozen[0]->snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_TRUE(dispatcher.frozenTasks().empty());
    dispatcher.waitForTransfers();
}

TEST_F(FrozenExecutionTest, ExceptionAfterInlineCompletionDoesNotReplaceOrRepeatItsOutcome) {
    const auto target = allocateTarget();
    for (const bool completion_unknown : {false, true}) {
        SCOPED_TRACE(completion_unknown);
        auto outcome = completion_unknown ? unknownExecution() : TransferResult(TransferStatus::DISK_IO_ERROR);
        auto context = std::make_shared<ObservableContext>(true, outcome);
        auto executor = engine({context});
        BlockTransferDispatcher dispatcher(executor);
        size_t completions = 0;
        dispatcher.runTransfer(parent({target}), [&](TransferResult result) {
            ++completions;
            EXPECT_EQ(result.status.code(), outcome.status.code());
            EXPECT_EQ(result.copy_error, outcome.copy_error);
        });
        context->deliver(unknownExecution());
        EXPECT_EQ(completions, 1u);
        const auto frozen = dispatcher.frozenTasks();
        ASSERT_EQ(frozen.size(), completion_unknown ? 1u : 0u);
        EXPECT_EQ(host_->treeRefCount(target), completion_unknown ? 2u : 1u);
        if (completion_unknown)
            frozen[0]->unfreeze();
        dispatcher.waitForTransfers();
    }
    host_->decTreeRef(target, BlockTreeRefType::LOAD);
}

TEST_F(FrozenExecutionTest, WorkerLocalExecutionDoesNotCreateAnotherCoordinatorHold) {
    const auto target = allocateTarget();
    auto context = std::make_shared<ObservableContext>();
    auto executor = engine({context});
    BlockTransferDispatcher dispatcher(executor);
    auto task = parent({target});
    const auto trace = task.traceId();
    auto worker_result = dispatcher.executePerRank(task);
    context->deliver(unknownExecution());
    ASSERT_TRUE(worker_result->done());
    EXPECT_EQ(worker_result->result().status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    ASSERT_EQ(executor->submitted_traces.size(), 1u);
    EXPECT_EQ(executor->submitted_traces[0], trace);
    EXPECT_TRUE(dispatcher.frozenTasks().empty());
    EXPECT_FALSE(host_->isFrozen(target));
    EXPECT_EQ(host_->treeRefCount(target), 1u);
    host_->decTreeRef(target, BlockTreeRefType::LOAD);
    dispatcher.waitForTransfers();
}


TEST_F(FrozenExecutionTest, ReleasedEnumerationDestroysManagedRecordOutsideTheDispatcherLock) {
    const auto target = allocateTarget();
    auto context = std::make_shared<ObservableContext>();
    auto executor = engine({context});
    auto dispatcher = std::make_shared<BlockTransferDispatcher>(executor);
    dispatcher->runTransfer(parent({target}), [](TransferResult result) {
        EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    });
    context->deliver(unknownExecution());
    auto records = dispatcher->frozenTasks();
    ASSERT_EQ(records.size(), 1u);
    auto record = records.front();
    records.clear();
    const auto trace = std::get<uint64_t>(record->execution().trace_id);
    ASSERT_EQ(executor->submitted_proofs.size(), 1u);
    ASSERT_NE(executor->submitted_proofs.front(), nullptr);
    executor->submitted_proofs.front()->report(true);

    auto destroyed = std::make_shared<std::atomic<size_t>>(0);
    std::weak_ptr<BlockTransferDispatcher> weak_dispatcher = dispatcher;
    // The retained diagnostic's destructor reenters enumeration. Erasing a
    // last-owned record while holding the dispatcher lock would deadlock here.
    record->recordRecoveryQuery(std::shared_ptr<const TransferCompletionQueryResult>(
        new TransferCompletionQueryResult,
        [weak_dispatcher, destroyed](const TransferCompletionQueryResult* query) {
            if (auto owner = weak_dispatcher.lock()) {
                EXPECT_TRUE(owner->frozenTasks().empty());
            }
            ++*destroyed;
            delete query;
        }));
    std::weak_ptr<FrozenTask> weak_record = record;
    EXPECT_TRUE(record->unfreeze());
    record.reset();
    EXPECT_FALSE(weak_record.expired());
    BoundedThread<bool> cleanup([dispatcher] { return dispatcher->frozenTasks().empty(); });
    ASSERT_EQ(cleanup.waitFor(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(cleanup.get());
    EXPECT_TRUE(weak_record.expired());
    EXPECT_EQ(destroyed->load(), 1u);
    EXPECT_TRUE(dispatcher->frozenTasks().empty());
    EXPECT_EQ(dispatcher->taskRegistry().size(), 1u);
    EXPECT_EQ(dispatcher->taskRegistry().queryLocal(std::vector<uint64_t>{trace}).front(),
              TransferTaskRegistry::State::BUFFER_SAFE);
    host_->decTreeRef(target, BlockTreeRefType::LOAD);
    dispatcher->waitForTransfers();
}

TEST_F(FrozenExecutionTest, ConcurrentEnumerationKeepsReleasingRecordUntilCapacityNotificationReturns) {
    const auto target = allocateTarget();
    auto entered = std::make_shared<std::promise<void>>();
    auto notifying = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto released = release->get_future().share();
    host_->setCapacityChangeCallback([entered, released] {
        entered->set_value();
        EXPECT_EQ(released.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    });
    auto context = std::make_shared<ObservableContext>();
    auto executor = engine({context});
    auto dispatcher = std::make_shared<BlockTransferDispatcher>(executor);
    dispatcher->runTransfer(parent({target}), [](TransferResult result) {
        EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    });
    context->deliver(unknownExecution());
    auto records = dispatcher->frozenTasks();
    ASSERT_EQ(records.size(), 1u);
    auto record = records.front();
    host_->decTreeRef(target, BlockTreeRefType::LOAD);
    BoundedThread<bool> unfreeze([record] { return record->unfreeze(); });
    ASSERT_EQ(notifying.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_EQ(record->snapshot().state, FrozenTask::State::RELEASING);
    BoundedThread<bool> enumerate([dispatcher, record] {
        for (size_t i = 0; i < 20; ++i) {
            const auto active = dispatcher->frozenTasks();
            if (active.size() != 1 || active.front() != record
                || active.front()->snapshot().state != FrozenTask::State::RELEASING)
                return false;
        }
        return true;
    });
    ASSERT_EQ(enumerate.waitFor(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(enumerate.get());
    EXPECT_FALSE(record->unfreeze());
    release->set_value();
    ASSERT_EQ(unfreeze.waitFor(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(unfreeze.get());
    EXPECT_EQ(record->snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_TRUE(dispatcher->frozenTasks().empty());
    EXPECT_TRUE(dispatcher->frozenTasks().empty());
    EXPECT_FALSE(host_->isAllocated(target));
    EXPECT_EQ(dispatcher->taskRegistry().size(), 1u);
    // The external snapshot continues to own usable diagnostics after pruning.
    EXPECT_EQ(records.front(), record);
    EXPECT_EQ(record->failure().status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    dispatcher->waitForTransfers();
}

}  // namespace
}  // namespace rtp_llm
