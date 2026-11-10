#include <gtest/gtest.h>

#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using namespace block_transfer_engine_test;

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

    std::shared_ptr<TransferAsyncContext> execute(TransferTask task) override {
        submitted_traces.push_back(task.traceId());
        submitted_descriptors.push_back(task.descriptors());
        if (contexts_.empty()) {
            throw std::runtime_error("unexpected extra execution");
        }
        auto context = contexts_.front();
        contexts_.pop_front();
        return context;
    }

    std::vector<uint64_t>                        submitted_traces;
    std::vector<std::vector<TransferDescriptor>> submitted_descriptors;

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
            auto desc = TransferDescriptor::diskToHost(0, 7, block);
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
    EXPECT_EQ(frozen[0]->execution().trace_id, executor->submitted_traces[0]);
    EXPECT_EQ(frozen[0]->execution().parent_trace_id, parent_trace);
    EXPECT_EQ(frozen[0]->execution().workflow_id, 900u);
    EXPECT_EQ(frozen[0]->execution().request_id, 73);
    EXPECT_EQ(frozen[0]->execution().operation, "load");
    EXPECT_FALSE(frozen[0]->execution().remote);
    EXPECT_EQ(frozen[0]->execution().expected_rank_count, 1u);
    EXPECT_EQ(frozen[0]->execution().descriptors.size(), 2u);
    EXPECT_EQ(frozen[0]->snapshot().target_block_count, 1u);
    ASSERT_NE(frozen[0]->failure().execution_diagnostics, nullptr);
    EXPECT_EQ(frozen[0]->failure().execution_diagnostics->trace_id, executor->submitted_traces[0]);
    EXPECT_TRUE(host_->isAllocated(target));
    EXPECT_EQ(host_->treeRefCount(target), 1u);
    frozen[0]->unfreeze();
    EXPECT_FALSE(host_->isAllocated(target));
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
    frozen[1]->unfreeze();
    EXPECT_FALSE(host_->isAllocated(target));
    EXPECT_EQ(host_->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
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

TEST_F(FrozenExecutionTest, StoreUnknownSkipsFrozenOwnershipWhileEvictStillFreezesTheSameDirections) {
    TempDirGuard dir("store_frozen_filter");
    auto disk = makeDiskPool(16, 8, dir.path, std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK));
    groups_ = {makeTestGroupSet(0, makeTestTopology({makeTestGroupBase()}), {"group0"}, {}, host_, disk)};
    for (const auto tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(tierName(tier));
        for (const bool store : {true, false}) {
            SCOPED_TRACE(store ? "store" : "evict");
            const auto ref_type = store ? BlockTreeRefType::STORE : BlockTreeRefType::EVICTION;
            const auto target = groups_[0]->allocateSingleBlock(tier, ref_type);
            ASSERT_NE(target, NULL_BLOCK_IDX);
            std::shared_ptr<IBlockPool> pool = tier == Tier::HOST ?
                std::static_pointer_cast<IBlockPool>(host_) : std::static_pointer_cast<IBlockPool>(disk);
            auto context = std::make_shared<ObservableContext>();
            auto executor = engine({context});
            BlockTransferDispatcher dispatcher(executor);
            auto descriptor = tier == Tier::HOST ? TransferDescriptor::deviceToHost(0, {1}, target) :
                                                  TransferDescriptor::deviceToDisk(0, {1}, target);
            size_t completions = 0;
            dispatcher.runTransfer(makeTransferTask({descriptor}), [&](TransferResult result) {
                ++completions;
                EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
                EXPECT_FALSE(result.buffer_safe);
                EXPECT_EQ(pool->isFrozen(target), !store);
                EXPECT_EQ(dispatcher.frozenTasks().size(), store ? 0u : 1u);
                pool->decTreeRef(target, ref_type);
            }, store ? "store" : "evict");
            context->deliver(unknownExecution());
            context->deliver(unknownExecution());
            EXPECT_EQ(completions, 1u);
            EXPECT_EQ(pool->isAllocated(target), !store);
            if (!store) {
                const auto frozen = dispatcher.frozenTasks();
                ASSERT_EQ(frozen.size(), 1u);
                frozen[0]->unfreeze();
            }
            EXPECT_FALSE(pool->isAllocated(target));
            dispatcher.waitForTransfers();
        }
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
    context->deliver(unknownExecution());
    EXPECT_EQ(completions, 1u);
    EXPECT_FALSE(host_->isAllocated(target));
    EXPECT_EQ(frozen[0]->snapshot().state, FrozenTask::State::RELEASED);
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

}  // namespace
}  // namespace rtp_llm
