#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {
namespace {

TransferDescriptor diskToHost(size_t group, BlockIdxType disk, BlockIdxType host) {
    auto desc = TransferDescriptor::hostToDisk(group, host, disk);
    std::swap(desc.source_tier, desc.target_tier);
    std::swap(desc.source_blocks, desc.target_blocks);
    return desc;
}

TransferDescriptor removedDeviceToDisk(size_t group, std::vector<BlockIdxType> device, BlockIdxType disk) {
    auto desc        = TransferDescriptor::deviceToHost(group, std::move(device), disk);
    desc.target_tier = Tier::DISK;
    return desc;
}

TEST(TransferTaskTest, SubtasksPreserveQueryIdAndDeadline) {
    TransferTask task({TransferDescriptor::hostToDevice(0, 1, {2})}, std::chrono::seconds(10), 73);
    auto         child      = task.subtask(0, task.descriptors().size());
    auto         grandchild = child.subtask(0, child.descriptors().size());
    EXPECT_EQ(child.deadline(), task.deadline());
    EXPECT_EQ(grandchild.deadline(), task.deadline());
    EXPECT_EQ(grandchild.requestId(), 73);
}

TEST(TransferTaskTest, RpcBatchesHaveDistinctExecutionTracesWithinWorkflow) {
    TransferTask task({TransferDescriptor::hostToDevice(0, 1, {2})}, std::chrono::seconds(10), 73);
    auto         first  = task.rpcSubtask({TransferDescriptor::hostToDevice(1, 3, {4})});
    auto         second = task.rpcSubtask({TransferDescriptor::hostToDevice(2, 5, {6})});

    EXPECT_NE(first.traceId(), 0u);
    EXPECT_NE(second.traceId(), 0u);
    EXPECT_NE(first.traceId(), task.traceId());
    EXPECT_NE(second.traceId(), task.traceId());
    EXPECT_NE(first.traceId(), second.traceId());
    EXPECT_EQ(first.deadline(), task.deadline());
    EXPECT_EQ(second.deadline(), task.deadline());
    EXPECT_EQ(first.requestId(), task.requestId());
    EXPECT_EQ(second.requestId(), task.requestId());
    EXPECT_EQ(first.workflowId(), task.workflowId());
    EXPECT_EQ(second.workflowId(), task.workflowId());
    EXPECT_EQ(first.parentTraceId(), task.traceId());
    EXPECT_EQ(second.parentTraceId(), task.traceId());

    auto nested_rpc = first.rpcSubtask();
    EXPECT_NE(nested_rpc.traceId(), first.traceId());
    EXPECT_EQ(nested_rpc.workflowId(), task.workflowId());
    EXPECT_EQ(nested_rpc.parentTraceId(), first.traceId());
}

TEST(TransferTaskTest, WorkerSubtasksPreserveReceivedExecutionTrace) {
    TransferTask task({TransferDescriptor::hostToDevice(0, 1, {2})}, std::chrono::seconds(10), 73);
    const auto   rpc = task.rpcSubtask();
    TransferTask worker(std::vector<TransferDescriptor>(rpc.descriptors().begin(), rpc.descriptors().end()),
                        std::chrono::seconds(10),
                        rpc.requestId());
    worker.setTraceId(rpc.traceId());
    worker.setWorkflow(rpc.workflowId(), rpc.parentTraceId());

    auto child  = worker.subtask({TransferDescriptor::hostToDevice(1, 3, {4})});
    auto nested = child.subtask({TransferDescriptor::hostToDevice(2, 5, {6})});

    EXPECT_EQ(worker.traceId(), rpc.traceId());
    EXPECT_EQ(child.traceId(), rpc.traceId());
    EXPECT_EQ(nested.traceId(), rpc.traceId());
    EXPECT_EQ(nested.workflowId(), task.workflowId());
    EXPECT_EQ(nested.parentTraceId(), task.traceId());
    EXPECT_EQ(nested.deadline(), worker.deadline());
    EXPECT_EQ(nested.requestId(), worker.requestId());
}

TEST(TransferTaskTest, CopyAndMovePreserveExecutionTrace) {
    TransferTask parent({TransferDescriptor::hostToDevice(0, 1, {2})}, std::chrono::seconds(10), 73);
    const auto   task   = parent.rpcSubtask();
    auto         copied = task;
    auto         moved  = std::move(copied);
    EXPECT_EQ(moved.traceId(), task.traceId());
    EXPECT_EQ(moved.workflowId(), task.workflowId());
    EXPECT_EQ(moved.parentTraceId(), task.parentTraceId());

    TransferTask assigned({}, std::chrono::seconds(1));
    assigned = task;
    EXPECT_EQ(assigned.traceId(), task.traceId());
    assigned = std::move(moved);
    EXPECT_EQ(assigned.traceId(), task.traceId());
    EXPECT_EQ(assigned.workflowId(), task.workflowId());
    EXPECT_EQ(assigned.parentTraceId(), task.parentTraceId());
}

TEST(TransferTaskTest, CompletionProofFollowsOwnershipButNotNewSubtasks) {
    TransferTask task({TransferDescriptor::hostToDevice(0, 1, {2})}, std::chrono::seconds(10));
    EXPECT_EQ(task.completionProof(), nullptr);
    auto proof = std::make_shared<TransferBufferSafety>([] {});
    task.setCompletionProof(proof);
    EXPECT_EQ(task.completionProof(), proof);

    auto copied = task;
    auto moved  = std::move(copied);
    EXPECT_EQ(moved.completionProof(), proof);
    EXPECT_EQ(task.subtask(0, task.descriptors().size()).completionProof(), nullptr);
    EXPECT_EQ(task.rpcSubtask().completionProof(), nullptr);

    auto owned = task.subtask(0, task.descriptors().size());
    owned.setCompletionProof(task.completionProof());
    EXPECT_EQ(owned.completionProof(), proof);
    task.setCompletionProof(nullptr);
    EXPECT_EQ(task.completionProof(), nullptr);
    EXPECT_EQ(owned.completionProof(), proof);
}

TEST(TransferTaskTest, RpcSubmissionSharesDescriptorsWithoutChangingParent) {
    TransferTask task({TransferDescriptor::hostToDevice(0, 1, {2, 3})}, std::chrono::seconds(10), 73);
    const auto   proof = std::make_shared<TransferBufferSafety>([] {});
    task.setCompletionProof(proof);
    const auto rpc = task.rpcSubtask();

    EXPECT_EQ(task.completionProof(), proof);
    EXPECT_EQ(rpc.descriptors().data(), task.descriptors().data());
    EXPECT_EQ(rpc.deadline(), task.deadline());
    EXPECT_EQ(rpc.requestId(), task.requestId());
    EXPECT_EQ(rpc.workflowId(), task.workflowId());
    EXPECT_EQ(rpc.parentTraceId(), task.traceId());
    EXPECT_NE(rpc.traceId(), task.traceId());
    EXPECT_EQ(rpc.completionProof(), nullptr);
}

TEST(TransferTaskTest, CopiesAndNestedRangesShareDescriptorsAfterParentDestruction) {
    const TransferDescriptor* original        = nullptr;
    const BlockIdxType*       original_blocks = nullptr;
    const auto                proof           = std::make_shared<TransferBufferSafety>([] {});
    auto                      child           = [&] {
        TransferTask task({TransferDescriptor::hostToDevice(0, 1, {2}),
                           TransferDescriptor::hostToDevice(0, 3, {4}),
                           TransferDescriptor::hostToDevice(0, 5, {6, 7})},
                          std::chrono::seconds(10),
                          73);
        original        = &task.descriptors()[2];
        original_blocks = original->target_blocks.data();
        auto copy       = task;
        EXPECT_EQ(copy.descriptors().data(), task.descriptors().data());
        const auto range = copy.subtask(1, 3);
        EXPECT_EQ(range.descriptors().data(), task.descriptors().data() + 1);
        EXPECT_EQ(range.descriptors().size(), 2u);
        const auto nested = range.subtask(1, 2, proof);
        EXPECT_EQ(nested.deadline(), task.deadline());
        EXPECT_EQ(nested.traceId(), task.traceId());
        EXPECT_EQ(nested.requestId(), task.requestId());
        EXPECT_EQ(range.completionProof(), nullptr);
        return nested;
    }();
    ASSERT_EQ(child.descriptors().size(), 1u);
    EXPECT_EQ(child.descriptors().data(), original);
    EXPECT_EQ(child.descriptors().front().target_blocks.data(), original_blocks);
    EXPECT_EQ(child.descriptors().front().target_blocks, (std::vector<BlockIdxType>{6, 7}));
    EXPECT_EQ(child.completionProof(), proof);
    auto assigned = child.subtask(0, 0);
    EXPECT_TRUE(assigned.descriptors().empty());
    assigned = child;
    EXPECT_EQ(assigned.descriptors().data(), original);
}

TEST(TransferTaskTest, SanityCheckAcceptsProducerGroupedRequests) {
    TransferTask task({TransferDescriptor::hostToDevice(0, 1, {2}),
                       TransferDescriptor::hostToDevice(0, 3, {4}),
                       TransferDescriptor::hostToDevice(1, 5, {6})},
                      std::chrono::seconds(10));
    EXPECT_TRUE(task.sanityCheck());
    EXPECT_TRUE(task.subtask(1, 3).sanityCheck());
    auto moved = std::move(task);
    EXPECT_TRUE(task.descriptors().empty());
    EXPECT_EQ(moved.descriptors().size(), 3u);
}

TEST(TransferResultTest, MergePreservesMessagesAndHighestPriorityCode) {
    for (bool crc_first : {false, true}) {
        TransferResult crc(TransferStatus::CRC_MISMATCH);
        TransferResult invalid(TransferStatus::INVALID_ARGS);
        TransferResult result;
        result.merge(crc_first ? crc : invalid);
        result.merge(TransferResult{});
        result.merge(crc_first ? invalid : crc);
        EXPECT_EQ(result.copy_error, CacheCopyError::CRC_MISMATCH);
        EXPECT_EQ(result.status.code(), ErrorCode::EXECUTION_EXCEPTION);
        EXPECT_EQ(result.status.ToString(),
                  crc_first ? "cache integrity verification failed; invalid block transfer request" :
                              "invalid block transfer request; cache integrity verification failed");
    }
}

TEST(TransferResultTest, BusinessStatusAndBlockDiagnosticsNeverImplyBufferSafety) {
    EXPECT_FALSE(TransferResult{}.buffer_safe);
    EXPECT_FALSE(TransferResult(3).buffer_safe);
    EXPECT_FALSE(TransferResult(TransferStatus::OK).buffer_safe);
    EXPECT_FALSE(TransferResult(ErrorInfo::OkStatus()).buffer_safe);
    EXPECT_FALSE(TransferResult(TransferStatus::CRC_MISMATCH).buffer_safe);
    EXPECT_FALSE(TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "not submitted"), CacheCopyError::NOT_EXECUTED)
                     .buffer_safe);
    const auto aggregate = TransferResult::emptyAggregate(3);
    EXPECT_TRUE(aggregate.buffer_safe);
    EXPECT_TRUE(aggregate.ok());
    EXPECT_EQ(aggregate.block_errors, std::vector<CacheCopyError>(3, CacheCopyError::NOT_EXECUTED));
}

TEST(TransferResultTest, ConfirmedCopyRequiresEverySafeSuccessfulBlock) {
    TransferResult result;
    result.expandBlockErrors(2);
    EXPECT_FALSE(result.confirmedCopy(2));

    result.buffer_safe = true;
    EXPECT_TRUE(result.confirmedCopy(2));
    EXPECT_FALSE(result.confirmedCopy(3));

    result.block_errors[1] = CacheCopyError::NOT_EXECUTED;
    EXPECT_TRUE(result.ok());
    EXPECT_FALSE(result.confirmedCopy(2));
}

TEST(TransferResultTest, EveryContributorMustProveSafetyRegardlessOfBusinessOutcome) {
    for (const bool first_safe : {false, true}) {
        for (const bool second_safe : {false, true}) {
            for (const bool first_ok : {false, true}) {
                for (const bool second_ok : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << first_safe << second_safe << first_ok << second_ok);
                    TransferResult first(first_ok ? TransferStatus::OK : TransferStatus::CRC_MISMATCH);
                    TransferResult second(second_ok ? TransferStatus::OK : TransferStatus::DISK_IO_ERROR);
                    first.buffer_safe  = first_safe;
                    second.buffer_safe = second_safe;
                    auto result        = TransferResult::emptyAggregate();
                    result.merge(first);
                    result.merge(second);
                    EXPECT_EQ(result.buffer_safe, first_safe && second_safe);
                    EXPECT_EQ(result.ok(), first_ok && second_ok);
                    EXPECT_EQ(result.copy_error,
                              !first_ok  ? CacheCopyError::CRC_MISMATCH :
                              !second_ok ? CacheCopyError::IO_FAILED :
                                           CacheCopyError::NONE);
                    // Further proven-safe successes cannot erase missing proof.
                    TransferResult safe_success(TransferStatus::OK);
                    safe_success.buffer_safe = true;
                    result.merge(safe_success);
                    EXPECT_EQ(result.buffer_safe, first_safe && second_safe);
                }
            }
        }
    }
}

TEST(TransferResultTest, UnknownCompletionKeepsBusinessPriorityAndBlocksAggregateSafety) {
    for (const bool unknown_first : {false, true}) {
        TransferResult crc(TransferStatus::CRC_MISMATCH);
        crc.buffer_safe = true;
        TransferResult unknown(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "RPC timed out"),
                               CacheCopyError::RPC_FAILED);
        auto           result = TransferResult::emptyAggregate();
        result.merge(unknown_first ? unknown : crc);
        result.merge(unknown_first ? crc : unknown);
        EXPECT_FALSE(result.buffer_safe);
        EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        EXPECT_EQ(result.copy_error, CacheCopyError::CRC_MISMATCH);
    }
}

TEST(TransferResultTest, NestedMergePreservesAllMessagesAndFirstCodeAtEqualPriority) {
    TransferResult result(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "first"));
    TransferResult batches(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "second"));
    batches.merge(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "third"));
    result.merge(std::move(batches));
    result.merge(TransferResult{});
    EXPECT_EQ(result.status.code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(result.status.ToString(), "first; second; third");
}

TEST(TransferDescriptorTest, FactoriesBuildExecutableDescriptors) {
    EXPECT_TRUE(TransferDescriptor::deviceToHost(0, {1, 2}, 3).isExecutable());
    EXPECT_TRUE(TransferDescriptor::hostToDevice(0, 3, {1, 2}).isExecutable());
    EXPECT_TRUE(TransferDescriptor::hostToDisk(0, 1, 2).isExecutable());
    EXPECT_TRUE(diskToHost(0, 1, 2).isExecutable());
    EXPECT_FALSE(diskToHost(0, 1, NULL_BLOCK_IDX).isExecutable());
    EXPECT_FALSE(removedDeviceToDisk(0, {1, 2}, 3).isExecutable());
    EXPECT_TRUE(TransferDescriptor::diskToDevice(0, 3, {1, 2}).isExecutable());
}

TEST(TransferDescriptorTest, OwnershipTracksGenerationForLoadEvictAndDrop) {
    for (const auto& direction : {std::make_pair(Tier::HOST, Tier::DEVICE),
                                  std::make_pair(Tier::DISK, Tier::DEVICE),
                                  std::make_pair(Tier::DEVICE, Tier::HOST),
                                  std::make_pair(Tier::HOST, Tier::DISK),
                                  std::make_pair(Tier::DEVICE, Tier::NONE)}) {
        SCOPED_TRACE(tierName(direction.first));
        SCOPED_TRACE(tierName(direction.second));
        TreeNode node;
        node.group_set_resources.resize(2);
        TransferDescriptor owner(&node, 1, 0, direction.first, direction.second, {1}, /*initial_owner=*/true);
        TransferDescriptor non_owner(&node, 1, 0, direction.first, direction.second, {1}, /*initial_owner=*/false);
        EXPECT_TRUE(owner.isCurrentOwner());
        EXPECT_FALSE(non_owner.isCurrentOwner());

        ++node.group_set_resources[0].generation;
        EXPECT_TRUE(owner.isCurrentOwner());
        ++node.group_set_resources[1].generation;
        EXPECT_TRUE(owner.initial_owner);
        EXPECT_FALSE(owner.isCurrentOwner());
        TransferDescriptor next_owner(&node, 1, 0, direction.first, direction.second, {1}, /*initial_owner=*/true);
        EXPECT_TRUE(next_owner.isCurrentOwner());
    }
    EXPECT_FALSE(TransferDescriptor{}.isCurrentOwner());
}

TEST(TransferDescriptorTest, IsExecutableRequiresResolvedLocalEndpoints) {
    EXPECT_FALSE(TransferDescriptor{}.isExecutable());
    EXPECT_FALSE(TransferDescriptor::deviceToHost(0, {}, 3).isExecutable());
    EXPECT_FALSE(TransferDescriptor::deviceToHost(0, {1, NULL_BLOCK_IDX}, 3).isExecutable());
    EXPECT_FALSE(TransferDescriptor::hostToDevice(0, NULL_BLOCK_IDX, {1}).isExecutable());
    EXPECT_FALSE(TransferDescriptor::hostToDisk(0, 1, NULL_BLOCK_IDX).isExecutable());
    EXPECT_FALSE(TransferDescriptor::diskToDevice(0, 3, {NULL_BLOCK_IDX}).isExecutable());

    TransferDescriptor remote_to_device;
    remote_to_device.source_tier   = Tier::REMOTE;
    remote_to_device.target_tier   = Tier::DEVICE;
    remote_to_device.source_blocks = {1};
    remote_to_device.target_blocks = {2};
    EXPECT_FALSE(remote_to_device.isExecutable());
}

TEST(TransferDescriptorTest, NeedsTransferDistinguishesLogicalSettlement) {
    TransferDescriptor resident_desc;
    resident_desc.source_tier   = Tier::DEVICE;
    resident_desc.target_tier   = Tier::DEVICE;
    resident_desc.source_blocks = {1, 2};
    resident_desc.target_blocks = {1, 2};
    EXPECT_FALSE(resident_desc.needsTransfer());
    EXPECT_FALSE(resident_desc.isExecutable());

    TransferDescriptor released_desc;
    released_desc.source_tier   = Tier::DISK;
    released_desc.target_tier   = Tier::NONE;
    released_desc.source_blocks = {3};
    EXPECT_FALSE(released_desc.needsTransfer());
    EXPECT_FALSE(released_desc.isExecutable());

    EXPECT_TRUE(TransferDescriptor::diskToDevice(0, 3, {1, 2}).needsTransfer());
}

TEST(TransferDescriptorTest, TransferValidationIgnoresPathMetadata) {
    TransferDescriptor desc = TransferDescriptor::hostToDevice(7, 3, {1, 2});
    desc.path_index         = 11;

    EXPECT_TRUE(desc.isExecutable());
    EXPECT_EQ(desc.debugString(),
              "TransferDescriptor{group_set_id=7, direction=HOST->DEVICE, source_blocks=[3], target_blocks=[1,2]}");
}

TEST(TransferTaskTest, ExpiredTaskHasNoRemainingTimeout) {
    TransferTask task({TransferDescriptor::hostToDevice(0, 3, {1, 2})}, std::chrono::seconds(1));
    task.deadline_             = TransferTask::Clock::now() - std::chrono::milliseconds(1);
    const auto parent_deadline = task.deadline_;

    auto subtask = task.subtask({TransferDescriptor::hostToDevice(1, 4, {5, 6})});

    EXPECT_TRUE(task.expired());
    EXPECT_FALSE(task.remainingTimeout().has_value());
    EXPECT_TRUE(subtask.expired());
    EXPECT_EQ(subtask.deadline_, parent_deadline);
}

TEST(TransferTaskTest, SubtaskPreservesAbsoluteDeadline) {
    TransferTask task({TransferDescriptor::hostToDevice(0, 3, {1, 2})}, std::chrono::seconds(1));
    task.deadline_             = TransferTask::Clock::now() + std::chrono::seconds(10) + std::chrono::microseconds(500);
    const auto parent_deadline = task.deadline_;

    auto subtask = task.subtask({TransferDescriptor::hostToDevice(1, 4, {5, 6})});
    auto nested  = subtask.subtask({TransferDescriptor::hostToDevice(2, 7, {8, 9})});

    EXPECT_EQ(subtask.deadline_, parent_deadline);
    EXPECT_EQ(nested.deadline_, parent_deadline);
    ASSERT_EQ(subtask.descriptors().size(), 1u);
    EXPECT_EQ(subtask.descriptors().front().group_set_id, 1u);
}

}  // namespace
}  // namespace rtp_llm
