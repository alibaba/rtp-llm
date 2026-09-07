#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {
namespace {

TEST(TransferDescriptorTest, FactoriesBuildExecutableDescriptors) {
    EXPECT_TRUE(TransferDescriptor::deviceToHost(0, {1, 2}, 3).isExecutable());
    EXPECT_TRUE(TransferDescriptor::hostToDevice(0, 3, {1, 2}).isExecutable());
    EXPECT_TRUE(TransferDescriptor::hostToDisk(0, 1, 2).isExecutable());
    EXPECT_TRUE(TransferDescriptor::diskToHost(0, 1, 2).isExecutable());
    EXPECT_TRUE(TransferDescriptor::deviceToDisk(0, {1, 2}, 3).isExecutable());
    EXPECT_TRUE(TransferDescriptor::diskToDevice(0, 3, {1, 2}).isExecutable());
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
