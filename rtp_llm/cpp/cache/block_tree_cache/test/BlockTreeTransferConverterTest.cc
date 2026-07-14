#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTransferConverter.h"

namespace rtp_llm {
namespace {

TEST(BlockTreeTransferConverterTest, ConvertsDeviceToHost) {
    const TransferDescriptor input = TransferDescriptor::deviceToHost(2, {11, 12}, 21);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(input, request));
    ASSERT_EQ(request.copy_items_size(), 1);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::D2H);
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(item.component_group_id(), 2);
    EXPECT_EQ(item.mem_block(), 21);
    ASSERT_EQ(item.gpu_blocks_size(), 2);
    EXPECT_EQ(item.gpu_blocks(0), 11);
    EXPECT_EQ(item.gpu_blocks(1), 12);

    TransferDescriptor output;
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 0, output));
    EXPECT_EQ(output.component_group_id, 2);
    EXPECT_EQ(output.source_tier, Tier::DEVICE);
    EXPECT_EQ(output.target_tier, Tier::HOST);
    EXPECT_EQ(output.host_block, 21);
    EXPECT_EQ(output.device_blocks, (std::vector<BlockIdxType>{11, 12}));
}

TEST(BlockTreeTransferConverterTest, ConvertsHostToDevice) {
    const TransferDescriptor input = TransferDescriptor::hostToDevice(1, 31, {41, NULL_BLOCK_IDX});
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(input, request));
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2D);

    TransferDescriptor output;
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 0, output));
    EXPECT_EQ(output.component_group_id, 1);
    EXPECT_EQ(output.source_tier, Tier::HOST);
    EXPECT_EQ(output.target_tier, Tier::DEVICE);
    EXPECT_EQ(output.host_block, 31);
    EXPECT_EQ(output.device_blocks, (std::vector<BlockIdxType>{41, NULL_BLOCK_IDX}));
}

TEST(BlockTreeTransferConverterTest, ConvertsHostToDisk) {
    const TransferDescriptor input = TransferDescriptor::hostToDisk(3, 51, 61);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(input, request));
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2DISK);
    EXPECT_EQ(item.backing_type(), MemoryOperationRequestPB::DISK);
    EXPECT_EQ(item.src_backing_type(), MemoryOperationRequestPB::MEMORY);
    EXPECT_EQ(item.src_mem_block(), 51);
    EXPECT_EQ(item.disk_slot(), 61);

    TransferDescriptor output;
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 0, output));
    EXPECT_EQ(output.component_group_id, 3);
    EXPECT_EQ(output.source_tier, Tier::HOST);
    EXPECT_EQ(output.target_tier, Tier::DISK);
    EXPECT_EQ(output.host_block, 51);
    EXPECT_EQ(output.disk_block, 61);
}

TEST(BlockTreeTransferConverterTest, ConvertsDiskToHost) {
    const TransferDescriptor input = TransferDescriptor::diskToHost(4, 71, 81);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(input, request));
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::DISK2H);
    EXPECT_EQ(item.backing_type(), MemoryOperationRequestPB::MEMORY);
    EXPECT_EQ(item.src_backing_type(), MemoryOperationRequestPB::DISK);
    EXPECT_EQ(item.src_disk_slot(), 71);
    EXPECT_EQ(item.mem_block(), 81);

    TransferDescriptor output;
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 0, output));
    EXPECT_EQ(output.component_group_id, 4);
    EXPECT_EQ(output.source_tier, Tier::DISK);
    EXPECT_EQ(output.target_tier, Tier::HOST);
    EXPECT_EQ(output.disk_block, 71);
    EXPECT_EQ(output.host_block, 81);
}

TEST(BlockTreeTransferConverterTest, PreservesGroupForIdenticalBlockIds) {
    MemoryOperationRequestPB request;
    const TransferDescriptor first  = TransferDescriptor::deviceToHost(0, {7}, 8);
    const TransferDescriptor second = TransferDescriptor::deviceToHost(2, {7}, 8);

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(first, request));
    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(second, request));
    ASSERT_EQ(request.copy_items_size(), 2);
    EXPECT_EQ(request.copy_items(0).component_group_id(), 0);
    EXPECT_EQ(request.copy_items(1).component_group_id(), 2);

    TransferDescriptor first_output;
    TransferDescriptor second_output;
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 0, first_output));
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 1, second_output));
    EXPECT_EQ(first_output.component_group_id, 0);
    EXPECT_EQ(second_output.component_group_id, 2);
}

TEST(BlockTreeTransferConverterTest, RejectsMissingOrInvalidGroup) {
    MemoryOperationRequestPB request;
    request.set_copy_direction(MemoryOperationRequestPB::D2H);
    MemoryOperationRequestPB::CopyItem* item = request.add_copy_items();
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    item->set_mem_block(2);
    item->add_gpu_blocks(3);

    TransferDescriptor output;
    EXPECT_FALSE(BlockTreeTransferConverter::decodeTransfer(request, 0, output));

    const TransferDescriptor invalid = TransferDescriptor::deviceToHost(-1, {3}, 2);
    MemoryOperationRequestPB invalid_request;
    EXPECT_FALSE(BlockTreeTransferConverter::appendTransfer(invalid, invalid_request));
    EXPECT_EQ(invalid_request.copy_items_size(), 0);

    const TransferDescriptor invalid_block = TransferDescriptor::deviceToHost(0, {3}, NULL_BLOCK_IDX);
    EXPECT_FALSE(BlockTreeTransferConverter::appendTransfer(invalid_block, invalid_request));
    EXPECT_EQ(invalid_request.copy_items_size(), 0);
}

TEST(BlockTreeTransferConverterTest, RejectsMixedDirections) {
    MemoryOperationRequestPB request;
    const TransferDescriptor d2h = TransferDescriptor::deviceToHost(0, {1}, 2);
    const TransferDescriptor h2d = TransferDescriptor::hostToDevice(0, 2, {1});

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(d2h, request));
    EXPECT_FALSE(BlockTreeTransferConverter::appendTransfer(h2d, request));
    EXPECT_EQ(request.copy_items_size(), 1);
}

}  // namespace
}  // namespace rtp_llm
