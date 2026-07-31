#include <gtest/gtest.h>

#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;

std::vector<GroupSetPtr> makeGroupSets() {
    MemoryLayoutConfig memory_layout;
    memory_layout.layer_num                = 1;
    memory_layout.block_num                = 128;
    memory_layout.kv_block_pool_size_bytes = 128;

    auto device_config                  = std::make_shared<DeviceBlockPoolConfig>();
    device_config->pool_type            = BlockPoolType::DEVICE;
    device_config->pool_name            = "converter_device";
    device_config->physical_block_count = 128;
    device_config->memory_layouts       = {memory_layout};
    DeviceBlockPoolPtr device_pool      = std::make_shared<DeviceBlockPool>(device_config);

    auto host_config                  = std::make_shared<HostBlockPoolConfig>();
    host_config->pool_type            = BlockPoolType::HOST;
    host_config->pool_name            = "converter_host";
    host_config->physical_block_count = 128;
    auto host_pool                    = std::make_shared<HostBlockPool>(host_config);

    auto disk_config             = std::make_shared<BlockTreeDiskBlockPoolConfig>();
    disk_config->pool_type       = BlockPoolType::DISK;
    disk_config->pool_name       = "converter_disk";
    disk_config->stride_bytes    = 4096;
    disk_config->disk_size_bytes = 128 * disk_config->stride_bytes;
    auto disk_pool               = std::make_shared<BlockTreeDiskBlockPool>(disk_config);

    std::vector<GroupBase> groups(7, makeTestGroupBase());
    for (auto& group : groups) {
        group.kv_block_stride_bytes = 128;
    }
    const auto topology = makeTestTopology(std::move(groups));

    const std::vector<std::vector<size_t>> memberships = {{0}, {1, 2}, {3, 4}, {5}, {6}};
    std::vector<GroupSetPtr>               group_sets;
    for (size_t group_set_id = 0; group_set_id < memberships.size(); ++group_set_id) {
        const auto& membership = memberships[group_set_id];
        auto group_set = makeTestGroupSet(group_set_id,
                                              topology,
                                              membership,
                                              std::vector<DeviceBlockPoolPtr>(membership.size(), device_pool),
                                              host_pool,
                                              disk_pool);
        group_sets.push_back(std::move(group_set));
    }
    return group_sets;
}

const std::vector<GroupSetPtr>& groupSets() {
    static const std::vector<GroupSetPtr> group_sets = makeGroupSets();
    return group_sets;
}

std::unordered_map<size_t, BlockIdxType> wireBlocks(const MemoryOperationRequestPB::CopyItem& item) {
    std::unordered_map<size_t, BlockIdxType> blocks;
    for (const auto& group_block : item.group_blocks()) {
        blocks.emplace(static_cast<size_t>(group_block.group_id()), group_block.block_id());
    }
    return blocks;
}

void expectDecodeFailure(const MemoryOperationRequestPB& request) {
    EXPECT_FALSE(BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets()).isExecutable());
}

TEST(TransferDescriptorTest, IsExecutableRequiresResolvedEndpointsPerDirection) {
    EXPECT_TRUE(TransferDescriptor::deviceToHost(0, {1, 2}, 3).isExecutable());
    EXPECT_TRUE(TransferDescriptor::hostToDevice(0, 3, {1, 2}).isExecutable());
    EXPECT_TRUE(TransferDescriptor::hostToDisk(0, 1, 2).isExecutable());
    EXPECT_TRUE(TransferDescriptor::diskToHost(0, 1, 2).isExecutable());
    EXPECT_TRUE(TransferDescriptor::deviceToDisk(0, {1}, 2).isExecutable());
    EXPECT_TRUE(TransferDescriptor::diskToDevice(0, 2, {1}).isExecutable());

    EXPECT_FALSE(TransferDescriptor{}.isExecutable());
    EXPECT_FALSE(TransferDescriptor::deviceToHost(0, {1, NULL_BLOCK_IDX, 2}, 3).isExecutable());
    EXPECT_FALSE(TransferDescriptor::hostToDevice(0, 3, {}).isExecutable());
    EXPECT_FALSE(TransferDescriptor::deviceToHost(0, {1}, NULL_BLOCK_IDX).isExecutable());
    EXPECT_FALSE(TransferDescriptor::hostToDisk(0, NULL_BLOCK_IDX, 2).isExecutable());
    EXPECT_FALSE(TransferDescriptor::deviceToDisk(0, {1}, NULL_BLOCK_IDX).isExecutable());
    EXPECT_FALSE(TransferDescriptor::diskToDevice(0, 2, {NULL_BLOCK_IDX}).isExecutable());

    // Host↔Disk must not carry device blocks: this is the only single-layer constraint.
    TransferDescriptor host_to_disk = TransferDescriptor::hostToDisk(0, 1, 2);
    host_to_disk.source_blocks      = {1, 3};
    EXPECT_FALSE(host_to_disk.isExecutable());
    TransferDescriptor disk_to_host = TransferDescriptor::diskToHost(0, 1, 2);
    disk_to_host.target_blocks      = {2, 3};
    EXPECT_FALSE(disk_to_host.isExecutable());
}

TEST(BlockTransferRequestConverterTest, ConvertsDeviceToHost) {
    const TransferDescriptor input = TransferDescriptor::deviceToHost(2, {11, 12}, 21);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(input, groupSets(), request));
    ASSERT_EQ(request.copy_items_size(), 1);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::D2H);
    const auto& item = request.copy_items(0);
    EXPECT_EQ(item.group_set_id(), 2);
    EXPECT_EQ(item.mem_block(), 21);
    EXPECT_EQ(wireBlocks(item), (std::unordered_map<size_t, BlockIdxType>{{3, 11}, {4, 12}}));

    const auto output = BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets());
    ASSERT_TRUE(output.isExecutable());
    EXPECT_EQ(output.group_set_id, 2);
    EXPECT_EQ(output.source_tier, Tier::DEVICE);
    EXPECT_EQ(output.target_tier, Tier::HOST);
    EXPECT_EQ(output.singleBlockAt(Tier::HOST), 21);
    EXPECT_EQ(output.blocksAt(Tier::DEVICE), (std::vector<BlockIdxType>{11, 12}));
}

TEST(BlockTransferRequestConverterTest, ConvertsHostToDevice) {
    const TransferDescriptor input = TransferDescriptor::hostToDevice(1, 31, {41, 42});
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(input, groupSets(), request));
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2D);
    EXPECT_EQ(request.copy_items(0).group_set_id(), 1);
    EXPECT_EQ(wireBlocks(request.copy_items(0)), (std::unordered_map<size_t, BlockIdxType>{{1, 41}, {2, 42}}));

    const auto output = BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets());
    ASSERT_TRUE(output.isExecutable());
    EXPECT_EQ(output.group_set_id, 1);
    EXPECT_EQ(output.source_tier, Tier::HOST);
    EXPECT_EQ(output.target_tier, Tier::DEVICE);
    EXPECT_EQ(output.singleBlockAt(Tier::HOST), 31);
    EXPECT_EQ(output.blocksAt(Tier::DEVICE), (std::vector<BlockIdxType>{41, 42}));
}

TEST(BlockTransferRequestConverterTest, ConvertsHostToDisk) {
    const TransferDescriptor input = TransferDescriptor::hostToDisk(3, 51, 61);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(input, groupSets(), request));
    const auto& item = request.copy_items(0);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2DISK);
    EXPECT_EQ(item.group_set_id(), 3);
    EXPECT_EQ(item.backing_type(), MemoryOperationRequestPB::DISK);
    EXPECT_EQ(item.src_backing_type(), MemoryOperationRequestPB::MEMORY);
    EXPECT_EQ(item.src_mem_block(), 51);
    EXPECT_EQ(item.disk_slot(), 61);
    EXPECT_EQ(item.group_blocks_size(), 0);

    const auto output = BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets());
    ASSERT_TRUE(output.isExecutable());
    EXPECT_EQ(output.group_set_id, 3);
    EXPECT_EQ(output.source_tier, Tier::HOST);
    EXPECT_EQ(output.target_tier, Tier::DISK);
    EXPECT_EQ(output.singleBlockAt(Tier::HOST), 51);
    EXPECT_EQ(output.singleBlockAt(Tier::DISK), 61);
}

TEST(BlockTransferRequestConverterTest, ConvertsDiskToHost) {
    const TransferDescriptor input = TransferDescriptor::diskToHost(4, 71, 81);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(input, groupSets(), request));
    const auto& item = request.copy_items(0);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::DISK2H);
    EXPECT_EQ(item.group_set_id(), 4);
    EXPECT_EQ(item.backing_type(), MemoryOperationRequestPB::MEMORY);
    EXPECT_EQ(item.src_backing_type(), MemoryOperationRequestPB::DISK);
    EXPECT_EQ(item.src_disk_slot(), 71);
    EXPECT_EQ(item.mem_block(), 81);
    EXPECT_EQ(item.group_blocks_size(), 0);

    const auto output = BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets());
    ASSERT_TRUE(output.isExecutable());
    EXPECT_EQ(output.group_set_id, 4);
    EXPECT_EQ(output.source_tier, Tier::DISK);
    EXPECT_EQ(output.target_tier, Tier::HOST);
    EXPECT_EQ(output.singleBlockAt(Tier::DISK), 71);
    EXPECT_EQ(output.singleBlockAt(Tier::HOST), 81);
}

TEST(BlockTransferRequestConverterTest, ConvertsDeviceToDisk) {
    const TransferDescriptor input = TransferDescriptor::deviceToDisk(3, {51}, 61);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(input, groupSets(), request));
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::D2DISK);
    EXPECT_EQ(item.group_set_id(), 3);
    EXPECT_EQ(item.backing_type(), MemoryOperationRequestPB::DISK);
    EXPECT_EQ(item.disk_slot(), 61);
    EXPECT_TRUE(isNullBlockIdx(item.mem_block()));
    EXPECT_EQ(wireBlocks(item), (std::unordered_map<size_t, BlockIdxType>{{5, 51}}));

    const auto output = BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets());
    ASSERT_TRUE(output.isExecutable());
    EXPECT_EQ(output.group_set_id, 3);
    EXPECT_EQ(output.source_tier, Tier::DEVICE);
    EXPECT_EQ(output.target_tier, Tier::DISK);
    EXPECT_EQ(output.singleBlockAt(Tier::DISK), 61);
    EXPECT_EQ(output.blocksAt(Tier::DEVICE), (std::vector<BlockIdxType>{51}));
}

TEST(BlockTransferRequestConverterTest, ConvertsDiskToDevice) {
    const TransferDescriptor input = TransferDescriptor::diskToDevice(4, 71, {81});
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(input, groupSets(), request));
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::DISK2D);
    EXPECT_EQ(item.group_set_id(), 4);
    // DISK2D has no primary non-device backing; the field is left at its proto3 default.
    EXPECT_EQ(item.backing_type(), MemoryOperationRequestPB::MEMORY);
    EXPECT_EQ(item.src_backing_type(), MemoryOperationRequestPB::DISK);
    EXPECT_EQ(item.src_disk_slot(), 71);
    EXPECT_TRUE(isNullBlockIdx(item.mem_block()));
    EXPECT_EQ(wireBlocks(item), (std::unordered_map<size_t, BlockIdxType>{{6, 81}}));

    const auto output = BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets());
    ASSERT_TRUE(output.isExecutable());
    EXPECT_EQ(output.group_set_id, 4);
    EXPECT_EQ(output.source_tier, Tier::DISK);
    EXPECT_EQ(output.target_tier, Tier::DEVICE);
    EXPECT_EQ(output.singleBlockAt(Tier::DISK), 71);
    EXPECT_EQ(output.blocksAt(Tier::DEVICE), (std::vector<BlockIdxType>{81}));
}

TEST(BlockTransferRequestConverterTest, PreservesGroupSetForIdenticalBlockIds) {
    MemoryOperationRequestPB request;
    const TransferDescriptor first  = TransferDescriptor::deviceToHost(0, {7}, 8);
    const TransferDescriptor second = TransferDescriptor::deviceToHost(2, {7, 9}, 8);

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(first, groupSets(), request));
    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(second, groupSets(), request));
    ASSERT_EQ(request.copy_items_size(), 2);
    EXPECT_EQ(request.copy_items(0).group_set_id(), 0);
    EXPECT_EQ(request.copy_items(1).group_set_id(), 2);

    const auto first_output  = BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets());
    const auto second_output = BlockTransferRequestConverter::decodeTransfer(request, 1, groupSets());
    ASSERT_TRUE(first_output.isExecutable());
    ASSERT_TRUE(second_output.isExecutable());
    EXPECT_EQ(first_output.group_set_id, 0);
    EXPECT_EQ(second_output.group_set_id, 2);
}

TEST(BlockTransferRequestConverterTest, RejectsInvalidGroupSetId) {
    MemoryOperationRequestPB request;
    request.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto* item = request.add_copy_items();
    item->set_group_set_id(99);
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    item->set_mem_block(2);
    auto* group_block = item->add_group_blocks();
    group_block->set_group_id(0);
    group_block->set_block_id(3);

    expectDecodeFailure(request);

    const TransferDescriptor invalid = TransferDescriptor::deviceToHost(99, {3}, 2);
    MemoryOperationRequestPB invalid_request;
    EXPECT_FALSE(BlockTransferRequestConverter::appendTransfer(invalid, groupSets(), invalid_request));
    EXPECT_EQ(invalid_request.copy_items_size(), 0);
}

TEST(BlockTransferRequestConverterTest, RejectsBlocksInvalidForBlockPools) {
    const TransferDescriptor invalid_device_block = TransferDescriptor::deviceToHost(0, {128}, 1);
    const TransferDescriptor invalid_host_block   = TransferDescriptor::deviceToHost(0, {1}, 0);
    const TransferDescriptor invalid_disk_block   = TransferDescriptor::hostToDisk(0, 1, 128);
    MemoryOperationRequestPB request;

    EXPECT_FALSE(BlockTransferRequestConverter::appendTransfer(invalid_device_block, groupSets(), request));
    EXPECT_FALSE(BlockTransferRequestConverter::appendTransfer(invalid_host_block, groupSets(), request));
    EXPECT_FALSE(BlockTransferRequestConverter::appendTransfer(invalid_disk_block, groupSets(), request));
    EXPECT_EQ(request.copy_items_size(), 0);

    request.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto* item = request.add_copy_items();
    item->set_group_set_id(0);
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    item->set_mem_block(1);
    auto* group_block = item->add_group_blocks();
    group_block->set_group_id(0);
    group_block->set_block_id(0);
    expectDecodeFailure(request);
}

TEST(BlockTransferRequestConverterTest, RestoresCanonicalPoolOrderFromMemberIds) {
    MemoryOperationRequestPB request;
    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(
        TransferDescriptor::hostToDevice(2, 21, {11, 12}), groupSets(), request));
    request.mutable_copy_items(0)->mutable_group_blocks()->SwapElements(0, 1);

    const auto output = BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets());
    ASSERT_TRUE(output.isExecutable());
    EXPECT_EQ(output.blocksAt(Tier::DEVICE), (std::vector<BlockIdxType>{11, 12}));
}

TEST(BlockTransferRequestConverterTest, RejectsMemberIdMismatch) {
    MemoryOperationRequestPB valid;
    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(
        TransferDescriptor::hostToDevice(2, 21, {11, 12}), groupSets(), valid));

    MemoryOperationRequestPB unknown = valid;
    unknown.mutable_copy_items(0)->mutable_group_blocks(0)->set_group_id(99);
    expectDecodeFailure(unknown);

    MemoryOperationRequestPB duplicate = valid;
    duplicate.mutable_copy_items(0)->mutable_group_blocks(1)->set_group_id(
        duplicate.copy_items(0).group_blocks(0).group_id());
    expectDecodeFailure(duplicate);

    MemoryOperationRequestPB incomplete = valid;
    incomplete.mutable_copy_items(0)->mutable_group_blocks()->RemoveLast();
    expectDecodeFailure(incomplete);

    MemoryOperationRequestPB wrong_set = valid;
    wrong_set.mutable_copy_items(0)->set_group_set_id(1);
    expectDecodeFailure(wrong_set);
}

TEST(BlockTransferRequestConverterTest, RejectsMixedDirections) {
    MemoryOperationRequestPB request;
    const TransferDescriptor d2h = TransferDescriptor::deviceToHost(0, {1}, 2);
    const TransferDescriptor h2d = TransferDescriptor::hostToDevice(0, 2, {1});

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(d2h, groupSets(), request));
    EXPECT_FALSE(BlockTransferRequestConverter::appendTransfer(h2d, groupSets(), request));
    EXPECT_EQ(request.copy_items_size(), 1);
}

}  // namespace
}  // namespace rtp_llm
