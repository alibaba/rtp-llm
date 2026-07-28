#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::TestGroupSpec;

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

    std::shared_ptr<HostBlockPoolConfig> host_config = std::make_shared<HostBlockPoolConfig>();
    host_config->pool_type                           = BlockPoolType::HOST;
    host_config->pool_name                           = "converter_host";
    host_config->physical_block_count                = 128;
    std::shared_ptr<HostBlockPool> host_pool         = std::make_shared<HostBlockPool>(host_config);

    std::shared_ptr<BlockTreeDiskBlockPoolConfig> disk_config = std::make_shared<BlockTreeDiskBlockPoolConfig>();
    disk_config->pool_type                                    = BlockPoolType::DISK;
    disk_config->pool_name                                    = "converter_disk";
    disk_config->stride_bytes                                 = 4096;
    disk_config->disk_size_bytes                              = 128 * disk_config->stride_bytes;
    std::shared_ptr<BlockTreeDiskBlockPool> disk_pool         = std::make_shared<BlockTreeDiskBlockPool>(disk_config);

    const std::vector<std::string> topology_tags = {
        "group_0_pool_0", "group_1_pool_0", "group_1_pool_1", "zeta", "alpha", "group_3_pool_0", "group_4_pool_0"};
    std::vector<TestGroupSpec> specs;
    specs.reserve(topology_tags.size());
    for (const auto& tag : topology_tags) {
        TestGroupSpec spec;
        spec.tag                   = tag;
        spec.kv_block_stride_bytes = 128;
        specs.push_back(std::move(spec));
    }
    const auto topology = makeTestTopology(std::move(specs));

    const std::vector<std::vector<size_t>> memberships = {{0}, {1, 2}, {3, 4}, {5}, {6}};
    std::vector<GroupSetPtr>               group_sets;
    for (size_t group_set_id = 0; group_set_id < 5; ++group_set_id) {
        const auto& membership = memberships[group_set_id];
        auto        group_set  = makeTestGroupSet(
            group_set_id, topology, membership, std::vector<DeviceBlockPoolPtr>(membership.size(), device_pool));
        group_set->setHostPool(host_pool);
        group_set->setDiskPool(disk_pool);
        group_sets.push_back(std::move(group_set));
    }
    return group_sets;
}

const std::vector<GroupSetPtr>& groupSets() {
    static const std::vector<GroupSetPtr> group_sets = makeGroupSets();
    return group_sets;
}

std::vector<std::string> wireTags(const MemoryOperationRequestPB::CopyItem& item) {
    return {item.group_set_tags().begin(), item.group_set_tags().end()};
}

std::unordered_map<std::string, BlockIdxType> taggedBlocks(const MemoryOperationRequestPB::CopyItem& item) {
    std::unordered_map<std::string, BlockIdxType> blocks;
    for (const auto& tagged_block : item.tagged_gpu_blocks()) {
        blocks.emplace(tagged_block.tag(), tagged_block.block_id());
    }
    return blocks;
}

void expectDecodeFailureWithoutMutation(const MemoryOperationRequestPB& request) {
    TransferDescriptor output = TransferDescriptor::deviceToHost(4, {7}, 8);
    EXPECT_FALSE(BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets(), output));
    EXPECT_EQ(output.group_set_id, 4);
    EXPECT_EQ(output.device_blocks, (std::vector<BlockIdxType>{7}));
    EXPECT_EQ(output.host_block, 8);
}

TEST(BlockTransferRequestConverterTest, ConvertsDeviceToHost) {
    const TransferDescriptor input = TransferDescriptor::deviceToHost(2, {11, 12}, 21);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(input, groupSets(), request));
    ASSERT_EQ(request.copy_items_size(), 1);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::D2H);
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(wireTags(item), (std::vector<std::string>{"alpha", "zeta"}));
    EXPECT_EQ(item.mem_block(), 21);
    EXPECT_EQ(taggedBlocks(item), (std::unordered_map<std::string, BlockIdxType>{{"zeta", 11}, {"alpha", 12}}));

    TransferDescriptor output;
    ASSERT_TRUE(BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets(), output));
    EXPECT_EQ(output.group_set_id, 2);
    EXPECT_EQ(output.source_tier, Tier::DEVICE);
    EXPECT_EQ(output.target_tier, Tier::HOST);
    EXPECT_EQ(output.host_block, 21);
    EXPECT_EQ(output.device_blocks, (std::vector<BlockIdxType>{11, 12}));
}

TEST(BlockTransferRequestConverterTest, ConvertsHostToDevice) {
    const TransferDescriptor input = TransferDescriptor::hostToDevice(1, 31, {41, NULL_BLOCK_IDX});
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(input, groupSets(), request));
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2D);
    EXPECT_EQ(wireTags(request.copy_items(0)), (std::vector<std::string>{"group_1_pool_0", "group_1_pool_1"}));

    TransferDescriptor output;
    ASSERT_TRUE(BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets(), output));
    EXPECT_EQ(output.group_set_id, 1);
    EXPECT_EQ(output.source_tier, Tier::HOST);
    EXPECT_EQ(output.target_tier, Tier::DEVICE);
    EXPECT_EQ(output.host_block, 31);
    EXPECT_EQ(output.device_blocks, (std::vector<BlockIdxType>{41, NULL_BLOCK_IDX}));
}

TEST(BlockTransferRequestConverterTest, ConvertsHostToDisk) {
    const TransferDescriptor input = TransferDescriptor::hostToDisk(3, 51, 61);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(input, groupSets(), request));
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2DISK);
    EXPECT_EQ(item.backing_type(), MemoryOperationRequestPB::DISK);
    EXPECT_EQ(item.src_backing_type(), MemoryOperationRequestPB::MEMORY);
    EXPECT_EQ(item.src_mem_block(), 51);
    EXPECT_EQ(item.disk_slot(), 61);
    EXPECT_EQ(wireTags(item), (std::vector<std::string>{"group_3_pool_0"}));

    TransferDescriptor output;
    ASSERT_TRUE(BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets(), output));
    EXPECT_EQ(output.group_set_id, 3);
    EXPECT_EQ(output.source_tier, Tier::HOST);
    EXPECT_EQ(output.target_tier, Tier::DISK);
    EXPECT_EQ(output.host_block, 51);
    EXPECT_EQ(output.disk_block, 61);
}

TEST(BlockTransferRequestConverterTest, ConvertsDiskToHost) {
    const TransferDescriptor input = TransferDescriptor::diskToHost(4, 71, 81);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(input, groupSets(), request));
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::DISK2H);
    EXPECT_EQ(item.backing_type(), MemoryOperationRequestPB::MEMORY);
    EXPECT_EQ(item.src_backing_type(), MemoryOperationRequestPB::DISK);
    EXPECT_EQ(item.src_disk_slot(), 71);
    EXPECT_EQ(item.mem_block(), 81);
    EXPECT_EQ(wireTags(item), (std::vector<std::string>{"group_4_pool_0"}));

    TransferDescriptor output;
    ASSERT_TRUE(BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets(), output));
    EXPECT_EQ(output.group_set_id, 4);
    EXPECT_EQ(output.source_tier, Tier::DISK);
    EXPECT_EQ(output.target_tier, Tier::HOST);
    EXPECT_EQ(output.disk_block, 71);
    EXPECT_EQ(output.host_block, 81);
}

TEST(BlockTransferRequestConverterTest, PreservesGroupForIdenticalBlockIds) {
    MemoryOperationRequestPB request;
    const TransferDescriptor first  = TransferDescriptor::deviceToHost(0, {7}, 8);
    const TransferDescriptor second = TransferDescriptor::deviceToHost(2, {7, NULL_BLOCK_IDX}, 8);

    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(first, groupSets(), request));
    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(second, groupSets(), request));
    ASSERT_EQ(request.copy_items_size(), 2);
    EXPECT_EQ(wireTags(request.copy_items(0)), (std::vector<std::string>{"group_0_pool_0"}));
    EXPECT_EQ(wireTags(request.copy_items(1)), (std::vector<std::string>{"alpha", "zeta"}));

    TransferDescriptor first_output;
    TransferDescriptor second_output;
    ASSERT_TRUE(BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets(), first_output));
    ASSERT_TRUE(BlockTransferRequestConverter::decodeTransfer(request, 1, groupSets(), second_output));
    EXPECT_EQ(first_output.group_set_id, 0);
    EXPECT_EQ(second_output.group_set_id, 2);
}

TEST(BlockTransferRequestConverterTest, RejectsMissingOrInvalidGroup) {
    MemoryOperationRequestPB request;
    request.set_copy_direction(MemoryOperationRequestPB::D2H);
    MemoryOperationRequestPB::CopyItem* item = request.add_copy_items();
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    item->set_mem_block(2);
    auto* tagged_block = item->add_tagged_gpu_blocks();
    tagged_block->set_tag("group_0_pool_0");
    tagged_block->set_block_id(3);

    TransferDescriptor output;
    EXPECT_FALSE(BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets(), output));

    const TransferDescriptor invalid = TransferDescriptor::deviceToHost(-1, {3}, 2);
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
    MemoryOperationRequestPB::CopyItem* item = request.add_copy_items();
    item->add_group_set_tags("group_0_pool_0");
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    item->set_mem_block(1);
    auto* tagged_block = item->add_tagged_gpu_blocks();
    tagged_block->set_tag("group_0_pool_0");
    tagged_block->set_block_id(0);
    TransferDescriptor output;
    EXPECT_FALSE(BlockTransferRequestConverter::decodeTransfer(request, 0, groupSets(), output));
}

TEST(BlockTransferRequestConverterTest, RejectsInvalidTagSetsBeforeDescriptorMutation) {
    MemoryOperationRequestPB valid;
    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(
        TransferDescriptor::deviceToHost(2, {11, 12}, 21), groupSets(), valid));

    MemoryOperationRequestPB empty = valid;
    empty.mutable_copy_items(0)->clear_group_set_tags();
    expectDecodeFailureWithoutMutation(empty);

    MemoryOperationRequestPB duplicate = valid;
    duplicate.mutable_copy_items(0)->add_group_set_tags("alpha");
    expectDecodeFailureWithoutMutation(duplicate);

    MemoryOperationRequestPB unknown = valid;
    unknown.mutable_copy_items(0)->set_group_set_tags(0, "unknown");
    expectDecodeFailureWithoutMutation(unknown);

    MemoryOperationRequestPB non_exact = valid;
    non_exact.mutable_copy_items(0)->clear_group_set_tags();
    non_exact.mutable_copy_items(0)->add_group_set_tags("alpha");
    expectDecodeFailureWithoutMutation(non_exact);
}

TEST(BlockTransferRequestConverterTest, RejectsTaggedBlockSetMismatchBeforeDescriptorMutation) {
    MemoryOperationRequestPB valid;
    ASSERT_TRUE(BlockTransferRequestConverter::appendTransfer(
        TransferDescriptor::hostToDevice(2, 21, {11, 12}), groupSets(), valid));

    MemoryOperationRequestPB unknown_tag = valid;
    unknown_tag.mutable_copy_items(0)->mutable_tagged_gpu_blocks(0)->set_tag("unknown");
    expectDecodeFailureWithoutMutation(unknown_tag);

    MemoryOperationRequestPB duplicate_tag = valid;
    duplicate_tag.mutable_copy_items(0)->mutable_tagged_gpu_blocks(1)->set_tag(
        duplicate_tag.copy_items(0).tagged_gpu_blocks(0).tag());
    expectDecodeFailureWithoutMutation(duplicate_tag);

    MemoryOperationRequestPB incomplete = valid;
    incomplete.mutable_copy_items(0)->mutable_tagged_gpu_blocks()->RemoveLast();
    expectDecodeFailureWithoutMutation(incomplete);
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
