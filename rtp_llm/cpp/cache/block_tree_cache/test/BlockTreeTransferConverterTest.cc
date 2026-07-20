#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTransferConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"

namespace rtp_llm {
namespace {

std::vector<ComponentGroupPtr> makeComponentGroups() {
    MemoryLayoutConfig memory_layout;
    memory_layout.layer_num                = 1;
    memory_layout.block_num                = 128;
    memory_layout.kv_block_pool_size_bytes = 128;

    BlockPoolConfig device_config;
    device_config.pool_name        = "converter_device";
    device_config.block_num        = 128;
    device_config.memory_layouts   = {memory_layout};
    DeviceBlockPoolPtr device_pool = std::make_shared<DeviceBlockPool>(std::make_shared<BlockPool>(device_config));

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

    std::vector<ComponentGroupPtr> component_groups;
    for (int group_id = 0; group_id < 5; ++group_id) {
        std::shared_ptr<FullComponentGroup> component_group = std::make_shared<FullComponentGroup>();
        component_group->component_group_id                 = group_id;
        if (group_id == 1 || group_id == 2) {
            const std::vector<std::string> tags = group_id == 2 ?
                                                      std::vector<std::string>{"zeta", "alpha"} :
                                                      std::vector<std::string>{"group_1_pool_0", "group_1_pool_1"};
            component_group->setDevicePools({device_pool, device_pool}, tags);
        } else {
            component_group->setDevicePools({device_pool}, {"group_" + std::to_string(group_id) + "_pool_0"});
        }
        component_group->setHostPool(host_pool);
        component_group->setDiskPool(disk_pool);
        component_groups.push_back(component_group);
    }
    return component_groups;
}

const std::vector<ComponentGroupPtr>& componentGroups() {
    static const std::vector<ComponentGroupPtr> component_groups = makeComponentGroups();
    return component_groups;
}

std::vector<std::string> wireTags(const MemoryOperationRequestPB::CopyItem& item) {
    return {item.component_group_tags().begin(), item.component_group_tags().end()};
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
    EXPECT_FALSE(BlockTreeTransferConverter::decodeTransfer(request, 0, componentGroups(), output));
    EXPECT_EQ(output.component_group_id, 4);
    EXPECT_EQ(output.device_blocks, (std::vector<BlockIdxType>{7}));
    EXPECT_EQ(output.host_block, 8);
}

TEST(BlockTreeTransferConverterTest, ConvertsDeviceToHost) {
    const TransferDescriptor input = TransferDescriptor::deviceToHost(2, {11, 12}, 21);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(input, componentGroups(), request));
    ASSERT_EQ(request.copy_items_size(), 1);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::D2H);
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(wireTags(item), (std::vector<std::string>{"alpha", "zeta"}));
    EXPECT_EQ(item.mem_block(), 21);
    EXPECT_EQ(taggedBlocks(item), (std::unordered_map<std::string, BlockIdxType>{{"zeta", 11}, {"alpha", 12}}));

    TransferDescriptor output;
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 0, componentGroups(), output));
    EXPECT_EQ(output.component_group_id, 2);
    EXPECT_EQ(output.source_tier, Tier::DEVICE);
    EXPECT_EQ(output.target_tier, Tier::HOST);
    EXPECT_EQ(output.host_block, 21);
    EXPECT_EQ(output.device_blocks, (std::vector<BlockIdxType>{11, 12}));
}

TEST(BlockTreeTransferConverterTest, ConvertsHostToDevice) {
    const TransferDescriptor input = TransferDescriptor::hostToDevice(1, 31, {41, NULL_BLOCK_IDX});
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(input, componentGroups(), request));
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2D);
    EXPECT_EQ(wireTags(request.copy_items(0)), (std::vector<std::string>{"group_1_pool_0", "group_1_pool_1"}));

    TransferDescriptor output;
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 0, componentGroups(), output));
    EXPECT_EQ(output.component_group_id, 1);
    EXPECT_EQ(output.source_tier, Tier::HOST);
    EXPECT_EQ(output.target_tier, Tier::DEVICE);
    EXPECT_EQ(output.host_block, 31);
    EXPECT_EQ(output.device_blocks, (std::vector<BlockIdxType>{41, NULL_BLOCK_IDX}));
}

TEST(BlockTreeTransferConverterTest, ConvertsHostToDisk) {
    const TransferDescriptor input = TransferDescriptor::hostToDisk(3, 51, 61);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(input, componentGroups(), request));
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2DISK);
    EXPECT_EQ(item.backing_type(), MemoryOperationRequestPB::DISK);
    EXPECT_EQ(item.src_backing_type(), MemoryOperationRequestPB::MEMORY);
    EXPECT_EQ(item.src_mem_block(), 51);
    EXPECT_EQ(item.disk_slot(), 61);
    EXPECT_EQ(wireTags(item), (std::vector<std::string>{"group_3_pool_0"}));

    TransferDescriptor output;
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 0, componentGroups(), output));
    EXPECT_EQ(output.component_group_id, 3);
    EXPECT_EQ(output.source_tier, Tier::HOST);
    EXPECT_EQ(output.target_tier, Tier::DISK);
    EXPECT_EQ(output.host_block, 51);
    EXPECT_EQ(output.disk_block, 61);
}

TEST(BlockTreeTransferConverterTest, ConvertsDiskToHost) {
    const TransferDescriptor input = TransferDescriptor::diskToHost(4, 71, 81);
    MemoryOperationRequestPB request;

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(input, componentGroups(), request));
    const MemoryOperationRequestPB::CopyItem& item = request.copy_items(0);
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::DISK2H);
    EXPECT_EQ(item.backing_type(), MemoryOperationRequestPB::MEMORY);
    EXPECT_EQ(item.src_backing_type(), MemoryOperationRequestPB::DISK);
    EXPECT_EQ(item.src_disk_slot(), 71);
    EXPECT_EQ(item.mem_block(), 81);
    EXPECT_EQ(wireTags(item), (std::vector<std::string>{"group_4_pool_0"}));

    TransferDescriptor output;
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 0, componentGroups(), output));
    EXPECT_EQ(output.component_group_id, 4);
    EXPECT_EQ(output.source_tier, Tier::DISK);
    EXPECT_EQ(output.target_tier, Tier::HOST);
    EXPECT_EQ(output.disk_block, 71);
    EXPECT_EQ(output.host_block, 81);
}

TEST(BlockTreeTransferConverterTest, PreservesGroupForIdenticalBlockIds) {
    MemoryOperationRequestPB request;
    const TransferDescriptor first  = TransferDescriptor::deviceToHost(0, {7}, 8);
    const TransferDescriptor second = TransferDescriptor::deviceToHost(2, {7, NULL_BLOCK_IDX}, 8);

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(first, componentGroups(), request));
    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(second, componentGroups(), request));
    ASSERT_EQ(request.copy_items_size(), 2);
    EXPECT_EQ(wireTags(request.copy_items(0)), (std::vector<std::string>{"group_0_pool_0"}));
    EXPECT_EQ(wireTags(request.copy_items(1)), (std::vector<std::string>{"alpha", "zeta"}));

    TransferDescriptor first_output;
    TransferDescriptor second_output;
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 0, componentGroups(), first_output));
    ASSERT_TRUE(BlockTreeTransferConverter::decodeTransfer(request, 1, componentGroups(), second_output));
    EXPECT_EQ(first_output.component_group_id, 0);
    EXPECT_EQ(second_output.component_group_id, 2);
}

TEST(BlockTreeTransferConverterTest, RejectsMissingOrInvalidGroup) {
    MemoryOperationRequestPB request;
    request.set_copy_direction(MemoryOperationRequestPB::D2H);
    MemoryOperationRequestPB::CopyItem* item = request.add_copy_items();
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    item->set_mem_block(2);
    auto* tagged_block = item->add_tagged_gpu_blocks();
    tagged_block->set_tag("group_0_pool_0");
    tagged_block->set_block_id(3);

    TransferDescriptor output;
    EXPECT_FALSE(BlockTreeTransferConverter::decodeTransfer(request, 0, componentGroups(), output));

    const TransferDescriptor invalid = TransferDescriptor::deviceToHost(-1, {3}, 2);
    MemoryOperationRequestPB invalid_request;
    EXPECT_FALSE(BlockTreeTransferConverter::appendTransfer(invalid, componentGroups(), invalid_request));
    EXPECT_EQ(invalid_request.copy_items_size(), 0);
}

TEST(BlockTreeTransferConverterTest, RejectsBlocksInvalidForBlockPools) {
    const TransferDescriptor invalid_device_block = TransferDescriptor::deviceToHost(0, {128}, 1);
    const TransferDescriptor invalid_host_block   = TransferDescriptor::deviceToHost(0, {1}, 0);
    const TransferDescriptor invalid_disk_block   = TransferDescriptor::hostToDisk(0, 1, 128);
    MemoryOperationRequestPB request;

    EXPECT_FALSE(BlockTreeTransferConverter::appendTransfer(invalid_device_block, componentGroups(), request));
    EXPECT_FALSE(BlockTreeTransferConverter::appendTransfer(invalid_host_block, componentGroups(), request));
    EXPECT_FALSE(BlockTreeTransferConverter::appendTransfer(invalid_disk_block, componentGroups(), request));
    EXPECT_EQ(request.copy_items_size(), 0);

    request.set_copy_direction(MemoryOperationRequestPB::D2H);
    MemoryOperationRequestPB::CopyItem* item = request.add_copy_items();
    item->add_component_group_tags("group_0_pool_0");
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    item->set_mem_block(1);
    auto* tagged_block = item->add_tagged_gpu_blocks();
    tagged_block->set_tag("group_0_pool_0");
    tagged_block->set_block_id(0);
    TransferDescriptor output;
    EXPECT_FALSE(BlockTreeTransferConverter::decodeTransfer(request, 0, componentGroups(), output));
}

TEST(BlockTreeTransferConverterTest, RejectsInvalidTagSetsBeforeDescriptorMutation) {
    MemoryOperationRequestPB valid;
    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(
        TransferDescriptor::deviceToHost(2, {11, 12}, 21), componentGroups(), valid));

    MemoryOperationRequestPB empty = valid;
    empty.mutable_copy_items(0)->clear_component_group_tags();
    expectDecodeFailureWithoutMutation(empty);

    MemoryOperationRequestPB duplicate = valid;
    duplicate.mutable_copy_items(0)->add_component_group_tags("alpha");
    expectDecodeFailureWithoutMutation(duplicate);

    MemoryOperationRequestPB unknown = valid;
    unknown.mutable_copy_items(0)->set_component_group_tags(0, "unknown");
    expectDecodeFailureWithoutMutation(unknown);

    MemoryOperationRequestPB non_exact = valid;
    non_exact.mutable_copy_items(0)->clear_component_group_tags();
    non_exact.mutable_copy_items(0)->add_component_group_tags("alpha");
    expectDecodeFailureWithoutMutation(non_exact);
}

TEST(BlockTreeTransferConverterTest, RejectsTaggedBlockSetMismatchBeforeDescriptorMutation) {
    MemoryOperationRequestPB valid;
    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(
        TransferDescriptor::hostToDevice(2, 21, {11, 12}), componentGroups(), valid));

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

TEST(BlockTreeTransferConverterTest, RejectsMixedDirections) {
    MemoryOperationRequestPB request;
    const TransferDescriptor d2h = TransferDescriptor::deviceToHost(0, {1}, 2);
    const TransferDescriptor h2d = TransferDescriptor::hostToDevice(0, 2, {1});

    ASSERT_TRUE(BlockTreeTransferConverter::appendTransfer(d2h, componentGroups(), request));
    EXPECT_FALSE(BlockTreeTransferConverter::appendTransfer(h2d, componentGroups(), request));
    EXPECT_EQ(request.copy_items_size(), 1);
}

}  // namespace
}  // namespace rtp_llm
