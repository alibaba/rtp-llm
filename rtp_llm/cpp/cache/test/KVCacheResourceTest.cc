#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {
namespace test {

TEST(BlockIdsTest, NonFull_MirrorsKernelBlocks) {
    BlockIds ids(/*kernel_blocks_per_kv_block=*/1);

    ids.add(BlockIndicesType{1, 2, 3});
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{1, 2, 3}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{1, 2, 3}));

    ids.remove(std::vector<size_t>{1});
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{1, NULL_BLOCK_IDX, 3}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{1, NULL_BLOCK_IDX, 3}));

    ids.swap(0, 2);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{3, NULL_BLOCK_IDX, 1}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{3, NULL_BLOCK_IDX, 1}));

    ids.setAt(1, 9);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{3, 9, 1}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{3, 9, 1}));
}

TEST(BlockIdsTest, Full_ExpandsKernelBlocks) {
    BlockIds ids(/*kernel_blocks_per_kv_block=*/2);

    ids.add(BlockIndicesType{5, 7});
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{5, 7}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{10, 11, 14, 15}));

    ids.remove(std::vector<size_t>{0});
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{NULL_BLOCK_IDX, 7}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 14, 15}));

    ids.setAt(1, 3);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{NULL_BLOCK_IDX, 3}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 6, 7}));

    ids.resize(3, 2);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{NULL_BLOCK_IDX, 3, 2}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 6, 7, 4, 5}));

    ids.swap(1, 2);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{NULL_BLOCK_IDX, 2, 3}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 4, 5, 6, 7}));

    const auto popped = ids.popBack();
    ASSERT_EQ(popped, 3);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{NULL_BLOCK_IDX, 2}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 4, 5}));
}

TEST(KVCacheResourceTest, InitGroups_RespectsGroupTypesAndBlocksPerKvBlock) {
    KVCacheResource resource;
    resource.initGroups(/*group_num=*/2,
                        /*layer_num=*/3,
                        /*layer_to_group_id=*/{0, 1, 0},
                        /*kernel_blocks_per_kv_block=*/4,
                        /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR});

    ASSERT_EQ(resource.groupNums(), 2);
    ASSERT_EQ(resource.layerBlocks().size(), 3u);

    auto& g0 = resource.mutableBlockIds(0);
    auto& g1 = resource.mutableBlockIds(1);

    ASSERT_EQ(g0.kernelBlocksPerKvBlock(), 4u);
    ASSERT_EQ(g1.kernelBlocksPerKvBlock(), 1u);

    g0.add(BlockIndicesType{1});
    g1.add(BlockIndicesType{1});

    ASSERT_EQ(resource.blocks(0), (BlockIndicesType{1}));
    ASSERT_EQ(resource.kernelBlocks(0), (BlockIndicesType{4, 5, 6, 7}));

    ASSERT_EQ(resource.blocks(1), (BlockIndicesType{1}));
    ASSERT_EQ(resource.kernelBlocks(1), (BlockIndicesType{1}));
}

TEST(CacheConfigTest, KernelBlocksPerKvBlockSafeByDefault) {
    CacheConfig config;
    config.seq_size_per_block        = 1;
    config.kernel_seq_size_per_block = 0;
    ASSERT_EQ(config.kernelBlocksPerKvBlock(), 1u);

    config.seq_size_per_block        = 8;
    config.kernel_seq_size_per_block = 2;
    ASSERT_EQ(config.kernelBlocksPerKvBlock(), 4u);
}

TEST(BatchKVCacheResourceTest, BasicBatchOperations_WorkAsExpected) {
    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(/*group_nums=*/2,
                     /*layer_num=*/3,
                     /*layer_to_group_id=*/{0, 1, 0},
                     /*kernel_blocks_per_kv_block=*/4,
                     /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR});

    ASSERT_EQ(batch.batchSize(), 2);
    ASSERT_EQ(batch.groupNums(), 2);

    batch.setBatchBlocks(/*batch_id=*/0, /*group_id=*/0, BlockIndicesType{1, 2});
    ASSERT_EQ(batch.blocks(0, 0), (BlockIndicesType{1, 2}));
    ASSERT_EQ(batch.kernelBlocks(0, 0), (BlockIndicesType{4, 5, 6, 7, 8, 9, 10, 11}));

    batch.setBatchBlocks(/*batch_id=*/0, /*group_id=*/1, BlockIndicesType{9, 10});
    ASSERT_EQ(batch.blocks(0, 1), (BlockIndicesType{9, 10}));
    ASSERT_EQ(batch.kernelBlocks(0, 1), (BlockIndicesType{9, 10}));

    auto all_g0 = batch.getAllBatchBlocks(/*group_id=*/0);
    ASSERT_EQ(all_g0.size(), 2u);
    ASSERT_EQ(all_g0[0], (BlockIndicesType{1, 2}));

    batch.pushBackCacheKey(0, 100);
    batch.pushBackCacheKey(1, 200);
    ASSERT_TRUE(batch.hasCacheKeys());

    batch.popBackAllBatchCacheKeys();
    ASSERT_EQ(batch.cacheKeys(0).size(), 0u);
    ASSERT_EQ(batch.cacheKeys(1).size(), 0u);
    ASSERT_FALSE(batch.hasCacheKeys());

    batch.setLastBlockAligned(true);
    ASSERT_TRUE(batch.lastBlockAligned());
    batch.cacheResource(1).setLastBlockAligned(false);
    ASSERT_FALSE(batch.lastBlockAligned());

    std::vector<KVCacheResource> old_resources;
    batch.resetAndReturnOldResources(/*new_batch_size=*/1, old_resources);
    ASSERT_EQ(old_resources.size(), 2u);
    ASSERT_EQ(batch.batchSize(), 1);

    KVCacheResource moved;
    moved.initGroups(/*group_num=*/1,
                     /*layer_num=*/1,
                     /*layer_to_group_id=*/{0},
                     /*kernel_blocks_per_kv_block=*/2,
                     /*group_types=*/{CacheGroupType::FULL});
    moved.mutableBlockIds(0).add(BlockIndicesType{3});
    batch.moveBatchResource(0, std::move(moved));
    ASSERT_EQ(batch.cacheResource(0).groupNums(), 1);
    ASSERT_EQ(batch.cacheResource(0).kernelBlocks(0), (BlockIndicesType{6, 7}));
}

}  // namespace test
}  // namespace rtp_llm
