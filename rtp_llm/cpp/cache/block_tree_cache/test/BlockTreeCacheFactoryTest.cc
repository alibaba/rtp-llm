#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

#include <cstdlib>
#include <stdexcept>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/allocator/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/CopyEngineTestUtils.h"
#include "rtp_llm/cpp/cache/spec/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {

struct BatchCopyParams;

// kv_cache_allocator is one archive containing init and blockBatchCopy(). This target
// exercises only init; keep the unexecuted platform hook fail-fast instead of linking
// the complete model CUDA runtime into a focused Factory test.
void execBatchCopy(const BatchCopyParams&) {
    throw std::logic_error("BlockTreeCacheFactoryTest must not execute allocator blockBatchCopy");
}

TEST(BlockTreeCacheFactoryTest, UsableBlockCountReservesBlockZeroWithinBudget) {
    // Budget fits 4 blocks: reserved block 0 counts within the budget, so usable = 3.
    EXPECT_EQ(computeHostUsableBlockCount(4 * 4096, 4096), 3u);
    // Budget fits exactly 1 block: only the reserved block, usable = 0.
    EXPECT_EQ(computeHostUsableBlockCount(4096, 4096), 0u);
    // Budget smaller than one block: usable = 0.
    EXPECT_EQ(computeHostUsableBlockCount(100, 4096), 0u);
    // Defensive: zero stride returns 0 instead of dividing by zero.
    EXPECT_EQ(computeHostUsableBlockCount(4096, 0), 0u);
}

TEST(BlockTreeCacheFactoryTest, ShouldPinHostBlockPoolHonorsEnv) {
    ::unsetenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
    EXPECT_TRUE(shouldPinHostBlockPool());  // default on when unset

    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "0", 1);
    EXPECT_FALSE(shouldPinHostBlockPool());
    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "off", 1);
    EXPECT_FALSE(shouldPinHostBlockPool());
    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "FALSE", 1);
    EXPECT_FALSE(shouldPinHostBlockPool());
    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "1", 1);
    EXPECT_TRUE(shouldPinHostBlockPool());

    ::unsetenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
}

TEST(BlockTreeCacheFactoryTest, ResolveDiskMountPathSelectsByLocalRank) {
    EXPECT_EQ(resolveDiskMountPath("/mnt/d0,/mnt/d1,/mnt/d2", 3, 0), "/mnt/d0");
    EXPECT_EQ(resolveDiskMountPath("/mnt/d0,/mnt/d1,/mnt/d2", 3, 2), "/mnt/d2");
    // split() trims surrounding whitespace.
    EXPECT_EQ(resolveDiskMountPath(" /mnt/d0 , /mnt/d1 ", 2, 1), "/mnt/d1");
}

TEST(BlockTreeCacheFactoryTest, ResolveDiskMountPathRejectsCountMismatch) {
    // RTP_LLM_CHECK aborts unless core-dump-on-exception is disabled; flip it so the
    // guard is observable as a throw in this test env.
    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(resolveDiskMountPath("/mnt/d0,/mnt/d1", 3, 0));
    EXPECT_ANY_THROW(resolveDiskMountPath("", 1, 0));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;
}

TEST(BlockTreeCacheFactoryTest, ResolveDiskMountPathRejectsOutOfRangeRank) {
    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(resolveDiskMountPath("/mnt/d0,/mnt/d1", 2, 2));
    EXPECT_ANY_THROW(resolveDiskMountPath("/mnt/d0,/mnt/d1", 2, -1));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;
}

// BTC-ISSUE-19: Factory creates one managed DiskBlockPool per component group on the
// same mount path. The second pool cannot acquire DiskMountGuard's exclusive .lock.
// Re-enable when a Full+SWA factory product can share one mount safely while keeping
// both groups' Disk pools executable.
TEST(BlockTreeCacheFactoryTest, DISABLED_Factory_CreatesExecutableFullSWAConfig) {
    if (!block_tree_cache_test::cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    CacheConfig cache_config;
    cache_config.dtype                       = TYPE_FP16;
    cache_config.layer_num                   = 3;
    cache_config.layer_all_num               = 3;
    cache_config.block_num                   = 8;
    cache_config.seq_size_per_block          = 1;
    cache_config.kernel_seq_size_per_block   = 1;
    cache_config.use_independent_block_pools = true;

    std::vector<KVCacheSpecPtr> specs;
    for (size_t index = 0; index < 3; ++index) {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->dtype              = TYPE_FP16;
        spec->local_head_num_kv  = 1;
        spec->size_per_head      = 8;
        spec->seq_size_per_block = 1;
        specs.push_back(spec);
    }
    cache_config.fromGroupedSpecs(specs,
                                  {{0}, {1}, {2}},
                                  {CacheGroupType::FULL, CacheGroupType::FULL, CacheGroupType::SWA},
                                  {"full_kv", "full_aux", "swa_kv"});
    auto policies                   = cache_config.groupPoliciesSnapshot();
    policies[2].sliding_window_size = 2;
    cache_config.setGroupPolicies(policies);
    cache_config.group_seq_size_per_block = {1, 1, 1};
    cache_config.setGroupBlockLayout({8, 8, 8}, {32, 32, 32}, {0, 0, 0});
    cache_config.kv_block_stride_bytes       = 32;
    cache_config.kv_block_size_bytes         = 32;
    cache_config.block_size_bytes            = 32;
    cache_config.layer_to_block_stride_bytes = {32, 32, 32};

    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(cache_config);
    ASSERT_TRUE(allocator->init());
    ASSERT_EQ(allocator->groupBlockPools().size(), 3u);

    copy_engine_test::TempDirGuard disk_dir("block_tree_cache_factory_full_swa");
    KVCacheConfig                  kv_cache_config;
    kv_cache_config.enable_device_cache           = true;
    kv_cache_config.enable_memory_cache           = true;
    kv_cache_config.memory_cache_size_mb          = 1;
    kv_cache_config.enable_memory_cache_disk      = true;
    kv_cache_config.memory_cache_disk_size_mb     = 1;
    kv_cache_config.memory_cache_disk_paths       = disk_dir.path;
    kv_cache_config.memory_cache_disk_buffered_io = true;

    BlockTreeCachePtr factory_cache =
        createBlockTreeCache(cache_config, kv_cache_config, allocator, ParallelismConfig{});
    ASSERT_NE(factory_cache, nullptr);
    ASSERT_TRUE(factory_cache->isInitialized());
    ASSERT_EQ(factory_cache->componentGroups().size(), 2u);
    ASSERT_EQ(factory_cache->components().size(), 3u);
    ASSERT_EQ(factory_cache->per_tag_mapping_.size(), 3u);
    EXPECT_EQ(factory_cache->per_tag_mapping_[0].component_group_id, 0);
    EXPECT_EQ(factory_cache->per_tag_mapping_[0].local_pool_index, 0);
    EXPECT_EQ(factory_cache->per_tag_mapping_[1].component_group_id, 0);
    EXPECT_EQ(factory_cache->per_tag_mapping_[1].local_pool_index, 1);
    EXPECT_EQ(factory_cache->per_tag_mapping_[2].component_group_id, 1);
    EXPECT_EQ(factory_cache->per_tag_mapping_[2].local_pool_index, 0);

    auto swa_group = std::dynamic_pointer_cast<SWAComponentGroup>(factory_cache->componentGroups()[1]);
    ASSERT_NE(swa_group, nullptr);
    EXPECT_EQ(swa_group->slidingWindowSize(), 2u);
    EXPECT_EQ(swa_group->seqSizePerBlock(), 1u);

    for (const ComponentGroupPtr& group : factory_cache->componentGroups()) {
        ASSERT_NE(group, nullptr);
        ASSERT_NE(group->hostPool(), nullptr);
        ASSERT_NE(group->diskPool(), nullptr);
        ASSERT_EQ(group->component_indices.size(), group->devicePoolCount());

        GroupBlockSet device_blocks = group->allocateBlocks(Tier::DEVICE, 1);
        ASSERT_EQ(device_blocks.per_node.size(), 1u);
        ASSERT_EQ(device_blocks.per_node[0].size(), group->devicePoolCount());
        const BlockIdxType host_block = group->allocateSingleBlock(Tier::HOST);
        const BlockIdxType disk_block = group->allocateSingleBlock(Tier::DISK);
        ASSERT_NE(host_block, NULL_BLOCK_IDX);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);

        EXPECT_EQ(factory_cache->executeTransfer(TransferDescriptor::deviceToHost(
                      group->component_group_id, device_blocks.per_node[0], host_block)),
                  CopyStatus::OK);
        EXPECT_EQ(factory_cache->executeTransfer(
                      TransferDescriptor::hostToDisk(group->component_group_id, host_block, disk_block)),
                  CopyStatus::OK);

        group->unreferenceBlocks(device_blocks);
        group->releaseSingleBlock(Tier::HOST, host_block);
        group->releaseSingleBlock(Tier::DISK, disk_block);
    }
}

}  // namespace rtp_llm
