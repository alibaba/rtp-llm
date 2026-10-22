#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
#include "rtp_llm/cpp/cache/SWACacheManager.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace test {

using DeviceSWACacheManager = SWACacheManager;

namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value): name_(name) {
        const char* old_value = std::getenv(name_);
        if (old_value != nullptr) {
            old_value_ = old_value;
            had_value_ = true;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvVar() {
        if (had_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    std::string old_value_;
    bool        had_value_ = false;
};

std::shared_ptr<FixedStateCacheSpec> makeDsv4StateSpec(int seq_size_per_block) {
    return makeTestFixedStateSpec(static_cast<uint32_t>(seq_size_per_block));
}

CacheGroupPolicy makePolicy(bool enable_prefix_reuse = true) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.enable_prefix_reuse = enable_prefix_reuse;
    if (!enable_prefix_reuse) {
        policy.active_tail_blocks   = 1;
        policy.validate_tail_blocks = false;
    }
    return policy;
}

size_t validBlockCount(const BlockIndicesType& blocks) {
    return static_cast<size_t>(
        std::count_if(blocks.begin(), blocks.end(), [](BlockIdxType block) { return !isNullBlockIdx(block); }));
}

}  // namespace

class DeviceSWACacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        old_core_dump_on_exception_                  = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
        block_pool_                                  = createDeviceBlockPool();
        block_pool_->init();
        total_blocks_ = block_pool_->freeBlocksNum();
    }

    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_on_exception_;
    }

    DeviceSWACacheManager makeGroup(int seq_size_per_block) {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->seq_size_per_block = seq_size_per_block;
        return DeviceSWACacheManager({}, spec, block_pool_, 0, 0);
    }

    DeviceSWACacheManager makeGroupWithStep(int seq_size_per_block, int linear_step) {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->seq_size_per_block = seq_size_per_block;
        return DeviceSWACacheManager({}, spec, block_pool_, 0, linear_step, makePolicy());
    }

    DeviceBlockPoolPtr block_pool_;
    size_t             total_blocks_ = 0;
    bool               old_core_dump_on_exception_{false};
};

TEST_F(DeviceSWACacheManagerTest, DefaultPolicyDrivesBehaviorInterfaces) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;
    DeviceSWACacheManager group({}, spec, block_pool_, 0, 0);

    EXPECT_FALSE(group.prefixReusable());
    EXPECT_TRUE(group.hasSparseSlots());
    EXPECT_FALSE(group.hasKernelBlockSubdiv());
    EXPECT_TRUE(group.transferTailBlocks());
    EXPECT_TRUE(group.isReservable());
    EXPECT_NE(group.memoryPlacement(), CacheMemoryPlacement::HOST_PINNED);
}

TEST_F(DeviceSWACacheManagerTest, PrefillChunkBackfillsTailAndPreservesHistory) {
    auto group = makeGroup(256);
    BlockIds ids;
    ASSERT_TRUE(group.malloc(ids, 32768, false));
    ASSERT_EQ(ids.blocksNum(), 128);
    EXPECT_EQ(validBlockCount(ids.blocks()), 2);
    std::vector<size_t> filled;
    ASSERT_TRUE(group.preparePrefillChunk(ids, 8192, &filled));
    EXPECT_EQ(filled, (std::vector<size_t>{30, 31}));
    const auto previous_tail = ids.blocks()[31];
    group.releaseBeforePrefillChunk(ids, 0, false);
    ASSERT_TRUE(group.preparePrefillChunk(ids, 16384, &filled));
    EXPECT_EQ(filled, (std::vector<size_t>{62, 63}));
    group.releaseBeforePrefillChunk(ids, 8192, false);
    EXPECT_EQ(ids.blocks()[31], previous_tail);
    ASSERT_TRUE(group.preparePrefillChunk(ids, 16384, &filled));
    EXPECT_TRUE(filled.empty());
    ASSERT_TRUE(group.preparePrefillChunk(ids, 24576, &filled));
    group.releaseBeforePrefillChunk(ids, 16384, false);
    EXPECT_TRUE(isNullBlockIdx(ids.blocks()[31]));
    EXPECT_FALSE(isNullBlockIdx(ids.blocks()[63]));
    EXPECT_FALSE(isNullBlockIdx(ids.blocks()[95]));
    EXPECT_FALSE(isNullBlockIdx(ids.blocks()[127]));
    EXPECT_EQ(validBlockCount(ids.blocks()), 6);
    group.unreference(ids.blocks());
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

TEST_F(DeviceSWACacheManagerTest, PrefillPartialTailAndRollbackPositions) {
    auto group = makeGroup(256);
    BlockIds ids;
    ASSERT_TRUE(group.malloc(ids, 20000, false));
    std::vector<size_t> filled;
    ASSERT_TRUE(group.preparePrefillChunk(ids, 8192, &filled));
    BlockIndicesType allocated;
    for (const auto i : filled) allocated.push_back(ids.blocks()[i]);
    group.unreference(allocated);
    ids.remove(filled);
    EXPECT_EQ(validBlockCount(ids.blocks()), 2);
    ASSERT_TRUE(group.preparePrefillChunk(ids, 20000, &filled));
    EXPECT_TRUE(filled.empty());
    group.unreference(ids.blocks());
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

// ==================== needBlocksNum ====================

TEST_F(DeviceSWACacheManagerTest, NeedBlocksNum_Basic) {
    auto group = makeGroup(4);
    EXPECT_EQ(group.needBlocksNum(1, 0), 1);
    EXPECT_EQ(group.needBlocksNum(4, 0), 1);
    EXPECT_EQ(group.needBlocksNum(5, 0), 2);
    EXPECT_EQ(group.needBlocksNum(8, 0), 2);
    EXPECT_EQ(group.needBlocksNum(9, 0), 3);
}

TEST_F(DeviceSWACacheManagerTest, NeedBlocksNum_WithCurrentBlocks) {
    auto group = makeGroup(4);
    EXPECT_EQ(group.needBlocksNum(10, 1), 2);
    EXPECT_EQ(group.needBlocksNum(10, 3), 0);
    EXPECT_EQ(group.needBlocksNum(10, 5), 0);
}

TEST_F(DeviceSWACacheManagerTest, NeedBlocksNum_WithReserveStep) {
    auto group = makeGroup(4);
    // reserve_step formula: ceil((seq_len + reserve_step) / block_size) - current
    EXPECT_EQ(group.needBlocksNum(8, 0, 0), 2);  // ceil((8+0)/4) = 2
    EXPECT_EQ(group.needBlocksNum(8, 0, 1), 3);  // ceil((8+1)/4) = 3
    EXPECT_EQ(group.needBlocksNum(8, 0, 2), 3);  // ceil((8+2)/4) = 3
    EXPECT_EQ(group.needBlocksNum(8, 0, 5), 4);  // ceil((8+5)/4) = 4
}

// ==================== getNeedBlocks ====================

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_SeqLenZero) {
    auto group = makeGroup(4);
    auto need  = group.getNeedBlocks(0, 0, 0, 0, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 0);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_ReuseDisabledCountsActiveTail) {
    auto group = makeGroupWithStep(4, 2);
    // seq_len=12 => seq_slots=3, reuse disabled => last two active tail blocks.
    auto need = group.getNeedBlocks(0, 12, 0, 0, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_ReuseEnabledUsesSparse) {
    auto group = makeGroupWithStep(4, 2);
    // seq_len=12 => seq_slots=3
    // count_sparse(0,3): eligible=(3+1)/2-(0+1)/2=2-0=2, tail=(3+1)%2==0 => 0, total=2
    auto need = group.getNeedBlocks(0, 12, 0, 0, true);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_HCAStateReuseEnabledCountsTailOnly) {
    auto spec  = makeDsv4StateSpec(4);
    auto group = DeviceSWACacheManager({}, spec, block_pool_, 5, /*linear_step=*/3, makePolicy(false));

    // seq_len=40 => seq_slots=10. If reuse sparse allocation were enabled, step hits
    // would keep positions 2/5/8 plus tail position 9. HCA_STATE skips reuse and keeps only tail 9.
    auto need = group.getNeedBlocks(0, 40, 0, 0, true);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 1);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_CSAStateReuseEnabledStillUsesSparse) {
    auto spec  = makeDsv4StateSpec(4);
    auto group = DeviceSWACacheManager({}, spec, block_pool_, 4, /*linear_step=*/3, makePolicy());

    auto need = group.getNeedBlocks(0, 40, 0, 0, true);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 4);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_WithReserveStep) {
    auto group = makeGroupWithStep(4, 2);
    // seq_len=8 => two active tail blocks, plus one reserve block.
    auto need = group.getNeedBlocks(0, 8, 2, 0, false);
    EXPECT_EQ(need.extra_blocks, 3);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_ReusePartialOverlap) {
    auto group = makeGroupWithStep(4, 2);
    // seq_len=12 => seq_slots=3
    // reuse_blocks_len=2: count_sparse(2,3)
    // eligible=(3+1)/2-(2+1)/2=2-1=1, tail=(3+1)%2==0 => 0, total=1
    auto need = group.getNeedBlocks(0, 12, 0, 2, true);
    EXPECT_EQ(need.extra_blocks, 1);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_ReuseFullOverlap) {
    auto group = makeGroupWithStep(4, 2);
    // seq_len=12 => seq_slots=3
    // reuse_blocks_len=3: count_sparse(3,3) = 0
    auto need = group.getNeedBlocks(0, 12, 0, 3, true);
    EXPECT_EQ(need.extra_blocks, 0);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_CommonSeqLenIgnored) {
    auto group = makeGroup(4);
    auto need1 = group.getNeedBlocks(0, 20, 0, 0, false);
    auto need2 = group.getNeedBlocks(20, 20, 0, 0, false);
    auto need3 = group.getNeedBlocks(100, 20, 0, 0, false);
    EXPECT_EQ(need1.extra_blocks, need2.extra_blocks);
    EXPECT_EQ(need2.extra_blocks, need3.extra_blocks);
    EXPECT_EQ(need1.common_blocks, 0);
}

// ==================== malloc (default step=0, acts like step=1, tail-only) ====================

TEST_F(DeviceSWACacheManagerTest, Malloc_ShortSeq_OnlyOneBlock) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 3));
    EXPECT_EQ(block_ids.blocksNum(), 1u);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 1);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_ManyBlocks_LastTwoActiveBlocksReal) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 20));
    // reuse_cache=false still keeps the last two active blocks.
    ASSERT_EQ(block_ids.blocksNum(), 5u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[i])) << "position " << i << " should be NULL";
    }
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[4]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_DSV4PromptTailKeepsPenultimateBlock) {
    auto     group = makeGroup(256);
    BlockIds block_ids(1);

    ASSERT_TRUE(group.malloc(block_ids, 5121, /*enable_reuse_cache=*/false, /*reserve_step=*/0));

    ASSERT_EQ(block_ids.blocksNum(), 21u);
    for (int i = 0; i < 19; ++i) {
        EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[i])) << "position " << i << " should be NULL";
    }
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[19]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[20]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_NoOpWhenEnoughBlocks) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 8));
    size_t free_after_first = block_pool_->freeBlocksNum();

    ASSERT_TRUE(group.malloc(block_ids, 8));
    EXPECT_EQ(block_ids.blocksNum(), 2u);
    EXPECT_EQ(block_pool_->freeBlocksNum(), free_after_first);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_DSV4TrapSkipsHCAStateNullTail) {
    ScopedEnvVar env("DSV4_TRAP_INVALID_KV_ACCESS", "1");
    auto         spec  = makeDsv4StateSpec(4);
    auto         group = DeviceSWACacheManager({}, spec, block_pool_, 5, 0, makePolicy(false));
    BlockIds     block_ids(1);
    block_ids.assign(BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});

    EXPECT_NO_THROW((void)group.malloc(block_ids, 12));
}

TEST_F(DeviceSWACacheManagerTest, Malloc_HCAStateReuseEnabledAllocatesTailOnly) {
    auto     spec  = makeDsv4StateSpec(4);
    auto     group = DeviceSWACacheManager({}, spec, block_pool_, 5, /*linear_step=*/3, makePolicy(false));
    BlockIds block_ids(1);

    ASSERT_TRUE(group.malloc(block_ids, 40, /*enable_reuse_cache=*/true, /*reserve_step=*/0));

    ASSERT_EQ(block_ids.blocksNum(), 10u);
    EXPECT_EQ(validBlockCount(block_ids.blocks()), 1u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[8]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[9]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 1);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_CSAStateReuseEnabledKeepsSparseBlocks) {
    auto     spec  = makeDsv4StateSpec(4);
    auto     group = DeviceSWACacheManager({}, spec, block_pool_, 4, /*linear_step=*/3, makePolicy());
    BlockIds block_ids(1);

    ASSERT_TRUE(group.malloc(block_ids, 40, /*enable_reuse_cache=*/true, /*reserve_step=*/0));

    ASSERT_EQ(block_ids.blocksNum(), 10u);
    EXPECT_EQ(validBlockCount(block_ids.blocks()), 4u);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[5]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[8]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[9]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 4);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_DSV4TrapChecksSWAKVNullTail) {
    ScopedEnvVar env("DSV4_TRAP_INVALID_KV_ACCESS", "1");
    auto         spec  = makeDsv4StateSpec(4);
    auto         group = DeviceSWACacheManager({}, spec, block_pool_, 6, 0, makePolicy());
    BlockIds     block_ids(1);
    block_ids.assign(BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});

    EXPECT_THROW((void)group.malloc(block_ids, 12), std::exception);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_DSV4TrapChecksNonSkipStateNullTail) {
    ScopedEnvVar env("DSV4_TRAP_INVALID_KV_ACCESS", "1");
    auto         spec  = makeDsv4StateSpec(4);
    auto         group = DeviceSWACacheManager({}, spec, block_pool_, 4, 0, makePolicy());
    BlockIds     block_ids(1);
    block_ids.assign(BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});

    EXPECT_THROW((void)group.malloc(block_ids, 12), std::exception);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_WithReserveStep) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    // seq_len=4 => seq_slots=1, reserve_step=2 => total=2 (1 + (2-1))
    // index 0: seq_tail => REAL, index 1: reserve => REAL
    ASSERT_TRUE(group.malloc(block_ids, 4, false, 2));
    ASSERT_EQ(block_ids.blocksNum(), 2u);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_FailsWhenPoolExhausted) {
    auto                  group = makeGroup(4);
    std::vector<BlockIds> holders;
    for (size_t i = 0; i < total_blocks_; ++i) {
        holders.emplace_back(1);
        if (!group.malloc(holders.back(), 4)) {
            break;
        }
    }
    EXPECT_EQ(block_pool_->freeBlocksNum(), 0u);

    BlockIds block_ids(1);
    EXPECT_FALSE(group.malloc(block_ids, 4));
}

// ==================== malloc with linear_step ====================

TEST_F(DeviceSWACacheManagerTest, Malloc_WithStep_ReuseEnabled) {
    auto     group = makeGroupWithStep(4, 2);
    BlockIds block_ids(1);
    // seq_len=16 => 4 slots; keep step hits plus the last two active blocks.
    ASSERT_TRUE(group.malloc(block_ids, 16, /*enable_reuse_cache=*/true));
    ASSERT_EQ(block_ids.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 3);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_WithStep_ReuseDisabled) {
    auto     group = makeGroupWithStep(4, 2);
    BlockIds block_ids(1);
    // seq_len=16 => 4 slots, reuse_cache=false => active tail indices 2 and 3.
    ASSERT_TRUE(group.malloc(block_ids, 16, /*enable_reuse_cache=*/false));
    ASSERT_EQ(block_ids.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_WithStep_ReserveAllocated) {
    auto     group = makeGroupWithStep(4, 2);
    BlockIds block_ids(1);
    // seq_len=16 => seq_slots=4, reserve_step=2 => total_slots=5
    // reuse disabled: active tail(2,3) and reserve(4) allocated
    ASSERT_TRUE(group.malloc(block_ids, 16, /*enable_reuse_cache=*/false, /*reserve_step=*/2));
    ASSERT_EQ(block_ids.blocksNum(), 5u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[4]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 3);
}

TEST_F(DeviceSWACacheManagerTest, MallocRequiredPositionsBackfillsPolicySkippedSlot) {
    auto group = makeGroupWithStep(4, 2);

    BlockIds block_ids(1);
    block_ids.assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    std::vector<size_t> backfilled_positions;
    ASSERT_TRUE(group.malloc(block_ids,
                             /*seq_len=*/16,
                             /*enable_reuse_cache=*/false,
                             /*reserve_step=*/0,
                             &backfilled_positions,
                             /*required_positions=*/{1, 2, 3}));

    ASSERT_EQ(block_ids.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_EQ(backfilled_positions, (std::vector<size_t>{1, 2, 3}));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 3);

    group.unreference(block_ids.blocks());
}

// ==================== removeSkippedBlocks ====================

TEST_F(DeviceSWACacheManagerTest, RemoveSkippedBlocks_TwoOrFewer_NoOp) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 5));
    ASSERT_EQ(block_ids.blocksNum(), 2u);

    group.removeSkippedBlocks(block_ids);
    EXPECT_EQ(block_ids.blocksNum(), 2u);
}

TEST_F(DeviceSWACacheManagerTest, RemoveSkippedBlocks_FreesNonTailReal) {
    auto     group = makeGroupWithStep(4, 2);
    BlockIds block_ids(1);
    // First: 2 blocks with reuse
    ASSERT_TRUE(group.malloc(block_ids, 5, true));
    // Extend to 5 blocks with reuse
    ASSERT_TRUE(group.malloc(block_ids, 20, true));
    ASSERT_EQ(block_ids.blocksNum(), 5u);
    size_t free_before = block_pool_->freeBlocksNum();

    group.removeSkippedBlocks(block_ids, true);

    // step=2: keep step_hit blocks + last 2
    // step_hit: index 1 ((1+1)%2==0), index 3 ((3+1)%2==0)
    // last 2: index 3, 4
    // With reuse_cache=true for the first malloc (5 tokens), blocks at 0,1 are:
    // active tail at 0,1 and step_hit at 1 => both REAL
    // Then extending to 20 tokens with reuse: new blocks at 2,3,4
    // step_hit at 3 and active tail at 3,4 => REAL. index 2: NULL
    // So blocks are: [REAL, REAL, NULL, REAL, REAL]
    // Removal continues across the NULL hole at index 2, keeps the step hit at
    // index 1, and releases the non-step block at index 0.
    EXPECT_EQ(block_pool_->freeBlocksNum(), free_before + 1);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[4]));
}

TEST_F(DeviceSWACacheManagerTest, RemoveSkippedBlocks_WithStep_FreesNonStepBlocks) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;
    DeviceSWACacheManager group({}, spec, block_pool, 0, 2, makePolicy());

    // Start with 6 allocated blocks (no NULLs). malloc() reserves capacity with refCount 0;
    // incRef gives each a single holder so removeSkippedBlocks' decRef can free them.
    auto allocated = block_pool->malloc(6).value();
    ASSERT_EQ(allocated.size(), 6u);
    block_pool->incRef(allocated);
    BlockIds blocks;
    blocks.assign(allocated);

    const size_t free_before = block_pool->freeBlocksNum();
    group.removeSkippedBlocks(blocks, true);

    // step=2, size=6: keep step_hit + last 2
    // step_hit: index 1 ((1+1)%2==0), 3 ((3+1)%2==0), 5 ((5+1)%2==0 but in last 2)
    // last 2: index 4, 5
    // loop from i=3 down: (block_size-3=3)
    //   i=3: step_hit => continue
    //   i=2: not step_hit => free
    //   i=1: step_hit => continue
    //   i=0: not step_hit => free
    ASSERT_EQ(blocks.blocksNum(), 6u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[5]));

    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 2);
}

TEST_F(DeviceSWACacheManagerTest, RemoveSkippedBlocks_HCAStateReuseEnabledKeepsTailOnly) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto spec  = makeDsv4StateSpec(4);
    auto group = DeviceSWACacheManager({}, spec, block_pool, 5, /*linear_step=*/2, makePolicy(false));

    auto allocated = block_pool->malloc(6).value();
    ASSERT_EQ(allocated.size(), 6u);
    block_pool->incRef(allocated);
    BlockIds blocks;
    blocks.assign(allocated);

    const size_t free_before = block_pool->freeBlocksNum();
    group.removeSkippedBlocks(blocks, /*enable_reuse_cache=*/true);

    ASSERT_EQ(blocks.blocksNum(), 6u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[i])) << "position " << i << " should be freed";
    }
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[5]));
    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 5);
}

TEST_F(DeviceSWACacheManagerTest, RemoveSkippedBlocks_WithReserveStep) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;
    DeviceSWACacheManager group({}, spec, block_pool, 0, 2);

    auto allocated = block_pool->malloc(6).value();
    ASSERT_EQ(allocated.size(), 6u);
    block_pool->incRef(allocated);
    BlockIds blocks;
    blocks.assign(allocated);

    const size_t free_before = block_pool->freeBlocksNum();
    // reserve_step=1: keep last 2 + 1 more (index 3)
    group.removeSkippedBlocks(blocks, false, 1);

    // reuse_cache=false so no step_hit check
    // loop from i=block_size-3-1=2 down:
    //   i=2: free, i=1: free, i=0: free
    ASSERT_EQ(blocks.blocksNum(), 6u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[5]));

    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 3);
}

TEST_F(DeviceSWACacheManagerTest, RemoveSkippedBlocks_ReserveTokensUsePhysicalBlockUnits) {
    struct Case {
        int tokens_per_block;
        int reserve_tokens;
        int retained_blocks;
    };
    for (const auto& c : std::vector<Case>{
             {256, 0, 1}, {256, 1, 2}, {256, 4, 2}, {256, 256, 2}, {256, 257, 3}, {4, 4, 2}, {4, 5, 3}, {1, 4, 5}}) {
        SCOPED_TRACE(::testing::Message()
                     << "block_tokens=" << c.tokens_per_block << " reserve_tokens=" << c.reserve_tokens);
        auto            spec = makeDsv4StateSpec(c.tokens_per_block);
        SWACacheManager group({}, spec, block_pool_, 0, 2, makePolicy(false));
        const size_t    free_before = block_pool_->freeBlocksNum();
        auto            allocated   = block_pool_->malloc(6).value();
        ASSERT_EQ(allocated.size(), 6u);
        block_pool_->incRef(allocated);
        BlockIds blocks;
        blocks.assign(allocated);
        group.removeSkippedBlocks(blocks, false, c.reserve_tokens);
        EXPECT_EQ(validBlockCount(blocks.blocks()), static_cast<size_t>(c.retained_blocks));
        for (int i = 0; i < 6; ++i) {
            EXPECT_EQ(isNullBlockIdx(blocks.blocks()[i]), i < 6 - c.retained_blocks);
        }
        group.unreference(blocks.blocks());
        EXPECT_EQ(block_pool_->freeBlocksNum(), free_before);
    }
}

TEST_F(DeviceSWACacheManagerTest, Mtp3HcaStateSupportsEightyStreamsAcrossBlockBoundaries) {
    // Match the service's 255 usable HCA state blocks and 80 concurrent streams.
    // Only allocator bookkeeping is exercised, so use a four-byte state block.
    auto backing_spec = std::dynamic_pointer_cast<FixedStateCacheSpec>(
        makeResolvedOpaqueSpec(true, "default", DataType::TYPE_FP32, 4, 256));
    ASSERT_NE(backing_spec, nullptr);
    auto config = makeSingleGroupCacheConfig(backing_spec, CacheGroupType::FULL, 1, 256);
    auto device_config = std::make_shared<DeviceBlockPoolConfig>(
        DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.topology().groups().front()));
    device_config->use_device_malloc_backing = true;
    std::shared_ptr<const DeviceBlockPoolConfig> const_config = device_config;
    auto pool = std::make_shared<DeviceBlockPool>(const_config);
    ASSERT_TRUE(pool->init());
    ASSERT_EQ(pool->freeBlocksNum(), 255u);
    SWACacheManager      group({}, backing_spec, pool, 0, 0, makePolicy(false));
    std::vector<BlockIds> streams(80);
    for (auto& blocks : streams) {
        ASSERT_TRUE(group.malloc(blocks, 4096, false, 4));
    }
    // Alternate one-token acceptance and full acceptance. This repeatedly moves
    // the four-token speculative window across 256-token page boundaries.
    for (int seq_len = 4097, step = 0; seq_len <= 4096 + 1500; seq_len += (++step % 2 == 0 ? 1 : 4)) {
        for (auto& blocks : streams) {
            ASSERT_TRUE(group.malloc(blocks, seq_len, false, 4)) << "seq_len=" << seq_len;
            group.removeSkippedBlocks(blocks, false, 4);
            const int committed_tail = (seq_len - 1) / 256;
            const int reserved_tail  = (seq_len + 4 - 1) / 256;
            ASSERT_FALSE(isNullBlockIdx(blocks.blocks()[committed_tail]));
            ASSERT_FALSE(isNullBlockIdx(blocks.blocks()[reserved_tail]));
            EXPECT_LE(validBlockCount(blocks.blocks()), 2u);
        }
        EXPECT_GE(pool->freeBlocksNum(), 95u);
    }
    for (auto& blocks : streams) {
        group.unreference(blocks.blocks());
    }
    EXPECT_EQ(pool->freeBlocksNum(), 255u);
}

// ==================== free ====================

TEST_F(DeviceSWACacheManagerTest, Free_ReleasesRealBlocks) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 20));
    EXPECT_LT(block_pool_->freeBlocksNum(), total_blocks_);

    group.unreference(block_ids.blocks());
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

TEST_F(DeviceSWACacheManagerTest, Free_Empty) {
    auto group = makeGroup(4);
    group.unreference({});
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

TEST_F(DeviceSWACacheManagerTest, Free_SkipsNullBlocks) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 20));
    EXPECT_LT(block_pool_->freeBlocksNum(), total_blocks_);

    group.unreference(block_ids.blocks());
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

// ==================== reference ====================

TEST_F(DeviceSWACacheManagerTest, Reference_AddsAndRefsBlocks) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 5));
    auto original = block_ids.blocks();

    BlockIds block_ids2(1);
    group.reference(block_ids2, original);
    EXPECT_EQ(block_ids2.blocksNum(), original.size());
    EXPECT_EQ(block_ids2.blocks(), original);
}

TEST_F(DeviceSWACacheManagerTest, Reference_NullBlocksNotReffed) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 20));
    auto original = block_ids.blocks();

    BlockIds block_ids2(1);
    group.reference(block_ids2, original);
    EXPECT_EQ(block_ids2.blocksNum(), original.size());
}

// ==================== batch allocation atomicity (regression: mid-loop leak) ====================

// Reproduces the historical bug where DeviceSWACacheManager::malloc called block_pool_->malloc(1)
// repeatedly inside a loop. If a later iteration failed (e.g. concurrent allocators raced for
// the last free blocks), the previously allocated blocks were leaked because they had only
// been recorded in a stack-local vector and were never written back to block_ids; the upper
// rollback in CoordinatorCacheManager::initMallocForCommonLen could not see them.
//
// After the fix, DeviceSWACacheManager::malloc performs a single atomic batch malloc on the pool,
// so a failed allocation must leave the pool's free counter unchanged.
TEST_F(DeviceSWACacheManagerTest, Malloc_FailsAtomicallyWithoutLeak) {
    auto group = makeGroupWithStep(4, 2);

    // Hold 7 blocks so that only 2 free blocks remain. shared_cache_ is empty here, so
    // ensureFreeBlocks() cannot evict and refill the pool.
    auto pre_alloc = block_pool_->malloc(7).value();
    ASSERT_EQ(pre_alloc.size(), 7u);
    block_pool_->incRef(pre_alloc);
    const size_t free_before = block_pool_->freeBlocksNum();
    ASSERT_EQ(free_before, total_blocks_ - 7);

    // seq_len=16, step=2, reuse=true => seq_slots=4. The group needs 3 real blocks at
    // positions {1, 2, 3}, which exceeds the 2 free blocks currently in the pool.
    BlockIds block_ids(1);
    EXPECT_FALSE(group.malloc(block_ids, 16, /*enable_reuse_cache=*/true));

    // Free count must stay identical to the pre-call value (no stranded blocks).
    EXPECT_EQ(block_pool_->freeBlocksNum(), free_before);
    // No partial state should have leaked into block_ids either.
    EXPECT_EQ(block_ids.blocksNum(), 0u);

    // The pre-allocated blocks must still be releasable, proving that DeviceBlockPool ref
    // counters were not corrupted by the failed malloc path.
    block_pool_->decRef(pre_alloc);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

// Verifies the new behavior: DeviceSWACacheManager::malloc reserves all required physical blocks
// via a single batch DeviceBlockPool::malloc(N) call instead of N individual malloc(1) calls.
TEST_F(DeviceSWACacheManagerTest, Malloc_AllocatesAtomicallyAsBatch) {
    auto         group       = makeGroupWithStep(4, 2);
    const size_t free_before = block_pool_->freeBlocksNum();

    // seq_len=16, step=2, reuse=true => 4 slots. Real blocks expected at positions {1, 2, 3}.
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 16, /*enable_reuse_cache=*/true));
    ASSERT_EQ(block_ids.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));

    // The pool's free count must drop by exactly the number of physical blocks (3).
    EXPECT_EQ(block_pool_->freeBlocksNum(), free_before - 3);

    group.unreference(block_ids.blocks());
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

// Larger sparse layout: with linear_step=2 and seq_len=24 (=> 6 slots) and reuse enabled,
// the active-tail-2 plus step-hits set {1, 3, 4, 5} forms 4 physical blocks. Validates
// that the batch path correctly distributes the 4 allocated indices across NULL/REAL slots.
TEST_F(DeviceSWACacheManagerTest, Malloc_BatchPlacementMatchesShouldAllocate) {
    auto         group       = makeGroupWithStep(4, 2);
    const size_t free_before = block_pool_->freeBlocksNum();

    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 24, /*enable_reuse_cache=*/true));
    ASSERT_EQ(block_ids.blocksNum(), 6u);
    // Expected: idx0=NULL, idx1=REAL(step), idx2=NULL, idx3=REAL(step+tail), idx4=REAL(tail), idx5=REAL(tail).
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[4]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[5]));

    // All 4 real blocks must be distinct (the batch DeviceBlockPool::malloc returns unique ids).
    std::vector<BlockIdxType> reals = {
        block_ids.blocks()[1], block_ids.blocks()[3], block_ids.blocks()[4], block_ids.blocks()[5]};
    std::sort(reals.begin(), reals.end());
    EXPECT_EQ(std::adjacent_find(reals.begin(), reals.end()), reals.end());

    EXPECT_EQ(block_pool_->freeBlocksNum(), free_before - 4);

    group.unreference(block_ids.blocks());
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
