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
        return DeviceSWACacheManager({}, spec, block_pool_, 0);
    }

    DeviceSWACacheManager makeReusableGroup(int seq_size_per_block) {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->seq_size_per_block = seq_size_per_block;
        return DeviceSWACacheManager({}, spec, block_pool_, 0, makePolicy(), 2);
    }

    DeviceBlockPoolPtr block_pool_;
    size_t             total_blocks_ = 0;
    bool               old_core_dump_on_exception_{false};
};

TEST_F(DeviceSWACacheManagerTest, DefaultPolicyDrivesBehaviorInterfaces) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;
    DeviceSWACacheManager group({}, spec, block_pool_, 0);

    EXPECT_FALSE(group.prefixReusable());
    EXPECT_TRUE(group.hasSparseSlots());
    EXPECT_FALSE(group.hasKernelBlockSubdiv());
    EXPECT_TRUE(group.transferTailBlocks());
    EXPECT_TRUE(group.isReservable());
    EXPECT_NE(group.memoryPlacement(), CacheMemoryPlacement::HOST_PINNED);
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

TEST_F(DeviceSWACacheManagerTest, EstimatePeakCountsTailInsteadOfHistoricalHoles) {
    auto group = makeGroup(16);
    EXPECT_EQ(group.estimatePeakNeedBlocks(160, {}, 0, 0, false), 2);
    EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(160, 160, 0, 0, false, 1), 2);
    EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(160, 160, 1600, 0, false, 1), 3);

    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, 160));
    ASSERT_EQ(blocks.blocksNum(), 10u);
    EXPECT_EQ(group.estimatePeakNeedBlocks(160, blocks.blocks(), 0, 0, false), 0);
    EXPECT_EQ(group.estimatePeakNeedBlocks(160, blocks.blocks(), 16, 0, false), 1);
    EXPECT_EQ(group.estimatePeakNeedBlocks(160, blocks.blocks(), 1600, 0, false), 1);
    group.unreference(blocks.blocks());
}

TEST_F(DeviceSWACacheManagerTest, EstimatePeakMatchesActualAllocationBeforeCleanup) {
    struct Case {
        int block_size;
        int seq_len;
        int remaining_tokens;
        int reserve_step;
    };
    const std::vector<Case> cases = {{16, 160, 0, 0},
                                     {16, 160, 1, 0},
                                     {16, 160, 160, 0},
                                     {4, 39, 1, 0},
                                     {4, 39, 2, 0},
                                     {4, 1, 32, 0},
                                     {1, 10, 10, 3},
                                     {4, 40, 32, 3},
                                     {4, 39, 33, 1},
                                     {16, 160, 32, 1},
                                     {4, 0, 0, 0},
                                     {4, 0, 16, 3},
                                     {4, 160, 0, 3}};
    for (int tail_blocks : {1, 2, 3}) {
        for (bool reuse : {false, true}) {
            for (const auto& c : cases) {
                SCOPED_TRACE(::testing::Message()
                             << "tail=" << tail_blocks << " reuse=" << reuse << " block_size=" << c.block_size
                             << " seq=" << c.seq_len << " remaining=" << c.remaining_tokens
                             << " reserve=" << c.reserve_step);
                auto spec                 = std::make_shared<MHAKVCacheSpec>();
                spec->seq_size_per_block  = c.block_size;
                auto policy               = makePolicy();
                policy.active_tail_blocks = tail_blocks;
                DeviceSWACacheManager group({}, spec, block_pool_, 0, policy, tail_blocks);
                const int             fresh_peak =
                    group.estimatePeakNeedBlocks(c.seq_len, {}, c.remaining_tokens, c.reserve_step, reuse);
                BlockIds blocks;
                ASSERT_TRUE(group.malloc(blocks, c.seq_len, reuse, c.reserve_step));
                const int initial_blocks = static_cast<int>(total_blocks_ - block_pool_->freeBlocksNum());
                const int extra_peak =
                    group.estimatePeakNeedBlocks(c.seq_len, blocks.blocks(), c.remaining_tokens, c.reserve_step, reuse);
                int observed_peak = initial_blocks;
                for (int step = 1; step <= c.remaining_tokens; ++step) {
                    ASSERT_TRUE(group.mallocForDecode(blocks, c.seq_len + step, reuse, c.reserve_step));
                    observed_peak =
                        std::max(observed_peak, static_cast<int>(total_blocks_ - block_pool_->freeBlocksNum()));
                    group.removeSkippedBlocks(blocks, reuse, c.reserve_step);
                }
                if (c.reserve_step <= c.block_size) {
                    EXPECT_EQ(fresh_peak, observed_peak);
                    EXPECT_EQ(extra_peak, observed_peak - initial_blocks);
                } else {
                    EXPECT_GE(fresh_peak, observed_peak);
                    EXPECT_GE(extra_peak, observed_peak - initial_blocks);
                }
                group.unreference(blocks.blocks());
                ASSERT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
            }
        }
    }
}

TEST_F(DeviceSWACacheManagerTest, EstimatePeakIncludesFirstDecodeReserveBeforeReleasingOldBlocks) {
    auto     group = makeGroup(1);
    BlockIds blocks;
    ASSERT_TRUE(group.malloc(blocks, 10, false, 0, nullptr, {0}));
    ASSERT_EQ(total_blocks_ - block_pool_->freeBlocksNum(), 3u);
    EXPECT_EQ(group.estimatePeakNeedBlocks(10, blocks.blocks(), 1, 3, false), 4);
    ASSERT_TRUE(group.mallocForDecode(blocks, 11, false, 3));
    EXPECT_EQ(total_blocks_ - block_pool_->freeBlocksNum(), 7u);
    group.removeSkippedBlocks(blocks, false, 3);
    EXPECT_EQ(total_blocks_ - block_pool_->freeBlocksNum(), 5u);
    group.unreference(blocks.blocks());
}

TEST_F(DeviceSWACacheManagerTest, EstimatePeakCoversMultipleAcceptedTokensBeforeCleanup) {
    auto     group = makeGroup(1);
    BlockIds blocks;
    ASSERT_TRUE(group.mallocForDecode(blocks, 10, false, 3));
    ASSERT_EQ(total_blocks_ - block_pool_->freeBlocksNum(), 5u);
    EXPECT_EQ(group.estimatePeakNeedBlocks(10, blocks.blocks(), 6, 3, false), 3);
    EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(10, 10, 6, 3, false, 1), 8);
    for (int seq_len : {13, 16}) {
        ASSERT_TRUE(group.mallocForDecode(blocks, seq_len, false, 3));
        EXPECT_EQ(total_blocks_ - block_pool_->freeBlocksNum(), 8u);
        group.removeSkippedBlocks(blocks, false, 3);
        EXPECT_EQ(total_blocks_ - block_pool_->freeBlocksNum(), 5u);
    }
    group.unreference(blocks.blocks());
}

TEST_F(DeviceSWACacheManagerTest, EstimateInitialBatchPeakMatchesSharedPrefixAndPrivateTails) {
    struct Case {
        int seq_len;
        int common_seq_len;
        int remaining_tokens;
        int reserve_step;
        int batch_size;
    };
    const std::vector<Case> cases = {{40, 40, 0, 0, 2},
                                     {40, 20, 0, 0, 2},
                                     {40, 20, 17, 0, 2},
                                     {40, 40, 17, 0, 2},
                                     {40, 0, 17, 0, 2},
                                     {39, 20, 17, 1, 2},
                                     {39, 36, 17, 1, 2},
                                     {40, 20, 17, 1, 2},
                                     {1, 1, 17, 0, 2},
                                     {1, 0, 17, 1, 2},
                                     {0, 0, 0, 0, 2},
                                     {40, 20, 17, 1, 1}};
    for (int tail_blocks : {1, 2}) {
        for (const auto& c : cases) {
            SCOPED_TRACE(::testing::Message() << "tail=" << tail_blocks << " seq=" << c.seq_len
                                              << " common=" << c.common_seq_len << " remaining=" << c.remaining_tokens
                                              << " reserve=" << c.reserve_step << " batch=" << c.batch_size);
            auto spec                 = std::make_shared<MHAKVCacheSpec>();
            spec->seq_size_per_block  = 4;
            auto policy               = makePolicy();
            policy.active_tail_blocks = tail_blocks;
            DeviceSWACacheManager group({}, spec, block_pool_, 0, policy, tail_blocks);
            const int             estimated_peak = group.estimateInitialBatchPeakNeedBlocks(
                c.seq_len, c.common_seq_len, c.remaining_tokens, c.reserve_step, true, c.batch_size);
            std::vector<BlockIds> batches(c.batch_size);
            ASSERT_TRUE(group.malloc(batches[0], c.common_seq_len, true));
            for (int batch = 1; batch < c.batch_size; ++batch) {
                group.reference(batches[batch], batches[0].blocks());
            }
            for (auto& blocks : batches) {
                ASSERT_TRUE(group.malloc(blocks, c.seq_len, true, c.reserve_step));
            }
            int observed_peak = static_cast<int>(total_blocks_ - block_pool_->freeBlocksNum());
            EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(
                          c.seq_len, c.common_seq_len, 0, c.reserve_step, true, c.batch_size),
                      observed_peak);
            for (int step = 1; step <= c.remaining_tokens; ++step) {
                for (auto& blocks : batches) {
                    ASSERT_TRUE(group.mallocForDecode(blocks, c.seq_len + step, true, c.reserve_step));
                }
                observed_peak = std::max(observed_peak, static_cast<int>(total_blocks_ - block_pool_->freeBlocksNum()));
                for (auto& blocks : batches) {
                    group.removeSkippedBlocks(blocks, true, c.reserve_step);
                }
            }
            EXPECT_EQ(estimated_peak, observed_peak);
            for (auto& blocks : batches) {
                group.unreference(blocks.blocks());
            }
            ASSERT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
        }
    }
}

// ==================== getNeedBlocks ====================

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_SeqLenZero) {
    auto group = makeGroup(4);
    auto need  = group.getNeedBlocks(0, 0, 0, 0, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 0);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_ReuseDisabledCountsActiveTail) {
    auto group = makeReusableGroup(4);
    // seq_len=12 => seq_slots=3, reuse disabled => last two active tail blocks.
    auto need = group.getNeedBlocks(0, 12, 0, 0, false);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_ReuseEnabledUsesSparse) {
    auto group = makeReusableGroup(4);
    // seq_len=12 => seq_slots=3
    auto need = group.getNeedBlocks(0, 12, 0, 0, true);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_HCAStateReuseEnabledCountsTailOnly) {
    auto spec  = makeDsv4StateSpec(4);
    auto group = DeviceSWACacheManager({}, spec, block_pool_, 5, makePolicy(false));
    // would keep positions 2/5/8 plus tail position 9. HCA_STATE skips reuse and keeps only tail 9.
    auto need = group.getNeedBlocks(0, 40, 0, 0, true);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 1);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_CSAStateReuseEnabledCountsTailOnly) {
    auto spec  = makeDsv4StateSpec(4);
    auto group = DeviceSWACacheManager({}, spec, block_pool_, 4, makePolicy(), 2);

    auto need = group.getNeedBlocks(0, 40, 0, 0, true);
    EXPECT_EQ(need.common_blocks, 0);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_WithReserveStep) {
    auto group = makeReusableGroup(4);
    // Prefill ignores decode reserve.
    auto need = group.getNeedBlocks(0, 8, 2, 0, false);
    EXPECT_EQ(need.extra_blocks, 2);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_ReusePartialOverlap) {
    auto group = makeReusableGroup(4);
    // seq_len=12 => seq_slots=3
    auto need = group.getNeedBlocks(0, 12, 0, 2, true);
    EXPECT_EQ(need.extra_blocks, 1);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_ReuseFullOverlap) {
    auto group = makeReusableGroup(4);
    // seq_len=12 => seq_slots=3
    auto need = group.getNeedBlocks(0, 12, 0, 3, true);
    EXPECT_EQ(need.extra_blocks, 0);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocks_CommonTailIsShared) {
    auto group = makeGroup(4);
    auto need1 = group.getNeedBlocks(0, 20, 0, 0, false);
    auto need2 = group.getNeedBlocks(20, 20, 0, 0, false);
    auto need3 = group.getNeedBlocks(16, 20, 0, 0, false);
    EXPECT_EQ(need1.common_blocks, 0);
    EXPECT_EQ(need1.extra_blocks, 2);
    EXPECT_EQ(need2.common_blocks, 2);
    EXPECT_EQ(need2.extra_blocks, 0);
    EXPECT_EQ(need3.common_blocks, 2);
    EXPECT_EQ(need3.extra_blocks, 1);
}

TEST_F(DeviceSWACacheManagerTest, GetNeedBlocksCountsSharedPrefixOnce) {
    auto group = makeGroup(4);
    auto need  = group.getNeedBlocks(20, 40, 3, 0, false);
    EXPECT_EQ(need.common_blocks, 2);
    EXPECT_EQ(need.extra_blocks, 2);
    need = group.getNeedBlocks(100, 20, 0, 0, false);
    EXPECT_EQ(need.common_blocks, 2);
    EXPECT_EQ(need.extra_blocks, 0);
}

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
    auto         group = DeviceSWACacheManager({}, spec, block_pool_, 5, makePolicy(false));
    BlockIds     block_ids(1);
    block_ids.assign(BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});

    EXPECT_NO_THROW((void)group.malloc(block_ids, 12));
}

TEST_F(DeviceSWACacheManagerTest, Malloc_HCAStateReuseEnabledAllocatesTailOnly) {
    auto     spec  = makeDsv4StateSpec(4);
    auto     group = DeviceSWACacheManager({}, spec, block_pool_, 5, makePolicy(false));
    BlockIds block_ids(1);

    ASSERT_TRUE(group.malloc(block_ids, 40, /*enable_reuse_cache=*/true, /*reserve_step=*/0));

    ASSERT_EQ(block_ids.blocksNum(), 10u);
    EXPECT_EQ(validBlockCount(block_ids.blocks()), 1u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[8]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[9]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 1);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_CSAStateReuseEnabledKeepsTailOnly) {
    auto     spec  = makeDsv4StateSpec(4);
    auto     group = DeviceSWACacheManager({}, spec, block_pool_, 4, makePolicy(), 2);
    BlockIds block_ids(1);

    ASSERT_TRUE(group.malloc(block_ids, 40, /*enable_reuse_cache=*/true, /*reserve_step=*/0));

    ASSERT_EQ(block_ids.blocksNum(), 10u);
    EXPECT_EQ(validBlockCount(block_ids.blocks()), 2u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[5]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[8]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[9]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_DSV4TrapChecksSWAKVNullTail) {
    ScopedEnvVar env("DSV4_TRAP_INVALID_KV_ACCESS", "1");
    auto         spec  = makeDsv4StateSpec(4);
    auto         group = DeviceSWACacheManager({}, spec, block_pool_, 6, makePolicy());
    BlockIds     block_ids(1);
    block_ids.assign(BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});

    EXPECT_THROW((void)group.malloc(block_ids, 12), std::exception);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_DSV4TrapChecksNonSkipStateNullTail) {
    ScopedEnvVar env("DSV4_TRAP_INVALID_KV_ACCESS", "1");
    auto         spec  = makeDsv4StateSpec(4);
    auto         group = DeviceSWACacheManager({}, spec, block_pool_, 4, makePolicy(), 2);
    BlockIds     block_ids(1);
    block_ids.assign(BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});

    EXPECT_THROW((void)group.malloc(block_ids, 12), std::exception);
}

TEST_F(DeviceSWACacheManagerTest, Decode_WithReserveStep) {
    auto     group = makeGroup(4);
    BlockIds block_ids(1);
    // index 0: seq_tail => REAL, index 1: reserve => REAL
    ASSERT_TRUE(group.mallocForDecode(block_ids, 4, false, 2));
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

TEST_F(DeviceSWACacheManagerTest, Malloc_WithTail_ReuseEnabled) {
    auto     group = makeReusableGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 16, /*enable_reuse_cache=*/true));
    ASSERT_EQ(block_ids.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 2);
}

TEST_F(DeviceSWACacheManagerTest, Malloc_WithTail_ReuseDisabled) {
    auto     group = makeReusableGroup(4);
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

TEST_F(DeviceSWACacheManagerTest, Decode_WithTail_ReserveAllocated) {
    auto     group = makeReusableGroup(4);
    BlockIds block_ids(1);
    // reuse disabled: active tail(2,3) and reserve(4) allocated
    ASSERT_TRUE(group.mallocForDecode(block_ids, 16, /*enable_reuse_cache=*/false, /*reserve_step=*/2));
    ASSERT_EQ(block_ids.blocksNum(), 5u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[4]));
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_ - 3);
}

TEST_F(DeviceSWACacheManagerTest, MallocRequiredPositionsBackfillsPolicySkippedSlot) {
    auto group = makeReusableGroup(4);

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
    auto     group = makeReusableGroup(4);
    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 5, true));
    ASSERT_TRUE(group.malloc(block_ids, 20, true));
    ASSERT_EQ(block_ids.blocksNum(), 5u);
    size_t free_before = block_pool_->freeBlocksNum();

    group.removeSkippedBlocks(block_ids, true);

    EXPECT_EQ(block_pool_->freeBlocksNum(), free_before + 2);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[4]));
}

TEST_F(DeviceSWACacheManagerTest, RemoveSkippedBlocks_FreesAllOutsideTail) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;
    DeviceSWACacheManager group({}, spec, block_pool, 0, makePolicy());

    auto allocated = block_pool->malloc(6).value();
    ASSERT_EQ(allocated.size(), 6u);
    block_pool->incRef(allocated);
    BlockIds blocks;
    blocks.assign(allocated);

    const size_t free_before = block_pool->freeBlocksNum();
    group.removeSkippedBlocks(blocks, true);

    ASSERT_EQ(blocks.blocksNum(), 6u);
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[2]));
    EXPECT_TRUE(isNullBlockIdx(blocks.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[4]));
    EXPECT_FALSE(isNullBlockIdx(blocks.blocks()[5]));

    EXPECT_EQ(block_pool->freeBlocksNum(), free_before + 4);
}

TEST_F(DeviceSWACacheManagerTest, RemoveSkippedBlocks_HCAStateReuseEnabledKeepsTailOnly) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9u);

    auto spec  = makeDsv4StateSpec(4);
    auto group = DeviceSWACacheManager({}, spec, block_pool, 5, makePolicy(false));

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
    DeviceSWACacheManager group({}, spec, block_pool, 0);

    auto allocated = block_pool->malloc(6).value();
    ASSERT_EQ(allocated.size(), 6u);
    block_pool->incRef(allocated);
    BlockIds blocks;
    blocks.assign(allocated);

    const size_t free_before = block_pool->freeBlocksNum();
    group.removeSkippedBlocks(blocks, false, 1);
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
    auto group = makeReusableGroup(4);

    auto pre_alloc = block_pool_->malloc(8).value();
    ASSERT_EQ(pre_alloc.size(), 8u);
    block_pool_->incRef(pre_alloc);
    const size_t free_before = block_pool_->freeBlocksNum();
    ASSERT_EQ(free_before, total_blocks_ - 8);

    BlockIds block_ids(1);
    EXPECT_FALSE(group.malloc(block_ids, 16, /*enable_reuse_cache=*/true));

    EXPECT_EQ(block_pool_->freeBlocksNum(), free_before);
    EXPECT_EQ(block_ids.blocksNum(), 0u);

    block_pool_->decRef(pre_alloc);
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

// Allocate the tail in a single pool transaction.
TEST_F(DeviceSWACacheManagerTest, Malloc_AllocatesAtomicallyAsBatch) {
    auto         group       = makeReusableGroup(4);
    const size_t free_before = block_pool_->freeBlocksNum();
    BlockIds     block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 16, /*enable_reuse_cache=*/true));
    ASSERT_EQ(block_ids.blocksNum(), 4u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[3]));

    // The pool's free count must drop by exactly the number of physical blocks (2).
    EXPECT_EQ(block_pool_->freeBlocksNum(), free_before - 2);

    group.unreference(block_ids.blocks());
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}
// Only the final two logical slots receive physical blocks.
TEST_F(DeviceSWACacheManagerTest, Malloc_BatchPlacementMatchesShouldAllocate) {
    auto         group       = makeReusableGroup(4);
    const size_t free_before = block_pool_->freeBlocksNum();

    BlockIds block_ids(1);
    ASSERT_TRUE(group.malloc(block_ids, 24, /*enable_reuse_cache=*/true));
    ASSERT_EQ(block_ids.blocksNum(), 6u);
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[1]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[2]));
    EXPECT_TRUE(isNullBlockIdx(block_ids.blocks()[3]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[4]));
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[5]));

    std::vector<BlockIdxType> reals = {block_ids.blocks()[4], block_ids.blocks()[5]};
    std::sort(reals.begin(), reals.end());
    EXPECT_EQ(std::adjacent_find(reals.begin(), reals.end()), reals.end());

    EXPECT_EQ(block_pool_->freeBlocksNum(), free_before - 2);

    group.unreference(block_ids.blocks());
    EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
}

TEST_F(DeviceSWACacheManagerTest, PrefillTailDependsOnRequestAndGroupReuse) {
    for (const bool group_reuse : {false, true}) {
        for (const bool request_reuse : {false, true}) {
            for (const int tail : {1, 3, std::numeric_limits<int>::max()}) {
                SCOPED_TRACE(::testing::Message() << group_reuse << "/" << request_reuse << " tail=" << tail);
                auto policy               = makePolicy(group_reuse);
                policy.active_tail_blocks = 2;
                DeviceSWACacheManager group({}, makeDsv4StateSpec(4), block_pool_, 0, policy, tail);
                const int             expected = group_reuse && request_reuse ? std::clamp(tail, 2, 6) : 2;
                BlockIds              blocks;
                const auto            need = group.getNeedBlocks(24, 24, 4, 0, request_reuse);
                EXPECT_EQ(need.common_blocks, expected);
                EXPECT_EQ(need.extra_blocks, 0);
                EXPECT_EQ(group.estimatePeakNeedBlocks(24, {}, 0, 4, request_reuse), expected);
                EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(24, 24, 0, 4, request_reuse, 1), expected);
                ASSERT_TRUE(group.malloc(blocks, 24, request_reuse, 4));
                ASSERT_EQ(blocks.blocksNum(), 6u);
                EXPECT_EQ(validBlockCount(blocks.blocks()), expected);
                for (int slot = 0; slot < 6; ++slot) {
                    EXPECT_EQ(!isNullBlockIdx(blocks.blocks()[slot]), slot >= 6 - expected);
                }
                ASSERT_TRUE(group.mallocForDecode(blocks, 25, request_reuse));
                group.removeSkippedBlocks(blocks, request_reuse);
                EXPECT_EQ(validBlockCount(blocks.blocks()), 2u);
                group.unreference(blocks.blocks());
                ASSERT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
            }
        }
    }
}

TEST_F(DeviceSWACacheManagerTest, PrefillReusesHitAndAllocatesOnlyFollowingTail) {
    for (const int reuse_slots : {2, 4, 5, 6}) {
        for (const bool remote_load : {false, true}) {
            auto policy               = makePolicy();
            policy.active_tail_blocks = 1;
            DeviceSWACacheManager group({}, makeDsv4StateSpec(4), block_pool_, 0, policy, 3);
            BlockIds              blocks;
            blocks.resize(reuse_slots);
            RequiredPositions required;
            BlockIdxType      reused = NULL_BLOCK_IDX;
            if (remote_load) {
                required.insert(reuse_slots - 1);
            } else {
                const auto allocated = block_pool_->malloc(1).value();
                block_pool_->incRef(allocated);
                reused = allocated.front();
                blocks.setAt(reuse_slots - 1, reused);
            }
            const int  new_blocks = std::min(3, 6 - reuse_slots);
            const auto need       = group.getNeedBlocks(24, 24, 4, reuse_slots, true, required);
            EXPECT_EQ(need.common_blocks, new_blocks + static_cast<int>(remote_load));
            EXPECT_EQ(need.extra_blocks, 0);
            ASSERT_TRUE(group.malloc(blocks, 24, true, 4, nullptr, required));
            EXPECT_EQ(validBlockCount(blocks.blocks()), new_blocks + 1);
            for (int slot = 0; slot < 6; ++slot) {
                EXPECT_EQ(!isNullBlockIdx(blocks.blocks()[slot]), slot == reuse_slots - 1 || slot >= 6 - new_blocks);
            }
            if (!remote_load) {
                EXPECT_EQ(blocks.blocks()[reuse_slots - 1], reused);
            }
            group.unreference(blocks.blocks());
            ASSERT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
        }
    }
}

TEST_F(DeviceSWACacheManagerTest, SmallPrefillTailPreservesActiveBlocksThroughFirstDecode) {
    for (const int seq_len : {1, 256, 257, 382, 383, 511, 512, 513}) {
        SCOPED_TRACE(seq_len);
        DeviceSWACacheManager group({}, makeDsv4StateSpec(256), block_pool_, 0, makePolicy(), 1);
        BlockIds              blocks;
        const int             expected = std::min((seq_len + 255) / 256, 2);
        EXPECT_EQ(group.getNeedBlocks(seq_len, seq_len, 0, 0, true).common_blocks, expected);
        EXPECT_EQ(group.estimatePeakNeedBlocks(seq_len, {}, 0, 0, true), expected);
        EXPECT_EQ(group.estimateInitialBatchPeakNeedBlocks(seq_len, seq_len, 0, 0, true, 1), expected);
        ASSERT_TRUE(group.malloc(blocks, seq_len, true));
        ASSERT_EQ(validBlockCount(blocks.blocks()), expected);
        const auto prefill_blocks = blocks.blocks();
        const int  new_blocks     = seq_len % 256 == 0 ? 1 : 0;
        EXPECT_EQ(group.estimatePeakNeedBlocks(seq_len, blocks.blocks(), 1, 0, true), new_blocks);
        std::vector<size_t> backfilled;
        ASSERT_TRUE(group.mallocForDecode(blocks, seq_len + 1, true, 0, &backfilled));
        EXPECT_TRUE(backfilled.empty());
        for (size_t slot = 0; slot < prefill_blocks.size(); ++slot) {
            EXPECT_EQ(blocks.blocks()[slot], prefill_blocks[slot]);
        }
        group.removeSkippedBlocks(blocks, true);
        EXPECT_EQ(validBlockCount(blocks.blocks()), std::min(expected + new_blocks, 2));
        group.unreference(blocks.blocks());
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks_);
    }
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
