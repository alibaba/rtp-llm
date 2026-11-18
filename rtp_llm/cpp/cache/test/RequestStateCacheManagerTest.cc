#include <gtest/gtest.h>

#include <numeric>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/RequestStateCacheManager.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm::test {
namespace {

ModelConfig model() {
    ModelConfig cfg;
    cfg.num_layers                                     = 2;
    cfg.max_seq_len                                    = 16384;
    cfg.data_type                                      = DataType::TYPE_BF16;
    cfg.attn_config.head_num                           = 2;
    cfg.attn_config.kv_head_num                        = 2;
    cfg.attn_config.size_per_head                      = 16;
    cfg.attn_config.tokens_per_block                   = 64;
    cfg.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::LINEAR, HybridAttentionType::NONE};
    auto& linear                                       = cfg.linear_attention_config;
    linear.linear_conv_kernel_dim                      = 4;
    linear.linear_key_head_dim = linear.linear_value_head_dim = 128;
    linear.linear_num_key_heads = linear.linear_num_value_heads = 2;
    linear.ssm_state_dtype                                      = DataType::TYPE_FP32;
    KVCacheSpecDesc state{"linear", KVCacheSpecType::LinearAttention};
    state.dtype               = DataType::TYPE_BF16;
    state.linear_state_layout = LinearStateLayout::SPLIT_EXPORT;
    cfg.kv_cache_spec_descs   = {{state}, {KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention}}};
    return cfg;
}

ParallelismConfig parallel() {
    ParallelismConfig pc;
    pc.tp_size = pc.world_size = pc.local_world_size = 2;
    pc.role_type                                     = RoleType::DECODE;
    return pc;
}

KVCacheConfig options(uint32_t slots = 9, bool single = false) {
    KVCacheConfig opts;
    opts.linear_ssm_double_buffer = !single;
    opts.linear_ssm_replay        = single;
    opts.linear_pool_blocks       = slots;
    return opts;
}

CacheConfig config(uint32_t slots = 9, bool single = false) {
    auto cfg = CacheConfigCreator::createWarmupConfig(model(), parallel(), options(slots, single), 2);
    cfg.finalizeBlockNums(128, RuntimeConfig{});
    return cfg;
}

MallocInfo request(const CacheConfig& cfg, int length = 8192) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(cfg.topologyPtr());
    CacheKeysType keys((length + 63) / 64);
    std::iota(keys.begin(), keys.end(), 1);
    resource->setBatchCacheKeys(0, keys);
    auto tokens            = std::make_shared<CompleteTokenIds>(1, 1, length + 1024, 64);
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = torch::arange(length, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    tokens->init(input);
    tokens->setReserveStep(3);
    MallocInfo info{resource, tokens};
    info.reuse_cache = info.enable_cache_lookup = false;
    return info;
}

class RequestStateCacheTest: public ::testing::Test {
    void SetUp() override {
        old_core_                                    = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
        createDevice();
    }
    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = old_core_;
    }
    bool old_core_ = false;
};

TEST_F(RequestStateCacheTest, ModelDeclarationSplitsBytesAndBudgetsEveryPool) {
    const auto  cfg    = config();
    const auto& ssm    = cfg.group("linear_ssm");
    const auto& conv   = cfg.group("linear");
    const auto& replay = cfg.group("linear_replay");
    EXPECT_EQ(ssm.policy.request_state_slots, 2u);
    EXPECT_EQ(replay.policy.request_state_slots, 1u);
    EXPECT_EQ(ssm.block_num, 9u);
    EXPECT_EQ(conv.block_num, 22u);  // four requests with five slots, one rollover and block 0
    EXPECT_EQ(replay.block_num, 5u);
    EXPECT_EQ(ssm.kvBlockStrideBytes(), 128u * 128 * 4);
    EXPECT_EQ(conv.kvBlockStrideBytes(), 3u * 3 * 128 * 2);
    EXPECT_EQ(replay.kvBlockStrideBytes(), 3u * 3 * 128 * 4);
    for (const auto* group : {&ssm, &conv, &replay}) {
        EXPECT_TRUE(group->policy.charge_to_paged_budget);
        EXPECT_FALSE(group->policy.enable_prefix_reuse);
    }
}

TEST_F(RequestStateCacheTest, WarmupOwnsDistinctStateForEveryBatchRow) {
    const auto cfg = CacheConfigCreator::createWarmupConfig(model(), parallel(), options(), 2, 4);
    EXPECT_EQ(cfg.group("linear_ssm").block_num, 9u);
    EXPECT_EQ(cfg.group("linear_replay").block_num, 5u);
    EXPECT_EQ(cfg.group("linear").block_num, 5u);
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), parallel(), options(), 2, 5));
}

TEST_F(RequestStateCacheTest, SingleStateChargesOneSsmPerRequestAndForecastsActualSlots) {
    const auto cfg = config(5, true);
    EXPECT_EQ(cfg.group("linear_ssm").policy.linear_state_layout, LinearStateLayout::SINGLE_SSM);
    EXPECT_EQ(cfg.group("linear_ssm").policy.request_state_slots, 1u);
    EXPECT_EQ(cfg.group("linear_ssm").block_num, 5u);
    EXPECT_EQ(cfg.group("linear").block_num, 22u);
    EXPECT_EQ(cfg.group("linear_replay").block_num, 5u);
    const auto warmup = CacheConfigCreator::createWarmupConfig(model(), parallel(), options(5, true), 2, 4);
    EXPECT_EQ(warmup.group("linear_ssm").block_num, 5u);
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), parallel(), options(5, true), 2, 5));

    auto pool = createDeviceBlockPool();
    ASSERT_TRUE(pool->init());
    RequestStateCacheManager manager(cfg.group("linear_ssm"), pool, 0);
    ASSERT_TRUE(manager.init());
    const auto initial_free = pool->freeBlocksNum();
    ASSERT_GT(initial_free, 0u);
    BlockIds blocks;
    for (int length : {1, 63, 64, 65, 8192, 16384}) {
        EXPECT_EQ(manager.estimatePeakNeedBlocks(length, {}, 1000000, 3, false), 1);
        EXPECT_EQ(manager.estimateInitialBatchPeakNeedBlocks(length, length, 1000000, 3, false, 1), 1);
        const auto need = manager.getNeedBlocks(length, length + 1, 3, 0, false);
        EXPECT_EQ(need.common_blocks + need.extra_blocks, 1);
        ASSERT_TRUE(manager.mallocForDecode(blocks, length, false, 3));
        EXPECT_EQ(blocks.blocksNum(), 1u);
        EXPECT_EQ(manager.estimatePeakNeedBlocks(length, blocks.blocks(), 1000000, 3, false), 0);
        manager.removeSkippedBlocks(blocks, false, 3);
        EXPECT_EQ(pool->freeBlocksNum(), initial_free - 1);
    }
    manager.unreference(blocks.blocks());
    EXPECT_EQ(pool->freeBlocksNum(), initial_free);

    auto invalid                     = options(5, true);
    invalid.linear_ssm_double_buffer = true;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), parallel(), invalid, 2));
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), parallel(), options(1, true), 2));
    invalid             = options(5, true);
    invalid.reuse_cache = true;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), parallel(), invalid, 2));
}

TEST_F(RequestStateCacheTest, SingleStateReturnsSavedSsmBytesToPagedBudget) {
    auto draft                                           = model();
    draft.num_layers                                     = 1;
    draft.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE};
    draft.kv_cache_spec_descs = {{KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention}}};
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_MTP;
    sp.gen_num_per_cycle = 2;
    auto opts            = options(5, true);
    opts.kv_cache_mem_mb = 4;
    const auto cfg       = CacheConfigCreator::createConfig(model(), parallel(), opts, sp, &draft, true);
    EXPECT_EQ(
        CacheConfigCreator::computeLocalBlockNum(cfg, model(), RuntimeConfig{}, opts, parallel(), std::nullopt, sp),
        (4u * 1024 * 1024 - 5u * 65536 - 22u * 2304 - 5u * 4608) / 8192);
    ASSERT_EQ(cfg.mtp_sub_configs.size(), 1u);
    EXPECT_TRUE(cfg.mtp_sub_configs.front()->layerIdsForGroup("linear_ssm").empty());
}

TEST_F(RequestStateCacheTest, ChainedMtpAccountsForStatePoolsAndOnePhysicalDraft) {
    auto draft                                           = model();
    draft.num_layers                                     = 1;
    draft.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE};
    draft.kv_cache_spec_descs = {{KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention}}};
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_MTP;
    sp.gen_num_per_cycle = 2;
    auto opts            = options();
    opts.kv_cache_mem_mb = 4;
    const auto cfg       = CacheConfigCreator::createConfig(model(), parallel(), opts, sp, &draft, true);
    ASSERT_EQ(cfg.mtp_sub_configs.size(), 1u);
    EXPECT_TRUE(cfg.mtp_sub_configs.front()->layerIdsForGroup("linear_ssm").empty());
    EXPECT_EQ(cfg.mtp_sub_configs.front()->layerIdsForGroup("full").size(), 1u);
    // 9 FP32 SSM rows, 22 BF16 conv rows and 5 FP32 replay rows;
    // target and the one physical draft each require 4096 bytes/paged block.
    EXPECT_EQ(
        CacheConfigCreator::computeLocalBlockNum(cfg, model(), RuntimeConfig{}, opts, parallel(), std::nullopt, sp),
        (4u * 1024 * 1024 - 9u * 65536 - 22u * 2304 - 5u * 4608) / 8192);
}

TEST_F(RequestStateCacheTest, UnsupportedDeploymentFailsBeforeAllocation) {
    auto opts        = options();
    opts.reuse_cache = true;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), parallel(), opts, 2));
    auto pc      = parallel();
    pc.role_type = RoleType::PREFILL;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), pc, options(), 2));
    pc         = parallel();
    pc.dp_size = 2;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), pc, options(), 2));
    pc                                                 = parallel();
    pc.ffn_disaggregate_config.enable_ffn_disaggregate = true;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), pc, options(), 2));
    auto cfg                                          = model();
    cfg.kv_cache_spec_descs[0][0].linear_state_layout = LinearStateLayout::LEGACY;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(cfg, parallel(), options(), 2));
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), parallel(), options(), 0));
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), parallel(), options(), 4));
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model(), parallel(), options(0), 2));
    auto short_page                         = model();
    short_page.attn_config.tokens_per_block = 1;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(short_page, parallel(), options(), 2));
    SpeculativeExecutionConfig sp;
    sp.type               = SP_TYPE_MTP;
    sp.gen_num_per_cycle  = 2;
    sp.tree_decode_config = "{}";
    EXPECT_ANY_THROW(CacheConfigCreator::createConfig(model(), parallel(), options(), sp, &cfg, true));
}

TEST_F(RequestStateCacheTest, AllocationForecastAndReferencesUseRequestSlots) {
    auto pool = createDeviceBlockPool();
    ASSERT_TRUE(pool->init());
    auto                     group = config().group("linear_ssm");
    RequestStateCacheManager manager(group, pool, 0);
    ASSERT_TRUE(manager.init());
    BlockIds blocks;
    for (int length : {1, 63, 64, 65, 8192, 16384}) {
        EXPECT_EQ(manager.estimatePeakNeedBlocks(length, {}, 1000000, 3, false), 2);
        EXPECT_EQ(manager.estimateInitialBatchPeakNeedBlocks(length, length, 1000000, 3, false, 1), 2);
        auto need = manager.getNeedBlocks(length, length + 1, 3, 0, false);
        EXPECT_EQ(need.common_blocks + need.extra_blocks, 2);
        ASSERT_TRUE(manager.mallocForDecode(blocks, length, false, 3));
        EXPECT_EQ(blocks.blocksNum(), 2u);
        EXPECT_EQ(manager.estimatePeakNeedBlocks(length, blocks.blocks(), 1000000, 3, false), 0);
        manager.removeSkippedBlocks(blocks, false, 3);
        EXPECT_EQ(pool->freeBlocksNum(), 7u);
    }
    const auto lease = blocks.blocks();
    manager.reference(lease);
    blocks.swap(0, 1);
    EXPECT_EQ(blocks.kernelBlocks()[0], lease[1]);
    manager.unreference(blocks.blocks());
    EXPECT_EQ(pool->freeBlocksNum(), 7u);
    manager.unreference(lease);
    EXPECT_EQ(pool->freeBlocksNum(), 9u);
}

TEST_F(RequestStateCacheTest, CoordinatorAdmits192AndKeepsBanksAcrossPagesAndAsyncRelease) {
    auto cfg = config(385);
    cfg.finalizeBlockNums(32768, RuntimeConfig{});
    auto allocator = std::make_shared<BlockTreeCacheTestAllocator<CoordinatorCacheManager>>(cfg);
    ASSERT_TRUE(allocator->init());
    const auto              initial_free = allocator->freeBlocksNum();
    std::vector<MallocInfo> requests;
    for (int i = 0; i < 192; ++i) {
        auto info = request(cfg);
        ASSERT_TRUE(allocator->malloc(info).success) << i;
        EXPECT_EQ(info.batch_kv_cache_resource->blocksNum(0, "linear_ssm"), 2);
        EXPECT_EQ(info.batch_kv_cache_resource->blocksNum(0, "linear_replay"), 1);
        requests.push_back(std::move(info));
    }
    auto overflow = request(cfg);
    EXPECT_FALSE(allocator->malloc(overflow).success);
    auto&      first  = requests.front();
    const auto before = first.batch_kv_cache_resource->blocks(0, "linear_ssm");
    for (int length : {8193, 8195, 8257, 8260}) {
        first.incr_seq_len_override = length;
        ASSERT_TRUE(allocator->malloc(first).success);
        EXPECT_EQ(first.batch_kv_cache_resource->blocks(0, "linear_ssm"), before);
    }
    auto lease = allocator->incrKVCacheRef(first.batch_kv_cache_resource->cacheResource(0), {128}, true);
    ASSERT_NE(lease, nullptr);
    EXPECT_EQ(lease->blocks("linear_ssm"), before);
    for (const auto& info : requests) {
        allocator->free({info.batch_kv_cache_resource, info.complete_token_ids});
    }
    EXPECT_LT(allocator->freeBlocksNum(), initial_free);
    lease.reset();
    EXPECT_EQ(allocator->freeBlocksNum(), initial_free);
}
TEST_F(RequestStateCacheTest, PrefixReuseFlagDoesNotShareWritableRequestState) {
    auto cfg = config(3, true);
    auto allocator = std::make_shared<BlockTreeCacheTestAllocator<CoordinatorCacheManager>>(cfg);
    ASSERT_TRUE(allocator->init());

    auto first = request(cfg, 128);
    auto second = request(cfg, 128);
    first.reuse_cache = second.reuse_cache = true;
    ASSERT_TRUE(allocator->malloc(first).success);
    ASSERT_TRUE(allocator->malloc(second).success);
    for (const auto* tag : {"linear_ssm", "linear_replay"}) {
        const auto& first_ids = first.batch_kv_cache_resource->blocks(0, tag);
        const auto& second_ids = second.batch_kv_cache_resource->blocks(0, tag);
        ASSERT_EQ(first_ids.size(), 1);
        ASSERT_EQ(second_ids.size(), 1);
        EXPECT_NE(first_ids.front(), second_ids.front());
    }

    const auto first_ssm = first.batch_kv_cache_resource->blocks(0, "linear_ssm");
    first.enable_remove_skipped_blocks = true;
    first.incr_seq_len_override = 192;
    ASSERT_TRUE(allocator->malloc(first).success);
    EXPECT_EQ(first.batch_kv_cache_resource->blocks(0, "linear_ssm"), first_ssm);

    allocator->free({first.batch_kv_cache_resource, first.complete_token_ids});
    allocator->free({second.batch_kv_cache_resource, second.complete_token_ids});
}

TEST_F(RequestStateCacheTest, SingleSsmAdmits192AndKeepsOneRowAcrossPagesAndAsyncRelease) {
    auto cfg = config(193, true);
    cfg.finalizeBlockNums(32768, RuntimeConfig{});
    auto allocator = std::make_shared<BlockTreeCacheTestAllocator<CoordinatorCacheManager>>(cfg);
    ASSERT_TRUE(allocator->init());
    const auto              initial_free = allocator->freeBlocksNum();
    std::vector<MallocInfo> requests;
    for (int i = 0; i < 192; ++i) {
        auto info = request(cfg);
        ASSERT_TRUE(allocator->malloc(info).success) << i;
        EXPECT_EQ(info.batch_kv_cache_resource->blocksNum(0, "linear_ssm"), 1);
        EXPECT_EQ(info.batch_kv_cache_resource->blocksNum(0, "linear_replay"), 1);
        requests.push_back(std::move(info));
    }
    auto overflow = request(cfg);
    EXPECT_FALSE(allocator->malloc(overflow).success);
    auto&      first  = requests.front();
    const auto before = first.batch_kv_cache_resource->blocks(0, "linear_ssm");
    for (int length : {8193, 8195, 8257, 8260}) {
        first.incr_seq_len_override = length;
        ASSERT_TRUE(allocator->malloc(first).success);
        EXPECT_EQ(first.batch_kv_cache_resource->blocks(0, "linear_ssm"), before);
    }
    auto lease = allocator->incrKVCacheRef(first.batch_kv_cache_resource->cacheResource(0), {128}, true);
    ASSERT_NE(lease, nullptr);
    EXPECT_EQ(lease->blocks("linear_ssm"), before);
    for (const auto& info : requests) {
        allocator->free({info.batch_kv_cache_resource, info.complete_token_ids});
    }
    EXPECT_LT(allocator->freeBlocksNum(), initial_free);
    lease.reset();
    EXPECT_EQ(allocator->freeBlocksNum(), initial_free);
}

}  // namespace
}  // namespace rtp_llm::test
