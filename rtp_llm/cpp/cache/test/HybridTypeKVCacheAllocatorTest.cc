#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/LinearReplayPool.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {
namespace test {

static CacheConfig makeTinyHybridConfig() {
    // 4 layers: [0,1] linear, [2,3] full. gcd(2,2)=2 => group_size=2.
    CacheConfig config;
    config.dtype                     = rtp_llm::DataType::TYPE_FP16;
    config.layer_num                 = 4;
    config.layer_all_num             = 4;
    config.block_num                 = 10;
    config.seq_size_per_block        = 4;
    config.kernel_seq_size_per_block = 2;
    config.linear_step               = 2;
    config.group_layer_num           = 2;

    // Linear spec (small but valid).
    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type               = KVCacheSpecType::LinearAttention;
    linear_spec->dtype              = config.dtype;
    linear_spec->layer_num          = 2;
    linear_spec->local_num_k_heads  = 1;
    linear_spec->local_num_v_heads  = 1;
    linear_spec->head_k_dim         = 1;
    linear_spec->head_v_dim         = 1;
    linear_spec->conv_kernel_dim    = 2;
    linear_spec->local_head_num_kv  = 1;
    linear_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    // Full spec.
    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->dtype              = config.dtype;
    full_spec->layer_num          = 2;
    full_spec->local_head_num_kv  = 1;
    full_spec->size_per_head      = 1;
    full_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    // Order matters: linear groups first, then full groups (as in CacheConfigCreator).
    config.layer_ids        = {{0, 1}, {2, 3}};
    config.global_layer_ids = config.layer_ids;
    config.cache_specs      = {linear_spec, full_spec};
    config.linear_group_num = 1;
    config.full_group_num   = 1;

    // Physical block strides: take max between full and linear.
    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;

    // No kv scale for fp16.
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;

    config.block_size_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    config.layer_to_group_id.assign(static_cast<size_t>(config.layer_num), 0);
    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        for (int layer_id : config.layer_ids[gid]) {
            config.layer_to_group_id[static_cast<size_t>(layer_id)] = static_cast<int>(gid);
        }
    }
    return config;
}

static ModelConfig makeTinyModelConfig(uint32_t num_layers) {
    ModelConfig cfg;
    cfg.num_layers                   = static_cast<int64_t>(num_layers);
    cfg.max_seq_len                  = 128;
    cfg.hidden_size                  = 64;
    cfg.vocab_size                   = 1024;
    cfg.data_type                    = rtp_llm::DataType::TYPE_FP16;
    cfg.attn_config.head_num         = 2;
    cfg.attn_config.kv_head_num      = 2;
    cfg.attn_config.size_per_head    = 16;
    cfg.attn_config.tokens_per_block = 4;
    cfg.attn_config.use_mla          = false;
    cfg.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;
    return cfg;
}

static CacheConfig makeTinyHybridMtpConfigByCreateSpConfig() {
    auto score_model_cfg   = makeTinyModelConfig(/*num_layers=*/4);
    auto propose_model_cfg = makeTinyModelConfig(/*num_layers=*/1);

    score_model_cfg.hybrid_attention_config.enable_hybrid_attention = true;
    score_model_cfg.hybrid_attention_config.hybrid_attention_types  = {
        HybridAttentionType::LINEAR, HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::NONE};
    score_model_cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    score_model_cfg.linear_attention_config.linear_key_head_dim    = 8;
    score_model_cfg.linear_attention_config.linear_value_head_dim  = 8;
    score_model_cfg.linear_attention_config.linear_num_key_heads   = 2;
    score_model_cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;

    RuntimeConfig runtime_cfg;
    KVCacheConfig kv_cache_cfg;
    kv_cache_cfg.test_block_num = 8;

    SpeculativeExecutionConfig sp_cfg;
    sp_cfg.type              = SP_TYPE_MTP;
    sp_cfg.gen_num_per_cycle = 2;

    return CacheConfigCreator::createSpConfig(score_model_cfg,
                                              propose_model_cfg,
                                              parallelism_cfg,
                                              runtime_cfg,
                                              kv_cache_cfg,
                                              sp_cfg,
                                              /*warm_up_result=*/std::nullopt,
                                              /*is_mtp=*/true,
                                              /*is_eagle=*/false);
}

TEST(CacheConfigCreatorTest, IndependentHybridEagleUsesDedicatedThirdPool) {
    auto score_model_cfg   = makeTinyModelConfig(/*num_layers=*/4);
    auto propose_model_cfg = makeTinyModelConfig(/*num_layers=*/1);
    score_model_cfg.hybrid_attention_config.enable_hybrid_attention           = true;
    score_model_cfg.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    score_model_cfg.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::NONE,
        HybridAttentionType::LINEAR,
        HybridAttentionType::NONE,
        HybridAttentionType::LINEAR};
    propose_model_cfg.hybrid_attention_config.enable_hybrid_attention           = true;
    propose_model_cfg.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    propose_model_cfg.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::SLIDING_WINDOW};
    propose_model_cfg.attn_config.sliding_window = 2048;
    score_model_cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    score_model_cfg.linear_attention_config.linear_key_head_dim    = 8;
    score_model_cfg.linear_attention_config.linear_value_head_dim  = 8;
    score_model_cfg.linear_attention_config.linear_num_key_heads   = 2;
    score_model_cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    RuntimeConfig runtime_cfg;
    KVCacheConfig kv_cache_cfg;
    kv_cache_cfg.test_block_num = 8;
    SpeculativeExecutionConfig sp_cfg;
    sp_cfg.type              = SP_TYPE_EAGLE3;
    sp_cfg.gen_num_per_cycle = 3;

    auto config = CacheConfigCreator::createSpConfig(score_model_cfg,
                                                     propose_model_cfg,
                                                     parallelism_cfg,
                                                     runtime_cfg,
                                                     kv_cache_cfg,
                                                     sp_cfg,
                                                     /*warm_up_result=*/std::nullopt,
                                                     /*is_mtp=*/true,
                                                     /*is_eagle=*/true);
    ASSERT_TRUE(config.use_independent_block_pools);
    ASSERT_EQ(config.group_types.size(), 3u);
    EXPECT_EQ(config.group_types[0], CacheGroupType::FULL);
    EXPECT_EQ(config.group_types[1], CacheGroupType::LINEAR);
    EXPECT_EQ(config.group_types[2], CacheGroupType::SWA);
    ASSERT_EQ(config.layer_to_group_id.size(), 5u);
    EXPECT_EQ(config.layer_to_group_id[4], 2);
    ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
    ASSERT_EQ(config.mtp_sub_configs[0]->global_layer_ids.size(), 1u);
    EXPECT_EQ(config.mtp_sub_configs[0]->global_layer_ids[0], std::vector<int>({4}));
}

TEST(CacheConfigCreatorTest, KimiMtpFullAttentionUsesDedicatedThirdPool) {
    auto score_model_cfg   = makeTinyModelConfig(/*num_layers=*/4);
    auto propose_model_cfg = makeTinyModelConfig(/*num_layers=*/1);
    score_model_cfg.hybrid_attention_config.enable_hybrid_attention           = true;
    score_model_cfg.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    score_model_cfg.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::NONE,
        HybridAttentionType::LINEAR,
        HybridAttentionType::NONE,
        HybridAttentionType::LINEAR};
    propose_model_cfg.hybrid_attention_config.enable_hybrid_attention           = true;
    propose_model_cfg.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    propose_model_cfg.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::NONE};
    propose_model_cfg.attn_config.sliding_window = 0;
    score_model_cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    score_model_cfg.linear_attention_config.linear_key_head_dim    = 8;
    score_model_cfg.linear_attention_config.linear_value_head_dim  = 8;
    score_model_cfg.linear_attention_config.linear_num_key_heads   = 2;
    score_model_cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    RuntimeConfig runtime_cfg;
    KVCacheConfig kv_cache_cfg;
    kv_cache_cfg.test_block_num = 8;
    SpeculativeExecutionConfig sp_cfg;
    sp_cfg.type              = SP_TYPE_MTP;
    sp_cfg.model_type        = "kimi_k3_mtp";
    for (int candidates : {1, 3}) {
        sp_cfg.gen_num_per_cycle = candidates;

        auto config = CacheConfigCreator::createSpConfig(score_model_cfg,
                                                         propose_model_cfg,
                                                         parallelism_cfg,
                                                         runtime_cfg,
                                                         kv_cache_cfg,
                                                         sp_cfg,
                                                         /*warm_up_result=*/std::nullopt,
                                                         /*is_mtp=*/true,
                                                         /*is_eagle=*/false);
        ASSERT_TRUE(config.use_independent_block_pools);
        ASSERT_EQ(config.group_types.size(), 3u);
        EXPECT_EQ(config.group_types[0], CacheGroupType::FULL);
        EXPECT_EQ(config.group_types[1], CacheGroupType::LINEAR);
        EXPECT_EQ(config.group_types[2], CacheGroupType::FULL);
        ASSERT_EQ(config.layer_to_group_id.size(), 5u);
        EXPECT_EQ(config.layer_to_group_id[4], 2);
        ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
        ASSERT_EQ(config.mtp_sub_configs[0]->global_layer_ids.size(), 1u);
        EXPECT_EQ(config.mtp_sub_configs[0]->global_layer_ids[0], std::vector<int>({4}));
    }
}

static CompleteTokenIdsPtr makeCompleteTokenIds(int batch_size, int seq_length, int seq_size_per_block) {
    auto complete_token_ids =
        std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_length + 64, seq_size_per_block);
    auto  input_ids  = torch::empty({(int64_t)seq_length}, torch::kInt32);
    auto* token_data = input_ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_length; ++i) {
        token_data[i] = i + 1;
    }
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = input_ids;
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);
    return complete_token_ids;
}

static BatchKVCacheResourcePtr makeBatchResource(
    int batch_size, int group_nums, int layer_num, const std::vector<int>& layer_to_group_id, CacheKeysType keys) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(group_nums, layer_num, layer_to_group_id);
    for (int b = 0; b < batch_size; ++b) {
        res->setBatchCacheKeys(b, keys);
    }
    return res;
}

static std::vector<BlockIdxType> allocateAndCache(BlockPoolPtr         block_pool,
                                                  SharedBlockCachePtr  shared_cache,
                                                  int                  group_id,
                                                  int                  group_num,
                                                  const CacheKeysType& keys,
                                                  bool                 is_resident = true) {
    auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
    EXPECT_EQ(blocks.size(), keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<BlockIdxType> group_slots(static_cast<size_t>(group_num), NULL_BLOCK_IDX);
        group_slots[static_cast<size_t>(group_id)] = blocks[i];
        shared_cache->put(keys[i], group_slots, is_resident);
    }

    // Drop request references so these blocks behave like "cached but available" blocks.
    block_pool->requestFree(blocks);
    return blocks;
}

static std::vector<BlockIdxType> allocateAndCacheKeepAllocated(BlockPoolPtr         block_pool,
                                                               SharedBlockCachePtr  shared_cache,
                                                               int                  group_id,
                                                               int                  group_num,
                                                               const CacheKeysType& keys,
                                                               bool                 is_resident = true) {
    auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
    EXPECT_EQ(blocks.size(), keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<BlockIdxType> group_slots(static_cast<size_t>(group_num), NULL_BLOCK_IDX);
        group_slots[static_cast<size_t>(group_id)] = blocks[i];
        shared_cache->put(keys[i], group_slots, is_resident);
    }

    // NOTE: intentionally keep these blocks allocated/unavailable to avoid accidental reuse via malloc().
    return blocks;
}

static size_t countValidBlocks(const BlockIndicesType& blocks) {
    size_t n = 0;
    for (auto b : blocks) {
        if (!isNullBlockIdx(b)) {
            ++n;
        }
    }
    return n;
}

class HybridTypeKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

TEST_F(HybridTypeKVCacheAllocatorTest, InitAndAddressLookupSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->seqSizePerBlock(), 4);
    EXPECT_EQ(allocator->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator->freeBlocksNum(), config.block_num - 1);

    // Should be able to fetch address for any global layer and non-zero block id.
    auto addr0 = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    auto addr3 = allocator->convertIndexToAddr(/*layer_id=*/3, /*block_id=*/1);
    EXPECT_NE(addr0.kv_addr, nullptr);
    EXPECT_NE(addr3.kv_addr, nullptr);
}

TEST_F(HybridTypeKVCacheAllocatorTest, ConvertToGlobalLayerIdHybridNoMtp) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);

    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/0), 0u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/3), 3u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/4),
              std::numeric_limits<uint32_t>::max());

    // no mtp sub-model
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(HybridTypeKVCacheAllocatorTest, ConvertToGlobalLayerIdHybridWithMtpSubConfigs) {
    auto config    = makeTinyHybridMtpConfigByCreateSpConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);

    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/2), 2u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0), 4u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/0), 5u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/1),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/3, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(HybridTypeKVCacheAllocatorTest, GetNeedBlocksUsesGroupGetNeedBlocksAndReuseFlag) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    // batch=2, seq_len=12 (3 slots), reserve_step=2
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/12, /*seq_size_per_block=*/4);
    token_ids->setReserveStep(2);

    // Reuse disabled: linear group keeps tail and tail-1 for common blocks; reserve_step contributes extra blocks.
    // full group contributes common=3, extra=1.
    {
        auto       batch_res = makeBatchResource(/*batch_size=*/2,
                                           /*group_nums=*/2,
                                           /*layer_num=*/static_cast<int>(config.layer_all_num),
                                           /*layer_to_group_id=*/config.layer_to_group_id,
                                           CacheKeysType{100, 101, 102, 103});
        MallocInfo info{batch_res, token_ids};
        info.enable_device_cache = false;
        info.reuse_cache         = false;
        // common_total = full(3) + linear(2) = 5
        // extra_total  = full(1) + linear(reserve_step-1=1) = 2
        // total = 5 + 2*2 = 9
        EXPECT_EQ(allocator->getNeedBlocks(info), 9);
    }

    // Reuse enabled but no existing blocks: linear group keeps step hits plus tail/tail-1.
    {
        auto       batch_res = makeBatchResource(/*batch_size=*/2,
                                           /*group_nums=*/2,
                                           /*layer_num=*/static_cast<int>(config.layer_all_num),
                                           /*layer_to_group_id=*/config.layer_to_group_id,
                                           CacheKeysType{100, 101, 102, 103});
        MallocInfo info{batch_res, token_ids};
        info.enable_device_cache = true;
        info.reuse_cache         = true;
        // full: common=3 extra=1
        // linear: common=2, extra=reserve_step-1(=1)
        // common_total = 3 + 2 = 5
        // extra_total  = 1 + 1 = 2
        // total = 5 + 2*2 = 9
        EXPECT_EQ(allocator->getNeedBlocks(info), 9);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, JointReuseUsesFullPrefixAndLinearTailOnly) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto block_pool   = allocator->getBlockPool();
    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(block_pool, nullptr);
    ASSERT_NE(shared_cache, nullptr);

    // Config order: gid=0 linear, gid=1 full.
    const int gid_linear = 0;
    const int gid_full   = 1;
    const int group_num  = 2;

    // Full group has prefix matches for {100,101,102}.
    CacheKeysType full_keys   = {100, 101, 102};
    auto          full_blocks = allocateAndCache(block_pool, shared_cache, gid_full, group_num, full_keys);

    // Linear group only matches key 101 (so joint match should backoff to pos=1 => reuse_blocks_len=2).
    CacheKeysType linear_keys   = {101};
    auto          linear_blocks = allocateAndCache(block_pool, shared_cache, gid_linear, group_num, linear_keys);
    ASSERT_EQ(linear_blocks.size(), 1u);

    // Request has 4 keys, but allocator drops the last for matching.
    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102, 103});
    // Enable device cache reuse for joint match.

    // seq_len=12 => 3 slots (4 tokens per block).
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Full group: should reuse the first 2 blocks and allocate the third.
    const auto& full_out = batch_res->blocks(0, gid_full);
    ASSERT_EQ(full_out.size(), 3u);
    EXPECT_EQ(full_out[0], full_blocks[0]);
    EXPECT_EQ(full_out[1], full_blocks[1]);
    EXPECT_FALSE(isNullBlockIdx(full_out[2]));

    // Linear group: only the tail slot of the reused prefix is filled; earlier slots stay NULL.
    const auto& linear_out = batch_res->blocks(0, gid_linear);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_EQ(linear_out[1], linear_blocks[0]);   // reused tail at pos=1
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));  // allocated tail for common length
}

TEST_F(HybridTypeKVCacheAllocatorTest, DisableReuseKeepsLinearTailAndTailMinusOneOnInitMalloc) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102, 103});
    // Disable device cache reuse.

    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Linear group should keep tail and tail-1 across common length slots.
    const auto& linear_out = batch_res->blocks(0, /*group_id=*/0);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
}

TEST_F(HybridTypeKVCacheAllocatorTest, DisableDeviceCacheSkipsReuseMatchAndAllocatesLinearTailAndTailMinusOne) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto block_pool   = allocator->getBlockPool();
    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(block_pool, nullptr);
    ASSERT_NE(shared_cache, nullptr);

    // Config order: gid=0 linear, gid=1 full.
    const int gid_linear = 0;
    const int gid_full   = 1;
    const int group_num  = 2;

    // Prepare cached blocks for full group; keep them allocated so allocator's malloc() cannot accidentally return same
    // ids.
    CacheKeysType full_keys   = {100, 101, 102};
    auto          full_blocks = allocateAndCacheKeepAllocated(block_pool, shared_cache, gid_full, group_num, full_keys);
    ASSERT_EQ(full_blocks.size(), 3u);

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102, 103});
    // Disable device cache reuse: allocator should skip reuse match even if cache exists.

    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);  // 3 slots

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Device cache disabled => must not reuse match.
    EXPECT_EQ(result.reuse_len, 0);

    // Full group should allocate fresh blocks (not reuse cached ones).
    const auto& full_out = batch_res->blocks(0, gid_full);
    ASSERT_EQ(full_out.size(), 3u);
    EXPECT_FALSE(isNullBlockIdx(full_out[0]));
    EXPECT_FALSE(isNullBlockIdx(full_out[1]));
    EXPECT_FALSE(isNullBlockIdx(full_out[2]));
    EXPECT_NE(full_out[0], full_blocks[0]);
    EXPECT_NE(full_out[1], full_blocks[1]);
    EXPECT_NE(full_out[2], full_blocks[2]);

    // Linear group keeps tail and tail-1 when reuse is disabled.
    const auto& linear_out = batch_res->blocks(0, gid_linear);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
    EXPECT_EQ(countValidBlocks(linear_out), 2u);
}

TEST_F(HybridTypeKVCacheAllocatorTest, IncrDecrKVCacheRefReferencesOnlyMatchedValidBlocksAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::HOST);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator->freeBlocksNum();
    auto         blocks      = block_pool->malloc(4);
    ASSERT_EQ(blocks.size(), 4u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 4);

    KVCacheResource resource;
    resource.initGroups(/*group_nums=*/2,
                        /*layer_num=*/static_cast<int>(config.layer_all_num),
                        /*layer_to_group_id=*/config.layer_to_group_id);
    resource.cacheKeys() = CacheKeysType{100, 101, 102};
    resource.mutableBlockIds(/*gid=*/0).assign(
        BlockIndicesType{blocks[0], 0, blocks[1]});  // linear group (contains a 0)
    resource.mutableBlockIds(/*gid=*/1).assign(BlockIndicesType{blocks[2], blocks[3], 0});  // full group (contains a 0)

    // keys: 101(pos1)->gid0:0(ignore), gid1:blocks[3](ref); 102(pos2)->gid0:blocks[1](ref), gid1:0(ignore)
    auto ref = allocator->incrKVCacheRef(resource, CacheKeysType{101, 999, 102});
    ASSERT_NE(ref, nullptr);
    ASSERT_EQ(ref->groupNums(), 2);
    ASSERT_EQ(ref->cacheKeys(), (CacheKeysType{101, 102}));
    ASSERT_EQ(ref->blocks(0).size(), 2u);
    ASSERT_EQ(ref->blocks(1).size(), 2u);

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2) << "Only blocks[1] and blocks[3] should remain referenced";

    ref.reset();
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, IncrKVCacheRefPreservesConnectorDummyTail) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::HOST);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator->freeBlocksNum();
    auto         blocks      = block_pool->malloc(2);
    ASSERT_EQ(blocks.size(), 2u);

    KVCacheResource resource;
    resource.initGroups(/*group_nums=*/2,
                        /*layer_num=*/static_cast<int>(config.layer_all_num),
                        /*layer_to_group_id=*/config.layer_to_group_id);
    resource.cacheKeys() = CacheKeysType{101, 103, 999};
    resource.rebuildLinearBlockDependencies();
    resource.setLastBlockAligned(false);
    resource.mutableBlockIds(/*gid=*/0).assign(BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    resource.mutableBlockIds(/*gid=*/1).assign(BlockIndicesType{blocks[0], blocks[1]});

    auto ref = allocator->incrKVCacheRef(resource, CacheKeysType{101, 103, 999}, /*is_connector=*/true);
    ASSERT_NE(ref, nullptr);
    EXPECT_FALSE(ref->lastBlockAligned());
    EXPECT_EQ(ref->cacheKeys(), (CacheKeysType{101, 103, 999}));
    ASSERT_EQ(ref->blocks(0).size(), 3u);
    ASSERT_EQ(ref->blocks(1).size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(ref->blocks(0)[2]));
    EXPECT_TRUE(isNullBlockIdx(ref->blocks(1)[2]));

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2);

    ref.reset();
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, InsertIntoCachePreservesLegacyNonCpAggregateSurface) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto block_pool   = allocator->getBlockPool();
    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(block_pool, nullptr);
    ASSERT_NE(shared_cache, nullptr);

    // gid=0 linear, gid=1 full.
    const int gid_linear = 0;
    const int gid_full   = 1;

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102});
    // Disable device cache reuse.

    // Non-CP insert keeps the legacy aggregate surface: every materialized
    // group slot is merged under its key, including hybrid tail slots.
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/10, /*seq_size_per_block=*/4);

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    auto malloc_result              = allocator->malloc(malloc_info);
    ASSERT_TRUE(malloc_result.success);
    ASSERT_EQ(batch_res->blocksNum(0, gid_full), 3);
    ASSERT_EQ(batch_res->blocksNum(0, gid_linear), 3);

    InsertInfo insert_info{batch_res, token_ids, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    // Full group has all allocated slots cached, including the trailing block.
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(100, gid_full)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(101, gid_full)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(102, gid_full)));

    // Linear group keeps its tail and tail-minus-one slots.
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(100, gid_linear)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(101, gid_linear)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(102, gid_linear)));
}

TEST_F(HybridTypeKVCacheAllocatorTest, ConvertIndexToBufferAndAllLayerCacheBaseSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    KVCacheAllocator* base = allocator.get();
    auto              buf0 = base->convertIndexToBuffer(/*layer_id=*/0, /*block_id=*/1);
    ASSERT_FALSE(buf0.empty());
    EXPECT_NE(buf0[0].addr, nullptr);

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined());
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, IncrMallocRollbackFreesPartiallyAllocatedBlocks) {
    auto config      = makeTinyHybridConfig();
    config.block_num = 6;  // free=5
    auto allocator   = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102});
    // Disable device cache reuse (linear group still materializes tail and tail-1).

    // Initial small allocation: seq_len=4 => 1 slot per group.
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    auto init_result              = allocator->malloc(init_info);
    ASSERT_TRUE(init_result.success);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1);

    const auto linear_block_before = batch_res->blocks(0, /*gid=*/0)[0];
    const auto full_block_before   = batch_res->blocks(0, /*gid=*/1)[0];

    // Leave exactly 1 free block in pool, so linear allocates 1 and full fails on the next allocation.
    const size_t free_before_incr = block_pool->freeBlocksNum();
    ASSERT_GE(free_before_incr, 1u);
    auto keep = block_pool->malloc(static_cast<int>(free_before_incr - 1));
    ASSERT_EQ(block_pool->freeBlocksNum(), 1u);

    // Incr to seq_len=9 => 3 slots per group. Linear adds 2 slots but allocates only 1 real block; full needs 2.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    auto incr_result              = allocator->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    // Rollback should restore original sizes and keep original blocks.
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/0)[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/1)[0], full_block_before);

    // Free blocks count should return to 1 (no leaks).
    EXPECT_EQ(block_pool->freeBlocksNum(), 1u);

    // Cleanup.
    block_pool->requestFree(keep);
}

TEST_F(HybridTypeKVCacheAllocatorTest, ReplayRemovesTargetReserveAndKeepsTwoTails) {
    auto config                    = makeTinyHybridConfig();
    config.block_num               = 32;
    config.group_types             = {CacheGroupType::LINEAR, CacheGroupType::FULL};
    config.linear_replay_group_ids = {0};
    auto allocator                 = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());
    auto tokens = makeCompleteTokenIds(1, 12, 4);
    tokens->setReserveStep(4);
    auto       resource = makeBatchResource(1, 2, config.layer_all_num, config.layer_to_group_id, {100, 101, 102});
    MallocInfo info{resource, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    EXPECT_EQ(allocator->getNeedBlocks(info), 6);
    ASSERT_TRUE(allocator->malloc(info).success);
    ASSERT_EQ(resource->blocksNum(0, 0), 3);
    EXPECT_EQ(resource->blocks(0, 0)[0], NULL_BLOCK_IDX);
    EXPECT_GT(resource->blocks(0, 0)[1], 0);
    EXPECT_GT(resource->blocks(0, 0)[2], 0);
    EXPECT_EQ(resource->blocksNum(0, 1), 4);
    allocator->free({resource, tokens});
}

TEST(LinearReplayCacheTest, SlotOwnershipAndGeneration) {
    auto pool  = std::make_shared<LinearReplaySlotPool>(1);
    auto first = pool->acquire();
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->slot_id, 0);
    EXPECT_EQ(first->generation, 1);
    auto window = first;
    first.reset();
    EXPECT_EQ(pool->acquire(), nullptr);
    window.reset();
    auto second = pool->acquire();
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->slot_id, 0);
    EXPECT_EQ(second->generation, 2);
}

TEST_F(HybridTypeKVCacheAllocatorTest, ReplayReuseKeepsPrefillSnapshotsWithoutAccumulatingDecodeStates) {
    auto config                    = makeTinyHybridConfig();
    config.block_num               = 48;
    config.linear_step             = 1;
    config.group_types             = {CacheGroupType::LINEAR, CacheGroupType::FULL};
    config.linear_replay_group_ids = {0};
    auto allocator                 = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());
    auto       pool     = allocator->getBlockPool();
    auto       tokens   = makeCompleteTokenIds(1, 20, 4);
    auto       resource = makeBatchResource(1, 2, config.layer_all_num, config.layer_to_group_id, {});
    MallocInfo initial{resource, tokens};
    initial.enable_device_cache = false;
    initial.reuse_cache         = true;
    ASSERT_TRUE(allocator->malloc(initial).success);
    const auto prefill_blocks = resource->blocks(0, 0);
    ASSERT_EQ(prefill_blocks.size(), 5u);
    resource->cacheResource().restrictLinearReplayPrefix(0, 3);

    for (int slots = 6; slots <= 14; ++slots) {
        const auto&            previous = resource->blocks(0, 0);
        const BlockIndicesType in_flight{previous[previous.size() - 2], previous.back()};
        pool->replayReference(in_flight);
        ASSERT_GT(pool->freeBlocksNum(), 2u);
        const auto pressure = pool->malloc(static_cast<int>(pool->freeBlocksNum()) - 2);
        ASSERT_EQ(pool->freeBlocksNum(), 2u);

        tokens = makeCompleteTokenIds(1, slots * 4, 4);
        MallocInfo decode{resource, tokens};
        decode.enable_device_cache = false;
        decode.reuse_cache         = true;
        EXPECT_EQ(allocator->singleBatchNeedBlocks(resource, slots * 4, 0), 2);
        EXPECT_EQ(allocator->getNeedBlocks(decode), 2);
        ASSERT_TRUE(allocator->malloc(decode).success);
        const auto& blocks = resource->blocks(0, 0);
        ASSERT_EQ(blocks.size(), static_cast<size_t>(slots));
        for (size_t pos = 0; pos < 3; ++pos) {
            EXPECT_EQ(blocks[pos], prefill_blocks[pos]);
            EXPECT_TRUE(resource->cacheResource().canPublishLinearReplayBlock(0, pos));
        }
        for (int pos = 3; pos < slots - 2; ++pos) {
            EXPECT_EQ(blocks[static_cast<size_t>(pos)], NULL_BLOCK_IDX);
        }
        EXPECT_GT(blocks[blocks.size() - 2], 0);
        EXPECT_GT(blocks.back(), 0);
        EXPECT_EQ(pool->freeBlocksNum(), 0u);
        pool->replayFree(in_flight);
        EXPECT_EQ(pool->freeBlocksNum(), 1u);
        pool->requestFree(pressure);
    }
    allocator->free({resource, tokens});
    EXPECT_EQ(pool->freeBlocksNum(), config.block_num - 1);
}

TEST_F(HybridTypeKVCacheAllocatorTest, ReplaySmallBlocksMaterializeReachablePagesAndRollbackFailure) {
    auto config                      = makeTinyHybridConfig();
    config.block_num                 = 16;
    config.seq_size_per_block        = 1;
    config.kernel_seq_size_per_block = 1;
    config.group_types               = {CacheGroupType::LINEAR, CacheGroupType::FULL};
    config.linear_replay_group_ids   = {0};
    config.linear_replay_slot_count  = 1;
    config.linear_replay_max_steps   = 4;
    for (auto& spec : config.cache_specs) {
        spec->seq_size_per_block = 1;
    }
    auto manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());
    auto       pool     = manager->allocator_->getBlockPool();
    auto       resource = makeBatchResource(1, 2, config.layer_all_num, config.layer_to_group_id, {});
    const auto tails    = pool->malloc(2);
    ASSERT_EQ(tails.size(), 2u);
    resource->mutableBlockIds(0, 0).assign(
        {NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, tails[0], tails[1]});
    const auto original = resource->blocks(0, 0);
    const auto pressure = pool->malloc(static_cast<int>(pool->freeBlocksNum()) - 1);
    EXPECT_FALSE(manager->makeLinearReplayTailsPrivate(resource, 2));
    EXPECT_EQ(pool->freeBlocksNum(), 1u);
    EXPECT_EQ(resource->blocks(0, 0), original);
    pool->requestFree(pressure);
    ASSERT_TRUE(manager->makeLinearReplayTailsPrivate(resource, 2));
    const auto& blocks = resource->blocks(0, 0);
    EXPECT_EQ(blocks[0], NULL_BLOCK_IDX);
    EXPECT_EQ(blocks[1], NULL_BLOCK_IDX);
    for (size_t pos = 2; pos < 6; ++pos) {
        EXPECT_GT(blocks[pos], 0);
    }
    EXPECT_EQ(blocks[4], tails[0]);
    EXPECT_EQ(blocks[5], tails[1]);
    EXPECT_FALSE(resource->cacheResource().canPublishLinearReplayBlock(0, 2));
    auto       hold                = manager->holdLinearReplayBlocks(resource);
    const auto free_before_release = pool->freeBlocksNum();
    manager->free({resource, makeCompleteTokenIds(1, 6, 1)});
    EXPECT_EQ(pool->freeBlocksNum(), free_before_release);
    hold.reset();
    EXPECT_EQ(pool->freeBlocksNum(), free_before_release + 4);
}

TEST(LinearReplayCacheTest, DraftGroupsKeepTheirOwnReserve) {
    CacheConfig config;
    config.group_types             = {CacheGroupType::FULL, CacheGroupType::LINEAR, CacheGroupType::LINEAR};
    config.linear_replay_group_ids = {1};
    EXPECT_EQ(config.effectiveReserveStep(0, 4), 4);
    EXPECT_EQ(config.effectiveReserveStep(1, 4), 0);
    EXPECT_EQ(config.effectiveReserveStep(2, 4), 4);
}

#if USING_CUDA
static CacheConfig makeReplayConfig(bool                independent_pools    = false,
                                    SpeculativeType     speculative_type     = SP_TYPE_MTP,
                                    HybridAttentionType draft_attention_type = HybridAttentionType::LINEAR,
                                    const std::string&  target_model_type    = "qwen3_next") {
    auto target                                                      = makeTinyModelConfig(4);
    target.model_type                                                = target_model_type;
    target.hybrid_attention_config.enable_hybrid_attention           = true;
    target.hybrid_attention_config.enable_independent_kv_cache_pools = independent_pools;
    target.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::LINEAR, HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::NONE};
    target.linear_attention_config.linear_conv_kernel_dim = 4;
    target.linear_attention_config.linear_key_head_dim    = 8;
    target.linear_attention_config.linear_value_head_dim  = 8;
    target.linear_attention_config.linear_num_key_heads   = 2;
    target.linear_attention_config.linear_num_value_heads = 4;
    auto draft                                            = target;
    draft.num_layers                                      = 1;
    draft.model_type                                      = draft_attention_type == HybridAttentionType::LINEAR ?
                                                                "test_linear_mtp" :
                                                                (target_model_type == "kimi_k3" ? "kimi_k3_mla_swa_eagle3" : target_model_type + "_mtp");
    draft.hybrid_attention_config.hybrid_attention_types  = {draft_attention_type};
    RuntimeConfig runtime;
    runtime.max_generate_batch_size = 5;
    KVCacheConfig cache;
    cache.test_block_num = 32;
    SpeculativeExecutionConfig speculative;
    speculative.type              = speculative_type;
    speculative.model_type        = draft.model_type;
    speculative.gen_num_per_cycle = 3;
    const bool is_eagle           = speculative_type == SP_TYPE_EAGLE || speculative_type == SP_TYPE_EAGLE3;
    return CacheConfigCreator::createSpConfig(
        target, draft, ParallelismConfig{}, runtime, cache, speculative, std::nullopt, true, is_eagle);
}

TEST(LinearReplayCacheTest, ConfigReservesCompactLogsOnlyForTargetLayers) {
    const auto config = makeReplayConfig();
    ASSERT_EQ(config.linear_replay_group_ids, std::vector<int>({1}));
    ASSERT_EQ(config.group_types.size(), 3u);
    EXPECT_EQ(config.group_types[2], CacheGroupType::LINEAR);
    EXPECT_EQ(config.global_layer_ids[1], std::vector<int>({0, 1}));
    EXPECT_EQ(config.global_layer_ids[2], std::vector<int>({4, 5, 6}));
    EXPECT_EQ(config.linear_replay_slot_count, 5u);
    EXPECT_EQ(config.linear_replay_max_steps, 4u);
    EXPECT_FALSE(config.linear_replay_channelwise_gate);
    EXPECT_EQ(config.linear_replay_reserve_bytes, 13560u);
    for (const auto& draft_config : config.mtp_sub_configs) {
        EXPECT_TRUE(draft_config->linear_replay_group_ids.empty());
        EXPECT_EQ(draft_config->linear_replay_reserve_bytes, 0u);
    }
    for (size_t group = 2; group < config.group_types.size(); ++group) {
        EXPECT_EQ(config.effectiveReserveStep(group, 4), 4);
    }
}

TEST(LinearReplayCacheTest, IndependentLinearDraftPoolIncludesEveryMtpModule) {
    const auto config = makeReplayConfig(true);
    ASSERT_TRUE(config.use_independent_block_pools);
    ASSERT_EQ(config.group_types.size(), 3u);
    ASSERT_EQ(config.global_layer_ids[2], std::vector<int>({4, 5, 6}));
    EXPECT_EQ(config.group_block_size_bytes[2], 3 * config.cache_specs[2]->block_size_bytes());
    EXPECT_EQ(config.effectiveReserveStep(2, 4), 4);
}

TEST(LinearReplayCacheTest, QwenEagleFullDraftEnablesTargetReplay) {
    const auto config = makeReplayConfig(false, SP_TYPE_EAGLE, HybridAttentionType::NONE, "qwen35_moe");
    ASSERT_EQ(config.linear_replay_group_ids, std::vector<int>({1}));
    ASSERT_EQ(config.group_types.size(), 3u);
    EXPECT_EQ(config.group_types[2], CacheGroupType::FULL);
    EXPECT_EQ(config.global_layer_ids[2], std::vector<int>({4}));
    EXPECT_EQ(config.linear_replay_max_steps, 4u);
    EXPECT_FALSE(config.linear_replay_channelwise_gate);
    EXPECT_EQ(config.linear_replay_reserve_bytes, 13560u);
    ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
    EXPECT_TRUE(config.mtp_sub_configs[0]->linear_replay_group_ids.empty());
    EXPECT_EQ(config.effectiveReserveStep(1, 4), 0);
    EXPECT_EQ(config.effectiveReserveStep(2, 4), 4);
}

TEST(LinearReplayCacheTest, KimiEagle3FullDraftUsesTargetChannelwiseGates) {
    const auto config = makeReplayConfig(true, SP_TYPE_EAGLE3, HybridAttentionType::NONE, "kimi_k3");
    ASSERT_EQ(config.linear_replay_group_ids, std::vector<int>({1}));
    ASSERT_EQ(config.group_types.size(), 3u);
    EXPECT_EQ(config.group_types[2], CacheGroupType::FULL);
    EXPECT_EQ(config.global_layer_ids[2], std::vector<int>({4}));
    EXPECT_TRUE(config.linear_replay_channelwise_gate);
    EXPECT_EQ(config.linear_replay_reserve_bytes, 18040u);
    ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
    EXPECT_TRUE(config.mtp_sub_configs[0]->linear_replay_group_ids.empty());
}

TEST_F(HybridTypeKVCacheAllocatorTest, TargetReplayAndLinearDraftUseTheirOwnPhysicalLayouts) {
    auto manager = std::make_shared<KVCacheManager>(makeReplayConfig());
    ASSERT_TRUE(manager->init());
    const auto main = manager->getMainModelCacheLayerLayout();
    ASSERT_TRUE(main.linear_replay.has_value());
    ASSERT_EQ(main.linear_replay->keys.size(), 4u);
    EXPECT_TRUE(main.linear_replay->keys[0].defined());
    EXPECT_FALSE(main.linear_replay->keys[2].defined());
    std::vector<void*> draft_addresses;
    for (int module = 0; module < 3; ++module) {
        const auto draft = manager->getMTPModuleCacheLayerLayout(module);
        EXPECT_FALSE(draft.linear_replay.has_value());
        ASSERT_EQ(draft.layers_to_kv_buffer_ptrs.size(), 1u);
        const auto address = draft.layers_to_kv_buffer_ptrs[0].data_ptr();
        EXPECT_NE(address, main.layers_to_kv_buffer_ptrs[0].data_ptr());
        EXPECT_EQ(draft.layer_to_groups[0], 2);
        ASSERT_EQ(draft.layer_region_to_group_id.size(), 1u);
        EXPECT_EQ(draft.layer_region_to_group_id[0][static_cast<size_t>(KVCacheRegionName::DEFAULT)], 2);
        ASSERT_EQ(draft.group_types.size(), 3u);
        EXPECT_EQ(draft.group_types[2], CacheGroupType::LINEAR);
        for (auto previous : draft_addresses) {
            EXPECT_NE(address, previous);
        }
        draft_addresses.push_back(address);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, QwenEagleDraftDefaultRegionUsesPhysicalGroup) {
    const auto config  = makeReplayConfig(false, SP_TYPE_EAGLE, HybridAttentionType::NONE, "qwen35_moe");
    auto       manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());
    const auto layout = manager->getMTPModuleCacheLayerLayout(0);
    ASSERT_EQ(layout.layer_to_groups, std::vector<int>({2}));
    ASSERT_EQ(layout.layer_region_to_group_id.size(), 1u);
    EXPECT_EQ(layout.layer_region_to_group_id[0][static_cast<size_t>(KVCacheRegionName::DEFAULT)], 2);

    torch_ext::KVCache python_cache;
    python_cache.seq_size_per_block       = config.seq_size_per_block;
    python_cache.kv_cache_base_by_layer   = layout.layers_to_kv_buffer_ptrs;
    python_cache.layer_group_types        = layout.layer_group_types;
    python_cache.layer_region_to_group_id = layout.layer_region_to_group_id;
    const auto layer                      = python_cache.getLayerCache(0, KVCacheRegionName::DEFAULT);
    EXPECT_EQ(layer.group_id, 2);
    EXPECT_EQ(layer.kv_cache_base.data_ptr(), layout.layers_to_kv_buffer_ptrs[0].data_ptr());
}
#endif

TEST(LinearReplayCacheTest, MutableAnchorsCannotPublishPrefixSnapshots) {
    KVCacheResource resource;
    resource.initGroups(2, 2, {0, 1});
    resource.restrictLinearReplayPrefix(1, 3);
    EXPECT_TRUE(resource.canPublishLinearReplayBlock(0, 100));
    EXPECT_TRUE(resource.canPublishLinearReplayBlock(1, 2));
    EXPECT_FALSE(resource.canPublishLinearReplayBlock(1, 3));
    resource.restrictLinearReplayPrefix(1, 8);
    EXPECT_FALSE(resource.canPublishLinearReplayBlock(1, 3));
    resource.restrictLinearReplayPrefix(1, 2);
    EXPECT_FALSE(resource.canPublishLinearReplayBlock(1, 2));
    resource.initGroups(2, 2, {0, 1});
    EXPECT_TRUE(resource.canPublishLinearReplayBlock(1, 3));
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
