#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

static CacheConfig makeTinyHybridConfig() {
    auto config                      = makeSimpleHybridMhaCacheConfig(/*layer_num=*/4,
                                                 /*block_num=*/10,
                                                 /*tokens_per_block=*/4,
                                                 rtp_llm::DataType::TYPE_FP16,
                                                 /*group_layer_num=*/2,
                                                 /*local_head_num_kv=*/1,
                                                 /*size_per_head=*/1);
    config.kernel_seq_size_per_block = 2;
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
    cfg.kv_cache_spec_descs.resize(num_layers);
    for (uint32_t i = 0; i < num_layers; ++i) {
        cfg.kv_cache_spec_descs[i].push_back(KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention});
    }
    return cfg;
}

static KVCacheSpecPtr makeLinearSpecWithGlobalHeads(uint32_t key_heads, uint32_t value_heads, uint32_t tp) {
    LinearAttentionConfig linear_config;
    linear_config.linear_conv_kernel_dim = 2;
    linear_config.linear_key_head_dim    = 8;
    linear_config.linear_value_head_dim  = 8;
    linear_config.linear_num_key_heads   = static_cast<int>(key_heads);
    linear_config.linear_num_value_heads = static_cast<int>(value_heads);

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = tp;

    KVCacheSpecDesc desc;
    desc.tag        = "linear_test";
    desc.cache_type = KVCacheSpecType::LinearAttention;
    desc.dtype      = DataType::TYPE_FP16;

    SpecBuildContext ctx;
    ctx.dtype                   = DataType::TYPE_FP16;
    ctx.seq_size_per_block      = 1;
    ctx.linear_attention_config = &linear_config;
    ctx.parallelism_config      = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}

static void setHybridLayerDescs(ModelConfig& cfg, const std::vector<HybridAttentionType>& types) {
    cfg.hybrid_attention_config.enable_hybrid_attention = true;
    cfg.hybrid_attention_config.hybrid_attention_types  = types;
    cfg.kv_cache_spec_descs.assign(static_cast<size_t>(cfg.num_layers), {});
    for (size_t i = 0; i < types.size(); ++i) {
        if (types[i] == HybridAttentionType::LINEAR) {
            cfg.kv_cache_spec_descs[i].push_back(KVCacheSpecDesc{"linear", KVCacheSpecType::LinearAttention});
        } else {
            cfg.kv_cache_spec_descs[i].push_back(KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention});
        }
    }
}

static void setHybridLayerDescsWithTags(ModelConfig&                            cfg,
                                        const std::vector<HybridAttentionType>& types,
                                        const std::vector<std::string>&         tags) {
    cfg.hybrid_attention_config.enable_hybrid_attention = true;
    cfg.hybrid_attention_config.hybrid_attention_types  = types;
    cfg.kv_cache_spec_descs.assign(static_cast<size_t>(cfg.num_layers), {});
    for (size_t i = 0; i < types.size(); ++i) {
        const auto cache_type = types[i] == HybridAttentionType::LINEAR ? KVCacheSpecType::LinearAttention :
                                                                          KVCacheSpecType::MultiHeadAttention;
        cfg.kv_cache_spec_descs[i].push_back(KVCacheSpecDesc{tags[i], cache_type});
    }
}

static CacheConfig makeTinyHybridMtpConfigByCreateSpConfig() {
    auto score_model_cfg   = makeTinyModelConfig(/*num_layers=*/4);
    auto propose_model_cfg = makeTinyModelConfig(/*num_layers=*/1);

    setHybridLayerDescs(score_model_cfg,
                        {HybridAttentionType::LINEAR,
                         HybridAttentionType::LINEAR,
                         HybridAttentionType::NONE,
                         HybridAttentionType::NONE});
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

static BatchKVCacheResourcePtr makeBatchResource(int                                  batch_size,
                                                 int                                  group_nums,
                                                 int                                  layer_num,
                                                 const std::vector<std::vector<int>>& layer_group_ids,
                                                 CacheKeysType                        keys) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(group_nums, layer_num, layer_group_ids);
    for (int b = 0; b < batch_size; ++b) {
        res->setBatchCacheKeys(b, keys);
    }
    return res;
}

static int estimateBatchPeakForSingleSequence(const KVCacheAllocator&        allocator,
                                              const BatchKVCacheResourcePtr& batch_resource,
                                              int                            seq_len,
                                              int                            remaining_tokens,
                                              int                            reserve_step,
                                              bool                           enable_reuse_cache) {
    return allocator.estimateBatchPeakNeedBlocks(batch_resource,
                                                 seq_len,
                                                 /*common_seq_len=*/seq_len,
                                                 remaining_tokens,
                                                 reserve_step,
                                                 enable_reuse_cache,
                                                 /*target_batch_size=*/1);
}

static std::vector<BlockIdxType> allocateAndCache(BlockPoolPtr         block_pool,
                                                  SharedBlockCachePtr  shared_cache,
                                                  int                  group_nums,
                                                  int                  group_id,
                                                  const CacheKeysType& keys,
                                                  bool                 is_resident = true) {
    auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
    EXPECT_EQ(blocks.size(), keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<BlockIdxType> group_slots(static_cast<size_t>(group_nums), NULL_BLOCK_IDX);
        group_slots[static_cast<size_t>(group_id)] = blocks[i];
        shared_cache->put(keys[i], group_slots, is_resident);
    }

    block_pool->requestFree(blocks);
    return blocks;
}

static std::vector<BlockIdxType> allocateAndCacheKeepAllocated(BlockPoolPtr         block_pool,
                                                               SharedBlockCachePtr  shared_cache,
                                                               int                  group_nums,
                                                               int                  group_id,
                                                               const CacheKeysType& keys,
                                                               bool                 is_resident = true) {
    auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
    EXPECT_EQ(blocks.size(), keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<BlockIdxType> group_slots(static_cast<size_t>(group_nums), NULL_BLOCK_IDX);
        group_slots[static_cast<size_t>(group_id)] = blocks[i];
        shared_cache->put(keys[i], group_slots, is_resident);
    }

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

TEST_F(HybridTypeKVCacheAllocatorTest, CreateHybridConfigAllowsOnlyFullGroups) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescs(cfg, {HybridAttentionType::NONE, HybridAttentionType::NONE});

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    auto cache_config =
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
    ASSERT_EQ(cache_config.groupNums(), 1);
    EXPECT_EQ(cache_config.groupTypesSnapshot()[0], CacheGroupType::FULL);
    EXPECT_EQ(cache_config.groupTagsSnapshot()[0], "full");
}

TEST_F(HybridTypeKVCacheAllocatorTest, CreateHybridConfigRejectsMultipleFullGroups) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescsWithTags(cfg, {HybridAttentionType::NONE, HybridAttentionType::NONE}, {"full", "full1"});

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    try {
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
        FAIL() << "expected multiple full groups to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("multiple full attention cache groups"), std::string::npos);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, CreateHybridConfigKeepsModelTokensPerBlock) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescs(cfg, {HybridAttentionType::NONE, HybridAttentionType::NONE});

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;

    auto cache_config =
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
    EXPECT_EQ(cache_config.seq_size_per_block, 4);
    ASSERT_EQ(cache_config.groupNums(), 1);
    EXPECT_EQ(cache_config.specForGroup(0)->seq_size_per_block, 4);
}

TEST(HybridCacheConfigTest, LinearSpecRejectsHeadsNotDivisibleByAttentionTp) {
    try {
        (void)makeLinearSpecWithGlobalHeads(/*key_heads=*/6, /*value_heads=*/8, /*tp=*/4);
        FAIL() << "expected non-divisible linear heads to be rejected";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("tag=linear_test"), std::string::npos);
        EXPECT_NE(message.find("key=6 value=8 tp=4"), std::string::npos);
    }
}

TEST(HybridCacheConfigTest, LinearSpecRejectsInvalidValueToKeyHeadGrouping) {
    try {
        (void)makeLinearSpecWithGlobalHeads(/*key_heads=*/8, /*value_heads=*/4, /*tp=*/4);
        FAIL() << "expected invalid linear value/key head grouping to be rejected";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("tag=linear_test"), std::string::npos);
        EXPECT_NE(message.find("key=8 value=4 tp=4"), std::string::npos);
    }
}

TEST(HybridCacheConfigTest, LinearSpecRejectsNonMultipleValueHeadsAfterTpValidation) {
    try {
        (void)makeLinearSpecWithGlobalHeads(/*key_heads=*/4, /*value_heads=*/6, /*tp=*/2);
        FAIL() << "expected non-multiple linear value/key head grouping to be rejected";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("tag=linear_test"), std::string::npos);
        EXPECT_NE(message.find("key=4 value=6 tp=2"), std::string::npos);
    }
}

TEST(HybridCacheConfigTest, LinearSpecUsesTensorParallelLocalHeadsForBlockSizes) {
    const auto spec = makeLinearSpecWithGlobalHeads(/*key_heads=*/4, /*value_heads=*/8, /*tp=*/4);

    // local key/value heads are 1/2. With head dims 8 and conv kernel dim 2:
    // SSM = 2 * 8 * 8, convolution = (2 - 1) * (2 * 1 * 8 + 2 * 8).
    EXPECT_EQ(spec->k_block_size(), 128u);
    EXPECT_EQ(spec->v_block_size(), 32u);
    EXPECT_EQ(spec->block_size(), 160u);
}

TEST_F(HybridTypeKVCacheAllocatorTest, CreateHybridConfigRejectsOnlyLinearGroups) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescs(cfg, {HybridAttentionType::LINEAR, HybridAttentionType::LINEAR});
    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    try {
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
        FAIL() << "expected a linear-only hybrid config to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("exactly one FULL MHA/MLA cache group"), std::string::npos);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, CreateSingleConfigRejectsLinearDescriptor) {
    auto cfg                                            = makeTinyModelConfig(/*num_layers=*/1);
    cfg.hybrid_attention_config.enable_hybrid_attention = false;
    cfg.kv_cache_spec_descs = {{KVCacheSpecDesc{"linear", KVCacheSpecType::LinearAttention}}};
    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    try {
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
        FAIL() << "expected a linear-only single config to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("exactly one FULL MHA/MLA cache group"), std::string::npos);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, InitRejectsOnlyLinearGroupsBeforeCreatingBlockPool) {
    auto cache_config = makeSimpleLinearCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto linear0 = makeLinearSpec("linear0", /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16, 1, 1);
    auto linear1 = makeLinearSpec("linear1", /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16, 1, 1);
    cache_config.fromGroupedSpecs(
        {linear0, linear1}, {{0}, {1}}, {CacheGroupType::LINEAR, CacheGroupType::LINEAR}, {"linear0", "linear1"});
    ASSERT_EQ(cache_config.groupNums(), 2);

    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(cache_config, AllocationType::DEVICE);
    EXPECT_THROW(allocator->init(), std::runtime_error);
    EXPECT_EQ(allocator->getBlockPool(), nullptr);
}

TEST_F(HybridTypeKVCacheAllocatorTest, TopologyRejectsSpecPolicyTypeMismatch) {
    auto config = makeSimpleLinearCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto groups      = config.topology().groups();
    auto layers      = config.topology().layers();
    groups[0].policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(HybridTypeKVCacheAllocatorTest, TopologyRejectsGroupLayerMissingForwardGid) {
    auto config = makeTinyHybridConfig();
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    groups[0].layer_ids.push_back(2);

    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(HybridTypeKVCacheAllocatorTest, TopologyRejectsMissingLayerTagMapping) {
    auto config = makeTinyHybridConfig();
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    layers[0].group_tags.clear();

    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(HybridTypeKVCacheAllocatorTest, CreateHybridConfigUsesTaggedContiguousLinearGroupsAndFullFirst) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/8);
    setHybridLayerDescsWithTags(cfg,
                                {HybridAttentionType::LINEAR,
                                 HybridAttentionType::LINEAR,
                                 HybridAttentionType::LINEAR,
                                 HybridAttentionType::NONE,
                                 HybridAttentionType::LINEAR,
                                 HybridAttentionType::LINEAR,
                                 HybridAttentionType::LINEAR,
                                 HybridAttentionType::NONE},
                                {"linear0", "linear0", "linear1", "full", "linear1", "linear2", "linear2", "full"});
    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    auto cache_config =
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);

    std::vector<std::string>    expected_tags{"full", "linear0", "linear1", "linear2"};
    std::vector<CacheGroupType> expected_types{
        CacheGroupType::FULL, CacheGroupType::LINEAR, CacheGroupType::LINEAR, CacheGroupType::LINEAR};
    std::vector<int> expected_full{3, 7};
    std::vector<int> expected_linear0{0, 1};
    std::vector<int> expected_linear1{2, 4};
    std::vector<int> expected_linear2{5, 6};

    ASSERT_EQ(cache_config.groupNums(), 4);
    EXPECT_EQ(cache_config.groupTagsSnapshot(), expected_tags);
    EXPECT_EQ(cache_config.groupTypesSnapshot(), expected_types);
    EXPECT_EQ(cache_config.layerIdsForGroup(0), expected_full);
    EXPECT_EQ(cache_config.layerIdsForGroup(1), expected_linear0);
    EXPECT_EQ(cache_config.layerIdsForGroup(2), expected_linear1);
    EXPECT_EQ(cache_config.layerIdsForGroup(3), expected_linear2);
}

TEST_F(HybridTypeKVCacheAllocatorTest, InitAndAddressLookupSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
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

    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (size_t mtp_id = 0; mtp_id < config.mtp_sub_configs.size(); ++mtp_id) {
        const auto& sub = config.mtp_sub_configs[mtp_id];
        ASSERT_NE(sub, nullptr);
        ASSERT_EQ(sub->groupNums(), 2);
        std::vector<std::string> expected_tags{"full", "linear"};
        EXPECT_EQ(sub->groupTagsSnapshot(), expected_tags);
        ASSERT_EQ(sub->layerIdsForGroup(0).size(), 1u);
        EXPECT_EQ(sub->layerIdsForGroup(0)[0], 0);
        EXPECT_TRUE(sub->layerIdsForGroup(1).empty());
    }

    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/2), 2u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0), 4u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/0), 5u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/1),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/3, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(HybridTypeKVCacheAllocatorTest, MergeMtpRejectsShortTargetGroup) {
    CacheConfig main_config;
    main_config.layer_num       = 5;
    main_config.layer_all_num   = 5;
    main_config.group_layer_num = 3;
    main_config.fromGroupedSpecs(
        {makeMhaSpec("full", 4, DataType::TYPE_FP16, 1, 1), makeLinearSpec("linear", 4, DataType::TYPE_FP16, 1, 1)},
        {{0, 1, 2}, {3, 4}},
        {CacheGroupType::FULL, CacheGroupType::LINEAR},
        {"full", "linear"});
    main_config.layer_to_block_stride_bytes.assign(6, 1);

    auto propose_config = makeSimpleLinearCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    EXPECT_THROW(main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/5),
                 std::runtime_error);
}

TEST_F(HybridTypeKVCacheAllocatorTest, MergeMtpRejectsPartialOrReorderedSourceGroup) {
    auto main_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    main_config.group_layer_num = 2;
    main_config.layer_to_block_stride_bytes.assign(4, 1);

    CacheConfig partial_source;
    partial_source.layer_num     = 2;
    partial_source.layer_all_num = 2;
    partial_source.fromGroupedSpecs(
        {makeMhaSpec("default", 4, DataType::TYPE_FP16, 1, 1), makeMhaSpec("aux", 4, DataType::TYPE_FP16, 1, 1)},
        {{0}, {1}},
        {CacheGroupType::FULL, CacheGroupType::FULL},
        {"default", "aux"});
    partial_source.layer_to_block_stride_bytes.assign(2, 1);
    EXPECT_THROW(main_config.mergeMTPModule(partial_source, /*module_index=*/0, /*main_layer_num=*/2),
                 std::runtime_error);

    auto reordered_source = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto reordered_groups         = reordered_source.topology().groups();
    auto reordered_layers         = reordered_source.topology().layers();
    reordered_groups[0].layer_ids = {1, 0};
    reordered_source.setTopology(std::move(reordered_groups), std::move(reordered_layers));
    EXPECT_THROW(main_config.mergeMTPModule(reordered_source, /*module_index=*/0, /*main_layer_num=*/2),
                 std::runtime_error);
}

TEST_F(HybridTypeKVCacheAllocatorTest, MtpPhysicalSlotsDoNotAliasMainSlots) {
    auto config    = makeTinyHybridMtpConfigByCreateSpConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const auto main0 = allocator->convertIndexToAddr(/*layer_id=*/2, /*block_id=*/1);
    const auto main1 = allocator->convertIndexToAddr(/*layer_id=*/3, /*block_id=*/1);
    const auto mtp0  = allocator->convertIndexToAddr(/*layer_id=*/4, /*block_id=*/1);
    const auto mtp1  = allocator->convertIndexToAddr(/*layer_id=*/5, /*block_id=*/1);
    ASSERT_NE(main0.kv_addr, nullptr);
    ASSERT_NE(main1.kv_addr, nullptr);
    ASSERT_NE(mtp0.kv_addr, nullptr);
    ASSERT_NE(mtp1.kv_addr, nullptr);
    EXPECT_NE(mtp0.kv_addr, main0.kv_addr);
    EXPECT_NE(mtp0.kv_addr, main1.kv_addr);
    EXPECT_NE(mtp1.kv_addr, main0.kv_addr);
    EXPECT_NE(mtp1.kv_addr, main1.kv_addr);
    EXPECT_NE(mtp0.kv_addr, mtp1.kv_addr);
}

TEST_F(HybridTypeKVCacheAllocatorTest, MtpLayoutProjectionRecountsActiveLayersAndKeepsEmptyPlaceholder) {
    auto config  = makeTinyHybridMtpConfigByCreateSpConfig();
    auto manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());

    const auto layout = manager->getMTPModuleGroupedCacheLayerLayout(0);
    ASSERT_EQ(layout.topology().layers().size(), 1u);
    EXPECT_EQ(layout.group("full").activeLayerCount(), 1u);
    EXPECT_FALSE(layout.group("full").empty());
    EXPECT_EQ(layout.group("linear").activeLayerCount(), 0u);
    EXPECT_TRUE(layout.group("linear").empty());
    EXPECT_TRUE(layout.at("full", 0).kv_addr.defined());
}

TEST_F(HybridTypeKVCacheAllocatorTest, GetNeedBlocksUsesGroupGetNeedBlocksAndReuseFlag) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // batch=2, seq_len=12 (3 slots), reserve_step=2
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/12, /*seq_size_per_block=*/4);
    token_ids->setReserveStep(2);

    // Reuse disabled: linear group keeps only tail for common blocks; reserve_step contributes extra blocks.
    // full group contributes common=3, extra=1.
    {
        auto       batch_res = makeBatchResource(/*batch_size=*/2,
                                           /*group_nums=*/2,
                                           /*layer_num=*/static_cast<int>(config.layer_all_num),
                                           /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                           CacheKeysType{100, 101, 102, 103});
        MallocInfo info{batch_res, token_ids};
        info.enable_device_cache = false;
        info.reuse_cache         = false;
        // common_total = full(3) + linear(1) = 4
        // extra_total  = full(1) + linear(reserve_step-1=1) = 2
        // total = 4 + 2*2 = 8
        EXPECT_EQ(allocator->getNeedBlocks(info), 8);
    }

    // Reuse enabled but no existing blocks: linear group uses sparse counting from begin=0.
    {
        auto       batch_res = makeBatchResource(/*batch_size=*/2,
                                           /*group_nums=*/2,
                                           /*layer_num=*/static_cast<int>(config.layer_all_num),
                                           /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                           CacheKeysType{100, 101, 102, 103});
        MallocInfo info{batch_res, token_ids};
        info.enable_device_cache = true;
        info.reuse_cache         = true;
        // full: common=3 extra=1
        // linear: common=count(0,3]=2, extra=reserve_step-1(=1)
        // common_total = 3 + 2 = 5
        // extra_total  = 1 + 1 = 2
        // total = 5 + 2*2 = 9
        EXPECT_EQ(allocator->getNeedBlocks(info), 9);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, JointReuseUsesFullPrefixAndLinearTailOnly) {
    auto config       = makeTinyHybridConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    // Config order: gid=0 linear, gid=1 full.
    const int gid_linear = 0;
    const int gid_full   = 1;

    // Full group has prefix matches for {100,101,102}.
    CacheKeysType full_keys   = {100, 101, 102};
    auto          full_blocks = allocateAndCache(block_pool, shared_cache, config.groupNums(), gid_full, full_keys);

    // Linear group only matches key 101 (so joint match should backoff to pos=1 => reuse_blocks_len=2).
    CacheKeysType linear_keys = {101};
    auto linear_blocks        = allocateAndCache(block_pool, shared_cache, config.groupNums(), gid_linear, linear_keys);
    ASSERT_EQ(linear_blocks.size(), 1u);

    // Request has 4 keys, but allocator drops the last for matching.
    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
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

TEST_F(HybridTypeKVCacheAllocatorTest, DisableReuseKeepsOnlyLinearTailOnInitMalloc) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                       CacheKeysType{100, 101, 102, 103});
    // Disable device cache reuse.

    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Linear group should keep only the tail block across common length slots.
    const auto& linear_out = batch_res->blocks(0, /*group_id=*/0);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
}

TEST_F(HybridTypeKVCacheAllocatorTest, DisableDeviceCacheSkipsReuseMatchAndAllocatesOnlyLinearTail) {
    auto config       = makeTinyHybridConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    // Config order: gid=0 linear, gid=1 full.
    const int gid_linear = 0;
    const int gid_full   = 1;

    // Prepare cached blocks for full group; keep them allocated so allocator's malloc() cannot accidentally return same
    // ids.
    CacheKeysType full_keys = {100, 101, 102};
    auto full_blocks = allocateAndCacheKeepAllocated(block_pool, shared_cache, config.groupNums(), gid_full, full_keys);
    ASSERT_EQ(full_blocks.size(), 3u);

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
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

    // Linear group keeps only tail block (others NULL) when reuse is disabled.
    const auto& linear_out = batch_res->blocks(0, gid_linear);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
    EXPECT_EQ(countValidBlocks(linear_out), 1u);
}

TEST_F(HybridTypeKVCacheAllocatorTest, UpdateKVBlockForksSharedBlocksAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator->freeBlocksNum();
    auto         blocks      = block_pool->malloc(6);
    ASSERT_EQ(blocks.size(), 6u);
    ASSERT_EQ(allocator->freeBlocksNum(), free_before - 6);

    auto batch_res = makeBatchResource(/*batch_size=*/2,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                       CacheKeysType{100, 101});
    batch_res->mutableBlockIds(/*batch_id=*/0, /*gid=*/0).assign({blocks[0], NULL_BLOCK_IDX, blocks[1]});
    batch_res->mutableBlockIds(/*batch_id=*/0, /*gid=*/1).assign({blocks[2], blocks[3]});
    batch_res->mutableBlockIds(/*batch_id=*/1, /*gid=*/0).assign({blocks[4]});
    batch_res->mutableBlockIds(/*batch_id=*/1, /*gid=*/1).assign({blocks[5]});

    std::vector<BlockIdPair> update_mapping;
    ASSERT_TRUE(allocator->updateKVBlock(batch_res,
                                         /*block_src_batch=*/std::vector<int>{0, 0},
                                         /*copy_last_block=*/false,
                                         update_mapping));

    EXPECT_TRUE(update_mapping.empty());
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 4) << "unused old batch blocks should be released";
    ASSERT_EQ(batch_res->batchSize(), 2);
    EXPECT_EQ(batch_res->cacheKeys(0), (CacheKeysType{100, 101}));
    EXPECT_EQ(batch_res->cacheKeys(1), (CacheKeysType{100, 101}));
    EXPECT_EQ(batch_res->blocks(0, 0), (BlockIndicesType{blocks[0], NULL_BLOCK_IDX, blocks[1]}));
    EXPECT_EQ(batch_res->blocks(0, 1), (BlockIndicesType{blocks[2], blocks[3]}));
    EXPECT_EQ(batch_res->blocks(1, 0), (BlockIndicesType{blocks[0], NULL_BLOCK_IDX, blocks[1]}));
    EXPECT_EQ(batch_res->blocks(1, 1), (BlockIndicesType{blocks[2], blocks[3]}));

    allocator->free(FreeInfo{batch_res, nullptr});
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, UpdateKVBlockCopyLastBlockAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator->freeBlocksNum();
    auto         blocks      = block_pool->malloc(6);
    ASSERT_EQ(blocks.size(), 6u);
    ASSERT_EQ(allocator->freeBlocksNum(), free_before - 6);

    auto batch_res = makeBatchResource(/*batch_size=*/2,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                       CacheKeysType{100, 101});
    batch_res->mutableBlockIds(/*batch_id=*/0, /*gid=*/0).assign({blocks[0], NULL_BLOCK_IDX, blocks[1]});
    batch_res->mutableBlockIds(/*batch_id=*/0, /*gid=*/1).assign({blocks[2], blocks[3]});
    batch_res->mutableBlockIds(/*batch_id=*/1, /*gid=*/0).assign({blocks[4]});
    batch_res->mutableBlockIds(/*batch_id=*/1, /*gid=*/1).assign({blocks[5]});

    std::vector<BlockIdPair> update_mapping{BlockIdPair{1, 2}};
    ASSERT_TRUE(allocator->updateKVBlock(batch_res,
                                         /*block_src_batch=*/std::vector<int>{0, 0},
                                         /*copy_last_block=*/true,
                                         update_mapping));

    EXPECT_TRUE(update_mapping.empty());
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 6);
    ASSERT_EQ(batch_res->batchSize(), 2);
    EXPECT_EQ(batch_res->cacheKeys(0), (CacheKeysType{100, 101}));
    EXPECT_EQ(batch_res->cacheKeys(1), (CacheKeysType{100, 101}));

    const auto& forked_group0 = batch_res->blocks(0, 0);
    const auto& moved_group0  = batch_res->blocks(1, 0);
    const auto& forked_group1 = batch_res->blocks(0, 1);
    const auto& moved_group1  = batch_res->blocks(1, 1);
    ASSERT_EQ(forked_group0.size(), 3u);
    ASSERT_EQ(forked_group1.size(), 2u);
    EXPECT_EQ(moved_group0, (BlockIndicesType{blocks[0], NULL_BLOCK_IDX, blocks[1]}));
    EXPECT_EQ(moved_group1, (BlockIndicesType{blocks[2], blocks[3]}));
    EXPECT_EQ(forked_group0[0], blocks[0]);
    EXPECT_TRUE(isNullBlockIdx(forked_group0[1]));
    EXPECT_NE(forked_group0[2], blocks[1]);
    EXPECT_FALSE(isNullBlockIdx(forked_group0[2]));
    EXPECT_EQ(forked_group1[0], blocks[2]);
    EXPECT_NE(forked_group1[1], blocks[3]);
    EXPECT_FALSE(isNullBlockIdx(forked_group1[1]));

    allocator->free(FreeInfo{batch_res, nullptr});
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, IncrDecrKVCacheRefReferencesOnlyMatchedValidBlocksAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::HOST);
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
                        /*layer_group_ids=*/config.layerGroupIdsSnapshot());
    resource.cacheKeys() = CacheKeysType{100, 101, 102};
    resource.mutableBlockIds(/*gid=*/0).assign(
        BlockIndicesType{blocks[0], 0, blocks[1]});  // linear group (contains a 0)
    resource.mutableBlockIds(/*gid=*/1).assign(BlockIndicesType{blocks[2], blocks[3], 0});  // full group (contains a 0)

    // keys: 101(pos1)->gid0:0(ignore), gid1:blocks[3](ref); 102(pos2)->gid0:blocks[1](ref), gid1:0(ignore).
    // The migrated HybridKV base drops unmatched keys rather than preserving empty placeholders.
    auto ref = allocator->incrKVCacheRef(resource, CacheKeysType{101, 999, 102});
    ASSERT_NE(ref, nullptr);
    ASSERT_EQ(ref->groupNums(), 2);
    ASSERT_EQ(ref->cacheKeys().size(), 2u);
    ASSERT_EQ(ref->blocks(0).size(), 2u);
    ASSERT_EQ(ref->blocks(1).size(), 2u);

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2) << "Only blocks[1] and blocks[3] should remain referenced";

    ref.reset();
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, InsertIntoCacheInsertsOnlyFullBlocks) {
    auto config       = makeTinyHybridConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    // gid=0 linear, gid=1 full.
    const int gid_linear = 0;
    const int gid_full   = 1;

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                       CacheKeysType{100, 101, 102});
    // Disable device cache reuse.

    // Non-CP SharedBlockCache insertion records the available group slots for each cache key.
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

    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(100, gid_full)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(101, gid_full)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(102, gid_full)));

    // Linear group has NULL in early slots when reuse disabled, then materializes the tail slot.
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(100, gid_linear)));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(101, gid_linear)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(102, gid_linear)));
}

TEST_F(HybridTypeKVCacheAllocatorTest, DefaultHybridLinearPrefixReuseSupportsInsertThenReuse) {
    auto config = makeTinyHybridConfig();
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_TRUE(config.policyForGroup(/*gid=*/0).enable_prefix_reuse);

    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto seed_res    = makeBatchResource(/*batch_size=*/1,
                                      /*group_nums=*/2,
                                      /*layer_num=*/static_cast<int>(config.layer_all_num),
                                      /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                      CacheKeysType{100, 101, 102});
    auto seed_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.enable_device_cache = false;
    seed_malloc.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(seed_malloc).success);

    allocator->insertIntoCache(InsertInfo{seed_res, seed_tokens, /*is_resident=*/false});
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(102, /*group_id=*/0)));

    auto hit_res    = makeBatchResource(/*batch_size=*/1,
                                     /*group_nums=*/2,
                                     /*layer_num=*/static_cast<int>(config.layer_all_num),
                                     /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                     CacheKeysType{100, 101, 102, 103});
    auto hit_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo hit_malloc{hit_res, hit_tokens};
    hit_malloc.enable_device_cache = true;
    hit_malloc.reuse_cache         = true;
    auto result                    = allocator->malloc(hit_malloc);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 12);
}

TEST_F(HybridTypeKVCacheAllocatorTest, ConvertIndexToBufferAndAllLayerCacheBaseSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    KVCacheAllocator* base = allocator.get();
    auto              buf0 = base->convertIndexToBuffer(/*layer_id=*/0, /*block_id=*/1);
    ASSERT_FALSE(buf0.empty());
    EXPECT_NE(buf0[0].addr, nullptr);

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.groups().size(), static_cast<size_t>(config.groupNums()));
    ASSERT_EQ(layout.topology().layers().size(), static_cast<size_t>(config.layer_num));
    for (size_t i = 0; i < layout.topology().layers().size(); ++i) {
        for (const auto& tag : layout.topology().layer(static_cast<int>(i)).group_tags) {
            EXPECT_TRUE(layout.group(tag).hasLayer(i));
        }
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, IncrMallocRollbackFreesPartiallyAllocatedBlocks) {
    auto config      = makeTinyHybridConfig();
    config.block_num = 6;  // free=5
    auto allocator   = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                       CacheKeysType{100, 101, 102});
    // Disable device cache reuse (makes linear group allocate only tail for new slots).

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

// Prefill init path (StreamCacheResource::initKVBlock sets enable_remove_skipped_blocks=false).
// With step=2 and reuse_blocks_len=3, the reused linear tail lands at pos 2, which is NOT
// a step hit ((2+1)%2==1). Without sparse cleanup, that slot must survive so that
// causal_conv1d can still read it by prefix_length.
TEST_F(HybridTypeKVCacheAllocatorTest, PrefillInitSkipsSparseCleanupAndPreservesReusedLinearTail) {
    auto config       = makeTinyHybridConfig();
    config.block_num  = 16;  // 6 cached (resident, non-evictable) + 4 new + 1 null reserved
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const int gid_linear = 0;
    const int gid_full   = 1;

    CacheKeysType shared_keys = {100, 101, 102};
    auto cached_full_blocks   = allocateAndCache(block_pool, shared_cache, config.groupNums(), gid_full, shared_keys);
    auto cached_linear_blocks = allocateAndCache(block_pool, shared_cache, config.groupNums(), gid_linear, shared_keys);
    ASSERT_EQ(cached_linear_blocks.size(), 3u);

    // Request has 5 keys; allocator drops the last before matching, leaving {100,101,102,103}.
    // Full matches the first 3 (103 is absent); linear joint backoff stops at pos=2 => reuse_blocks_len=3.
    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                       CacheKeysType{100, 101, 102, 103, 104});

    // seq_len=20 => 5 slots. block_size-3-reserve_step = 2, so removeSkippedBlocks would scan pos 2.
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/20, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache          = true;
    info.reuse_cache                  = true;
    info.enable_remove_skipped_blocks = false;  // prefill init path
    auto result                       = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    const auto& linear_out = batch_res->blocks(0, gid_linear);
    ASSERT_EQ(linear_out.size(), 5u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_EQ(linear_out[2], cached_linear_blocks[2]) << "reused linear tail must survive prefill init";
    EXPECT_FALSE(isNullBlockIdx(linear_out[3]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[4]));
}

// Decode path (StreamCacheResource::incrKVBlock sets enable_remove_skipped_blocks=true).
// The allocator is invoked on an already-populated resource, so malloc() dispatches directly
// to incrMalloc(). Sparse cleanup must prune non-step blocks while preserving step hits and
// the configured active tail slot.
TEST_F(HybridTypeKVCacheAllocatorTest, DecodeIncrMallocAppliesSparseCleanupOnLinearGroups) {
    auto config      = makeTinyHybridConfig();
    config.block_num = 16;  // pre-allocates 6 + 6 = 12 blocks plus the reserved null block
    auto allocator   = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const int gid_linear = 0;
    const int gid_full   = 1;

    auto linear_alloc = block_pool->malloc(6);
    auto full_alloc   = block_pool->malloc(6);
    ASSERT_EQ(linear_alloc.size(), 6u);
    ASSERT_EQ(full_alloc.size(), 6u);

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                       CacheKeysType{});
    batch_res->mutableBlockIds(0, gid_linear).assign(linear_alloc);
    batch_res->mutableBlockIds(0, gid_full).assign(full_alloc);
    ASSERT_GT(batch_res->curBlocksNum(), 0);

    // seq_len=24 => 6 slots; current_blocks==6 so group malloc is a no-op and only cleanup runs.
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache          = false;
    info.reuse_cache                  = true;
    info.enable_remove_skipped_blocks = true;  // decode path
    auto result                       = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // active_tail_blocks=1 materializes the current tail, while decode cleanup retains at least two tails.
    // For step=2 and size=6: keep pos 1, 3 (step hits) and pos 4, 5 (decode tails).
    const auto& linear_out = batch_res->blocks(0, gid_linear);
    ASSERT_EQ(linear_out.size(), 6u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[2]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[3]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[4]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[5]));

    // Full group is untouched by sparse cleanup.
    const auto& full_out = batch_res->blocks(0, gid_full);
    ASSERT_EQ(full_out.size(), 6u);
    for (size_t i = 0; i < full_out.size(); ++i) {
        EXPECT_EQ(full_out[i], full_alloc[i]);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, EstimatePeakNeedBlocks) {
    // Config: [0,1]=linear group (gid=0), [2,3]=full group (gid=1). seq_size_per_block=4.
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int group_nums = 2;
    const int blk        = config.seq_size_per_block;  // 4

    // New resource (cur_slots=0 for both groups):
    // reuse disabled: full=ceil(108/4)=27, linear tail peak=3 => total=30.
    auto new_res = makeBatchResource(1, group_nums, config.layer_num, config.layerGroupIdsSnapshot(), {});
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/false), 30);

    // reuse enabled: linear keeps 14 blocks after cleanup and transiently holds a fifteenth tail block.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/true), 42);

    // With reserve_step=3: full=ceil(111/4)=28. linear: total_slots=29, tail=5,
    // step-hits before tail=24/2=12 => linear=17. total=45.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 3, /*enable_reuse_cache=*/true), 45);

    // Allocate blocks to simulate running decode (seqLen=8 → 2 slots per group)
    auto       token_ids = makeCompleteTokenIds(1, /*seq_length=*/8, config.seq_size_per_block);
    MallocInfo mi{new_res, token_ids};
    auto       result = allocator->malloc(mi);
    ASSERT_TRUE(result.success);

    const int full_slots   = new_res->blocksNum(0, 1);  // full group slots after malloc
    const int linear_slots = new_res->blocksNum(0, 0);  // linear group slots after malloc

    // remaining=0: no more slots needed for either group
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 0, 0, /*enable_reuse_cache=*/false), 0);

    // remaining=4: ceil((8+4)/4)=3 per group, minus cur_slots
    int expect_per_group = (8 + 4 + blk - 1) / blk;
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 4, 0, /*enable_reuse_cache=*/false),
              std::max(expect_per_group - full_slots, 0) + std::max(expect_per_group - linear_slots, 0));

    // Large remaining from current_slots=2:
    // reuse disabled: cleanup scans across the initial null slot. At the second boundary the running resource
    // transiently holds three physical linear blocks before releasing the oldest tail, two more than its current tail.
    int expect_full_large = (8 + 100 + blk - 1) / blk;  // 27
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/false),
              std::max(expect_full_large - full_slots, 0) + 2);

    // reuse enabled: target linear keeps tail 2 + step-hit slots before tail 12;
    // The fresh seq_len=8 allocation owns one physical linear block. Decode later peaks at 15 physical blocks.
    int expect_linear_large = 14;
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/true),
              std::max(expect_full_large - full_slots, 0) + expect_linear_large);
}

TEST_F(HybridTypeKVCacheAllocatorTest, EstimatePeakNeedBlocksUsesLinearActiveTailPolicy) {
    auto config   = makeTinyHybridConfig();
    auto policies = config.groupPoliciesSnapshot();
    ASSERT_EQ(policies.size(), 2u);
    ASSERT_EQ(policies[0].group_type, CacheGroupType::LINEAR);
    policies[0].active_tail_blocks = 4;
    config.setGroupPolicies(policies);

    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto resource = makeBatchResource(/*batch_size=*/1,
                                      config.groupNums(),
                                      config.layer_num,
                                      config.layerGroupIdsSnapshot(),
                                      /*keys=*/{});

    // At seq_len=24 the LINEAR group materializes four active tails and the FULL group owns six blocks.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(
                  *allocator, resource, /*seq_len=*/24, /*remaining_tokens=*/0, /*reserve_step=*/0, false),
              10);

    // One more block boundary adds a transient LINEAR tail and one permanent FULL block.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(
                  *allocator, resource, /*seq_len=*/24, /*remaining_tokens=*/4, /*reserve_step=*/0, false),
              12);
}

TEST_F(HybridTypeKVCacheAllocatorTest, EstimateBatchPeakNeedBlocksAccountsForNonEmptyTargetWidth) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto resource = makeBatchResource(/*batch_size=*/2,
                                      /*group_nums=*/2,
                                      /*layer_num=*/config.layer_num,
                                      /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                      /*keys=*/{});

    // common_seq_len=8 means the first two slots are shared. The NULL slot in the linear group consumes no block.
    resource->setBatchBlocks(/*batch_id=*/0, /*group_id=*/0, {NULL_BLOCK_IDX, 10, 11});
    resource->setBatchBlocks(/*batch_id=*/1, /*group_id=*/0, {NULL_BLOCK_IDX, 10, 12});
    resource->setBatchBlocks(/*batch_id=*/0, /*group_id=*/1, {20, 21, 22});
    resource->setBatchBlocks(/*batch_id=*/1, /*group_id=*/1, {20, 21, 23});

    // No growth is needed at the current batch width.
    EXPECT_EQ(allocator->estimateBatchPeakNeedBlocks(resource,
                                                     /*seq_len=*/12,
                                                     /*common_seq_len=*/8,
                                                     /*remaining_tokens=*/0,
                                                     /*reserve_step=*/0,
                                                     /*enable_reuse_cache=*/false,
                                                     /*target_batch_size=*/2),
              0);

    // No future growth is needed, regardless of the target width.
    EXPECT_EQ(allocator->estimateBatchPeakNeedBlocks(resource,
                                                     /*seq_len=*/12,
                                                     /*common_seq_len=*/8,
                                                     /*remaining_tokens=*/0,
                                                     /*reserve_step=*/0,
                                                     /*enable_reuse_cache=*/false,
                                                     /*target_batch_size=*/3),
              0);

    // Four more tokens add one block in each group for each current batch.
    EXPECT_EQ(allocator->estimateBatchPeakNeedBlocks(resource,
                                                     /*seq_len=*/12,
                                                     /*common_seq_len=*/8,
                                                     /*remaining_tokens=*/4,
                                                     /*reserve_step=*/0,
                                                     /*enable_reuse_cache=*/false,
                                                     /*target_batch_size=*/2),
              4);

    // One future block in each group is charged at the requested target width.
    EXPECT_EQ(allocator->estimateBatchPeakNeedBlocks(resource,
                                                     /*seq_len=*/12,
                                                     /*common_seq_len=*/8,
                                                     /*remaining_tokens=*/4,
                                                     /*reserve_step=*/0,
                                                     /*enable_reuse_cache=*/false,
                                                     /*target_batch_size=*/3),
              6);

    resource->setBatchBlocks(/*batch_id=*/0, /*group_id=*/0, {NULL_BLOCK_IDX, 10, 11, NULL_BLOCK_IDX});
    resource->setBatchBlocks(/*batch_id=*/1, /*group_id=*/0, {NULL_BLOCK_IDX, 10, 12, NULL_BLOCK_IDX});
    resource->setBatchBlocks(/*batch_id=*/0, /*group_id=*/1, {20, 21, 22, 24});
    resource->setBatchBlocks(/*batch_id=*/1, /*group_id=*/1, {20, 21, 23, 25});

    // Existing blocks already cover this unaligned sequence length.
    EXPECT_EQ(allocator->estimateBatchPeakNeedBlocks(resource,
                                                     /*seq_len=*/13,
                                                     /*common_seq_len=*/8,
                                                     /*remaining_tokens=*/0,
                                                     /*reserve_step=*/0,
                                                     /*enable_reuse_cache=*/false,
                                                     /*target_batch_size=*/2),
              0);
}

TEST_F(HybridTypeKVCacheAllocatorTest, FreshUnalignedMultiSequencePeakMatchesExactCapacity) {
    for (const bool reuse_cache : {false, true}) {
        SCOPED_TRACE(reuse_cache ? "reuse enabled" : "reuse disabled");

        auto config      = makeTinyHybridConfig();
        config.block_num = 7;  // Six usable blocks: exactly the two-stage initialization peak below.
        auto allocator   = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
        ASSERT_TRUE(allocator->init());

        auto resource = makeBatchResource(/*batch_size=*/2,
                                          /*group_nums=*/2,
                                          /*layer_num=*/config.layer_num,
                                          /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                          /*keys=*/{});

        // block_size=4, seq_len=5: initMallocForCommonLen shares one Linear and one Full block for the first four
        // tokens. incrMalloc then allocates one private tail in each group for each sequence: 2 + 2 * 2 = 6.
        EXPECT_EQ(allocator->estimateBatchPeakNeedBlocks(resource,
                                                         /*seq_len=*/5,
                                                         /*common_seq_len=*/4,
                                                         /*remaining_tokens=*/0,
                                                         /*reserve_step=*/0,
                                                         reuse_cache,
                                                         /*target_batch_size=*/2),
                  6);
        EXPECT_EQ(allocator->freeBlocksNum(), 6);

        // At the next block boundary both groups allocate one more private block per sequence. Linear cleanup only
        // happens after that allocation, so the lifecycle peak is ten blocks.
        EXPECT_EQ(allocator->estimateBatchPeakNeedBlocks(resource,
                                                         /*seq_len=*/5,
                                                         /*common_seq_len=*/4,
                                                         /*remaining_tokens=*/4,
                                                         /*reserve_step=*/0,
                                                         reuse_cache,
                                                         /*target_batch_size=*/2),
                  10);

        auto token_ids = makeCompleteTokenIds(
            /*batch_size=*/2, /*seq_length=*/5, /*seq_size_per_block=*/config.seq_size_per_block);
        MallocInfo info{resource, token_ids};
        info.enable_device_cache          = false;
        info.reuse_cache                  = reuse_cache;
        info.enable_remove_skipped_blocks = false;
        ASSERT_TRUE(allocator->malloc(info).success);
        EXPECT_EQ(allocator->freeBlocksNum(), 0);

        allocator->free(FreeInfo{resource, token_ids});
        EXPECT_EQ(allocator->freeBlocksNum(), 6);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, EstimatedPeakCoversDecodeMallocAndSparseCleanup) {
    auto config      = makeTinyHybridConfig();
    config.block_num = 28;  // 27 usable blocks.
    auto allocator   = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                       CacheKeysType{});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1,
                                          /*seq_length=*/8,
                                          /*seq_size_per_block=*/config.seq_size_per_block);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache          = false;
    info.reuse_cache                  = true;
    info.enable_remove_skipped_blocks = false;
    ASSERT_TRUE(allocator->malloc(info).success);
    ASSERT_EQ(allocator->freeBlocksNum(), 24);

    // From seq_len=8 to 68: full needs 15 more blocks; linear grows from one physical block to a transient peak of 10.
    ASSERT_EQ(estimateBatchPeakForSingleSequence(*allocator,
                                                 batch_res,
                                                 /*seq_len=*/8,
                                                 /*remaining_tokens=*/60,
                                                 /*reserve_step=*/0,
                                                 /*reuse_cache=*/true),
              24);

    info.enable_remove_skipped_blocks = true;
    size_t min_free_blocks            = allocator->freeBlocksNum();
    for (int seq_len = 9; seq_len <= 68; ++seq_len) {
        token_ids->setSeqLength(seq_len);
        ASSERT_TRUE(allocator->malloc(info).success) << "seq_len=" << seq_len;
        min_free_blocks = std::min(min_free_blocks, allocator->freeBlocksNum());
    }

    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, /*gid=*/0)), 9);
    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, /*gid=*/1)), 17);
    EXPECT_EQ(min_free_blocks, 1);
    EXPECT_EQ(allocator->freeBlocksNum(), 1);
}

TEST_F(HybridTypeKVCacheAllocatorTest, FreshReusePeakCoversThreeBoundaryDecodeAtExactCapacity) {
    auto config    = makeTinyHybridConfig();  // 9 usable blocks, seq_size_per_block=4, linear_step=2.
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_group_ids=*/config.layerGroupIdsSnapshot(),
                                       CacheKeysType{});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1,
                                          /*seq_length=*/8,
                                          /*seq_size_per_block=*/config.seq_size_per_block);

    // seq_len 8 -> 17 crosses the slot boundaries at 9, 13 and 17. Full peaks at 5 blocks and linear peaks at 4.
    ASSERT_EQ(allocator->freeBlocksNum(), 9);
    ASSERT_EQ(estimateBatchPeakForSingleSequence(*allocator,
                                                 batch_res,
                                                 /*seq_len=*/8,
                                                 /*remaining_tokens=*/9,
                                                 /*reserve_step=*/0,
                                                 /*reuse_cache=*/true),
              9);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache          = false;
    info.reuse_cache                  = true;
    info.enable_remove_skipped_blocks = false;
    ASSERT_TRUE(allocator->malloc(info).success);

    info.enable_remove_skipped_blocks = true;
    for (int seq_len = 9; seq_len <= 17; ++seq_len) {
        token_ids->setSeqLength(seq_len);
        ASSERT_TRUE(allocator->malloc(info).success) << "seq_len=" << seq_len;
    }

    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, /*gid=*/0)), 3);
    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, /*gid=*/1)), 5);
    EXPECT_EQ(allocator->freeBlocksNum(), 1);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
