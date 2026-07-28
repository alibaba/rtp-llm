#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
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
    cfg.hybrid_attention_config.hybrid_attention_types = types;
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
    cfg.hybrid_attention_config.hybrid_attention_types = types;
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

class HybridCacheConfigMTPTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

TEST_F(HybridCacheConfigMTPTest, CreateHybridConfigAllowsOnlyFullGroups) {
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

TEST_F(HybridCacheConfigMTPTest, CreateHybridConfigAllowsMultipleFullGroups) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescsWithTags(cfg, {HybridAttentionType::NONE, HybridAttentionType::NONE}, {"full", "full1"});

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    auto cache_config =
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);

    EXPECT_FALSE(cache_config.isStandardSingleTopology());
    ASSERT_EQ(cache_config.groupNums(), 2);
    EXPECT_EQ(cache_config.groupTagsSnapshot(), std::vector<std::string>({"full", "full1"}));
    EXPECT_EQ(cache_config.groupTypesSnapshot(),
              std::vector<CacheGroupType>({CacheGroupType::FULL, CacheGroupType::FULL}));
    EXPECT_EQ(cache_config.layerIdsForGroup(0), std::vector<int>({0}));
    EXPECT_EQ(cache_config.layerIdsForGroup(1), std::vector<int>({1}));
}

TEST_F(HybridCacheConfigMTPTest, CreateHybridConfigKeepsModelTokensPerBlock) {
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

TEST_F(HybridCacheConfigMTPTest, DecoupledKernelBlocksKeepPhysicalMhaStrideAndPreserveLinearGroupStride) {
    auto cfg                         = makeTinyModelConfig(/*num_layers=*/2);
    cfg.attn_config.tokens_per_block = 8;
    setHybridLayerDescs(cfg, {HybridAttentionType::NONE, HybridAttentionType::LINEAR});
    cfg.kv_cache_spec_descs[0][0].dtype                = DataType::TYPE_INT8;
    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    KVCacheConfig kv_cache_cfg;
    kv_cache_cfg.seq_size_per_block        = 8;
    kv_cache_cfg.kernel_seq_size_per_block = 2;

    const auto cache_config = HybridPoolConfigCreator::createConfig(cfg, parallelism_cfg, kv_cache_cfg, false, 0);
    ASSERT_EQ(cache_config.groupTagsSnapshot(), std::vector<std::string>({"full", "linear"}));

    const auto& mha_spec    = cache_config.specForGroup(0);
    const auto& linear_spec = cache_config.specForGroup(1);
    const auto  linear_bpk  = cache_config.kernelBlocksPerKvBlockForGroup(1);
    EXPECT_EQ(linear_bpk, 1);
    EXPECT_EQ(cache_config.kvBlockStrideBytesForGroup(0), mha_spec->block_size_bytes());
    EXPECT_EQ(cache_config.kvScaleStrideBytesForGroup(0), mha_spec->scale_block_size_bytes());
    EXPECT_EQ(cache_config.kvBlockStrideBytesForGroup(1), linear_spec->block_size_bytes() * linear_bpk);
    EXPECT_EQ(cache_config.kvScaleStrideBytesForGroup(1), linear_spec->scale_block_size_bytes() * linear_bpk);
    EXPECT_EQ(cache_config.block_size_bytes,
              mha_spec->block_size_bytes() + mha_spec->scale_block_size_bytes()
                  + linear_spec->block_size_bytes() * linear_bpk + linear_spec->scale_block_size_bytes() * linear_bpk);
}

TEST_F(HybridCacheConfigMTPTest, DecoupledKernelBlocksKeepPhysicalMlaStride) {
    auto cfg                         = makeTinyModelConfig(/*num_layers=*/1);
    cfg.attn_config.tokens_per_block = 8;
    cfg.attn_config.use_mla          = true;
    cfg.attn_config.kv_lora_rank     = 16;
    cfg.attn_config.rope_head_dim    = 4;
    cfg.kv_cache_spec_descs          = {{KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadLatentAttention}}};

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    KVCacheConfig kv_cache_cfg;
    kv_cache_cfg.seq_size_per_block        = 8;
    kv_cache_cfg.kernel_seq_size_per_block = 2;

    const auto cache_config = HybridPoolConfigCreator::createConfig(cfg, parallelism_cfg, kv_cache_cfg, false, 0);
    ASSERT_EQ(cache_config.groupNums(), 1);
    const auto& mla_spec = cache_config.specForGroup(0);
    EXPECT_EQ(cache_config.kvBlockStrideBytesForGroup(0), mla_spec->block_size_bytes());
    EXPECT_EQ(cache_config.kvScaleStrideBytesForGroup(0), mla_spec->scale_block_size_bytes());
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

TEST_F(HybridCacheConfigMTPTest, CreateHybridConfigAllowsOnlyLinearGroups) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescs(cfg, {HybridAttentionType::LINEAR, HybridAttentionType::LINEAR});
    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    auto cache_config =
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);

    EXPECT_FALSE(cache_config.isStandardSingleTopology());
    ASSERT_EQ(cache_config.groupNums(), 1);
    EXPECT_EQ(cache_config.groupTagsSnapshot(), std::vector<std::string>({"linear"}));
    EXPECT_EQ(cache_config.groupTypesSnapshot(), std::vector<CacheGroupType>({CacheGroupType::LINEAR}));
    EXPECT_EQ(cache_config.layerIdsForGroup(0), std::vector<int>({0, 1}));
}

TEST_F(HybridCacheConfigMTPTest, LinearDescriptorSelectsHybridPoolConfig) {
    auto cfg                = makeTinyModelConfig(/*num_layers=*/1);
    cfg.kv_cache_spec_descs = {{KVCacheSpecDesc{"linear", KVCacheSpecType::LinearAttention}}};
    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    const auto cache_config =
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
    EXPECT_FALSE(cache_config.isStandardSingleTopology());
    ASSERT_EQ(cache_config.groupNums(), 1);
    EXPECT_EQ(cache_config.groupTagsSnapshot(), std::vector<std::string>({"linear"}));
    EXPECT_EQ(cache_config.groupTypesSnapshot(), std::vector<CacheGroupType>({CacheGroupType::LINEAR}));
}

TEST_F(HybridCacheConfigMTPTest, InitAllowsOnlyLinearIndependentGroups) {
    auto cache_config = makeSimpleLinearCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto linear0 = makeLinearSpec("linear0", /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16, 1, 1);
    auto linear1 = makeLinearSpec("linear1", /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16, 1, 1);
    cache_config.fromGroupedSpecs(
        {linear0, linear1}, {{0}, {1}}, {CacheGroupType::LINEAR, CacheGroupType::LINEAR}, {"linear0", "linear1"});
    enableIndependentBlockPoolsForTest(cache_config);
    ASSERT_EQ(cache_config.groupNums(), 2);

    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(cache_config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->getBlockPool(), nullptr);
    EXPECT_EQ(allocator->groupBlockPools().size(), 2u);
}

TEST_F(HybridCacheConfigMTPTest, TopologyRejectsSpecPolicyTypeMismatch) {
    auto config = makeSimpleLinearCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto groups      = config.topology().groups();
    auto layers      = config.topology().layers();
    groups[0].policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(HybridCacheConfigMTPTest, TopologyRejectsGroupLayerMissingForwardGid) {
    auto config = makeTinyHybridConfig();
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    groups[0].layer_ids.push_back(2);

    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(HybridCacheConfigMTPTest, TopologyRejectsMissingLayerTagMapping) {
    auto config = makeTinyHybridConfig();
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    layers[0].group_tags.clear();

    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(HybridCacheConfigMTPTest, CreateHybridConfigAggregatesSemanticTagsInFirstSeenOrder) {
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
                                {"linear", "linear", "linear", "full", "linear", "linear", "linear", "full"});
    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    auto cache_config =
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);

    ASSERT_EQ(cache_config.groupNums(), 2);
    EXPECT_EQ(cache_config.groupTagsSnapshot(), std::vector<std::string>({"linear", "full"}));
    EXPECT_EQ(cache_config.groupTypesSnapshot(),
              std::vector<CacheGroupType>({CacheGroupType::LINEAR, CacheGroupType::FULL}));
    EXPECT_EQ(cache_config.layerIdsForGroup(0), std::vector<int>({0, 1, 2, 4, 5, 6}));
    EXPECT_EQ(cache_config.layerIdsForGroup(1), std::vector<int>({3, 7}));
}

TEST_F(HybridCacheConfigMTPTest, InitAndAddressLookupSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->seqSizePerBlock(), 4);
    EXPECT_EQ(allocator->totalBlocksNum(), static_cast<size_t>(config.groupNums()) * (config.block_num - 1));
    EXPECT_EQ(allocator->freeBlocksNum(), static_cast<size_t>(config.groupNums()) * (config.block_num - 1));

    // Should be able to fetch address for any global layer and non-zero block id.
    auto addr0 = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    auto addr3 = allocator->convertIndexToAddr(/*layer_id=*/3, /*block_id=*/1);
    EXPECT_NE(addr0.kv_addr, nullptr);
    EXPECT_NE(addr3.kv_addr, nullptr);
}

TEST_F(HybridCacheConfigMTPTest, ConvertToGlobalLayerIdHybridNoMtp) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);

    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/0), 0u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/3), 3u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/4),
              std::numeric_limits<uint32_t>::max());

    // no mtp sub-model
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(HybridCacheConfigMTPTest, ConvertToGlobalLayerIdHybridWithMtpSubConfigs) {
    auto config    = makeTinyHybridMtpConfigByCreateSpConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);

    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (size_t mtp_id = 0; mtp_id < config.mtp_sub_configs.size(); ++mtp_id) {
        const auto& sub = config.mtp_sub_configs[mtp_id];
        ASSERT_NE(sub, nullptr);
        ASSERT_EQ(sub->groupNums(), 2);
        std::vector<std::string> expected_tags{"linear", "full"};
        EXPECT_EQ(sub->groupTagsSnapshot(), expected_tags);
        EXPECT_TRUE(sub->layerIdsForGroup(0).empty());
        ASSERT_EQ(sub->layerIdsForGroup(1).size(), 1u);
        EXPECT_EQ(sub->layerIdsForGroup(1)[0], 0);
    }

    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/2), 2u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0), 4u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/0), 5u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/1),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/3, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(HybridCacheConfigMTPTest, MergeMtpAcceptsUnequalTargetGroupWidths) {
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
    auto module0 = main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/5);

    ASSERT_NE(module0, nullptr);
    EXPECT_EQ(main_config.layerIdsForGroup(0), std::vector<int>({0, 1, 2}));
    EXPECT_EQ(main_config.layerIdsForGroup(1), std::vector<int>({3, 4, 5}));
    EXPECT_TRUE(module0->layerIdsForGroup(0).empty());
    EXPECT_EQ(module0->layerIdsForGroup(1), std::vector<int>({0}));
}

TEST_F(HybridCacheConfigMTPTest, MergeMtpUsesPerTagMainAndModuleLayerCounts) {
    CacheConfig main_config;
    main_config.layer_num       = 5;
    main_config.layer_all_num   = 5;
    main_config.group_layer_num = 3;
    main_config.fromGroupedSpecs(
        {makeMhaSpec("full", 4, DataType::TYPE_FP16, 1, 1), makeLinearSpec("linear", 4, DataType::TYPE_FP16, 1, 1)},
        {{0, 2}, {1, 3, 4}},
        {CacheGroupType::FULL, CacheGroupType::LINEAR},
        {"full", "linear"});
    main_config.layer_to_block_stride_bytes.assign(9, 1);

    CacheConfig propose_config;
    propose_config.layer_num     = 2;
    propose_config.layer_all_num = 2;
    propose_config.fromGroupedSpecs(
        {makeMhaSpec("full", 4, DataType::TYPE_FP16, 1, 1), makeLinearSpec("linear", 4, DataType::TYPE_FP16, 1, 1)},
        {{0}, {1}},
        {CacheGroupType::FULL, CacheGroupType::LINEAR},
        {"full", "linear"});
    propose_config.layer_to_block_stride_bytes.assign(2, 1);

    auto module0 = main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/5);
    ASSERT_NE(module0, nullptr);
    EXPECT_EQ(main_config.layerIdsForGroup(0), std::vector<int>({0, 2, 5}));
    EXPECT_EQ(main_config.layerIdsForGroup(1), std::vector<int>({1, 3, 4, 6}));
    EXPECT_EQ(module0->layerIdsForGroup(0), std::vector<int>({0}));
    EXPECT_EQ(module0->layerIdsForGroup(1), std::vector<int>({1}));

    auto module1 = main_config.mergeMTPModule(propose_config, /*module_index=*/1, /*main_layer_num=*/5);
    ASSERT_NE(module1, nullptr);
    EXPECT_EQ(main_config.layerIdsForGroup(0), std::vector<int>({0, 2, 5, 7}));
    EXPECT_EQ(main_config.layerIdsForGroup(1), std::vector<int>({1, 3, 4, 6, 8}));
    EXPECT_EQ(module1->layerIdsForGroup(0), std::vector<int>({0}));
    EXPECT_EQ(module1->layerIdsForGroup(1), std::vector<int>({1}));
}

TEST_F(HybridCacheConfigMTPTest, MergeMtpRejectsPartialOrReorderedSourceGroup) {
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

TEST_F(HybridCacheConfigMTPTest, MtpPhysicalSlotsDoNotAliasMainSlots) {
    auto config    = makeTinyHybridMtpConfigByCreateSpConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
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

TEST_F(HybridCacheConfigMTPTest, MtpLayoutProjectionRecountsActiveLayersAndKeepsEmptyPlaceholder) {
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

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
