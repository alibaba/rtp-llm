#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

constexpr std::string_view kLinearTag = "linear";
constexpr std::string_view kFullTag   = "full1";

static CacheConfig makeTinyHybridConfig() {
    auto config = makeSimpleHybridMhaCacheConfig(/*layer_num=*/4,
                                                 /*block_num=*/10,
                                                 /*tokens_per_block=*/4,
                                                 rtp_llm::DataType::TYPE_FP16,
                                                 /*group_layer_num=*/2,
                                                 /*local_head_num_kv=*/1,
                                                 /*size_per_head=*/1);
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    for (auto& group : groups) {
        if (group.policy.group_type == CacheGroupType::FULL) {
            group.kernel_seq_size_per_block = 2;
        }
    }
    config.setTopology(std::move(groups), std::move(layers));
    return config;
}

static BlockPoolPtr poolFor(const KVCacheAllocatorPtr& allocator, std::string_view tag) {
    return allocator->blockPool(std::string(tag));
}

static void setGroupBlockNum(CacheConfig& config, uint32_t block_num) {
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    for (auto& group : groups) {
        group.block_num = block_num;
    }
    config.setTopology(std::move(groups), std::move(layers));
}

static void setGroupBlockNums(CacheConfig& config, uint32_t linear_block_num, uint32_t full_block_num) {
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    for (auto& group : groups) {
        group.block_num = group.tag == kLinearTag ? linear_block_num : full_block_num;
    }
    config.setTopology(std::move(groups), std::move(layers));
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

static CacheConfig makeTinyHybridMtpConfigByCreateSpConfig(SpeculativeType    sp_type     = SP_TYPE_MTP,
                                                           const std::string& propose_tag = "full",
                                                           int64_t            gen_num     = 2) {
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
    propose_model_cfg.kv_cache_spec_descs[0][0].tag                = propose_tag;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;

    RuntimeConfig runtime_cfg;
    KVCacheConfig kv_cache_cfg;
    kv_cache_cfg.test_block_num = 8;

    SpeculativeExecutionConfig sp_cfg;
    sp_cfg.type              = sp_type;
    sp_cfg.gen_num_per_cycle = gen_num;

    return CacheConfigCreator::createSpConfig(score_model_cfg,
                                              propose_model_cfg,
                                              parallelism_cfg,
                                              runtime_cfg,
                                              kv_cache_cfg,
                                              sp_cfg,
                                              /*warm_up_result=*/std::nullopt,
                                              /*is_mtp=*/true,
                                              /*is_eagle=*/sp_type == SP_TYPE_EAGLE);
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

static BatchKVCacheResourcePtr makeBatchResource(int batch_size, const CacheConfig& config, CacheKeysType keys) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(config.topologyPtr());
    for (int b = 0; b < batch_size; ++b) {
        for (const auto& group : config.topology().groups()) {
            if (group.policy.enable_prefix_reuse) {
                res->setBatchCacheKeys(b, group.tag, keys);
            }
        }
    }
    return res;
}

static void rebuildRequestPrefixes(const BatchKVCacheResourcePtr& resource, const CompleteTokenIdsPtr& token_ids) {
    for (int batch_id = 0; batch_id < token_ids->batchSize(); ++batch_id) {
        resource->cacheResource(batch_id).requestPrefix().rebuild(token_ids->data(batch_id), token_ids->seqLength());
    }
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
                                                  std::string_view     tag,
                                                  const CacheKeysType& keys,
                                                  bool                 is_resident = true) {
    auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
    EXPECT_EQ(blocks.size(), keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        shared_cache->put(
            keys[i], {SharedBlockCache::UnifiedCacheItem::GroupBlock{std::string(tag), blocks[i]}}, is_resident);
    }

    block_pool->requestFree(blocks);
    return blocks;
}

static std::vector<BlockIdxType> allocateAndCacheKeepAllocated(BlockPoolPtr         block_pool,
                                                               SharedBlockCachePtr  shared_cache,
                                                               std::string_view     tag,
                                                               const CacheKeysType& keys,
                                                               bool                 is_resident = true) {
    auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
    EXPECT_EQ(blocks.size(), keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        shared_cache->put(
            keys[i], {SharedBlockCache::UnifiedCacheItem::GroupBlock{std::string(tag), blocks[i]}}, is_resident);
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

class KVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

TEST_F(KVCacheAllocatorTest, CreateHybridConfigAllowsOnlyFullGroups) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescs(cfg, {HybridAttentionType::NONE, HybridAttentionType::NONE});

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    auto cache_config =
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
    ASSERT_EQ(cache_config.groupNums(), 1);
    EXPECT_EQ(cache_config.typeForGroup("full"), CacheGroupType::FULL);
}

TEST_F(KVCacheAllocatorTest, CreateHybridConfigRejectsMultipleFullGroups) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescsWithTags(cfg, {HybridAttentionType::NONE, HybridAttentionType::NONE}, {"full", "full1"});

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    try {
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
        FAIL() << "expected multiple full groups to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("multiple FULL MHA/MLA cache groups"), std::string::npos);
    }
}

TEST_F(KVCacheAllocatorTest, CreateHybridConfigRejectsAttentionTypeSpecMismatch) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescs(cfg, {HybridAttentionType::LINEAR, HybridAttentionType::NONE});
    cfg.kv_cache_spec_descs[0] = {KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention}};

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    EXPECT_THROW(CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0),
                 std::runtime_error);
}

TEST_F(KVCacheAllocatorTest, CreateHybridConfigKeepsModelTokensPerBlock) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescs(cfg, {HybridAttentionType::NONE, HybridAttentionType::NONE});

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;

    auto cache_config =
        CacheConfigCreator::createBasicConfig(cfg, parallelism_cfg, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
    ASSERT_EQ(cache_config.groupNums(), 1);
    for (const auto& group : cache_config.topology().groups()) {
        EXPECT_EQ(cache_config.specForGroup(group.tag)->seq_size_per_block, 4);
    }
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

TEST_F(KVCacheAllocatorTest, CreateHybridConfigRejectsOnlyLinearGroups) {
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

TEST_F(KVCacheAllocatorTest, CreateSingleConfigRejectsLinearDescriptor) {
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

TEST_F(KVCacheAllocatorTest, IndependentPoolsSupportOnlyLinearGroups) {
    auto cache_config = makeSimpleLinearCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto linear0 = makeLinearSpec("linear0", /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16, 1, 1);
    auto linear1 = makeLinearSpec("linear1", /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16, 1, 1);
    setTestTopology(cache_config,
                    {makeTestGroupForConfig(linear0, {0}, CacheGroupType::LINEAR, "linear0"),
                     makeTestGroupForConfig(linear1, {1}, CacheGroupType::LINEAR, "linear1")});
    applyUniformTestBlockCount(cache_config, 4);
    ASSERT_EQ(cache_config.groupNums(), 2);

    auto allocator = std::make_shared<KVCacheAllocator>(cache_config, AllocationType::DEVICE);
    EXPECT_TRUE(allocator->init());
    EXPECT_THROW(allocator->blockPool("default"), std::out_of_range);
    EXPECT_NE(allocator->blockPool("linear0"), nullptr);
    EXPECT_NE(allocator->blockPool("linear1"), nullptr);
    EXPECT_EQ(allocator->poolCount(), 2u);
}

TEST_F(KVCacheAllocatorTest, TopologyRejectsSpecPolicyTypeMismatch) {
    auto config = makeSimpleLinearCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto groups      = config.topology().groups();
    auto layers      = config.topology().layers();
    groups[0].policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(KVCacheAllocatorTest, TopologyRejectsGroupLayerMissingForwardGid) {
    auto config = makeTinyHybridConfig();
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    groups[0].layer_ids.push_back(2);

    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(KVCacheAllocatorTest, TopologyRejectsMissingLayerTagMapping) {
    auto config = makeTinyHybridConfig();
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    layers[0].group_tags.clear();

    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(KVCacheAllocatorTest, CreateHybridConfigAggregatesLinearLayersIntoOneGroup) {
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

    std::vector<int> expected_full{3, 7};
    std::vector<int> expected_linear{0, 1, 2, 4, 5, 6};

    ASSERT_EQ(cache_config.groupNums(), 2);
    EXPECT_EQ(cache_config.typeForGroup("full"), CacheGroupType::FULL);
    EXPECT_EQ(cache_config.typeForGroup("linear"), CacheGroupType::LINEAR);
    EXPECT_EQ(cache_config.layerIdsForGroup("full"), expected_full);
    EXPECT_EQ(cache_config.layerIdsForGroup("linear"), expected_linear);
}

TEST_F(KVCacheAllocatorTest, InitAndAddressLookupSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->seqSizePerBlock(), 4);
    size_t expected_blocks = 0;
    for (const auto& group : config.topology().groups()) {
        expected_blocks += group.block_num - 1;
    }
    EXPECT_EQ(allocator->totalBlocksNum(), expected_blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), expected_blocks);

    // Should be able to fetch address for any global layer and non-zero block id.
    auto addr0 = allocator->convertIndexToAddr(/*layer_id=*/0, std::string(kLinearTag), /*block_id=*/1);
    auto addr3 = allocator->convertIndexToAddr(/*layer_id=*/3, std::string(kFullTag), /*block_id=*/1);
    EXPECT_NE(addr0.kv_addr, nullptr);
    EXPECT_NE(addr3.kv_addr, nullptr);
}

TEST_F(KVCacheAllocatorTest, ConvertToGlobalLayerIdHybridNoMtp) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);

    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/0), 0u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/3), 3u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/4),
              std::numeric_limits<uint32_t>::max());

    // no mtp sub-model
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(KVCacheAllocatorTest, ConvertToGlobalLayerIdHybridWithMtpSubConfigs) {
    auto config    = makeTinyHybridMtpConfigByCreateSpConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);

    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (size_t mtp_id = 0; mtp_id < config.mtp_sub_configs.size(); ++mtp_id) {
        const auto& sub = config.mtp_sub_configs[mtp_id];
        ASSERT_NE(sub, nullptr);
        ASSERT_EQ(sub->groupNums(), 2);
        ASSERT_EQ(sub->layerIdsForGroup("full").size(), 1u);
        EXPECT_EQ(sub->layerIdsForGroup("full")[0], 0);
        EXPECT_TRUE(sub->layerIdsForGroup("linear").empty());
    }

    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/2), 2u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0), 4u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/0), 5u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/1),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/3, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(KVCacheAllocatorTest, EagleMapsSoleDefaultFullDraftGroupToUniqueFullTargetGroup) {
    auto config = makeTinyHybridMtpConfigByCreateSpConfig(SP_TYPE_EAGLE, "default");

    ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
    const auto& sub_config = config.mtp_sub_configs[0];
    ASSERT_NE(sub_config, nullptr);
    ASSERT_EQ(sub_config->topology().groups().size(), config.topology().groups().size());
    for (size_t group_index = 0; group_index < config.topology().groups().size(); ++group_index) {
        EXPECT_EQ(sub_config->topology().groups()[group_index].tag, config.topology().groups()[group_index].tag);
    }

    EXPECT_EQ(sub_config->groupForLayer(0, "full").tag, "full");
    EXPECT_EQ(sub_config->layerIdsForGroup("full"), std::vector<int>({0}));
    EXPECT_EQ(sub_config->specForGroup("full")->tag, "full");
    EXPECT_EQ(sub_config->specForGroup("full")->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_TRUE(sub_config->layerIdsForGroup("linear").empty());

    auto manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());
    const auto layout = manager->getMTPModuleGroupedCacheLayerLayout(0);
    EXPECT_TRUE(layout.at("full", 0).kv_addr.defined());
    EXPECT_TRUE(layout.group("linear").empty());
}

TEST_F(KVCacheAllocatorTest, MtpMapsDefaultFullDraftGroupForEveryModule) {
    auto config = makeTinyHybridMtpConfigByCreateSpConfig(SP_TYPE_MTP, "default", /*gen_num=*/2);

    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    EXPECT_EQ(config.layerIdsForGroup("full"), std::vector<int>({2, 3, 4, 5}));
    for (size_t module_index = 0; module_index < config.mtp_sub_configs.size(); ++module_index) {
        const auto& sub_config = config.mtp_sub_configs[module_index];
        ASSERT_NE(sub_config, nullptr);
        ASSERT_EQ(sub_config->topology().groups().size(), config.topology().groups().size());
        EXPECT_EQ(sub_config->layerIdsForGroup("full"), std::vector<int>({0}));
        EXPECT_TRUE(sub_config->layerIdsForGroup("linear").empty());
        EXPECT_EQ(config.groupForLayer(static_cast<int>(4 + module_index), "full").tag, "full");
    }
}

TEST_F(KVCacheAllocatorTest, MergeMtpAliasesCompatibleDefaultMlaGroup) {
    auto main_config    = makeSingleLayerCacheConfig(makeResolvedMlaSpec(DataType::TYPE_FP16,
                                                                      /*kv_lora_rank=*/1,
                                                                      /*rope_head_dim=*/1,
                                                                      /*seq_size_per_block=*/4,
                                                                      "full"),
                                                  CacheGroupType::FULL);
    auto propose_config = makeSingleLayerCacheConfig(makeResolvedMlaSpec(DataType::TYPE_FP16,
                                                                         /*kv_lora_rank=*/1,
                                                                         /*rope_head_dim=*/1,
                                                                         /*seq_size_per_block=*/4,
                                                                         "default"),
                                                     CacheGroupType::FULL);

    const auto sub_config = main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/1);
    ASSERT_NE(sub_config, nullptr);
    ASSERT_EQ(sub_config->topology().groups().size(), 1u);
    EXPECT_EQ(sub_config->topology().groups()[0].tag, "full");
    EXPECT_EQ(sub_config->specForGroup("full")->type, KVCacheSpecType::MultiHeadLatentAttention);
    EXPECT_EQ(sub_config->specForGroup("full")->tag, "full");
    EXPECT_EQ(sub_config->layerIdsForGroup("full"), std::vector<int>({0}));
    EXPECT_EQ(main_config.layerIdsForGroup("full"), std::vector<int>({0, 1}));
}

TEST_F(KVCacheAllocatorTest, MergeMtpRejectsAmbiguousDefaultFullGroupAlias) {
    CacheConfig main_config;
    main_config.layer_num     = 2;
    main_config.layer_all_num = 2;
    setTestTopology(
        main_config,
        {makeTestGroupForConfig(makeMhaSpec("full0", 4, DataType::TYPE_FP16, 1, 1), {0}, CacheGroupType::FULL, "full0"),
         makeTestGroupForConfig(
             makeMhaSpec("full1", 4, DataType::TYPE_FP16, 1, 1), {1}, CacheGroupType::FULL, "full1")});
    auto propose_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);

    try {
        main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/2);
        FAIL() << "expected an ambiguous default FULL group mapping to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ambiguous default FULL group mapping"), std::string::npos);
    }
}

TEST_F(KVCacheAllocatorTest, MergeMtpDoesNotAliasDefaultFullGroupToLinearTarget) {
    auto main_config = makeSimpleLinearCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto propose_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);

    try {
        main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/1);
        FAIL() << "expected a default FULL group without a compatible target to be rejected";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("no compatible target group for sole propose tag=default"), std::string::npos);
        EXPECT_NE(message.find("tag=linear"), std::string::npos);
    }
}

TEST_F(KVCacheAllocatorTest, MergeMtpRejectsIncompatibleDefaultFullGroupAlias) {
    const auto expect_no_compatible_alias = [](CacheConfig target_config, const CacheConfig& propose_config) {
        try {
            target_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/1);
            FAIL() << "expected an incompatible default FULL group alias to be rejected";
        } catch (const std::runtime_error& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find("no compatible target group for sole propose tag=default"), std::string::npos);
            EXPECT_NE(message.find("target_groups=[{tag=full"), std::string::npos);
        }
    };

    auto target = makeSingleLayerCacheConfig(
        makeMhaSpec("full", /*tokens_per_block=*/4, DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/1),
        CacheGroupType::FULL);
    auto compatible_propose = makeSingleLayerCacheConfig(makeMhaSpec("default",
                                                                     /*tokens_per_block=*/4,
                                                                     DataType::TYPE_FP16,
                                                                     /*local_head_num_kv=*/1,
                                                                     /*size_per_head=*/1),
                                                         CacheGroupType::FULL);
    auto different_tokens   = makeSingleLayerCacheConfig(makeMhaSpec("default",
                                                                   /*tokens_per_block=*/8,
                                                                   DataType::TYPE_FP16,
                                                                   /*local_head_num_kv=*/1,
                                                                   /*size_per_head=*/1),
                                                       CacheGroupType::FULL);
    expect_no_compatible_alias(target, different_tokens);

    auto different_geometry = makeSingleLayerCacheConfig(makeMhaSpec("default",
                                                                     /*tokens_per_block=*/4,
                                                                     DataType::TYPE_FP16,
                                                                     /*local_head_num_kv=*/2,
                                                                     /*size_per_head=*/1),
                                                         CacheGroupType::FULL);
    expect_no_compatible_alias(target, different_geometry);

    auto mla_target = makeSingleLayerCacheConfig(makeResolvedMlaSpec(DataType::TYPE_FP16,
                                                                     /*kv_lora_rank=*/1,
                                                                     /*rope_head_dim=*/1,
                                                                     /*seq_size_per_block=*/4,
                                                                     "full"),
                                                 CacheGroupType::FULL);
    expect_no_compatible_alias(mla_target, compatible_propose);

    auto different_group_stride  = compatible_propose;
    auto different_stride_groups = different_group_stride.topology().groups();
    different_stride_groups[0].kv_block_stride_bytes += 1;
    different_group_stride.setTopology(std::move(different_stride_groups), different_group_stride.topology().layers());
    expect_no_compatible_alias(target, different_group_stride);

    auto target_with_different_policy    = target;
    auto target_policy                   = target_with_different_policy.topology().group("full").policy;
    target_policy.fixed_block_num        = 2;
    target_policy.charge_to_paged_budget = true;
    target_with_different_policy.setGroupPolicies({{"full", target_policy}});
    expect_no_compatible_alias(target_with_different_policy, compatible_propose);
}

TEST_F(KVCacheAllocatorTest, MergeMtpPrefersExactDefaultGroupMatch) {
    CacheConfig main_config;
    main_config.layer_num     = 2;
    main_config.layer_all_num = 2;
    setTestTopology(
        main_config,
        {makeTestGroupForConfig(
             makeMhaSpec("default", 4, DataType::TYPE_FP16, 1, 1), {0}, CacheGroupType::FULL, "default"),
         makeTestGroupForConfig(makeMhaSpec("aux", 4, DataType::TYPE_FP16, 1, 1), {1}, CacheGroupType::FULL, "aux")});
    auto propose_config = makeSingleLayerCacheConfig(makeMhaSpec("default",
                                                                 /*tokens_per_block=*/4,
                                                                 DataType::TYPE_FP16,
                                                                 /*local_head_num_kv=*/1,
                                                                 /*size_per_head=*/1),
                                                     CacheGroupType::FULL);

    const auto sub_config = main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/2);
    ASSERT_NE(sub_config, nullptr);
    ASSERT_EQ(sub_config->topology().groups().size(), 2u);
    EXPECT_EQ(sub_config->topology().groups()[0].tag, "default");
    EXPECT_EQ(sub_config->topology().groups()[1].tag, "aux");
    EXPECT_EQ(sub_config->layerIdsForGroup("default"), std::vector<int>({0}));
    EXPECT_TRUE(sub_config->layerIdsForGroup("aux").empty());
    EXPECT_EQ(main_config.layerIdsForGroup("default"), std::vector<int>({0, 2}));
    EXPECT_EQ(main_config.layerIdsForGroup("aux"), std::vector<int>({1}));
}

TEST_F(KVCacheAllocatorTest, MergeMtpDoesNotAliasDefaultLinearProposeGroup) {
    auto main_config = makeSingleLayerCacheConfig(
        makeMhaSpec("full", /*tokens_per_block=*/4, DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/1),
        CacheGroupType::FULL);
    auto propose_config = makeSingleLayerCacheConfig(
        makeLinearSpec(
            "default", /*tokens_per_block=*/4, DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/1),
        CacheGroupType::LINEAR);

    try {
        main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/1);
        FAIL() << "expected a default Linear propose group not to use the FULL alias";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("missing group mapping for sub layer 0"), std::string::npos);
    }
}

TEST_F(KVCacheAllocatorTest, MergeMtpAliasErrorIdentifiesSourceAndTargetTags) {
    auto main_config = makeSingleGroupCacheConfig(
        makeMhaSpec("full", /*tokens_per_block=*/4, DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/1),
        CacheGroupType::FULL,
        /*layer_num=*/2,
        /*block_num=*/4);
    auto propose_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, DataType::TYPE_FP16);
    auto propose_groups         = propose_config.topology().groups();
    auto propose_layers         = propose_config.topology().layers();
    propose_groups[0].layer_ids = {1, 0};
    propose_config.setTopology(std::move(propose_groups), std::move(propose_layers));

    try {
        main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/2);
        FAIL() << "expected reordered aliased source layers to be rejected";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("source_tag=default"), std::string::npos);
        EXPECT_NE(message.find("target_tag=full"), std::string::npos);
    }
}

TEST_F(KVCacheAllocatorTest, MergeMtpDoesNotAliasMultiGroupProposeConfig) {
    auto main_config = makeSingleLayerCacheConfig(
        makeMhaSpec("full", /*tokens_per_block=*/4, DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/1),
        CacheGroupType::FULL);

    CacheConfig propose_config;
    propose_config.layer_num     = 1;
    propose_config.layer_all_num = 1;
    setTestTopology(
        propose_config,
        {makeTestGroupForConfig(
             makeMhaSpec("default", 4, DataType::TYPE_FP16, 1, 1), {0}, CacheGroupType::FULL, "default"),
         makeTestGroupForConfig(makeMhaSpec("aux", 4, DataType::TYPE_FP16, 1, 1), {0}, CacheGroupType::FULL, "aux")});
    try {
        main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/1);
        FAIL() << "expected a multi-group propose config not to use the default alias";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("missing group mapping for sub layer 0"), std::string::npos);
    }
}

TEST_F(KVCacheAllocatorTest, MergeMtpUsesGroupLocalMainLayerCount) {
    CacheConfig main_config;
    main_config.layer_num     = 5;
    main_config.layer_all_num = 5;
    setTestTopology(
        main_config,
        {makeTestGroupForConfig(
             makeMhaSpec("full", 4, DataType::TYPE_FP16, 1, 1), {0, 1, 2}, CacheGroupType::FULL, "full"),
         makeTestGroupForConfig(
             makeLinearSpec("linear", 4, DataType::TYPE_FP16, 1, 1), {3, 4}, CacheGroupType::LINEAR, "linear")});

    auto propose_config = makeSimpleLinearCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto sub_config = main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/5);
    ASSERT_NE(sub_config, nullptr);
    EXPECT_EQ(main_config.layerIdsForGroup("linear"), (std::vector<int>{3, 4, 5}));
}

TEST_F(KVCacheAllocatorTest, MergeMtpRejectsPartialOrReorderedSourceGroup) {
    auto main_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);

    CacheConfig partial_source;
    partial_source.layer_num     = 2;
    partial_source.layer_all_num = 2;
    setTestTopology(
        partial_source,
        {makeTestGroupForConfig(
             makeMhaSpec("default", 4, DataType::TYPE_FP16, 1, 1), {0}, CacheGroupType::FULL, "default"),
         makeTestGroupForConfig(makeMhaSpec("aux", 4, DataType::TYPE_FP16, 1, 1), {1}, CacheGroupType::FULL, "aux")});
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

TEST_F(KVCacheAllocatorTest, MtpPhysicalSlotsDoNotAliasMainSlots) {
    auto config    = makeTinyHybridMtpConfigByCreateSpConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const auto address_for_layer = [&](int layer_id) {
        BlockAddrInfo result;
        const auto    groups = config.groupsForLayer(layer_id);
        EXPECT_EQ(groups.size(), 1u);
        for (const auto& group_ref : groups) {
            result = allocator->convertIndexToAddr(layer_id, group_ref.get().tag, /*block_id=*/1);
        }
        return result;
    };
    const auto main0 = address_for_layer(2);
    const auto main1 = address_for_layer(3);
    const auto mtp0  = address_for_layer(4);
    const auto mtp1  = address_for_layer(5);
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

TEST_F(KVCacheAllocatorTest, MtpLayoutProjectionRecountsActiveLayersAndKeepsEmptyPlaceholder) {
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

TEST_F(KVCacheAllocatorTest, GetNeedBlocksUsesGroupGetNeedBlocksAndReuseFlag) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // batch=2, seq_len=12 (3 slots), reserve_step=2
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/12, /*seq_size_per_block=*/4);
    token_ids->setReserveStep(2);

    // Reuse disabled: linear group keeps only tail for common blocks; reserve_step contributes extra blocks.
    // full group contributes common=3, extra=1.
    {
        auto       batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100, 101, 102, 103});
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
        auto       batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100, 101, 102, 103});
        MallocInfo info{batch_res, token_ids};
        info.enable_device_cache = true;
        info.reuse_cache         = true;
        // full: common=3 extra=1
        // linear: common=count(0,3]=2, extra=reserve_step-1(=1)
        // common_total = 3 + 2 = 5
        // extra_total  = 1 + 1 = 2
        // total = 5 + 2*2 = 9
        EXPECT_EQ(allocator->getNeedBlocks(info), 10);
    }
}

TEST_F(KVCacheAllocatorTest, JointReuseUsesFullPrefixAndLinearTailOnly) {
    auto config       = makeTinyHybridConfig();
    auto allocator    = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto full_pool   = poolFor(allocator, kFullTag);
    auto linear_pool = poolFor(allocator, kLinearTag);

    // Full group has prefix matches for {100,101,102}.
    CacheKeysType full_keys   = {100, 101, 102};
    auto          full_blocks = allocateAndCache(full_pool, shared_cache, kFullTag, full_keys);

    // Linear group only matches key 101 (so joint match should backoff to pos=1 => reuse_blocks_len=2).
    CacheKeysType linear_keys   = {101};
    auto          linear_blocks = allocateAndCache(linear_pool, shared_cache, kLinearTag, linear_keys);
    ASSERT_EQ(linear_blocks.size(), 1u);

    // Request carries a trailing key beyond the three physical slots.
    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    // Enable device cache reuse for joint match.

    // seq_len=12 => 3 slots (4 tokens per block).
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    rebuildRequestPrefixes(batch_res, token_ids);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Full group: should reuse the first 2 blocks and allocate the third.
    const auto& full_out = batch_res->blocks(0, kFullTag);
    ASSERT_EQ(full_out.size(), 3u);
    EXPECT_EQ(full_out[0], full_blocks[0]);
    EXPECT_EQ(full_out[1], full_blocks[1]);
    EXPECT_FALSE(isNullBlockIdx(full_out[2]));

    // linear_step=1 materializes every physical block; the common-boundary
    // match is reused and the remaining slots are allocated.
    const auto& linear_out = batch_res->blocks(0, kLinearTag);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_FALSE(isNullBlockIdx(linear_out[0]));
    EXPECT_EQ(linear_out[1], linear_blocks[0]);   // reused tail at pos=1
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));  // allocated tail for common length
}

TEST_F(KVCacheAllocatorTest, HeterogeneousReuseLengthIsIndependentOfGroupOrder) {
    auto run = [](bool reverse_groups) {
        auto config = makeTinyHybridConfig();
        auto groups = config.topology().groups();
        for (auto& group : groups) {
            group.seq_size_per_block        = group.tag == kFullTag ? 4 : 2;
            group.kernel_seq_size_per_block = group.seq_size_per_block;
        }
        if (reverse_groups) {
            std::reverse(groups.begin(), groups.end());
        }
        config.setTopology(std::move(groups), config.topology().layers());

        auto allocator    = std::make_shared<KVCacheAllocator>(config, AllocationType::HOST);
        auto shared_cache = std::make_shared<SharedBlockCache>();
        allocator->setSharedBlockCache(shared_cache);
        EXPECT_TRUE(allocator->init());

        allocateAndCache(poolFor(allocator, kFullTag), shared_cache, kFullTag, {400, 800});
        allocateAndCache(poolFor(allocator, kLinearTag), shared_cache, kLinearTag, {200, 400, 600, 800});

        auto resource = makeBatchResource(/*batch_size=*/1, config, {});
        resource->setBatchCacheKeys(0, kFullTag, {400, 800});
        resource->setBatchCacheKeys(0, kLinearTag, {200, 400, 600, 800});
        auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/2);
        rebuildRequestPrefixes(resource, token_ids);

        MallocInfo info{resource, token_ids};
        info.enable_device_cache = true;
        info.reuse_cache         = true;
        const auto result        = allocator->malloc(info);
        EXPECT_TRUE(result.success);
        EXPECT_EQ(resource->blocksNum(0, kFullTag), 2u);
        EXPECT_EQ(resource->blocksNum(0, kLinearTag), 4u);
        return result.reuse_len;
    };

    // Prefill must retain at least one input token for model execution. The
    // greatest shared physical boundary below 8 tokens is 4 for both orders.
    EXPECT_EQ(run(false), 4);
    EXPECT_EQ(run(true), 4);
}

TEST_F(KVCacheAllocatorTest, MissingSparseBoundaryFallsBackToNoReuse) {
    auto config        = makeTinyHybridConfig();
    config.linear_step = 4;
    auto allocator     = std::make_shared<KVCacheAllocator>(config, AllocationType::HOST);
    auto shared_cache  = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    allocateAndCache(poolFor(allocator, kFullTag), shared_cache, kFullTag, {100, 101, 102});
    // The sparse group has a later materialized endpoint, but not the shorter
    // boundary selected by the FULL group and prefill's keep-one-token rule.
    allocateAndCache(poolFor(allocator, kLinearTag), shared_cache, kLinearTag, {103});

    auto resource  = makeBatchResource(/*batch_size=*/1, config, {100, 101, 102, 103});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    rebuildRequestPrefixes(resource, token_ids);
    MallocInfo info{resource, token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;

    MallocResult result;
    EXPECT_NO_THROW(result = allocator->malloc(info));
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
}

TEST_F(KVCacheAllocatorTest, ReuseSpanFallsBackToAllGroupsWhenReuseDisabled) {
    auto config = makeTinyHybridConfig();
    auto groups = config.topology().groups();
    for (auto& group : groups) {
        group.policy.enable_prefix_reuse = false;
        group.seq_size_per_block         = group.tag == kFullTag ? 4 : 2;
        group.kernel_seq_size_per_block  = group.seq_size_per_block;
    }
    config.setTopology(std::move(groups), config.topology().layers());

    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->seqSizePerBlock(), 4);
}

TEST_F(KVCacheAllocatorTest, LayerCopyStrideUsesMtpPhysicalLayout) {
    auto config = makeTinyHybridMtpConfigByCreateSpConfig();
    ASSERT_FALSE(config.mtp_sub_configs.empty());
    auto&      sub_config = *config.mtp_sub_configs.front();
    auto       sub_groups = sub_config.topology().groups();
    const auto sub_it =
        std::find_if(sub_groups.begin(), sub_groups.end(), [](const GroupBase& group) { return group.tag == "full"; });
    ASSERT_NE(sub_it, sub_groups.end());
    sub_it->kv_block_stride_bytes = 1234;
    sub_it->kv_scale_stride_bytes = 56;
    sub_config.setTopology(std::move(sub_groups), sub_config.topology().layers());

    const auto& global_layers = config.layerIdsForGroup("full");
    const auto  mtp_it        = std::find_if(global_layers.begin(), global_layers.end(), [&](int layer_id) {
        return layer_id >= static_cast<int>(config.layer_num);
    });
    ASSERT_NE(mtp_it, global_layers.end());
    const auto       mtp_layer = *mtp_it;
    KVCacheAllocator allocator(config, AllocationType::HOST);
    EXPECT_EQ(allocator.layerCopyStrides("full", mtp_layer), (std::pair<size_t, size_t>{1234, 56}));
}

TEST(CacheConfigCapacityContractTest, LinearStepDoesNotChangePhysicalPoolCapacity) {
    auto config        = makeTinyHybridConfig();
    config.linear_step = 4;
    config.applyTokenCapacity(32);
    // linear_step is a runtime materialization cadence. Pool capacity remains
    // topology-owned so an arbitrary request tail can still be materialized.
    EXPECT_EQ(config.topology().group(kLinearTag).block_num, 8u);
    EXPECT_EQ(config.topology().group(kFullTag).block_num, 8u);
}

TEST_F(KVCacheAllocatorTest, FullPromptHitKeepsOneTokenForPrefill) {
    auto config       = makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                           /*block_num=*/10,
                                           /*tokens_per_block=*/4,
                                           rtp_llm::DataType::TYPE_FP16);
    auto allocator    = std::make_shared<KVCacheAllocator>(config, AllocationType::HOST);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto cached_blocks = allocateAndCache(allocator->blockPool("default"), shared_cache, "default", {100, 200});
    auto resource      = makeBatchResource(/*batch_size=*/1, config, {100, 200});
    auto token_ids     = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    rebuildRequestPrefixes(resource, token_ids);

    MallocInfo info{resource, token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    const auto result        = allocator->malloc(info);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 4);
    ASSERT_EQ(resource->blocksNum(0, "default"), 2u);
    EXPECT_EQ(resource->blocks(0, "default")[0], cached_blocks[0]);
    EXPECT_NE(resource->blocks(0, "default")[1], cached_blocks[1]);
    EXPECT_EQ(shared_cache->matchGroup(200, "default"), cached_blocks[1]);
}

TEST_F(KVCacheAllocatorTest, DisableReuseKeepsOnlyLinearTailOnInitMalloc) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    // Disable device-cache matching, but materialize every reusable LINEAR
    // block so insertion can publish each completed group-local prefix.

    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Linear group should keep only the tail block across common length slots.
    const auto& linear_out = batch_res->blocks(0, kLinearTag);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
}

TEST_F(KVCacheAllocatorTest, DisableDeviceCacheSkipsReuseMatchAndAllocatesOnlyLinearTail) {
    auto config       = makeTinyHybridConfig();
    auto allocator    = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto full_pool = poolFor(allocator, kFullTag);

    // Prepare cached blocks for full group; keep them allocated so allocator's malloc() cannot accidentally return same
    // ids.
    CacheKeysType full_keys   = {100, 101, 102};
    auto          full_blocks = allocateAndCacheKeepAllocated(full_pool, shared_cache, kFullTag, full_keys);
    ASSERT_EQ(full_blocks.size(), 3u);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
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
    const auto& full_out = batch_res->blocks(0, kFullTag);
    ASSERT_EQ(full_out.size(), 3u);
    EXPECT_FALSE(isNullBlockIdx(full_out[0]));
    EXPECT_FALSE(isNullBlockIdx(full_out[1]));
    EXPECT_FALSE(isNullBlockIdx(full_out[2]));
    EXPECT_NE(full_out[0], full_blocks[0]);
    EXPECT_NE(full_out[1], full_blocks[1]);
    EXPECT_NE(full_out[2], full_blocks[2]);

    // Linear group keeps only tail block (others NULL) when reuse is disabled.
    const auto& linear_out = batch_res->blocks(0, kLinearTag);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
    EXPECT_EQ(countValidBlocks(linear_out), 1u);
}

TEST_F(KVCacheAllocatorTest, UpdateKVBlockForksSharedBlocksAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto linear_pool = poolFor(allocator, kLinearTag);
    auto full_pool   = poolFor(allocator, kFullTag);

    const size_t free_before   = allocator->freeBlocksNum();
    auto         linear_blocks = linear_pool->malloc(3);
    auto         full_blocks   = full_pool->malloc(3);
    ASSERT_EQ(linear_blocks.size(), 3u);
    ASSERT_EQ(full_blocks.size(), 3u);
    ASSERT_EQ(allocator->freeBlocksNum(), free_before - 6);

    auto                 batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100, 101});
    std::vector<int32_t> prefix_tokens(13, 1);
    for (int batch_id = 0; batch_id < 2; ++batch_id) {
        batch_res->cacheResource(batch_id).requestPrefix().rebuild(prefix_tokens.data(), prefix_tokens.size());
    }
    auto& source_prefix = batch_res->cacheResource(0).requestPrefix();
    source_prefix.setDeviceReuseTokens(4);
    source_prefix.setMemoryReuseTokens(4);
    source_prefix.setRemoteReuseTokens(4);
    const auto source_prefix_keys = source_prefix.keys();
    batch_res->mutableBlockIds(0, kLinearTag).assign({linear_blocks[0], NULL_BLOCK_IDX, linear_blocks[1]});
    batch_res->mutableBlockIds(0, kFullTag).assign({full_blocks[0], full_blocks[1]});
    batch_res->mutableBlockIds(1, kLinearTag).assign({linear_blocks[2]});
    batch_res->mutableBlockIds(1, kFullTag).assign({full_blocks[2]});

    std::vector<GroupBlockIdPair> update_mapping;
    ASSERT_TRUE(allocator->updateKVBlock(batch_res,
                                         /*block_src_batch=*/std::vector<int>{0, 0},
                                         /*copy_last_block=*/false,
                                         update_mapping));

    EXPECT_TRUE(update_mapping.empty());
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 4) << "unused old batch blocks should be released";
    ASSERT_EQ(batch_res->batchSize(), 2);
    EXPECT_EQ(batch_res->cacheKeys(0, kFullTag), (CacheKeysType{100, 101}));
    EXPECT_EQ(batch_res->cacheKeys(1, kFullTag), (CacheKeysType{100, 101}));
    EXPECT_EQ(batch_res->blocks(0, kLinearTag), (BlockIndicesType{linear_blocks[0], NULL_BLOCK_IDX, linear_blocks[1]}));
    EXPECT_EQ(batch_res->blocks(0, kFullTag), (BlockIndicesType{full_blocks[0], full_blocks[1]}));
    EXPECT_EQ(batch_res->blocks(1, kLinearTag), (BlockIndicesType{linear_blocks[0], NULL_BLOCK_IDX, linear_blocks[1]}));
    EXPECT_EQ(batch_res->blocks(1, kFullTag), (BlockIndicesType{full_blocks[0], full_blocks[1]}));
    for (int batch_id = 0; batch_id < 2; ++batch_id) {
        const auto& prefix = batch_res->cacheResource(batch_id).requestPrefix();
        EXPECT_EQ(prefix.keys(), source_prefix_keys);
        EXPECT_EQ(prefix.deviceReuseTokens(), 4u);
        EXPECT_EQ(prefix.memoryReuseTokens(), 4u);
        EXPECT_EQ(prefix.remoteReuseTokens(), 4u);
    }

    allocator->free(FreeInfo{batch_res, nullptr});
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(KVCacheAllocatorTest, UpdateKVBlockCopyLastBlockAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto linear_pool = poolFor(allocator, kLinearTag);
    auto full_pool   = poolFor(allocator, kFullTag);

    const size_t free_before   = allocator->freeBlocksNum();
    auto         linear_blocks = linear_pool->malloc(3);
    auto         full_blocks   = full_pool->malloc(3);
    ASSERT_EQ(linear_blocks.size(), 3u);
    ASSERT_EQ(full_blocks.size(), 3u);
    ASSERT_EQ(allocator->freeBlocksNum(), free_before - 6);

    auto batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100, 101});
    batch_res->mutableBlockIds(0, kLinearTag).assign({linear_blocks[0], NULL_BLOCK_IDX, linear_blocks[1]});
    batch_res->mutableBlockIds(0, kFullTag).assign({full_blocks[0], full_blocks[1]});
    batch_res->mutableBlockIds(1, kLinearTag).assign({linear_blocks[2]});
    batch_res->mutableBlockIds(1, kFullTag).assign({full_blocks[2]});

    std::vector<GroupBlockIdPair> update_mapping{{"stale", 1, 2}};
    ASSERT_TRUE(allocator->updateKVBlock(batch_res,
                                         /*block_src_batch=*/std::vector<int>{0, 0},
                                         /*copy_last_block=*/true,
                                         update_mapping));

    ASSERT_EQ(update_mapping.size(), 2u);
    EXPECT_EQ(update_mapping[0].tag, std::string(kLinearTag));
    EXPECT_EQ(update_mapping[1].tag, std::string(kFullTag));
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 6);
    ASSERT_EQ(batch_res->batchSize(), 2);
    EXPECT_EQ(batch_res->cacheKeys(0, kFullTag), (CacheKeysType{100, 101}));
    EXPECT_EQ(batch_res->cacheKeys(1, kFullTag), (CacheKeysType{100, 101}));

    const auto& forked_group0 = batch_res->blocks(0, kLinearTag);
    const auto& moved_group0  = batch_res->blocks(1, kLinearTag);
    const auto& forked_group1 = batch_res->blocks(0, kFullTag);
    const auto& moved_group1  = batch_res->blocks(1, kFullTag);
    ASSERT_EQ(forked_group0.size(), 3u);
    ASSERT_EQ(forked_group1.size(), 2u);
    EXPECT_EQ(moved_group0, (BlockIndicesType{linear_blocks[0], NULL_BLOCK_IDX, linear_blocks[1]}));
    EXPECT_EQ(moved_group1, (BlockIndicesType{full_blocks[0], full_blocks[1]}));
    EXPECT_EQ(forked_group0[0], linear_blocks[0]);
    EXPECT_TRUE(isNullBlockIdx(forked_group0[1]));
    EXPECT_NE(forked_group0[2], linear_blocks[1]);
    EXPECT_FALSE(isNullBlockIdx(forked_group0[2]));
    EXPECT_EQ(forked_group1[0], full_blocks[0]);
    EXPECT_NE(forked_group1[1], full_blocks[1]);
    EXPECT_FALSE(isNullBlockIdx(forked_group1[1]));

    allocator->free(FreeInfo{batch_res, nullptr});
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(KVCacheAllocatorTest, UpdateKVBlockReservationFailureLeavesResourceUnchanged) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto linear_pool = poolFor(allocator, kLinearTag);
    auto full_pool   = poolFor(allocator, kFullTag);

    const size_t free_before   = allocator->freeBlocksNum();
    auto         linear_blocks = linear_pool->malloc(1);
    auto         full_blocks   = full_pool->malloc(1);
    ASSERT_EQ(linear_blocks.size(), 1u);
    ASSERT_EQ(full_blocks.size(), 1u);
    auto full_keep = full_pool->malloc(static_cast<int>(full_pool->freeBlocksNum()));
    ASSERT_EQ(full_pool->freeBlocksNum(), 0u);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100});
    batch_res->mutableBlockIds(0, kLinearTag).assign({linear_blocks[0]});
    batch_res->mutableBlockIds(0, kFullTag).assign({full_blocks[0]});

    const auto before_batch0_group0 = batch_res->blocks(0, kLinearTag);
    const auto before_batch0_group1 = batch_res->blocks(0, kFullTag);
    const auto linear_free_before   = linear_pool->freeBlocksNum();
    const auto full_free_before     = full_pool->freeBlocksNum();
    const auto linear_refs_before   = linear_pool->requestRefBlocksNum();
    const auto full_refs_before     = full_pool->requestRefBlocksNum();

    std::vector<GroupBlockIdPair> update_mapping{{"stale", 1, 2}};
    EXPECT_FALSE(allocator->updateKVBlock(batch_res,
                                          /*block_src_batch=*/std::vector<int>{0, 0},
                                          /*copy_last_block=*/true,
                                          update_mapping));

    EXPECT_TRUE(update_mapping.empty());
    EXPECT_EQ(batch_res->batchSize(), 1);
    EXPECT_EQ(batch_res->blocks(0, kLinearTag), before_batch0_group0);
    EXPECT_EQ(batch_res->blocks(0, kFullTag), before_batch0_group1);
    EXPECT_EQ(linear_pool->freeBlocksNum(), linear_free_before);
    EXPECT_EQ(full_pool->freeBlocksNum(), full_free_before);
    EXPECT_EQ(linear_pool->requestRefBlocksNum(), linear_refs_before);
    EXPECT_EQ(full_pool->requestRefBlocksNum(), full_refs_before);

    allocator->free(FreeInfo{batch_res, nullptr});
    full_pool->requestFree(full_keep);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(KVCacheAllocatorTest, UpdateKVBlockReusesDroppedBatchCapacityTransactionally) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto linear_pool = poolFor(allocator, kLinearTag);
    auto full_pool   = poolFor(allocator, kFullTag);

    const size_t free_before   = allocator->freeBlocksNum();
    auto         linear_blocks = linear_pool->malloc(static_cast<int>(linear_pool->freeBlocksNum()));
    auto         full_blocks   = full_pool->malloc(static_cast<int>(full_pool->freeBlocksNum()));
    ASSERT_GE(linear_blocks.size(), 2u);
    ASSERT_GE(full_blocks.size(), 2u);
    ASSERT_EQ(linear_pool->freeBlocksNum(), 0u);
    ASSERT_EQ(full_pool->freeBlocksNum(), 0u);

    auto batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100});
    batch_res->mutableBlockIds(0, kLinearTag).assign({linear_blocks[0]});
    batch_res->mutableBlockIds(0, kFullTag).assign({full_blocks[0]});
    batch_res->mutableBlockIds(1, kLinearTag).assign({linear_blocks[1]});
    batch_res->mutableBlockIds(1, kFullTag).assign({full_blocks[1]});

    std::vector<GroupBlockIdPair> update_mapping;
    ASSERT_TRUE(allocator->updateKVBlock(batch_res,
                                         /*block_src_batch=*/std::vector<int>{1, 1},
                                         /*copy_last_block=*/true,
                                         update_mapping));

    ASSERT_EQ(update_mapping.size(), 2u);
    EXPECT_EQ(update_mapping[0].tag, std::string(kLinearTag));
    EXPECT_EQ(update_mapping[0].src, linear_blocks[1]);
    EXPECT_EQ(update_mapping[0].dst, linear_blocks[0]);
    EXPECT_EQ(update_mapping[1].tag, std::string(kFullTag));
    EXPECT_EQ(update_mapping[1].src, full_blocks[1]);
    EXPECT_EQ(update_mapping[1].dst, full_blocks[0]);
    EXPECT_EQ(linear_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(full_pool->freeBlocksNum(), 0u);

    allocator->free(FreeInfo{batch_res, nullptr});
    linear_pool->requestFree(BlockIndicesType(linear_blocks.begin() + 2, linear_blocks.end()));
    full_pool->requestFree(BlockIndicesType(full_blocks.begin() + 2, full_blocks.end()));
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(KVCacheAllocatorTest, IncrDecrKVCacheRefReferencesOnlyMatchedValidBlocksAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto linear_pool = poolFor(allocator, kLinearTag);
    auto full_pool   = poolFor(allocator, kFullTag);

    const size_t free_before   = allocator->freeBlocksNum();
    auto         linear_blocks = linear_pool->malloc(2);
    auto         full_blocks   = full_pool->malloc(2);
    ASSERT_EQ(linear_blocks.size(), 2u);
    ASSERT_EQ(full_blocks.size(), 2u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 4);

    KVCacheResource resource;
    resource.initGroups(config.topologyPtr());
    resource.cacheKeys(kLinearTag) = CacheKeysType{100, 101, 102};
    resource.cacheKeys(kFullTag)   = CacheKeysType{100, 101, 102};
    resource.mutableBlockIds(kLinearTag)
        .assign(BlockIndicesType{linear_blocks[0], 0, linear_blocks[1]});  // linear group (contains a 0)
    resource.mutableBlockIds(kFullTag).assign(
        BlockIndicesType{full_blocks[0], full_blocks[1], 0});  // full group (contains a 0)

    // keys: 101(pos1)->gid0:0(ignore), gid1:blocks[3](ref); 102(pos2)->gid0:blocks[1](ref), gid1:0(ignore).
    // The migrated HybridKV base drops unmatched keys rather than preserving empty placeholders.
    const CacheKeysByGroup requested_keys{{std::string(kLinearTag), CacheKeysType{101, 999, 102}},
                                          {std::string(kFullTag), CacheKeysType{101, 999, 102}}};
    auto                   ref = allocator->incrKVCacheRef(resource, requested_keys);
    ASSERT_NE(ref, nullptr);
    ASSERT_EQ(ref->groupNums(), 2);
    ASSERT_EQ(ref->cacheKeys(kLinearTag).size(), 1u);
    ASSERT_EQ(ref->cacheKeys(kFullTag).size(), 1u);
    ASSERT_EQ(ref->blocks(kLinearTag).size(), 1u);
    ASSERT_EQ(ref->blocks(kFullTag).size(), 1u);

    linear_pool->requestFree(linear_blocks);
    full_pool->requestFree(full_blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2) << "Only blocks[1] and blocks[3] should remain referenced";

    ref.reset();
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(KVCacheAllocatorTest, ReserveCapacitySkipsLogicalGroupsMarkedNonReservable) {
    auto config = makeTinyHybridConfig();
    setGroupBlockNums(config, /*linear=*/3, /*full=*/9);
    auto groups = config.topology().groups();
    for (auto& group : groups) {
        group.policy.reservable = group.tag == kFullTag;
    }
    config.setTopology(std::move(groups), config.topology().layers());

    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::HOST, nullptr, /*reserve_ratio=*/50);
    ASSERT_TRUE(allocator->init());

    const size_t full_available = allocator->blockPool(std::string(kFullTag))->availableBlocksNum();
    EXPECT_EQ(allocator->reservableAvailableBlocksNum(), full_available);
    EXPECT_EQ(allocator->reserveBlocksNum(), full_available / 2);

    for (const auto& snapshot : allocator->poolMetricsSnapshots()) {
        if (snapshot.pool_name == kLinearTag) {
            EXPECT_EQ(snapshot.reserve_blocks, 0u);
        } else if (snapshot.pool_name == kFullTag) {
            EXPECT_EQ(snapshot.reserve_blocks, allocator->reserveBlocksNum());
        }
    }
}

TEST_F(KVCacheAllocatorTest, InsertIntoCacheInsertsOnlyFullBlocks) {
    auto config       = makeTinyHybridConfig();
    auto allocator    = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102});
    // Disable device cache reuse.

    // Non-CP SharedBlockCache insertion records the available group block ids for each cache key.
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/10, /*seq_size_per_block=*/4);

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = true;
    auto malloc_result              = allocator->malloc(malloc_info);
    ASSERT_TRUE(malloc_result.success);
    ASSERT_EQ(batch_res->blocksNum(0, kFullTag), 3);
    ASSERT_EQ(batch_res->blocksNum(0, kLinearTag), 3);

    InsertInfo insert_info{batch_res, token_ids, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(100, kFullTag)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(101, kFullTag)));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(102, kFullTag)));

    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(100, kLinearTag)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(101, kLinearTag)));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(102, kLinearTag)));
}

TEST_F(KVCacheAllocatorTest, DefaultHybridLinearPrefixReuseSupportsInsertThenReuse) {
    auto config = makeTinyHybridConfig();
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_TRUE(config.policyForGroup(kLinearTag).enable_prefix_reuse);

    auto allocator    = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto seed_res    = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102});
    auto seed_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/13, /*seq_size_per_block=*/4);

    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.enable_device_cache = false;
    seed_malloc.reuse_cache         = true;
    ASSERT_TRUE(allocator->malloc(seed_malloc).success);

    allocator->insertIntoCache(InsertInfo{seed_res, seed_tokens, /*is_resident=*/false});
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(102, kLinearTag)));

    auto hit_res    = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    auto hit_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    rebuildRequestPrefixes(hit_res, hit_tokens);

    MallocInfo hit_malloc{hit_res, hit_tokens};
    hit_malloc.enable_device_cache = true;
    hit_malloc.reuse_cache         = true;
    auto result                    = allocator->malloc(hit_malloc);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 8);
}

TEST_F(KVCacheAllocatorTest, ConvertIndexToBufferAndAllLayerCacheBaseSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    KVCacheAllocator* base = allocator.get();
    auto              buf0 = base->convertIndexToBuffer(/*layer_id=*/0, "linear", /*block_id=*/1);
    ASSERT_FALSE(buf0.empty());
    EXPECT_NE(buf0[0].addr, nullptr);

    auto linear_buf = base->convertIndexToBuffer(/*layer_id=*/0, "linear", /*block_id=*/1);
    auto full_buf   = base->convertIndexToBuffer(/*layer_id=*/2, "full1", /*block_id=*/1);
    ASSERT_FALSE(linear_buf.empty());
    ASSERT_FALSE(full_buf.empty());
    EXPECT_NE(linear_buf[0].addr, nullptr);
    EXPECT_NE(full_buf[0].addr, nullptr);
    EXPECT_EQ(linear_buf[0].size_bytes, config.kvBlockStrideBytesForGroup(kLinearTag));
    EXPECT_EQ(full_buf[0].size_bytes, config.kvBlockStrideBytesForGroup(kFullTag));
    EXPECT_LT(linear_buf[0].size_bytes, config.kvBlockStrideBytesForGroup(kFullTag));

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.groups().size(), static_cast<size_t>(config.groupNums()));
    ASSERT_EQ(layout.topology().layers().size(), static_cast<size_t>(config.layer_num));
    for (size_t i = 0; i < layout.topology().layers().size(); ++i) {
        for (const auto& tag : layout.topology().layer(static_cast<int>(i)).group_tags) {
            EXPECT_TRUE(layout.group(tag).hasLayer(i));
        }
    }
}

TEST_F(KVCacheAllocatorTest, IncrMallocRollbackFreesPartiallyAllocatedBlocks) {
    auto config = makeTinyHybridConfig();
    setGroupBlockNum(config, 6);  // five usable blocks per group
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto linear_pool = poolFor(allocator, kLinearTag);
    auto full_pool   = poolFor(allocator, kFullTag);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102});
    // Disable device cache reuse (makes linear group allocate only tail for new slots).

    // Initial small allocation: seq_len=4 => 1 slot per group.
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    auto init_result              = allocator->malloc(init_info);
    ASSERT_TRUE(init_result.success);
    ASSERT_EQ(batch_res->blocksNum(0, kLinearTag), 1);
    ASSERT_EQ(batch_res->blocksNum(0, kFullTag), 1);

    const auto linear_block_before = batch_res->blocks(0, kLinearTag)[0];
    const auto full_block_before   = batch_res->blocks(0, kFullTag)[0];

    // Leave exactly 1 free block in pool, so linear allocates 1 and full fails on the next allocation.
    const size_t free_before_incr = full_pool->freeBlocksNum();
    ASSERT_GE(free_before_incr, 1u);
    auto keep = full_pool->malloc(static_cast<int>(free_before_incr - 1));
    ASSERT_EQ(full_pool->freeBlocksNum(), 1u);
    const auto linear_free_before_incr = linear_pool->freeBlocksNum();

    // Incr to seq_len=9 => 3 slots per group. Linear adds 2 slots but allocates only 1 real block; full needs 2.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    auto incr_result              = allocator->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    // Rollback should restore original sizes and keep original blocks.
    ASSERT_EQ(batch_res->blocksNum(0, kLinearTag), 1);
    ASSERT_EQ(batch_res->blocksNum(0, kFullTag), 1);
    EXPECT_EQ(batch_res->blocks(0, kLinearTag)[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, kFullTag)[0], full_block_before);

    // Free blocks count should return to 1 (no leaks).
    EXPECT_EQ(full_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(linear_pool->freeBlocksNum(), linear_free_before_incr);

    // Cleanup.
    full_pool->requestFree(keep);
}

// Prefill init path (StreamCacheResource::initKVBlock sets enable_remove_skipped_blocks=false).
// With step=2 and reuse_blocks_len=3, the reused linear tail lands at pos 2, which is NOT
// a step hit ((2+1)%2==1). Without sparse cleanup, that slot must survive so that
// causal_conv1d can still read it by prefix_length.
TEST_F(KVCacheAllocatorTest, PrefillInitSkipsSparseCleanupAndPreservesReusedLinearTail) {
    auto config = makeTinyHybridConfig();
    setGroupBlockNum(config, 16);
    auto allocator    = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto linear_pool = poolFor(allocator, kLinearTag);
    auto full_pool   = poolFor(allocator, kFullTag);

    CacheKeysType shared_keys          = {100, 101, 102};
    auto          cached_full_blocks   = allocateAndCache(full_pool, shared_cache, kFullTag, shared_keys);
    auto          cached_linear_blocks = allocateAndCache(linear_pool, shared_cache, kLinearTag, shared_keys);
    ASSERT_EQ(cached_linear_blocks.size(), 3u);

    // Request has 5 keys. Full matches the first 3 (103 is absent); linear joint backoff stops at pos=2, so the
    // joint reuse length is three blocks.
    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103, 104});

    // seq_len=20 => 5 slots. block_size-3-reserve_step = 2, so removeSkippedBlocks would scan pos 2.
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/20, /*seq_size_per_block=*/4);
    rebuildRequestPrefixes(batch_res, token_ids);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache          = true;
    info.reuse_cache                  = true;
    info.enable_remove_skipped_blocks = false;  // prefill init path
    auto result                       = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    const auto& linear_out = batch_res->blocks(0, kLinearTag);
    ASSERT_EQ(linear_out.size(), 5u);
    EXPECT_FALSE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_EQ(linear_out[2], cached_linear_blocks[2]) << "reused linear tail must survive prefill init";
    EXPECT_FALSE(isNullBlockIdx(linear_out[3]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[4]));
}

// Decode path (StreamCacheResource::incrKVBlock sets enable_remove_skipped_blocks=true).
// The allocator is invoked on an already-populated resource, so malloc() dispatches directly
// to incrMalloc(). Sparse cleanup must prune non-step blocks while preserving step hits and
// the configured active tail slot.
TEST_F(KVCacheAllocatorTest, DecodeIncrMallocAppliesSparseCleanupOnLinearGroups) {
    auto config = makeTinyHybridConfig();
    setGroupBlockNum(config, 16);
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto linear_pool  = poolFor(allocator, kLinearTag);
    auto full_pool    = poolFor(allocator, kFullTag);
    auto linear_alloc = linear_pool->malloc(6);
    auto full_alloc   = full_pool->malloc(6);
    ASSERT_EQ(linear_alloc.size(), 6u);
    ASSERT_EQ(full_alloc.size(), 6u);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{});
    batch_res->mutableBlockIds(0, kLinearTag).assign(linear_alloc);
    batch_res->mutableBlockIds(0, kFullTag).assign(full_alloc);
    ASSERT_GT(batch_res->curBlocksNum(), 0);

    // seq_len=24 => 6 slots; current_blocks==6 so group malloc is a no-op and only cleanup runs.
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache          = false;
    info.reuse_cache                  = true;
    info.enable_remove_skipped_blocks = true;  // decode path
    auto result                       = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Production linear_step=1 materializes every physical block.
    const auto& linear_out = batch_res->blocks(0, kLinearTag);
    ASSERT_EQ(linear_out.size(), 6u);
    EXPECT_FALSE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[3]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[4]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[5]));

    // Full group is untouched by sparse cleanup.
    const auto& full_out = batch_res->blocks(0, kFullTag);
    ASSERT_EQ(full_out.size(), 6u);
    for (size_t i = 0; i < full_out.size(); ++i) {
        EXPECT_EQ(full_out[i], full_alloc[i]);
    }
}

TEST_F(KVCacheAllocatorTest, EstimatePeakNeedBlocks) {
    // Config: [0,1]=linear group (gid=0), [2,3]=full group (gid=1). seq_size_per_block=4.
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int blk = static_cast<int>(config.seqSizePerBlockForGroup(kFullTag));  // 4

    // New resource (cur_slots=0 for both groups):
    // reuse disabled: full=ceil(108/4)=27, linear tail peak=3 => total=30.
    auto new_res = makeBatchResource(1, config, {});
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/false), 30);

    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/true), 54);

    // With reserve_step=3: full=ceil(111/4)=28. linear: total_slots=29, tail=5,
    // step-hits before tail=24/2=12 => linear=17. total=45.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 3, /*enable_reuse_cache=*/true), 57);

    // Allocate blocks to simulate running decode (seqLen=8 → 2 slots per group)
    auto       token_ids = makeCompleteTokenIds(1, /*seq_length=*/8, config.seqSizePerBlockForGroup(kFullTag));
    MallocInfo mi{new_res, token_ids};
    auto       result = allocator->malloc(mi);
    ASSERT_TRUE(result.success);

    const int full_slots   = new_res->blocksNum(0, kFullTag);    // full group slots after malloc
    const int linear_slots = new_res->blocksNum(0, kLinearTag);  // linear group slots after malloc

    // remaining=0: no more slots needed for either group
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 0, 0, /*enable_reuse_cache=*/false), 0);

    // remaining=4: ceil((8+4)/4)=3 per group, minus cur_slots
    int expect_per_group = (8 + 4 + blk - 1) / blk;
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 4, 0, /*enable_reuse_cache=*/false),
              std::max(expect_per_group - full_slots, 0) + std::max(expect_per_group - linear_slots, 0));

    // Large remaining from current_slots=2:
    // reuse disabled: cleanup scans across the initial null slot. At the second boundary the running resource
    // transiently holds three physical linear blocks before releasing the oldest tail, two more than its current tail.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/false), 26);

    // reuse enabled: target linear keeps tail 2 + step-hit slots before tail 12;
    // The fresh seq_len=8 allocation owns one physical linear block. Decode later peaks at 15 physical blocks.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/true), 50);
}

TEST_F(KVCacheAllocatorTest, EstimatePeakNeedBlocksUsesLinearActiveTailPolicy) {
    auto                                              config = makeTinyHybridConfig();
    std::unordered_map<std::string, CacheGroupPolicy> policies;
    for (const auto& group : config.topology().groups()) {
        policies.emplace(group.tag, group.policy);
    }
    ASSERT_EQ(policies.size(), 2u);
    ASSERT_EQ(policies.at(std::string(kLinearTag)).group_type, CacheGroupType::LINEAR);
    policies.at(std::string(kLinearTag)).active_tail_blocks = 4;
    config.setGroupPolicies(policies);

    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto resource = makeBatchResource(/*batch_size=*/1, config, /*keys=*/{});

    // At seq_len=24 the LINEAR group materializes four active tails and the FULL group owns six blocks.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(
                  *allocator, resource, /*seq_len=*/24, /*remaining_tokens=*/0, /*reserve_step=*/0, false),
              10);

    // One more block boundary adds a transient LINEAR tail and one permanent FULL block.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(
                  *allocator, resource, /*seq_len=*/24, /*remaining_tokens=*/4, /*reserve_step=*/0, false),
              12);
}

TEST_F(KVCacheAllocatorTest, EstimateBatchPeakNeedBlocksAccountsForNonEmptyTargetWidth) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto resource = makeBatchResource(/*batch_size=*/2, config, /*keys=*/{});

    // common_seq_len=8 means the first two slots are shared. The NULL slot in the linear group consumes no block.
    resource->setBatchBlocks(0, kLinearTag, {NULL_BLOCK_IDX, 10, 11});
    resource->setBatchBlocks(1, kLinearTag, {NULL_BLOCK_IDX, 10, 12});
    resource->setBatchBlocks(0, kFullTag, {20, 21, 22});
    resource->setBatchBlocks(1, kFullTag, {20, 21, 23});

    // No growth is needed at the current batch width.
    EXPECT_EQ(allocator->estimateBatchPeakNeedBlocks(resource,
                                                     /*seq_len=*/12,
                                                     /*common_seq_len=*/8,
                                                     /*remaining_tokens=*/0,
                                                     /*reserve_step=*/0,
                                                     /*enable_reuse_cache=*/false,
                                                     /*target_batch_size=*/2),
              0);

    // The aligned tail remains shared when one more sequence is forked.
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

    // One future block in each group is charged at the requested target width;
    // the currently aligned tails remain shared.
    EXPECT_EQ(allocator->estimateBatchPeakNeedBlocks(resource,
                                                     /*seq_len=*/12,
                                                     /*common_seq_len=*/8,
                                                     /*remaining_tokens=*/4,
                                                     /*reserve_step=*/0,
                                                     /*enable_reuse_cache=*/false,
                                                     /*target_batch_size=*/3),
              6);

    resource->setBatchBlocks(0, kLinearTag, {NULL_BLOCK_IDX, 10, 11, NULL_BLOCK_IDX});
    resource->setBatchBlocks(1, kLinearTag, {NULL_BLOCK_IDX, 10, 12, NULL_BLOCK_IDX});
    resource->setBatchBlocks(0, kFullTag, {20, 21, 22, 24});
    resource->setBatchBlocks(1, kFullTag, {20, 21, 23, 25});

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

TEST_F(KVCacheAllocatorTest, FreshUnalignedMultiSequencePeakMatchesExactCapacity) {
    for (const bool reuse_cache : {false, true}) {
        SCOPED_TRACE(reuse_cache ? "reuse enabled" : "reuse disabled");

        auto config = makeTinyHybridConfig();
        setGroupBlockNum(config, 4);  // Three usable blocks per pool, six in aggregate.
        auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
        ASSERT_TRUE(allocator->init());

        auto resource = makeBatchResource(/*batch_size=*/2, config, /*keys=*/{});

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
            /*batch_size=*/2, /*seq_length=*/5, /*seq_size_per_block=*/config.seqSizePerBlockForGroup(kFullTag));
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

TEST_F(KVCacheAllocatorTest, EstimatedPeakCoversDecodeMallocAndSparseCleanup) {
    auto config = makeTinyHybridConfig();
    setGroupBlockNums(config, /*linear_block_num=*/18, /*full_block_num=*/18);  // 34 usable blocks in aggregate.
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1,
                                          /*seq_length=*/8,
                                          /*seq_size_per_block=*/config.seqSizePerBlockForGroup(kFullTag));

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache          = false;
    info.reuse_cache                  = true;
    info.enable_remove_skipped_blocks = false;
    ASSERT_TRUE(allocator->malloc(info).success);
    ASSERT_EQ(allocator->freeBlocksNum(), 30);

    // From seq_len=8 to 68: full needs 15 more blocks; linear grows from one physical block to a transient peak of 10.
    ASSERT_EQ(estimateBatchPeakForSingleSequence(*allocator,
                                                 batch_res,
                                                 /*seq_len=*/8,
                                                 /*remaining_tokens=*/60,
                                                 /*reserve_step=*/0,
                                                 /*reuse_cache=*/true),
              30);

    info.enable_remove_skipped_blocks = true;
    size_t min_free_blocks            = allocator->freeBlocksNum();
    for (int seq_len = 9; seq_len <= 68; ++seq_len) {
        token_ids->setSeqLength(seq_len);
        ASSERT_TRUE(allocator->malloc(info).success) << "seq_len=" << seq_len;
        min_free_blocks = std::min(min_free_blocks, allocator->freeBlocksNum());
    }

    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, kLinearTag)), 17);
    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, kFullTag)), 17);
    EXPECT_EQ(min_free_blocks, 0);
    EXPECT_EQ(allocator->freeBlocksNum(), 0);
}

TEST_F(KVCacheAllocatorTest, FreshReusePeakCoversThreeBoundaryDecodeAtExactCapacity) {
    auto config = makeTinyHybridConfig();
    setGroupBlockNums(config, /*linear_block_num=*/6, /*full_block_num=*/6);  // 10 usable blocks in aggregate.
    auto allocator = std::make_shared<KVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1,
                                          /*seq_length=*/8,
                                          /*seq_size_per_block=*/config.seqSizePerBlockForGroup(kFullTag));

    // seq_len 8 -> 17 crosses the slot boundaries at 9, 13 and 17. With production
    // linear_step=1 both groups finish with five materialized blocks.
    ASSERT_EQ(allocator->freeBlocksNum(), 10);
    ASSERT_EQ(estimateBatchPeakForSingleSequence(*allocator,
                                                 batch_res,
                                                 /*seq_len=*/8,
                                                 /*remaining_tokens=*/9,
                                                 /*reserve_step=*/0,
                                                 /*reuse_cache=*/true),
              10);

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

    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, kLinearTag)), 5);
    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, kFullTag)), 5);
    EXPECT_EQ(allocator->freeBlocksNum(), 0);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
