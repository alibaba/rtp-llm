#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

using TestHybridTypeCoordinatorCacheManager = BlockTreeCacheTestAllocator<CoordinatorCacheManager>;

class PausableHybridPerRankBlockTransferEngine: public PerRankBlockTransferEngine {
public:
    explicit PausableHybridPerRankBlockTransferEngine(const std::vector<GroupSetPtr>& groups):
        PerRankBlockTransferEngine(groups) {}

    std::shared_ptr<AsyncContext> execute(TransferTask task) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++submit_count_;
            if (released_) {
                return PerRankBlockTransferEngine::execute(std::move(task));
            }
            auto context = std::make_shared<TransferBatchAsyncContext>();
            entered_     = true;
            pending_.push_back({std::move(task), context});
            cv_.notify_all();
            return context;
        }
    }

    bool waitUntilEnteredFor(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return entered_; });
    }

    void release() {
        std::vector<PendingSubmit> pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
            pending.swap(pending_);
            cv_.notify_all();
        }
        for (auto& submit : pending) {
            auto context = PerRankBlockTransferEngine::execute(std::move(submit.task));
            context->waitDone();
            submit.context->complete(context->errorInfo());
        }
    }

    size_t submitCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return submit_count_;
    }

private:
    struct PendingSubmit {
        TransferTask                               task;
        std::shared_ptr<TransferBatchAsyncContext> context;
    };

    mutable std::mutex         mutex_;
    std::condition_variable    cv_;
    bool                       entered_{false};
    bool                       released_{false};
    size_t                     submit_count_{0};
    std::vector<PendingSubmit> pending_;
};

class ScopedHybridTransferRelease {
public:
    explicit ScopedHybridTransferRelease(std::shared_ptr<PausableHybridPerRankBlockTransferEngine> transfer_engine):
        transfer_engine_(std::move(transfer_engine)) {}

    ~ScopedHybridTransferRelease() {
        transfer_engine_->release();
    }

private:
    std::shared_ptr<PausableHybridPerRankBlockTransferEngine> transfer_engine_;
};

static CacheConfig makeTinyHybridConfig() {
    auto config = makeSimpleHybridMhaCacheConfig(/*layer_num=*/4,
                                                 /*block_num=*/10,
                                                 /*tokens_per_block=*/4,
                                                 rtp_llm::DataType::TYPE_FP16,
                                                 /*group_layer_num=*/2,
                                                 /*local_head_num_kv=*/1,
                                                 /*size_per_head=*/1);
    auto groups = config.topology().groups();
    for (auto& group : groups) {
        if (group.policy.group_type == CacheGroupType::FULL) {
            auto spec                       = group.spec->clone();
            spec->kernel_seq_size_per_block = 2;
            group.spec                      = std::move(spec);
        }
    }
    config.setTopology(std::move(groups), config.topology().layers());
    return config;
}

static DeviceBlockPoolPtr poolForTag(const CoordinatorCacheManager& allocator, const std::string& tag) {
    for (const auto& group : allocator.cacheGroups()) {
        if (group->tag() == tag) {
            return group->blockPool();
        }
    }
    throw std::runtime_error("missing test pool for tag " + tag);
}

static BlockIndicesType allocateReferencedBlocks(const DeviceBlockPoolPtr& pool, size_t count) {
    auto blocks = pool->malloc(count).value();
    pool->incRef(blocks);
    return blocks;
}

static void setGroupBlockCounts(CacheConfig& config, size_t linear_count, size_t full_count) {
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    for (auto& group : groups) {
        switch (group.policy.group_type) {
            case CacheGroupType::LINEAR:
                group.block_num = linear_count;
                break;
            case CacheGroupType::FULL:
                group.block_num = full_count;
                break;
            default:
                throw std::runtime_error("unsupported group type in setGroupBlockCounts for tag " + group.tag);
        }
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

    auto       config              = CacheConfigCreator::createConfig(score_model_cfg,
                                                   parallelism_cfg,
                                                   kv_cache_cfg,
                                                   sp_cfg,
                                                   &propose_model_cfg,
                                                   /*is_mtp=*/true,
                                                   /*is_eagle=*/sp_type == SP_TYPE_EAGLE);
    const auto candidate_block_num = CacheConfigCreator::computeLocalBlockNum(
        config, score_model_cfg, runtime_cfg, kv_cache_cfg, parallelism_cfg, std::nullopt, sp_cfg);
    return rtp_llm::test::finalizeCacheConfig(std::move(config), candidate_block_num);
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

static BatchKVCacheResourcePtr
makeBatchResource(int batch_size, const CacheConfig& config, CacheKeysType keys, bool reorder_resource = false) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(config.topologyPtr());
    if (reorder_resource) {
        auto reversed = config;
        auto groups   = config.topology().groups();
        std::reverse(groups.begin(), groups.end());
        reversed.setTopology(std::move(groups), config.topology().layers());
        // Keep coordinator order unchanged and use different orders within a batch.
        for (int b = 0; b < batch_size; b += 2) {
            res->cacheResource(b).initGroups(reversed.topologyPtr());
        }
    }
    for (int b = 0; b < batch_size; ++b) {
        res->setBatchCacheKeys(b, keys);
    }
    return res;
}

static int estimateBatchPeakForSingleSequence(const CoordinatorCacheManager& allocator,
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

static size_t countValidBlocks(const BlockIndicesType& blocks) {
    size_t n = 0;
    for (auto b : blocks) {
        if (!isNullBlockIdx(b)) {
            ++n;
        }
    }
    return n;
}

class HybridTypeCoordinatorCacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

TEST_F(HybridTypeCoordinatorCacheManagerTest, CreateHybridConfigAllowsOnlyFullGroups) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescs(cfg, {HybridAttentionType::NONE, HybridAttentionType::NONE});

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    auto cache_config       = CacheConfigCreator::createWarmupConfig(cfg, parallelism_cfg, /*gen_num_per_cycle=*/0);
    ASSERT_EQ(cache_config.groupNums(), 1);
    EXPECT_EQ(cache_config.group("full").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(cache_config.group("full").tag, "full");
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, CreateHybridConfigRejectsMultipleFullGroups) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescsWithTags(cfg, {HybridAttentionType::NONE, HybridAttentionType::NONE}, {"full", "full1"});

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    try {
        CacheConfigCreator::createWarmupConfig(cfg, parallelism_cfg, /*gen_num_per_cycle=*/0);
        FAIL() << "expected multiple full groups to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("multiple FULL MHA/MLA cache groups"), std::string::npos);
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, CreateHybridConfigKeepsModelTokensPerBlock) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescs(cfg, {HybridAttentionType::NONE, HybridAttentionType::NONE});

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;

    auto cache_config = CacheConfigCreator::createWarmupConfig(cfg, parallelism_cfg, /*gen_num_per_cycle=*/0);
    EXPECT_EQ(cache_config.seq_size_per_block, 4);
    ASSERT_EQ(cache_config.groupNums(), 1);
    EXPECT_EQ(cache_config.group("full").spec->seq_size_per_block, 4);
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

TEST_F(HybridTypeCoordinatorCacheManagerTest, CreateConfigSupportsOneLinearIndependentPool) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/2);
    setHybridLayerDescs(cfg, {HybridAttentionType::LINEAR, HybridAttentionType::LINEAR});
    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;
    const auto config       = CacheConfigCreator::createWarmupConfig(cfg, parallelism_cfg, 0);
    ASSERT_EQ(config.groupNums(), 1);
    EXPECT_EQ(config.groupTags(), std::vector<std::string>{"linear"});
    EXPECT_EQ(config.group("linear").policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(config.layerIdsForGroup("linear"), std::vector<int>({0, 1}));
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, CreateConfigRejectsLinearDescriptorWithoutAttentionMetadata) {
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
        CacheConfigCreator::createWarmupConfig(cfg, parallelism_cfg, /*gen_num_per_cycle=*/0);
        FAIL() << "expected a linear-only single config to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("hybrid_attention_types size"), std::string::npos);
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, InitSupportsOnlyLinearGroupsWithIndependentPools) {
    auto cache_config = makeSimpleLinearCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto linear0 = makeLinearSpec("linear0", /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16, 1, 1);
    auto linear1 = makeLinearSpec("linear1", /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16, 1, 1);
    cache_config.fromGroupedSpecs(
        {linear0, linear1}, {{0}, {1}}, {CacheGroupType::LINEAR, CacheGroupType::LINEAR}, {"linear0", "linear1"});
    cache_config.finalizeBlockNums(/*baseline_block_num=*/4, RuntimeConfig{});
    ASSERT_EQ(cache_config.groupNums(), 2);

    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(cache_config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    EXPECT_NE(poolForTag(*allocator, "linear0"), poolForTag(*allocator, "linear1"));
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, TopologyRejectsSpecPolicyTypeMismatch) {
    auto config = makeSimpleLinearCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto groups      = config.topology().groups();
    auto layers      = config.topology().layers();
    groups[0].policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, TopologyRejectsUnknownLayerTag) {
    auto config = makeTinyHybridConfig();
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    layers[0].group_tags.push_back("missing");

    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, TopologyRejectsMissingLayerTagMapping) {
    auto config = makeTinyHybridConfig();
    auto groups = config.topology().groups();
    auto layers = config.topology().layers();
    layers[0].group_tags.clear();

    EXPECT_THROW(config.setTopology(std::move(groups), std::move(layers)), std::runtime_error);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, CreateHybridConfigUsesFirstSeenTagsWithFullAfterLinear) {
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
    auto cache_config       = CacheConfigCreator::createWarmupConfig(cfg, parallelism_cfg, /*gen_num_per_cycle=*/0);

    std::vector<std::string>    expected_tags{"linear0", "linear1", "full", "linear2"};
    std::vector<CacheGroupType> expected_types{
        CacheGroupType::LINEAR, CacheGroupType::LINEAR, CacheGroupType::FULL, CacheGroupType::LINEAR};
    std::vector<int> expected_full{3, 7};
    std::vector<int> expected_linear0{0, 1};
    std::vector<int> expected_linear1{2, 4};
    std::vector<int> expected_linear2{5, 6};

    ASSERT_EQ(cache_config.groupNums(), 4);
    EXPECT_EQ(publishedGroupTags(cache_config.topology()), expected_tags);
    EXPECT_EQ(publishedGroupTypes(cache_config.topology()), expected_types);
    EXPECT_EQ(cache_config.layerIdsForGroup("linear0"), expected_linear0);
    EXPECT_EQ(cache_config.layerIdsForGroup("linear1"), expected_linear1);
    EXPECT_EQ(cache_config.layerIdsForGroup("full"), expected_full);
    EXPECT_EQ(cache_config.layerIdsForGroup("linear2"), expected_linear2);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, CreateHybridConfigKeepsExplicitPhysicallyHeterogeneousLinearTags) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/4);
    setHybridLayerDescsWithTags(cfg,
                                {HybridAttentionType::LINEAR,
                                 HybridAttentionType::LINEAR,
                                 HybridAttentionType::NONE,
                                 HybridAttentionType::NONE},
                                {"recurrent_state", "convolution_state", "full", "full"});
    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;
    cfg.kv_cache_spec_descs[0][0].dtype                = DataType::TYPE_FP16;
    cfg.kv_cache_spec_descs[1][0].dtype                = DataType::TYPE_FP32;

    ParallelismConfig parallelism_cfg;
    auto              config = CacheConfigCreator::createWarmupConfig(cfg, parallelism_cfg, /*gen_num_per_cycle=*/0);

    ASSERT_EQ(config.groupNums(), 3);
    EXPECT_EQ(publishedGroupTags(config.topology()),
              (std::vector<std::string>{"recurrent_state", "convolution_state", "full"}));
    EXPECT_TRUE(config.group("recurrent_state").policy.enable_prefix_reuse);
    EXPECT_TRUE(config.group("convolution_state").policy.enable_prefix_reuse);
    EXPECT_EQ(config.group("recurrent_state").spec->memoryLayoutDType(), DataType::TYPE_FP16);
    EXPECT_EQ(config.group("convolution_state").spec->memoryLayoutDType(), DataType::TYPE_FP32);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, CreateHybridConfigRejectsDifferentLayoutsUnderOneLinearTag) {
    auto cfg = makeTinyModelConfig(/*num_layers=*/4);
    setHybridLayerDescsWithTags(cfg,
                                {HybridAttentionType::LINEAR,
                                 HybridAttentionType::LINEAR,
                                 HybridAttentionType::NONE,
                                 HybridAttentionType::NONE},
                                {"linear", "linear", "full", "full"});
    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;
    cfg.kv_cache_spec_descs[0][0].dtype                = DataType::TYPE_FP16;
    cfg.kv_cache_spec_descs[1][0].dtype                = DataType::TYPE_BF16;

    ParallelismConfig parallelism_cfg;
    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(cfg, parallelism_cfg, /*gen_num_per_cycle=*/0),
                 std::runtime_error);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, InitAndAddressLookupSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->seqSizePerBlock(), 4);
    EXPECT_EQ(allocator->totalBlocksNum(),
              (config.group("linear").block_num - 1) + (config.group("full1").block_num - 1));
    EXPECT_EQ(allocator->freeBlocksNum(),
              (config.group("linear").block_num - 1) + (config.group("full1").block_num - 1));

    const std::vector<SingleTypeCacheManagerPtr> groups = allocator->cacheGroups();
    ASSERT_GT(groups.size(), 1u);
    const std::vector<KVCachePoolMetricsSnapshot> snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_NE(poolForTag(*allocator, "linear"), poolForTag(*allocator, "full1"));
    for (const auto& snapshot : snapshots) {
        EXPECT_EQ(snapshot.used_blocks, snapshot.total_blocks - snapshot.free_blocks);
        EXPECT_EQ(snapshot.free_blocks, config.group(snapshot.pool_name).block_num - 1);
    }

    // Should be able to fetch address for any global layer and non-zero block id.
    auto addr0 = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    auto addr3 = allocator->convertIndexToAddr(/*layer_id=*/3, /*block_id=*/1);
    EXPECT_NE(addr0.kv_addr, nullptr);
    EXPECT_NE(addr3.kv_addr, nullptr);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, ConvertToGlobalLayerIdHybridNoMtp) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);

    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/0), 0u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/3), 3u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/4),
              std::numeric_limits<uint32_t>::max());

    // no mtp sub-model
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, ConvertToGlobalLayerIdHybridWithMtpSubConfigs) {
    auto config    = makeTinyHybridMtpConfigByCreateSpConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);

    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (size_t mtp_id = 0; mtp_id < config.mtp_sub_configs.size(); ++mtp_id) {
        const auto& sub = config.mtp_sub_configs[mtp_id];
        ASSERT_NE(sub, nullptr);
        ASSERT_EQ(sub->groupNums(), 2);
        std::vector<std::string> expected_tags{"linear", "full"};
        EXPECT_EQ(publishedGroupTags(sub->topology()), expected_tags);
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

TEST_F(HybridTypeCoordinatorCacheManagerTest, EagleMapsSoleDefaultFullDraftGroupToUniqueFullTargetGroup) {
    auto config = makeTinyHybridMtpConfigByCreateSpConfig(SP_TYPE_EAGLE, "default");

    ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
    const auto& sub_config = config.mtp_sub_configs[0];
    ASSERT_NE(sub_config, nullptr);
    EXPECT_EQ(publishedGroupTags(sub_config->topology()), publishedGroupTags(config.topology()));

    EXPECT_EQ(sub_config->groupForLayer(0, "full").tag, "full");
    EXPECT_EQ(sub_config->layerIdsForGroup(sub_config->topology().group("full").tag), std::vector<int>({0}));
    EXPECT_EQ(sub_config->group("full").spec->tag, "full");
    EXPECT_EQ(sub_config->group("full").spec->type, KVCacheSpecType::MultiHeadAttention);

    EXPECT_TRUE(sub_config->layerIdsForGroup(sub_config->topology().group("linear").tag).empty());

    auto manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());
    const auto layout = manager->getMTPModuleGroupedCacheLayerLayout(0);
    EXPECT_TRUE(layout.at("full", 0).kv_addr.defined());
    EXPECT_TRUE(layout.group("linear").empty());
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MtpMapsDefaultFullDraftGroupForEveryModule) {
    auto config = makeTinyHybridMtpConfigByCreateSpConfig(SP_TYPE_MTP, "default", /*gen_num=*/2);

    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    EXPECT_EQ(config.layerIdsForGroup(config.topology().group("full").tag), std::vector<int>({2, 3, 4, 5}));
    for (size_t module_index = 0; module_index < config.mtp_sub_configs.size(); ++module_index) {
        const auto& sub_config = config.mtp_sub_configs[module_index];
        ASSERT_NE(sub_config, nullptr);
        EXPECT_EQ(publishedGroupTags(sub_config->topology()), publishedGroupTags(config.topology()));
        EXPECT_EQ(sub_config->layerIdsForGroup(sub_config->topology().group("full").tag), std::vector<int>({0}));
        EXPECT_TRUE(sub_config->layerIdsForGroup(sub_config->topology().group("linear").tag).empty());
        EXPECT_EQ(config.groupForLayer(static_cast<int>(4 + module_index), "full").tag, "full");
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, CreateSpConfigPreservesQwenPackingAlignmentForFullMtpLayers) {
    auto score_model_cfg   = makeTinyModelConfig(/*num_layers=*/8);
    auto propose_model_cfg = makeTinyModelConfig(/*num_layers=*/1);

    setHybridLayerDescs(score_model_cfg,
                        {HybridAttentionType::LINEAR,
                         HybridAttentionType::LINEAR,
                         HybridAttentionType::LINEAR,
                         HybridAttentionType::NONE,
                         HybridAttentionType::LINEAR,
                         HybridAttentionType::LINEAR,
                         HybridAttentionType::LINEAR,
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

    CacheConfig config;
    ASSERT_NO_THROW({
        config                         = CacheConfigCreator::createConfig(score_model_cfg,
                                                  parallelism_cfg,
                                                  kv_cache_cfg,
                                                  sp_cfg,
                                                  &propose_model_cfg,
                                                  /*is_mtp=*/true,
                                                  /*is_eagle=*/false);
        const auto candidate_block_num = CacheConfigCreator::computeLocalBlockNum(
            config, score_model_cfg, runtime_cfg, kv_cache_cfg, parallelism_cfg, std::nullopt, sp_cfg);
        config = rtp_llm::test::finalizeCacheConfig(std::move(config), candidate_block_num);
    });

    EXPECT_EQ(publishedGroupTags(config.topology()), std::vector<std::string>({"linear", "full"}));
    EXPECT_EQ(config.layerIdsForGroup("full").size(), 4u);
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);

    EXPECT_EQ(config.layerIdsForGroup(config.topology().group("full").tag), std::vector<int>({3, 7, 8, 9}));
    EXPECT_EQ(config.groupForLayer(8, "full").tag, "full");
    EXPECT_EQ(config.groupForLayer(9, "full").tag, "full");

    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_NE(sub_config, nullptr);
        EXPECT_EQ(publishedGroupTags(sub_config->topology()), std::vector<std::string>({"linear", "full"}));
        EXPECT_EQ(sub_config->layerIdsForGroup(sub_config->topology().group("full").tag), std::vector<int>({0}));
        EXPECT_TRUE(sub_config->layerIdsForGroup(sub_config->topology().group("linear").tag).empty());
    }

    auto manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());
    auto allocator = std::dynamic_pointer_cast<CoordinatorCacheManager>(manager->coordinator_manager_);
    ASSERT_NE(allocator, nullptr);
    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    const auto full_pool_config =
        DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.topology().group("full"));
    const auto& full_layouts = full_pool_config.memory_layouts;
    ASSERT_EQ(full_layouts.size(), 3u);
    EXPECT_EQ(full_layouts[0].layer_num, 2u);
    EXPECT_EQ(full_layouts[1].layer_num, 1u);
    EXPECT_EQ(full_layouts[2].layer_num, 1u);
    const auto linear_pool_config =
        DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.topology().group("linear"));
    const auto& linear_layouts = linear_pool_config.memory_layouts;
    ASSERT_EQ(linear_layouts.size(), 1u);
    EXPECT_EQ(linear_layouts[0].layer_num, 6u);
    EXPECT_EQ(poolForTag(*allocator, "full")->getTotalSizeBytes(), full_pool_config.total_size_bytes);
    EXPECT_EQ(poolForTag(*allocator, "linear")->getTotalSizeBytes(), linear_pool_config.total_size_bytes);
    EXPECT_EQ(poolForTag(*allocator, "full")->allLayerCacheBase().size(), 4u);
    EXPECT_EQ(poolForTag(*allocator, "linear")->allLayerCacheBase().size(), 6u);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MergeMtpAliasesCompatibleDefaultMlaGroup) {
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
    EXPECT_EQ(publishedGroupTags(sub_config->topology()), std::vector<std::string>({"full"}));
    EXPECT_EQ(sub_config->group("full").spec->type, KVCacheSpecType::MultiHeadLatentAttention);
    EXPECT_EQ(sub_config->group("full").spec->tag, "full");
    EXPECT_EQ(sub_config->layerIdsForGroup("full"), std::vector<int>({0}));
    EXPECT_EQ(main_config.layerIdsForGroup("full"), std::vector<int>({0, 1}));
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MergeMtpRejectsAmbiguousDefaultFullGroupAlias) {
    CacheConfig main_config;
    main_config.layer_num = 2;
    main_config.fromGroupedSpecs(
        {makeMhaSpec("full0", 4, DataType::TYPE_FP16, 1, 1), makeMhaSpec("full1", 4, DataType::TYPE_FP16, 1, 1)},
        {{0}, {1}},
        {CacheGroupType::FULL, CacheGroupType::FULL},
        {"full0", "full1"});

    auto propose_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);

    try {
        main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/2);
        FAIL() << "expected an ambiguous default FULL group mapping to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ambiguous default FULL group mapping"), std::string::npos);
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MergeMtpDoesNotAliasDefaultFullGroupToLinearTarget) {
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

TEST_F(HybridTypeCoordinatorCacheManagerTest, MergeMtpRejectsIncompatibleDefaultFullGroupAlias) {
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

    auto different_group_stride = compatible_propose;
    setGroupBlockLayout(different_group_stride,
                        {different_group_stride.group("default").tag},
                        {different_group_stride.group("default").block_num},
                        {different_group_stride.group("default").kvBlockStrideBytes() + 1},
                        {different_group_stride.group("default").kvScaleStrideBytes()});
    expect_no_compatible_alias(target, different_group_stride);

    auto target_with_different_policy    = target;
    auto target_policy                   = target_with_different_policy.group("full").policy;
    target_policy.explicit_block_num     = 2;
    target_policy.charge_to_paged_budget = true;
    setTestGroupPolicies(target_with_different_policy, {target_policy});
    expect_no_compatible_alias(target_with_different_policy, compatible_propose);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MergeMtpPrefersExactDefaultGroupMatch) {
    CacheConfig main_config;
    main_config.layer_num = 2;
    main_config.fromGroupedSpecs(
        {makeMhaSpec("default", 4, DataType::TYPE_FP16, 1, 1), makeMhaSpec("aux", 4, DataType::TYPE_FP16, 1, 1)},
        {{0}, {1}},
        {CacheGroupType::FULL, CacheGroupType::FULL},
        {"default", "aux"});

    auto propose_config = makeSingleLayerCacheConfig(makeMhaSpec("default",
                                                                 /*tokens_per_block=*/4,
                                                                 DataType::TYPE_FP16,
                                                                 /*local_head_num_kv=*/1,
                                                                 /*size_per_head=*/1),
                                                     CacheGroupType::FULL);

    const auto sub_config = main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/2);
    ASSERT_NE(sub_config, nullptr);
    EXPECT_EQ(publishedGroupTags(sub_config->topology()), std::vector<std::string>({"default", "aux"}));
    EXPECT_EQ(sub_config->layerIdsForGroup(sub_config->topology().group("default").tag), std::vector<int>({0}));
    EXPECT_TRUE(sub_config->layerIdsForGroup(sub_config->topology().group("aux").tag).empty());
    EXPECT_EQ(main_config.layerIdsForGroup(main_config.topology().group("default").tag), std::vector<int>({0, 2}));
    EXPECT_EQ(main_config.layerIdsForGroup(main_config.topology().group("aux").tag), std::vector<int>({1}));
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MergeMtpDoesNotAliasDefaultLinearProposeGroup) {
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
        EXPECT_NE(std::string(e.what()).find("unmapped draft cache group tag=default"), std::string::npos);
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MergeMtpAliasErrorIdentifiesSourceAndTargetTags) {
    auto main_config = makeSingleGroupCacheConfig(
        makeMhaSpec("full", /*tokens_per_block=*/4, DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/1),
        CacheGroupType::FULL,
        /*layer_num=*/2,
        /*block_num=*/4);
    auto propose_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, DataType::TYPE_FP16);
    propose_config.layer_num = 3;

    try {
        main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/2);
        FAIL() << "expected incomplete aliased source layers to be rejected";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("source_tag=default"), std::string::npos);
        EXPECT_NE(message.find("target_tag=full"), std::string::npos);
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MergeMtpDoesNotAliasMultiGroupProposeConfig) {
    auto main_config = makeSingleLayerCacheConfig(
        makeMhaSpec("full", /*tokens_per_block=*/4, DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/1),
        CacheGroupType::FULL);

    CacheConfig propose_config;
    propose_config.layer_num = 1;
    propose_config.fromGroupedSpecs(
        {makeMhaSpec("default", 4, DataType::TYPE_FP16, 1, 1), makeMhaSpec("aux", 4, DataType::TYPE_FP16, 1, 1)},
        {{0}, {0}},
        {CacheGroupType::FULL, CacheGroupType::FULL},
        {"default", "aux"});

    try {
        main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/1);
        FAIL() << "expected a multi-group propose config not to use the default alias";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("unmapped draft cache group tag=default"), std::string::npos);
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MergeMtpAllowsUnequalIndependentGroupLayerCounts) {
    CacheConfig main_config;
    main_config.layer_num = 5;
    main_config.fromGroupedSpecs(
        {makeMhaSpec("full", 4, DataType::TYPE_FP16, 1, 1), makeLinearSpec("linear", 4, DataType::TYPE_FP16, 1, 1)},
        {{0, 1, 2}, {3, 4}},
        {CacheGroupType::FULL, CacheGroupType::LINEAR},
        {"full", "linear"});

    auto propose_config = makeSimpleLinearCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    auto sub_config = main_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/5);
    ASSERT_NE(sub_config, nullptr);
    EXPECT_EQ(main_config.layerIdsForGroup("linear"), (std::vector<int>{3, 4, 5}));
    EXPECT_EQ(sub_config->layerIdsForGroup("linear"), (std::vector<int>{0}));
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MergeMtpRejectsPartialSourceAndDerivesOrderedLayers) {
    auto main_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);

    CacheConfig partial_source;
    partial_source.layer_num = 2;
    partial_source.fromGroupedSpecs(
        {makeMhaSpec("default", 4, DataType::TYPE_FP16, 1, 1), makeMhaSpec("aux", 4, DataType::TYPE_FP16, 1, 1)},
        {{0}, {1}},
        {CacheGroupType::FULL, CacheGroupType::FULL},
        {"default", "aux"});
    EXPECT_THROW(main_config.mergeMTPModule(partial_source, /*module_index=*/0, /*main_layer_num=*/2),
                 std::runtime_error);

    auto reordered_source = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    EXPECT_EQ(reordered_source.layerIdsForGroup("default"), (std::vector<int>{0, 1}));
    EXPECT_NO_THROW(main_config.mergeMTPModule(reordered_source, /*module_index=*/0, /*main_layer_num=*/2));
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, MtpPhysicalSlotsDoNotAliasMainSlots) {
    auto config    = makeTinyHybridMtpConfigByCreateSpConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
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

TEST_F(HybridTypeCoordinatorCacheManagerTest, MtpLayoutProjectionRecountsActiveLayersAndKeepsEmptyPlaceholder) {
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

TEST_F(HybridTypeCoordinatorCacheManagerTest, GetNeedBlocksUsesGroupGetNeedBlocksAndReuseFlag) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // batch=2, seq_len=12 (3 resources), reserve_step=2
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/12, /*seq_size_per_block=*/4);
    token_ids->setReserveStep(2);

    // Reuse disabled: linear group keeps only tail for common blocks; reserve_step contributes extra blocks.
    // full group contributes common=3, extra=1.
    {
        auto       batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100, 101, 102, 103});
        MallocInfo info{batch_res, token_ids};
        info.enable_cache_lookup = false;
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
        info.enable_cache_lookup = true;
        info.reuse_cache         = true;
        // full: common=3 extra=1
        // linear: common=count(0,3]=2, extra=reserve_step-1(=1)
        // common_total = 3 + 2 = 5
        // extra_total  = 1 + 1 = 2
        // total = 5 + 2*2 = 9
        EXPECT_EQ(allocator->getNeedBlocks(info), 9);
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, TieredJoinedLoadMapsTargetsAcrossFullAndLinearGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);

    KVCacheConfig tiered_config;
    tiered_config.enable_memory_cache  = true;
    tiered_config.memory_cache_size_mb = 1;
    allocator->setBlockTreeCacheConfigForTest(std::move(tiered_config));
    ASSERT_TRUE(allocator->init());

    const auto& cache = allocator->blockTreeCacheOwner();
    ASSERT_NE(cache, nullptr);
    auto transfer_engine = std::make_shared<PausableHybridPerRankBlockTransferEngine>(cache->groupSets());
    cache->transfer_dispatcher_->per_rank_engine_ = transfer_engine;
    ScopedHybridTransferRelease transfer_release(transfer_engine);

    const CacheKeysType                        cached_keys{100, 101};
    std::vector<std::vector<GroupSetResource>> slots(cached_keys.size(),
                                                     std::vector<GroupSetResource>(cache->groupSets().size()));
    for (const GroupSetPtr& group_set : cache->groupSets()) {
        ASSERT_NE(group_set->hostPool(), nullptr);
        for (size_t path_index = 0; path_index < cached_keys.size(); ++path_index) {
            const BlockIdxType source_block = group_set->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            ASSERT_FALSE(isNullBlockIdx(source_block));
            slots[path_index][group_set->groupSetId()].host_block = source_block;
        }
    }
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, cached_keys, slots));

    const CacheKeysType request_keys{100, 101, 102};
    auto                first_resource = makeBatchResource(/*batch_size=*/1, config, request_keys);
    auto       first_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/4);
    MallocInfo first_info{first_resource, first_tokens};
    first_info.enable_cache_lookup = true;
    first_info.reuse_cache         = true;
    MallocResult first_result      = allocator->malloc(first_info);
    ASSERT_TRUE(first_result.success);
    const auto first_context = std::dynamic_pointer_cast<LoadAsyncContext>(first_result.async_context);
    ASSERT_NE(first_context, nullptr);
    EXPECT_EQ(first_result.reuse_len, 0);

    EXPECT_EQ(first_resource->cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(first_context->matchedBlocks(), cached_keys.size());
    EXPECT_EQ(first_context->matchedBlocks(Tier::HOST), cached_keys.size());
    ASSERT_TRUE(transfer_engine->waitUntilEnteredFor(std::chrono::seconds(5)));

    const size_t expected_submit_count = cache->groupSets().size();

    auto       second_resource = makeBatchResource(/*batch_size=*/1, config, request_keys);
    auto       second_tokens   = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/4);
    MallocInfo second_info{second_resource, second_tokens};
    second_info.enable_cache_lookup = true;
    second_info.reuse_cache         = true;
    MallocResult second_result      = allocator->malloc(second_info);
    ASSERT_TRUE(second_result.success);
    const auto second_context = std::dynamic_pointer_cast<LoadAsyncContext>(second_result.async_context);
    ASSERT_NE(second_context, nullptr);
    EXPECT_EQ(second_result.reuse_len, 0);

    EXPECT_EQ(second_resource->cacheResource(0).deviceReuseBlockNum(), 0u);
    ASSERT_EQ(second_context->loadDescs().size(), first_context->loadDescs().size());
    ASSERT_TRUE(std::all_of(second_context->joinedLoads().begin(),
                            second_context->joinedLoads().end(),
                            [](bool joined) { return joined; }));
    EXPECT_EQ(transfer_engine->submitCount(), expected_submit_count);

    for (size_t desc_index = 0; desc_index < second_context->loadDescs().size(); ++desc_index) {
        const TransferDescriptor& joined_desc = second_context->loadDescs()[desc_index];
        const auto                first_desc  = std::find_if(first_context->loadDescs().begin(),
                                             first_context->loadDescs().end(),
                                             [&joined_desc](const TransferDescriptor& desc) {
                                                 return desc.group_set_id == joined_desc.group_set_id
                                                        && desc.path_index == joined_desc.path_index;
                                             });
        ASSERT_NE(first_desc, first_context->loadDescs().end());
        EXPECT_EQ(joined_desc.target_blocks, first_desc->target_blocks);
    }

    ASSERT_EQ(config.group("linear").policy.group_type, CacheGroupType::LINEAR);
    ASSERT_EQ(config.group("full1").policy.group_type, CacheGroupType::FULL);
    const auto targetFor =
        [&](const std::shared_ptr<LoadAsyncContext>& context, const std::string& tag, size_t path_index) {
            for (const TransferDescriptor& desc : context->loadDescs()) {
                if (desc.path_index != path_index) {
                    continue;
                }
                const auto& group_tags = cache->groupSets()[desc.group_set_id]->groupTags();
                const auto  group_it   = std::find(group_tags.begin(), group_tags.end(), tag);
                if (group_it != group_tags.end()) {
                    return desc.target_blocks[static_cast<size_t>(group_it - group_tags.begin())];
                }
            }
            return NULL_BLOCK_IDX;
        };

    const BlockIdxType full_target_0   = targetFor(first_context, "full1", 0);
    const BlockIdxType full_target_1   = targetFor(first_context, "full1", 1);
    const BlockIdxType linear_target_1 = targetFor(first_context, "linear", 1);
    ASSERT_FALSE(isNullBlockIdx(full_target_0));
    ASSERT_FALSE(isNullBlockIdx(full_target_1));
    ASSERT_FALSE(isNullBlockIdx(linear_target_1));

    ASSERT_EQ(first_resource->blocks(0, "full1").size(), 3u);
    ASSERT_EQ(second_resource->blocks(0, "full1").size(), 3u);
    EXPECT_EQ(first_resource->blocks(0, "full1")[0], full_target_0);
    EXPECT_EQ(first_resource->blocks(0, "full1")[1], full_target_1);
    EXPECT_EQ(second_resource->blocks(0, "full1")[0], full_target_0);
    EXPECT_EQ(second_resource->blocks(0, "full1")[1], full_target_1);

    ASSERT_EQ(first_resource->blocks(0, "linear").size(), 3u);
    ASSERT_EQ(second_resource->blocks(0, "linear").size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(first_resource->blocks(0, "linear")[0]));
    EXPECT_TRUE(isNullBlockIdx(second_resource->blocks(0, "linear")[0]));
    EXPECT_EQ(first_resource->blocks(0, "linear")[1], linear_target_1);
    EXPECT_EQ(second_resource->blocks(0, "linear")[1], linear_target_1);

    transfer_engine->release();
    first_context->waitDone();
    second_context->waitDone();
    ASSERT_TRUE(first_context->success());
    ASSERT_TRUE(second_context->success());
    EXPECT_EQ(transfer_engine->submitCount(), expected_submit_count);

    for (const TransferDescriptor& desc : first_context->loadDescs()) {
        const GroupSetPtr& group_set = cache->groupSets()[desc.group_set_id];
        ASSERT_EQ(desc.source_blocks.size(), 1u);
        EXPECT_FALSE(group_set->hostPool()->isAllocated(desc.source_blocks.front()));
        ASSERT_EQ(desc.target_blocks.size(), group_set->devicePools().size());
        for (size_t member_group_id = 0; member_group_id < desc.target_blocks.size(); ++member_group_id) {
            EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(desc.target_blocks[member_group_id]), 3u);
        }
    }

    allocator->free(FreeInfo{first_resource, first_tokens});
    allocator->free(FreeInfo{second_resource, second_tokens});
    for (const TransferDescriptor& desc : first_context->loadDescs()) {
        const GroupSetPtr& group_set = cache->groupSets()[desc.group_set_id];
        for (size_t member_group_id = 0; member_group_id < desc.target_blocks.size(); ++member_group_id) {
            EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(desc.target_blocks[member_group_id]), 1u);
        }
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, DeviceLoadSourceMatchRefBecomesRequestRef) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);

    KVCacheConfig tiered_config;
    tiered_config.enable_memory_cache  = true;
    tiered_config.memory_cache_size_mb = 1;
    allocator->setBlockTreeCacheConfigForTest(std::move(tiered_config));
    ASSERT_TRUE(allocator->init());

    const auto& cache = allocator->blockTreeCacheOwner();
    ASSERT_NE(cache, nullptr);
    auto transfer_engine = std::make_shared<PausableHybridPerRankBlockTransferEngine>(cache->groupSets());
    cache->transfer_dispatcher_->per_rank_engine_ = transfer_engine;
    ScopedHybridTransferRelease transfer_release(transfer_engine);

    const CacheKeysType                                                         cached_keys{100, 101};
    std::vector<std::vector<GroupSetResource>>                                  slots(cached_keys.size(),
                                                     std::vector<GroupSetResource>(cache->groupSets().size()));
    std::vector<std::pair<GroupSetPtr, block_tree_cache_test::MultiNodeBlocks>> device_resources;
    for (const GroupSetPtr& group_set : cache->groupSets()) {
        const size_t group_set_id = group_set->groupSetId();
        if (group_set->groupType() == CacheGroupType::FULL) {
            auto blocks = block_tree_cache_test::allocateDeviceBlocksForTest(*group_set, cached_keys.size());
            ASSERT_EQ(blocks.size(), cached_keys.size());
            for (size_t path_index = 0; path_index < cached_keys.size(); ++path_index) {
                slots[path_index][group_set_id].device_blocks = blocks[path_index];
            }
            device_resources.emplace_back(group_set, std::move(blocks));
        } else {
            ASSERT_NE(group_set->hostPool(), nullptr);
            for (size_t path_index = 0; path_index < cached_keys.size(); ++path_index) {
                const BlockIdxType source_block = group_set->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
                ASSERT_FALSE(isNullBlockIdx(source_block));
                slots[path_index][group_set_id].host_block = source_block;
            }
        }
    }
    ASSERT_EQ(device_resources.size(), 1u);
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache, cached_keys, slots));
    for (const auto& [group_set, blocks] : device_resources) {
        block_tree_cache_test::unreferenceDeviceBlocksForTest(*group_set, blocks);
    }

    const CacheKeysType request_keys{100, 101, 102};
    auto                request_resource = makeBatchResource(/*batch_size=*/1, config, request_keys);
    auto       request_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{request_resource, request_tokens};
    malloc_info.enable_cache_lookup = true;
    malloc_info.reuse_cache         = true;

    MallocResult malloc_result = allocator->malloc(malloc_info);
    ASSERT_TRUE(malloc_result.success);
    const auto load_context = std::dynamic_pointer_cast<LoadAsyncContext>(malloc_result.async_context);
    ASSERT_NE(load_context, nullptr);
    ASSERT_TRUE(transfer_engine->waitUntilEnteredFor(std::chrono::seconds(5)));

    size_t device_source_descs = 0;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        const TransferDescriptor& desc = load_context->loadDescs()[desc_index];
        if (desc.source_tier != Tier::DEVICE || load_context->joinedLoads()[desc_index]) {
            continue;
        }
        ++device_source_descs;
        const GroupSetPtr& group_set = cache->groupSets()[desc.group_set_id];
        ASSERT_EQ(desc.source_blocks.size(), group_set->devicePools().size());
        for (size_t member_group_id = 0; member_group_id < desc.source_blocks.size(); ++member_group_id) {
            EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(desc.source_blocks[member_group_id]), 2u);
        }
    }
    EXPECT_EQ(device_source_descs, cached_keys.size());

    transfer_engine->release();
    load_context->waitDone();
    ASSERT_TRUE(load_context->success());
    allocator->free(FreeInfo{request_resource, request_tokens});

    for (const auto& [group_set, blocks] : device_resources) {
        for (size_t path_index = 0; path_index < blocks.size(); ++path_index) {
            for (size_t member_group_id = 0; member_group_id < blocks[path_index].size(); ++member_group_id) {
                EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(blocks[path_index][member_group_id]), 1u);
            }
        }
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, JointReuseUsesFullPrefixAndLinearTailOnly) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // Config order: group_id=0 linear, group_id=1 full.
    const int linear_group_id = 0;
    const int full_group_id   = 1;

    // Seed through the allocator's normal insert path. First capture the resource
    // dependency chain, then deliberately replace it with inconsistent metadata:
    // BlockTree must derive topology only from the ordered cache keys.
    const CacheKeysType seed_keys{100, 101};
    auto                seed_resource = makeBatchResource(/*batch_size=*/1, config, seed_keys);
    seed_resource->cacheResource(0).setCacheKeys(seed_keys);
    const BlockDependenciesType generated_dependencies = seed_resource->cacheResource(0).blockDependencies();
    ASSERT_EQ(generated_dependencies.size(), seed_keys.size());
    for (size_t index = 0; index < generated_dependencies.size(); ++index) {
        EXPECT_EQ(generated_dependencies[index].ordinal, index);
        EXPECT_EQ(generated_dependencies[index].has_parent, index > 0);
        if (index > 0) {
            EXPECT_EQ(generated_dependencies[index].parent_key, seed_keys[index - 1]);
        }
    }

    std::vector<BlockIndicesType> seeded_blocks(static_cast<size_t>(config.groupNums()));
    const auto                    cache_groups = allocator->cacheGroups();
    ASSERT_EQ(cache_groups.size(), seeded_blocks.size());
    for (size_t group_id = 0; group_id < cache_groups.size(); ++group_id) {
        const auto allocated = cache_groups[group_id]->blockPool()->malloc(seed_keys.size());
        ASSERT_TRUE(allocated.has_value());
        ASSERT_EQ(allocated->size(), seed_keys.size());
        seeded_blocks[group_id] = *allocated;
        cache_groups[group_id]->blockPool()->incRef(seeded_blocks[group_id]);
        seed_resource->setBatchBlocks(0, config.groupTags()[group_id], seeded_blocks[group_id]);
    }
    seed_resource->cacheResource(0).setBlockDependencies(
        {BlockDependency{true, 9999, 41}, BlockDependency{false, 0, 7}});
    {
        size_t resident_prefix_length = 0;
        allocator->insertIntoCache(InsertInfo{seed_resource, nullptr, /*is_resident=*/false}, resident_prefix_length);
    }
    for (size_t group_id = 0; group_id < cache_groups.size(); ++group_id) {
        cache_groups[group_id]->unreference(seeded_blocks[group_id]);
    }

    const auto tree_path = allocator->blockTreeCacheOwner()->tree()->findNode(seed_keys);
    ASSERT_EQ(tree_path.size(), seed_keys.size());
    for (size_t index = 0; index < tree_path.size(); ++index) {
        EXPECT_EQ(tree_path[index]->cache_key, seed_keys[index]);
        if (generated_dependencies[index].has_parent) {
            ASSERT_EQ(tree_path[index]->parent, tree_path[index - 1]);
            EXPECT_EQ(tree_path[index]->parent->cache_key, generated_dependencies[index].parent_key);
        } else {
            EXPECT_EQ(tree_path[index]->parent, allocator->blockTreeCacheOwner()->tree()->root());
        }
    }
    auto seeded_match = allocator->blockTreeCacheOwner()->match(seed_keys);
    ASSERT_EQ(seeded_match.matched_device_blocks, seed_keys.size());
    block_tree_cache_test::releaseRequestRefsForTest(*allocator->blockTreeCacheOwner(),
                                                     seeded_match.matched_device_resources);

    const auto& full_blocks   = seeded_blocks[static_cast<size_t>(full_group_id)];
    const auto& linear_blocks = seeded_blocks[static_cast<size_t>(linear_group_id)];

    // Request has 4 keys, but allocator drops the last for matching.
    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    // Enable device cache reuse for joint match.

    // seq_len=12 => 3 resources (4 tokens per block).
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_cache_lookup = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Full group: should reuse the first 2 blocks and allocate the third.
    const auto& full_out = batch_res->blocks(0, "full1");
    ASSERT_EQ(full_out.size(), 3u);
    EXPECT_EQ(full_out[0], full_blocks[0]);
    EXPECT_EQ(full_out[1], full_blocks[1]);
    EXPECT_FALSE(isNullBlockIdx(full_out[2]));

    // Linear group: only the tail resource of the reused prefix is filled; earlier resources stay NULL.
    const auto& linear_out = batch_res->blocks(0, "linear");
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_EQ(linear_out[1], linear_blocks.back());  // reused tail at pos=1
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));     // allocated tail for common length
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, DisableReuseKeepsOnlyLinearTailOnInitMalloc) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    // Disable device cache reuse.

    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_cache_lookup = false;
    info.reuse_cache         = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Linear group should keep only the tail block across common length resources.
    const auto& linear_out = batch_res->blocks(0, "linear");
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, DisableDeviceCacheSkipsReuseMatchAndAllocatesOnlyLinearTail) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // Config order: group_id=0 linear, group_id=1 full.

    // Seed a complete tree path. Tree holds keep the cached blocks unavailable
    // for fresh allocation while device-cache matching is disabled below.
    CacheKeysType full_keys = {100, 101, 102};
    const auto    seeded    = seedCompleteBlockTreePath(allocator, full_keys);
    ASSERT_TRUE(seeded.success);
    const auto& full_blocks = seeded.blocks_by_tag.at("full1");
    ASSERT_EQ(full_blocks.size(), 3u);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    // Disable device cache reuse: allocator should skip reuse match even if cache exists.

    auto token_ids =
        makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);  // 3 resources

    MallocInfo info{batch_res, token_ids};
    info.enable_cache_lookup = false;
    info.reuse_cache         = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Device cache disabled => must not reuse match.
    EXPECT_EQ(result.reuse_len, 0);

    // Full group should allocate fresh blocks (not reuse cached ones).
    const auto& full_out = batch_res->blocks(0, "full1");
    ASSERT_EQ(full_out.size(), 3u);
    EXPECT_FALSE(isNullBlockIdx(full_out[0]));
    EXPECT_FALSE(isNullBlockIdx(full_out[1]));
    EXPECT_FALSE(isNullBlockIdx(full_out[2]));
    EXPECT_NE(full_out[0], full_blocks[0]);
    EXPECT_NE(full_out[1], full_blocks[1]);
    EXPECT_NE(full_out[2], full_blocks[2]);

    // Linear group keeps only tail block (others NULL) when reuse is disabled.
    const auto& linear_out = batch_res->blocks(0, "linear");
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
    EXPECT_EQ(countValidBlocks(linear_out), 1u);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, PreparedLoadReclaimsIndependentPoolTreeCandidatesBeforeRetry) {
    auto config = makeTinyHybridConfig();
    setGroupBlockCounts(config, 5, 5);
    auto          allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    KVCacheConfig tree_config;
    // Exercise on-demand reclamation, without background watermark eviction
    // removing the seeded path before the capacity check begins.
    tree_config.block_tree_device_evict_low_watermark_ratio  = 0;
    tree_config.block_tree_device_evict_high_watermark_ratio = 0;
    allocator->setBlockTreeCacheConfigForTest(tree_config);
    ASSERT_TRUE(allocator->init());
    allocator->setReserveBlocksNum(0);

    const auto seeded = seedCompleteBlockTreePath(allocator, CacheKeysType{100, 101, 102, 103});
    ASSERT_TRUE(seeded.success);
    ASSERT_EQ(allocator->freeBlocksNum(), 0u);
    ASSERT_GT(allocator->blockTreeCacheOwner()->getStats().device_heap_total_size, 0u);

    auto       resource  = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{200});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache = true;
    malloc_info.verbose     = false;

    // A one-token-block request needs one LINEAR and one FULL physical block.
    // Both independent pools are full of reclaimable tree blocks; prepared
    // admission must reclaim in each pool instead of retrying forever.
    EXPECT_EQ(allocator->preparedReserveStatusForTest(malloc_info, /*reserve_blocks=*/0, {{}, {}}), MallocStatus::NONE);
    EXPECT_GE(poolForTag(*allocator, "linear")->freeBlocksNum(), 1u);
    EXPECT_GE(poolForTag(*allocator, "full1")->freeBlocksNum(), 1u);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, DeviceMissReclaimsIndependentPoolTreeCandidatesForReserve) {
    auto config = makeTinyHybridConfig();
    setGroupBlockCounts(config, 6, 6);
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    // Split the total reserve equally: one reserved block in each pool.
    allocator->setReserveBlocksNum(2);

    const auto seeded = seedCompleteBlockTreePath(allocator, CacheKeysType{100, 101, 102, 103});
    ASSERT_TRUE(seeded.success);
    ASSERT_EQ(allocator->freeBlocksNum(), 2u);

    auto       resource  = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{200});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache = true;
    malloc_info.verbose     = false;

    EXPECT_EQ(allocator->preparedReserveStatusForTest(
                  malloc_info, /*reserve_blocks=*/2, {{}, {}}, /*has_load_context=*/false),
              MallocStatus::NONE);
    EXPECT_GE(poolForTag(*allocator, "linear")->freeBlocksNum(), 2u);
    EXPECT_GE(poolForTag(*allocator, "full1")->freeBlocksNum(), 2u);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, UpdateKVBlockForksAliasedBlocksAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto                   linear_pool   = poolForTag(*allocator, "linear");
    auto                   full_pool     = poolForTag(*allocator, "full1");
    const size_t           free_before   = allocator->freeBlocksNum();
    const auto             linear_blocks = allocateReferencedBlocks(linear_pool, 3);
    const auto             full_blocks   = allocateReferencedBlocks(full_pool, 3);
    const BlockIndicesType blocks{
        linear_blocks[0], linear_blocks[1], full_blocks[0], full_blocks[1], linear_blocks[2], full_blocks[2]};
    ASSERT_EQ(allocator->freeBlocksNum(), free_before - 6);

    auto batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100, 101}, /*reorder_resource=*/true);
    batch_res->mutableBlockIds(/*batch_id=*/0, "linear").assign({blocks[0], NULL_BLOCK_IDX, blocks[1]});
    batch_res->mutableBlockIds(/*batch_id=*/0, "full1").assign({blocks[2], blocks[3]});
    batch_res->mutableBlockIds(/*batch_id=*/1, "linear").assign({blocks[4]});
    batch_res->mutableBlockIds(/*batch_id=*/1, "full1").assign({blocks[5]});

    std::vector<TaggedBlockIdPair> update_mapping;
    ASSERT_TRUE(allocator->updateKVBlock(batch_res,
                                         /*block_src_batch=*/std::vector<int>{0, 0},
                                         /*copy_last_block=*/false,
                                         update_mapping));

    EXPECT_TRUE(update_mapping.empty());
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 4) << "unused old batch blocks should be released";
    ASSERT_EQ(batch_res->batchSize(), 2);
    EXPECT_EQ(batch_res->cacheKeys(0), (CacheKeysType{100, 101}));
    EXPECT_EQ(batch_res->cacheKeys(1), (CacheKeysType{100, 101}));
    EXPECT_EQ(batch_res->blocks(0, "linear"), (BlockIndicesType{blocks[0], NULL_BLOCK_IDX, blocks[1]}));
    EXPECT_EQ(batch_res->blocks(0, "full1"), (BlockIndicesType{blocks[2], blocks[3]}));
    EXPECT_EQ(batch_res->blocks(1, "linear"), (BlockIndicesType{blocks[0], NULL_BLOCK_IDX, blocks[1]}));
    EXPECT_EQ(batch_res->blocks(1, "full1"), (BlockIndicesType{blocks[2], blocks[3]}));

    allocator->free(FreeInfo{batch_res, nullptr});
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, UpdateKVBlockCopyLastBlockAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto                   linear_pool   = poolForTag(*allocator, "linear");
    auto                   full_pool     = poolForTag(*allocator, "full1");
    const size_t           free_before   = allocator->freeBlocksNum();
    const auto             linear_blocks = allocateReferencedBlocks(linear_pool, 3);
    const auto             full_blocks   = allocateReferencedBlocks(full_pool, 3);
    const BlockIndicesType blocks{
        linear_blocks[0], linear_blocks[1], full_blocks[0], full_blocks[1], linear_blocks[2], full_blocks[2]};
    ASSERT_EQ(allocator->freeBlocksNum(), free_before - 6);

    auto batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100, 101}, /*reorder_resource=*/true);
    batch_res->mutableBlockIds(/*batch_id=*/0, "linear").assign({blocks[0], NULL_BLOCK_IDX, blocks[1]});
    batch_res->mutableBlockIds(/*batch_id=*/0, "full1").assign({blocks[2], blocks[3]});
    batch_res->mutableBlockIds(/*batch_id=*/1, "linear").assign({blocks[4]});
    batch_res->mutableBlockIds(/*batch_id=*/1, "full1").assign({blocks[5]});

    std::vector<TaggedBlockIdPair> update_mapping{{"stale", 1, 2}};
    ASSERT_TRUE(allocator->updateKVBlock(batch_res,
                                         /*block_src_batch=*/std::vector<int>{0, 0},
                                         /*copy_last_block=*/true,
                                         update_mapping));

    ASSERT_EQ(update_mapping.size(), 2u);
    EXPECT_EQ(update_mapping[0].tag, "linear");
    EXPECT_EQ(update_mapping[1].tag, "full1");
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 6);
    ASSERT_EQ(batch_res->batchSize(), 2);
    EXPECT_EQ(batch_res->cacheKeys(0), (CacheKeysType{100, 101}));
    EXPECT_EQ(batch_res->cacheKeys(1), (CacheKeysType{100, 101}));

    const auto& forked_group0 = batch_res->blocks(0, "linear");
    const auto& moved_group0  = batch_res->blocks(1, "linear");
    const auto& forked_group1 = batch_res->blocks(0, "full1");
    const auto& moved_group1  = batch_res->blocks(1, "full1");
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

TEST_F(HybridTypeCoordinatorCacheManagerTest, UpdateKVBlockReservationFailureLeavesResourceUnchanged) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto         linear_pool   = poolForTag(*allocator, "linear");
    auto         full_pool     = poolForTag(*allocator, "full1");
    const size_t free_before   = allocator->freeBlocksNum();
    const auto   linear_blocks = allocateReferencedBlocks(linear_pool, 1);
    const auto   full_blocks   = allocateReferencedBlocks(full_pool, full_pool->freeBlocksNum());
    ASSERT_GT(linear_pool->freeBlocksNum(), 0u);
    ASSERT_EQ(full_pool->freeBlocksNum(), 0u);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100}, /*reorder_resource=*/true);
    batch_res->mutableBlockIds(/*batch_id=*/0, "linear").assign(linear_blocks);
    batch_res->mutableBlockIds(/*batch_id=*/0, "full1").assign({full_blocks[0]});
    const auto          before_batch0_group0 = batch_res->blocks(0, "linear");
    const auto          before_batch0_group1 = batch_res->blocks(0, "full1");
    const auto          linear_free_before   = linear_pool->freeBlocksNum();
    const auto          full_free_before     = full_pool->freeBlocksNum();
    const auto          linear_ref_before    = linear_pool->refCount(linear_blocks[0]);
    std::vector<size_t> full_refs_before;
    for (auto block : full_blocks) {
        full_refs_before.push_back(full_pool->refCount(block));
    }

    std::vector<TaggedBlockIdPair> update_mapping{{"stale", 1, 2}};
    EXPECT_FALSE(allocator->updateKVBlock(batch_res,
                                          /*block_src_batch=*/std::vector<int>{0, 0},
                                          /*copy_last_block=*/true,
                                          update_mapping));

    EXPECT_TRUE(update_mapping.empty());
    EXPECT_EQ(batch_res->batchSize(), 1);
    EXPECT_EQ(batch_res->blocks(0, "linear"), before_batch0_group0);
    EXPECT_EQ(batch_res->blocks(0, "full1"), before_batch0_group1);
    EXPECT_EQ(linear_pool->freeBlocksNum(), linear_free_before);
    EXPECT_EQ(full_pool->freeBlocksNum(), full_free_before);
    EXPECT_EQ(linear_pool->refCount(linear_blocks[0]), linear_ref_before);
    for (size_t i = 0; i < full_blocks.size(); ++i) {
        EXPECT_EQ(full_pool->refCount(full_blocks[i]), full_refs_before[i]);
    }
    allocator->free(FreeInfo{batch_res, nullptr});
    full_pool->decRef(BlockIndicesType(full_blocks.begin() + 1, full_blocks.end()));
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, UpdateKVBlockReusesDroppedBatchCapacityTransactionally) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto                   linear_pool   = poolForTag(*allocator, "linear");
    auto                   full_pool     = poolForTag(*allocator, "full1");
    const size_t           free_before   = allocator->freeBlocksNum();
    const auto             linear_blocks = allocateReferencedBlocks(linear_pool, linear_pool->freeBlocksNum());
    const auto             full_blocks   = allocateReferencedBlocks(full_pool, full_pool->freeBlocksNum());
    const BlockIndicesType blocks{linear_blocks[0], full_blocks[0], linear_blocks[1], full_blocks[1]};
    ASSERT_EQ(allocator->freeBlocksNum(), 0u);

    auto batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100}, /*reorder_resource=*/true);
    batch_res->mutableBlockIds(/*batch_id=*/0, "linear").assign({blocks[0]});
    batch_res->mutableBlockIds(/*batch_id=*/0, "full1").assign({blocks[1]});
    batch_res->mutableBlockIds(/*batch_id=*/1, "linear").assign({blocks[2]});
    batch_res->mutableBlockIds(/*batch_id=*/1, "full1").assign({blocks[3]});

    std::vector<TaggedBlockIdPair> update_mapping;
    ASSERT_TRUE(allocator->updateKVBlock(batch_res,
                                         /*block_src_batch=*/std::vector<int>{1, 1},
                                         /*copy_last_block=*/true,
                                         update_mapping));

    ASSERT_EQ(update_mapping.size(), 2u);
    EXPECT_EQ(update_mapping[0].tag, "linear");
    EXPECT_EQ(update_mapping[0].src, blocks[2]);
    EXPECT_EQ(update_mapping[0].dst, blocks[0]);
    EXPECT_EQ(update_mapping[1].tag, "full1");
    EXPECT_EQ(update_mapping[1].src, blocks[3]);
    EXPECT_EQ(update_mapping[1].dst, blocks[1]);
    EXPECT_EQ(linear_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(full_pool->freeBlocksNum(), 0u);

    allocator->free(FreeInfo{batch_res, nullptr});
    linear_pool->decRef(BlockIndicesType(linear_blocks.begin() + 2, linear_blocks.end()));
    full_pool->decRef(BlockIndicesType(full_blocks.begin() + 2, full_blocks.end()));
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, IncrDecrKVCacheRefReferencesOnlyMatchedValidBlocksAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto                   linear_pool   = poolForTag(*allocator, "linear");
    auto                   full_pool     = poolForTag(*allocator, "full1");
    const size_t           free_before   = allocator->freeBlocksNum();
    const auto             linear_blocks = allocateReferencedBlocks(linear_pool, 2);
    const auto             full_blocks   = allocateReferencedBlocks(full_pool, 2);
    const BlockIndicesType blocks{linear_blocks[0], linear_blocks[1], full_blocks[0], full_blocks[1]};
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 4);

    KVCacheResource resource;
    resource.initGroups(config.topologyPtr());
    resource.cacheKeys() = CacheKeysType{100, 101, 102};
    resource.mutableBlockIds("linear").assign(BlockIndicesType{blocks[0], NULL_BLOCK_IDX, blocks[1]});
    resource.mutableBlockIds("full1").assign(BlockIndicesType{blocks[2], blocks[3], NULL_BLOCK_IDX});

    // keys: 101(pos1)->group_id0:NULL(ignore), group_id1:blocks[3](ref);
    // 102(pos2)->group_id0:blocks[1](ref), group_id1:NULL(ignore).
    // Unmatched keys are dropped rather than represented by empty placeholders.
    auto ref = allocator->incrKVCacheRef(resource, CacheKeysType{101, 999, 102});
    ASSERT_NE(ref, nullptr);
    ASSERT_EQ(ref->groupNums(), 2);
    ASSERT_EQ(ref->cacheKeys().size(), 2u);
    ASSERT_EQ(ref->blocks("linear").size(), 2u);
    ASSERT_EQ(ref->blocks("full1").size(), 2u);

    linear_pool->decRef(linear_blocks);
    full_pool->decRef(full_blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2) << "Only blocks[1] and blocks[3] should remain referenced";
    EXPECT_EQ(linear_pool->referencedBlocksNum(), 1u);
    EXPECT_EQ(full_pool->referencedBlocksNum(), 1u);

    ref.reset();
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, DenseManagersFollowBoundTopologyOrder) {
    for (const bool reverse_topology : {false, true}) {
        auto config = makeTinyHybridConfig();
        if (reverse_topology) {
            auto groups = config.groups();
            std::reverse(groups.begin(), groups.end());
            config.setTopology(std::move(groups), config.topology().layers());
        }
        const auto bound_topology = config.topologyPtr();
        auto       allocator      = std::make_shared<CoordinatorCacheManager>(config, AllocationType::HOST);

        // Replace the caller's topology before initialization: the allocator owns its binding.
        auto replacement_groups = config.groups();
        std::reverse(replacement_groups.begin(), replacement_groups.end());
        config.setTopology(std::move(replacement_groups), config.topology().layers());
        ASSERT_NE(config.topologyPtr(), bound_topology);
        ASSERT_TRUE(allocator->init());
        const auto  managers  = allocator->cacheGroups();
        const auto& pools     = allocator->groupBlockPools();
        const auto  snapshots = allocator->poolMetricsSnapshots();
        ASSERT_EQ(managers.size(), bound_topology->groupTags().size());
        ASSERT_EQ(pools.size(), managers.size());
        ASSERT_EQ(snapshots.size(), managers.size());
        for (size_t row = 0; row < managers.size(); ++row) {
            const auto& tag = bound_topology->groupTags()[row];
            EXPECT_EQ(managers[row]->tag(), tag);
            EXPECT_EQ(managers[row]->blockPool(), pools[row]);
            EXPECT_EQ(snapshots[row].pool_index, row);
            EXPECT_EQ(snapshots[row].pool_name, pools[row]->poolName());
            for (const int layer_id : bound_topology->layerIdsForGroup(tag)) {
                const auto expected = managers[row]->convertIndexToAddr(layer_id, 1);
                const auto actual   = allocator->convertIndexToAddr(layer_id, tag, 1);
                EXPECT_EQ(actual.kv_addr, expected.kv_addr);
                EXPECT_EQ(actual.kv_scale_addr, expected.kv_scale_addr);
            }
        }
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, ReferenceAndReleaseReorderedResourceByTag) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto         linear_pool   = poolForTag(*allocator, "linear");
    auto         full_pool     = poolForTag(*allocator, "full1");
    const size_t free_before   = allocator->freeBlocksNum();
    const auto   linear_blocks = allocateReferencedBlocks(linear_pool, 2);
    const auto   full_blocks   = allocateReferencedBlocks(full_pool, 2);

    auto reordered_groups = config.topology().groups();
    std::reverse(reordered_groups.begin(), reordered_groups.end());
    KVCacheResource resource;
    resource.initGroups(CacheTopology::create(std::move(reordered_groups), config.topology().layers()));
    resource.setCacheKeys({100, 101});
    resource.mutableBlockIds("linear").assign(linear_blocks);
    resource.mutableBlockIds("full1").assign(full_blocks);

    auto selected = allocator->incrKVCacheRef(resource, {101});
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->blocks("linear"), (BlockIndicesType{linear_blocks[1]}));
    EXPECT_EQ(selected->blocks("full1"), (BlockIndicesType{full_blocks[1]}));

    linear_pool->decRef(linear_blocks);
    full_pool->decRef(full_blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2);
    selected.reset();
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, ReferenceIdentityMismatchDoesNotMutatePools) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto         linear_pool   = poolForTag(*allocator, "linear");
    auto         full_pool     = poolForTag(*allocator, "full1");
    const auto   linear_blocks = allocateReferencedBlocks(linear_pool, 1);
    const auto   full_blocks   = allocateReferencedBlocks(full_pool, 1);
    const size_t linear_ref    = linear_pool->refCount(linear_blocks.front());
    const size_t full_ref      = full_pool->refCount(full_blocks.front());

    auto       groups     = config.topology().groups();
    const auto old_tag    = groups[1].tag;
    groups[1].tag         = "unexpected";
    auto replacement_spec = groups[1].spec->clone();
    replacement_spec->tag = groups[1].tag;
    groups[1].spec        = std::move(replacement_spec);
    auto layers           = config.topology().layers();
    for (auto& layer : layers) {
        std::replace(layer.group_tags.begin(), layer.group_tags.end(), old_tag, std::string("unexpected"));
    }
    KVCacheResource resource;
    resource.initGroups(CacheTopology::create(std::move(groups), std::move(layers)));
    resource.setCacheKeys({100});
    resource.mutableBlockIds("linear").assign(linear_blocks);
    resource.mutableBlockIds("unexpected").assign(full_blocks);

    EXPECT_ANY_THROW(allocator->incrKVCacheRef(resource, {100}));
    EXPECT_ANY_THROW(allocator->decrKVCacheRef(resource));
    EXPECT_EQ(linear_pool->refCount(linear_blocks.front()), linear_ref);
    EXPECT_EQ(full_pool->refCount(full_blocks.front()), full_ref);

    linear_pool->decRef(linear_blocks);
    full_pool->decRef(full_blocks);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, FreeIdentityMismatchDoesNotPartiallyReleaseOrClearRows) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto         linear_pool   = poolForTag(*allocator, "linear");
    auto         full_pool     = poolForTag(*allocator, "full1");
    const auto   linear_blocks = allocateReferencedBlocks(linear_pool, 1);
    const auto   full_blocks   = allocateReferencedBlocks(full_pool, 1);
    const size_t linear_ref    = linear_pool->refCount(linear_blocks.front());
    const size_t full_ref      = full_pool->refCount(full_blocks.front());

    auto       groups     = config.topology().groups();
    const auto old_tag    = groups[1].tag;
    groups[1].tag         = "unexpected";
    auto replacement_spec = groups[1].spec->clone();
    replacement_spec->tag = groups[1].tag;
    groups[1].spec        = std::move(replacement_spec);
    auto layers           = config.topology().layers();
    for (auto& layer : layers) {
        std::replace(layer.group_tags.begin(), layer.group_tags.end(), old_tag, std::string("unexpected"));
    }

    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(CacheTopology::create(std::move(groups), std::move(layers)));
    resource->mutableBlockIds(0, "linear").assign(linear_blocks);
    resource->mutableBlockIds(0, "unexpected").assign(full_blocks);
    const auto linear_row     = resource->blocks(0, "linear");
    const auto unexpected_row = resource->blocks(0, "unexpected");

    EXPECT_ANY_THROW(allocator->free(FreeInfo{resource, nullptr}));
    EXPECT_EQ(linear_pool->refCount(linear_blocks.front()), linear_ref);
    EXPECT_EQ(full_pool->refCount(full_blocks.front()), full_ref);
    EXPECT_EQ(resource->blocks(0, "linear"), linear_row);
    EXPECT_EQ(resource->blocks(0, "unexpected"), unexpected_row);

    linear_pool->decRef(linear_blocks);
    full_pool->decRef(full_blocks);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, ConnectorRefPreservesDummyTailAcrossGroupsAndReleasesValidBlocks) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto                   linear_pool   = poolForTag(*allocator, "linear");
    auto                   full_pool     = poolForTag(*allocator, "full1");
    const size_t           free_before   = allocator->freeBlocksNum();
    const auto             linear_blocks = allocateReferencedBlocks(linear_pool, 1);
    const auto             full_blocks   = allocateReferencedBlocks(full_pool, 1);
    const BlockIndicesType blocks{linear_blocks[0], full_blocks[0]};

    KVCacheResource resource;
    resource.initGroups(config.topologyPtr());
    resource.cacheKeys() = CacheKeysType{100, 101, 102};
    resource.setLastBlockAligned(false);
    resource.mutableBlockIds("linear").assign(BlockIndicesType{blocks[0], NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    resource.mutableBlockIds("full1").assign(BlockIndicesType{NULL_BLOCK_IDX, blocks[1], NULL_BLOCK_IDX});

    auto ref = allocator->incrKVCacheRef(resource, resource.cacheKeys(), /*is_connector=*/true);
    ASSERT_NE(ref, nullptr);
    EXPECT_FALSE(ref->lastBlockAligned());
    EXPECT_EQ(ref->cacheKeys(), resource.cacheKeys());
    EXPECT_EQ(ref->blocks("linear"), (BlockIndicesType{blocks[0], NULL_BLOCK_IDX, NULL_BLOCK_IDX}));
    EXPECT_EQ(ref->blocks("full1"), (BlockIndicesType{NULL_BLOCK_IDX, blocks[1], NULL_BLOCK_IDX}));
    EXPECT_EQ(linear_pool->refCount(blocks[0]), 2u);
    EXPECT_EQ(full_pool->refCount(blocks[1]), 2u);

    linear_pool->decRef(linear_blocks);
    full_pool->decRef(full_blocks);
    EXPECT_EQ(linear_pool->referencedBlocksNum(), 1u);
    EXPECT_EQ(full_pool->referencedBlocksNum(), 1u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2);

    ref.reset();
    EXPECT_EQ(linear_pool->referencedBlocksNum(), 0u);
    EXPECT_EQ(full_pool->referencedBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, InsertIntoCacheInsertsOnlyFullBlocks) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // group_id=0 linear, group_id=1 full.

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102});
    // Disable device cache reuse.

    // BlockTree insertion records the available reusable group coordinates.
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/10, /*seq_size_per_block=*/4);

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_cache_lookup = false;
    malloc_info.reuse_cache         = false;
    auto malloc_result              = allocator->malloc(malloc_info);
    ASSERT_TRUE(malloc_result.success);
    ASSERT_EQ(batch_res->blocksNum(0, "full1"), 3);
    ASSERT_EQ(batch_res->blocksNum(0, "linear"), 3);

    InsertInfo insert_info{batch_res, token_ids, /*is_resident=*/false};
    {
        size_t resident_prefix_length = 0;
        allocator->insertIntoCache(insert_info, resident_prefix_length);
    }

    auto match = allocator->blockTreeCacheOwner()->match(CacheKeysType{100, 101, 102});
    EXPECT_EQ(match.matched_device_blocks, 3u);
    EXPECT_EQ(allocator->blockTreeCacheOwner()->matchedBlocksForGroup("full1", match.matched_device_resources).size(),
              3u);
    EXPECT_EQ(allocator->blockTreeCacheOwner()->matchedBlocksForGroup("linear", match.matched_device_resources).size(),
              1u);
    block_tree_cache_test::releaseRequestRefsForTest(*allocator->blockTreeCacheOwner(), match.matched_device_resources);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, InsertIntoCachePreservesLinearHoleAndPublishesLaterState) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto                   linear_pool   = poolForTag(*allocator, "linear");
    auto                   full_pool     = poolForTag(*allocator, "full1");
    const auto             linear_blocks = allocateReferencedBlocks(linear_pool, 2);
    const auto             full_blocks   = allocateReferencedBlocks(full_pool, 3);
    const BlockIndicesType blocks{linear_blocks[0], linear_blocks[1], full_blocks[0], full_blocks[1], full_blocks[2]};

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102});
    batch_res->mutableBlockIds(/*batch_id=*/0, "linear").assign({blocks[0], NULL_BLOCK_IDX, blocks[1]});
    batch_res->mutableBlockIds(/*batch_id=*/0, "full1").assign({blocks[2], blocks[3], blocks[4]});

    {
        size_t resident_prefix_length = 0;
        EXPECT_NO_THROW(
            allocator->insertIntoCache(InsertInfo{batch_res, nullptr, /*is_resident=*/false}, resident_prefix_length));
    }
    const auto path = allocator->blockTreeCacheOwner()->tree()->findNode(CacheKeysType{100, 101, 102});
    ASSERT_EQ(path.size(), 3u);
    EXPECT_EQ(path[0]->group_set_resources[0].device_blocks, (BlockIndicesType{blocks[0]}));
    EXPECT_TRUE(path[1]->group_set_resources[0].is_empty());
    EXPECT_EQ(path[2]->group_set_resources[0].device_blocks, (BlockIndicesType{blocks[1]}));
    EXPECT_EQ(path[0]->group_set_resources[1].device_blocks, (BlockIndicesType{blocks[2]}));
    EXPECT_EQ(path[1]->group_set_resources[1].device_blocks, (BlockIndicesType{blocks[3]}));
    EXPECT_EQ(path[2]->group_set_resources[1].device_blocks, (BlockIndicesType{blocks[4]}));

    auto match = allocator->blockTreeCacheOwner()->match(CacheKeysType{100, 101, 102});
    EXPECT_EQ(match.matched_device_blocks, 3u);
    EXPECT_EQ(allocator->blockTreeCacheOwner()->matchedBlocksForGroup("linear", match.matched_device_resources),
              (BlockIndicesType{blocks[1]}));
    block_tree_cache_test::releaseRequestRefsForTest(*allocator->blockTreeCacheOwner(), match.matched_device_resources);

    linear_pool->decRef(linear_blocks);
    full_pool->decRef(full_blocks);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, InsertIntoCacheStopsBeforeFullHole) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto                   linear_pool   = poolForTag(*allocator, "linear");
    auto                   full_pool     = poolForTag(*allocator, "full1");
    const auto             linear_blocks = allocateReferencedBlocks(linear_pool, 3);
    const auto             full_blocks   = allocateReferencedBlocks(full_pool, 2);
    const BlockIndicesType blocks{linear_blocks[0], linear_blocks[1], linear_blocks[2], full_blocks[0], full_blocks[1]};

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102});
    batch_res->mutableBlockIds(/*batch_id=*/0, "linear").assign({blocks[0], blocks[1], blocks[2]});
    batch_res->mutableBlockIds(/*batch_id=*/0, "full1").assign({blocks[3], NULL_BLOCK_IDX, blocks[4]});

    {
        size_t resident_prefix_length = 0;
        allocator->insertIntoCache(InsertInfo{batch_res, nullptr, /*is_resident=*/false}, resident_prefix_length);
    }
    const auto path = allocator->blockTreeCacheOwner()->tree()->findNode(CacheKeysType{100, 101, 102});
    ASSERT_EQ(path.size(), 1u);
    EXPECT_EQ(path.front()->group_set_resources[0].device_blocks, (BlockIndicesType{blocks[0]}));
    EXPECT_EQ(path.front()->group_set_resources[1].device_blocks, (BlockIndicesType{blocks[3]}));

    linear_pool->decRef(linear_blocks);
    full_pool->decRef(full_blocks);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, DefaultHybridLinearPrefixReuseSupportsInsertThenReuse) {
    auto config = makeTinyHybridConfig();
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_TRUE(config.group("linear").policy.enable_prefix_reuse);

    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto seed_res    = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102});
    auto seed_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.enable_cache_lookup = false;
    seed_malloc.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(seed_malloc).success);

    {
        size_t resident_prefix_length = 0;
        allocator->insertIntoCache(InsertInfo{seed_res, seed_tokens, /*is_resident=*/false}, resident_prefix_length);
    }
    auto seed_match = allocator->blockTreeCacheOwner()->match(CacheKeysType{100, 101, 102});
    EXPECT_EQ(seed_match.matched_device_blocks, 3u);
    EXPECT_EQ(
        allocator->blockTreeCacheOwner()->matchedBlocksForGroup("linear", seed_match.matched_device_resources).size(),
        1u);
    block_tree_cache_test::releaseRequestRefsForTest(*allocator->blockTreeCacheOwner(),
                                                     seed_match.matched_device_resources);

    auto hit_res    = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    auto hit_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/16, /*seq_size_per_block=*/4);

    MallocInfo hit_malloc{hit_res, hit_tokens};
    hit_malloc.enable_cache_lookup = true;
    hit_malloc.reuse_cache         = true;
    auto result                    = allocator->malloc(hit_malloc);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 12);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, ConvertIndexToBufferAndAllLayerCacheBaseSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    CoordinatorCacheManager* base = allocator.get();
    auto                     buf0 = base->convertIndexToBuffer(/*layer_id=*/0, /*block_id=*/1);
    ASSERT_FALSE(buf0.empty());
    EXPECT_NE(buf0[0].addr, nullptr);

    auto linear_buf = base->convertIndexToBuffer(/*layer_id=*/0, "linear", /*block_id=*/1);
    auto full_buf   = base->convertIndexToBuffer(/*layer_id=*/2, "full1", /*block_id=*/1);
    ASSERT_FALSE(linear_buf.empty());
    ASSERT_FALSE(full_buf.empty());
    EXPECT_NE(linear_buf[0].addr, nullptr);
    EXPECT_NE(full_buf[0].addr, nullptr);
    EXPECT_EQ(linear_buf[0].size_bytes, config.topology().group("linear").kvBlockStrideBytes());
    EXPECT_EQ(full_buf[0].size_bytes, config.topology().group("full1").kvBlockStrideBytes());
    EXPECT_LT(linear_buf[0].size_bytes,
              std::max(config.topology().group("linear").kvBlockStrideBytes(),
                       config.topology().group("full1").kvBlockStrideBytes()));

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.groups().size(), static_cast<size_t>(config.groupNums()));
    ASSERT_EQ(layout.topology().layers().size(), static_cast<size_t>(config.layer_num));
    for (size_t i = 0; i < layout.topology().layers().size(); ++i) {
        for (const auto& tag : layout.topology().layer(static_cast<int>(i)).group_tags) {
            EXPECT_TRUE(layout.group(tag).hasLayer(i));
        }
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, IncrMallocRollbackFreesPartiallyAllocatedBlocks) {
    auto config = makeTinyHybridConfig();
    setGroupBlockCounts(config, 6, 6);  // Five usable blocks per pool.
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto linear_pool = poolForTag(*allocator, "linear");
    auto full_pool   = poolForTag(*allocator, "full1");

    auto batch_res =
        makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102}, /*reorder_resource=*/true);
    // Disable device cache reuse (makes linear group allocate only tail for new resources).

    // Initial small allocation: seq_len=4 => 1 resource per group.
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_cache_lookup = false;
    auto init_result              = allocator->malloc(init_info);
    ASSERT_TRUE(init_result.success);
    ASSERT_EQ(batch_res->blocksNum(0, "linear"), 1);
    ASSERT_EQ(batch_res->blocksNum(0, "full1"), 1);

    const auto linear_block_before = batch_res->blocks(0, "linear")[0];
    const auto full_block_before   = batch_res->blocks(0, "full1")[0];

    // LINEAR can grow by two blocks; FULL has only one spare and fails afterward.
    const auto linear_free_before = linear_pool->freeBlocksNum();
    ASSERT_GE(linear_free_before, 2u);
    auto keep = allocateReferencedBlocks(full_pool, full_pool->freeBlocksNum() - 1);
    ASSERT_EQ(full_pool->freeBlocksNum(), 1u);
    const auto linear_ref_before = linear_pool->refCount(linear_block_before);
    const auto full_ref_before   = full_pool->refCount(full_block_before);

    // Incr to seq_len=9 => 3 resources per group. Linear allocates 2 blocks and Full then needs 2 more.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_cache_lookup = false;
    auto incr_result              = allocator->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    // Rollback should restore original sizes and keep original blocks.
    ASSERT_EQ(batch_res->blocksNum(0, "linear"), 1);
    ASSERT_EQ(batch_res->blocksNum(0, "full1"), 1);
    EXPECT_EQ(batch_res->blocks(0, "linear")[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, "full1")[0], full_block_before);

    // Both pool capacities and original request refs survive rollback.
    EXPECT_EQ(linear_pool->freeBlocksNum(), linear_free_before);
    EXPECT_EQ(full_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(linear_pool->refCount(linear_block_before), linear_ref_before);
    EXPECT_EQ(full_pool->refCount(full_block_before), full_ref_before);

    // Cleanup.
    full_pool->decRef(keep);
    allocator->free(FreeInfo{batch_res, nullptr});
}

// Prefill init path (StreamCacheResource::initKVBlock sets enable_remove_skipped_blocks=false).
// With step=2 and reuse_blocks_len=3, the reused linear tail lands at pos 2, which is NOT
// a step hit ((2+1)%2==1). Without sparse cleanup, that resource must survive so that
// causal_conv1d can still read it by prefix_length.
TEST_F(HybridTypeCoordinatorCacheManagerTest, PrefillInitSkipsSparseCleanupAndPreservesReusedLinearTail) {
    auto config = makeTinyHybridConfig();
    setGroupBlockCounts(config, 16, 16);  // Enough capacity for cached and new blocks in each pool.
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    CacheKeysType shared_keys = {100, 101, 102};
    const auto    seeded      = seedCompleteBlockTreePath(allocator, shared_keys);
    ASSERT_TRUE(seeded.success);
    const auto& cached_linear_blocks = seeded.blocks_by_tag.at("linear");
    ASSERT_EQ(cached_linear_blocks.size(), 3u);

    // Request has 5 keys; allocator drops the last before matching, leaving {100,101,102,103}.
    // Full matches the first 3 (103 is absent); linear joint backoff stops at pos=2 => reuse_blocks_len=3.
    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103, 104});

    // seq_len=20 => 5 resources. block_size-3-reserve_step = 2, so removeSkippedBlocks would scan pos 2.
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/20, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_cache_lookup          = true;
    info.reuse_cache                  = true;
    info.enable_remove_skipped_blocks = false;  // prefill init path
    auto result                       = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    const auto& linear_out = batch_res->blocks(0, "linear");
    ASSERT_EQ(linear_out.size(), 5u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_EQ(linear_out[2], cached_linear_blocks[2]) << "reused linear tail must survive prefill init";
    EXPECT_FALSE(isNullBlockIdx(linear_out[3]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[4]));
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, ChunkPrefillStateSlotLifecycle) {
    auto config      = makeTinyHybridConfig();
    // 16 usable slots across independent pools: two requests, each with 5 FULL and at most 3 LINEAR.
    setGroupBlockCounts(config, 7, 11);
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    const auto                           free_before = allocator->freeBlocksNum();
    std::vector<BatchKVCacheResourcePtr> resources;
    std::vector<MallocInfo>              infos;
    auto                                 tokens = makeCompleteTokenIds(1, 18, 4);
    for (int request = 0; request < 2; ++request) {
        resources.push_back(makeBatchResource(1, config, CacheKeysType{}));
        infos.push_back(MallocInfo{resources.back(), tokens});
        infos.back().reuse_cache                  = false;
        infos.back().enable_cache_lookup          = false;
        infos.back().enable_remove_skipped_blocks = false;
        ASSERT_TRUE(allocator->malloc(infos.back()).success);
        ASSERT_TRUE(isNullBlockIdx(resources.back()->blocks(0, "linear")[0]));
    }
    const auto full0 = resources[0]->blocks(0, "full1");
    const auto full1 = resources[1]->blocks(0, "full1");
    // Block size 4 keeps this allocator test small; GPU coverage uses 64/128/130.
    int prefix_len = 0;
    for (int end : {4, 12, 16, 18}) {
        for (int request : {1, 0}) {
            auto expected                        = resources[request]->blocks(0, "linear");
            infos[request].incr_seq_len_override = end;
            infos[request].computed_prefix_len   = prefix_len;
            ASSERT_TRUE(allocator->malloc(infos[request]).success);
            const auto&  after    = resources[request]->blocks(0, "linear");
            const size_t boundary = (end - 1) / 4;
            ASSERT_EQ(after.size(), expected.size());
            ASSERT_LT(boundary, after.size());
            ASSERT_FALSE(isNullBlockIdx(after[boundary]));
            // Only obsolete states and the newly materialized boundary may change.
            if (prefix_len > 0) {
                std::fill(expected.begin(), expected.begin() + (prefix_len - 1) / 4, NULL_BLOCK_IDX);
            }
            if (isNullBlockIdx(expected[boundary])) {
                expected[boundary] = after[boundary];
            }
            EXPECT_EQ(after, expected);
            for (auto slot : resources[1 - request]->blocks(0, "linear")) {
                EXPECT_NE(after[boundary], slot);
            }
        }
        if (end == 12) {
            // The next chunk must reclaim obsolete states before allocating its boundary.
            EXPECT_EQ(allocator->freeBlocksNum(), 0u);
        }
        prefix_len = end;
    }
    EXPECT_EQ(resources[0]->blocks(0, "full1"), full0);
    EXPECT_EQ(resources[1]->blocks(0, "full1"), full1);
    for (int request : {0, 1}) {
        allocator->free(FreeInfo{resources[request], tokens});
    }
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, ChunkPrefillPreservesActiveLinearTailUntilItLeavesWindow) {
    auto config = makeTinyHybridConfig();
    auto groups = config.topology().groups();
    ASSERT_EQ(groups.size(), 2u);
    ASSERT_EQ(groups[0].policy.group_type, CacheGroupType::LINEAR);
    groups[0].policy.active_tail_blocks = 4;
    config.setTopology(std::move(groups), config.topology().layers());
    setGroupBlockCounts(config, 24, 24);

    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    const auto free_before = allocator->freeBlocksNum();

    auto resource = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{});
    auto tokens   = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    MallocInfo info{resource, tokens};
    info.reuse_cache                  = false;
    info.enable_cache_lookup          = false;
    info.enable_remove_skipped_blocks = false;
    ASSERT_TRUE(allocator->malloc(info).success);

    const auto& initial_linear_blocks = resource->blocks(0, "linear");
    ASSERT_EQ(initial_linear_blocks.size(), 3u);
    const auto first_block = initial_linear_blocks[0];
    ASSERT_FALSE(isNullBlockIdx(first_block));

    // Hold a second reference so a wrongly released block cannot be recycled
    // into the same slot and make the identity check pass accidentally.
    auto linear_group = allocator->cacheGroups()[0];
    linear_group->reference(BlockIndicesType{first_block});
    info.computed_prefix_len   = 8;
    info.incr_seq_len_override = 12;
    const auto next_result = allocator->malloc(info);
    linear_group->unreference(BlockIndicesType{first_block});
    ASSERT_TRUE(next_result.success);
    const auto& retained_blocks = resource->blocks(0, "linear");
    ASSERT_EQ(retained_blocks.size(), 3u);
    EXPECT_EQ(retained_blocks[0], first_block);

    // A longer grant moves the four-block tail past position zero, so it can
    // now be reclaimed while the final four positions remain materialized.
    info.incr_seq_len_override = 20;
    ASSERT_TRUE(allocator->malloc(info).success);
    const auto& linear_blocks = resource->blocks(0, "linear");
    ASSERT_EQ(linear_blocks.size(), 5u);
    EXPECT_TRUE(isNullBlockIdx(linear_blocks[0]));
    for (size_t pos = 1; pos < linear_blocks.size(); ++pos) {
        EXPECT_FALSE(isNullBlockIdx(linear_blocks[pos]));
    }
    allocator->free(FreeInfo{resource, tokens});
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// Decode path (StreamCacheResource::incrKVBlock sets enable_remove_skipped_blocks=true).
// The allocator is invoked on an already-populated resource, so malloc() dispatches directly
// to incrMalloc(). Sparse cleanup must prune non-step blocks while preserving step hits and
// the configured active tail resource.
TEST_F(HybridTypeCoordinatorCacheManagerTest, DecodeIncrMallocAppliesSparseCleanupOnLinearGroups) {
    auto config = makeTinyHybridConfig();
    setGroupBlockCounts(config, 7, 7);  // Six usable blocks per independent pool.
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto linear_pool = poolForTag(*allocator, "linear");
    auto full_pool   = poolForTag(*allocator, "full1");

    auto linear_alloc = linear_pool->malloc(6).value();
    auto full_alloc   = full_pool->malloc(6).value();
    ASSERT_EQ(linear_alloc.size(), 6u);
    ASSERT_EQ(full_alloc.size(), 6u);
    linear_pool->incRef(linear_alloc);
    full_pool->incRef(full_alloc);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{}, /*reorder_resource=*/true);
    batch_res->mutableBlockIds(0, "linear").assign(linear_alloc);
    batch_res->mutableBlockIds(0, "full1").assign(full_alloc);
    ASSERT_GT(batch_res->curBlocksNum(), 0);

    // seq_len=24 => 6 resources; current_blocks==6 so group malloc is a no-op and only cleanup runs.
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_cache_lookup          = false;
    info.reuse_cache                  = true;
    info.enable_remove_skipped_blocks = true;  // decode path
    auto result                       = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // active_tail_blocks=1 materializes the current tail, while decode cleanup retains at least two tails.
    // For step=2 and size=6: keep pos 1, 3 (step hits) and pos 4, 5 (decode tails).
    const auto& linear_out = batch_res->blocks(0, "linear");
    ASSERT_EQ(linear_out.size(), 6u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[2]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[3]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[4]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[5]));

    // Full group is untouched by sparse cleanup.
    const auto& full_out = batch_res->blocks(0, "full1");
    ASSERT_EQ(full_out.size(), 6u);
    for (size_t i = 0; i < full_out.size(); ++i) {
        EXPECT_EQ(full_out[i], full_alloc[i]);
        EXPECT_EQ(full_pool->refCount(full_alloc[i]), 1u);
    }
    EXPECT_EQ(linear_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(full_pool->freeBlocksNum(), 0u);
    allocator->free(FreeInfo{batch_res, nullptr});
    EXPECT_EQ(linear_pool->freeBlocksNum(), 6u);
    EXPECT_EQ(full_pool->freeBlocksNum(), 6u);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, EstimatePeakNeedBlocks) {
    // Config: [0,1]=linear group (group_id=0), [2,3]=full group (group_id=1). seq_size_per_block=4.
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int blk = config.seq_size_per_block;  // 4

    // New resource (cur_slots=0 for both groups):
    // reuse disabled: full=ceil(108/4)=27, linear tail peak=3 => total=30.
    auto new_res = makeBatchResource(1, config, {});
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/false), 30);

    // reuse enabled: linear keeps 14 blocks after cleanup and transiently holds a fifteenth tail block.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/true), 42);

    // With reserve_step=3: full=ceil(111/4)=28. linear: total_slots=29, tail=5,
    // step-hits before tail=24/2=12 => linear=17. total=45.
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 3, /*enable_reuse_cache=*/true), 45);

    // Allocate blocks to simulate running decode (seqLen=8 → 2 resources per group)
    auto       token_ids = makeCompleteTokenIds(1, /*seq_length=*/8, config.seq_size_per_block);
    MallocInfo mi{new_res, token_ids};
    auto       result = allocator->malloc(mi);
    ASSERT_TRUE(result.success);

    const int full_slots   = new_res->blocksNum(0, "full1");   // full-group blocks after malloc
    const int linear_slots = new_res->blocksNum(0, "linear");  // linear-group blocks after malloc

    // remaining=0: no more resources needed for either group
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 0, 0, /*enable_reuse_cache=*/false), 0);

    // remaining=4: ceil((8+4)/4)=3 per group, minus cur_slots
    int expect_per_group = (8 + 4 + blk - 1) / blk;
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 4, 0, /*enable_reuse_cache=*/false),
              std::max(expect_per_group - full_slots, 0) + std::max(expect_per_group - linear_slots, 0));

    // Large remaining from current_slots=2:
    // reuse disabled: cleanup scans across the initial null resource. At the second boundary the running resource
    // transiently holds three physical linear blocks before releasing the oldest tail, two more than its current tail.
    int expect_full_large = (8 + 100 + blk - 1) / blk;  // 27
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/false),
              std::max(expect_full_large - full_slots, 0) + 2);

    // reuse enabled: target linear keeps tail 2 + step-hit resources before tail 12;
    // The fresh seq_len=8 allocation owns one physical linear block. Decode later peaks at 15 physical blocks.
    int expect_linear_large = 14;
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator, new_res, 8, 100, 0, /*enable_reuse_cache=*/true),
              std::max(expect_full_large - full_slots, 0) + expect_linear_large);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, EstimatePeakNeedBlocksUsesLinearActiveTailPolicy) {
    auto                          config = makeTinyHybridConfig();
    std::vector<CacheGroupPolicy> policies;
    for (const auto& group : config.topology().groups()) {
        policies.push_back(group.policy);
    }
    ASSERT_EQ(policies.size(), 2u);
    ASSERT_EQ(policies[0].group_type, CacheGroupType::LINEAR);
    policies[0].active_tail_blocks = 4;
    setTestGroupPolicies(config, policies);

    auto allocator = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
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

TEST_F(HybridTypeCoordinatorCacheManagerTest, EstimateBatchPeakNeedBlocksAccountsForNonEmptyTargetWidth) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto resource = makeBatchResource(/*batch_size=*/2, config, /*keys=*/{});

    // common_seq_len=8 means the first two resources are shared. The NULL resource in the linear group consumes no
    // block.
    resource->setBatchBlocks(/*batch_id=*/0, "linear", {NULL_BLOCK_IDX, 10, 11});
    resource->setBatchBlocks(/*batch_id=*/1, "linear", {NULL_BLOCK_IDX, 10, 12});
    resource->setBatchBlocks(/*batch_id=*/0, "full1", {20, 21, 22});
    resource->setBatchBlocks(/*batch_id=*/1, "full1", {20, 21, 23});

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

    resource->setBatchBlocks(/*batch_id=*/0, "linear", {NULL_BLOCK_IDX, 10, 11, NULL_BLOCK_IDX});
    resource->setBatchBlocks(/*batch_id=*/1, "linear", {NULL_BLOCK_IDX, 10, 12, NULL_BLOCK_IDX});
    resource->setBatchBlocks(/*batch_id=*/0, "full1", {20, 21, 22, 24});
    resource->setBatchBlocks(/*batch_id=*/1, "full1", {20, 21, 23, 25});

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

TEST_F(HybridTypeCoordinatorCacheManagerTest, FreshUnalignedMultiSequencePeakMatchesExactCapacity) {
    for (const bool reuse_cache : {false, true}) {
        SCOPED_TRACE(reuse_cache ? "reuse enabled" : "reuse disabled");

        auto config = makeTinyHybridConfig();
        setGroupBlockCounts(config, 4, 4);  // Three usable blocks each: six total, the exact initialization peak.
        auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
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
            /*batch_size=*/2, /*seq_length=*/5, /*seq_size_per_block=*/config.seq_size_per_block);
        MallocInfo info{resource, token_ids};
        info.enable_cache_lookup          = false;
        info.reuse_cache                  = reuse_cache;
        info.enable_remove_skipped_blocks = false;
        ASSERT_TRUE(allocator->malloc(info).success);
        EXPECT_EQ(allocator->freeBlocksNum(), 0);

        allocator->free(FreeInfo{resource, token_ids});
        EXPECT_EQ(allocator->freeBlocksNum(), 6);
    }
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, EstimatedPeakCoversDecodeMallocAndSparseCleanup) {
    auto config = makeTinyHybridConfig();
    setGroupBlockCounts(config, 11, 18);  // LINEAR peak 10 + FULL peak 17 = 27 usable blocks.
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1,
                                          /*seq_length=*/8,
                                          /*seq_size_per_block=*/config.seq_size_per_block);

    MallocInfo info{batch_res, token_ids};
    info.enable_cache_lookup          = false;
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

    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, "linear")), 9);
    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, "full1")), 17);
    EXPECT_EQ(min_free_blocks, 1);
    EXPECT_EQ(allocator->freeBlocksNum(), 1);
}

TEST_F(HybridTypeCoordinatorCacheManagerTest, FreshReusePeakCoversThreeBoundaryDecodeAtExactCapacity) {
    auto config = makeTinyHybridConfig();
    setGroupBlockCounts(config, 5, 6);  // LINEAR peak 4 + FULL peak 5 = 9 usable blocks.
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1,
                                          /*seq_length=*/8,
                                          /*seq_size_per_block=*/config.seq_size_per_block);

    // seq_len 8 -> 17 crosses the resource boundaries at 9, 13 and 17. Full peaks at 5 blocks and linear peaks at 4.
    ASSERT_EQ(allocator->freeBlocksNum(), 9);
    ASSERT_EQ(estimateBatchPeakForSingleSequence(*allocator,
                                                 batch_res,
                                                 /*seq_len=*/8,
                                                 /*remaining_tokens=*/9,
                                                 /*reserve_step=*/0,
                                                 /*reuse_cache=*/true),
              9);

    MallocInfo info{batch_res, token_ids};
    info.enable_cache_lookup          = false;
    info.reuse_cache                  = true;
    info.enable_remove_skipped_blocks = false;
    ASSERT_TRUE(allocator->malloc(info).success);

    info.enable_remove_skipped_blocks = true;
    for (int seq_len = 9; seq_len <= 17; ++seq_len) {
        token_ids->setSeqLength(seq_len);
        ASSERT_TRUE(allocator->malloc(info).success) << "seq_len=" << seq_len;
    }

    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, "linear")), 3);
    EXPECT_EQ(countValidBlocks(batch_res->blocks(0, "full1")), 5);
    EXPECT_EQ(allocator->freeBlocksNum(), 1);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
