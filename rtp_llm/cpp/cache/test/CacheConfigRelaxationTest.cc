#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {
namespace test {
namespace {

// NemotronH-style hybrid: [GQA attention, attention-free MoE, mamba, attention-free MoE].
// Attention-free layers carry an empty kv_cache_spec_descs row: no cache group,
// no budget, no tensors. The mamba layer uses Mamba-2 state geometry where
// key_head_dim (dstate) differs from value_head_dim (headdim).
ModelConfig makeHybridEmptyLayerModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 4;
    mc.hidden_size                  = 128;
    mc.attn_config.head_num         = 4;
    mc.attn_config.kv_head_num      = 2;
    mc.attn_config.size_per_head    = 16;
    mc.attn_config.tokens_per_block = 8;
    mc.hybrid_attention_config.enable_hybrid_attention = true;
    mc.hybrid_attention_config.hybrid_attention_types  = {HybridAttentionType::NONE,
                                                          HybridAttentionType::NONE,
                                                          HybridAttentionType::LINEAR,
                                                          HybridAttentionType::NONE};

    mc.linear_attention_config.linear_conv_kernel_dim = 4;
    mc.linear_attention_config.linear_key_head_dim    = 128;  // dstate
    mc.linear_attention_config.linear_value_head_dim  = 64;   // headdim
    mc.linear_attention_config.linear_num_key_heads   = 2;
    mc.linear_attention_config.linear_num_value_heads = 4;

    KVCacheSpecDesc full_desc;
    full_desc.tag        = "full";
    full_desc.cache_type = KVCacheSpecType::MultiHeadAttention;

    KVCacheSpecDesc linear_desc;
    linear_desc.tag        = "linear";
    linear_desc.cache_type = KVCacheSpecType::LinearAttention;

    // Layers 1 and 3 are attention-free: empty descriptor rows.
    mc.kv_cache_spec_descs = {{full_desc}, {}, {linear_desc}, {}};
    return mc;
}

GroupBase makeRelaxationGroup(const std::string& tag, CacheGroupType type) {
    auto spec                       = std::make_shared<MHAKVCacheSpec>();
    spec->tag                       = tag;
    spec->seq_size_per_block        = 8;
    spec->kernel_seq_size_per_block = 2;

    GroupBase group;
    group.tag       = tag;
    group.spec      = std::move(spec);
    group.policy    = defaultCacheGroupPolicy(type);
    group.block_num = 16;
    return group;
}

}  // namespace

// ============================================================
// Relaxation 1: linear_key_head_dim != linear_value_head_dim
// ============================================================

TEST(CacheConfigRelaxationTest, LinearSpecGeometrySurvivesKeyAndValueHeadDimSplit) {
    // Mamba-2: 4 value heads each owning a 128(dstate) x 64(headdim) state slice.
    const auto mamba2 = makeResolvedLinearSpec(DataType::TYPE_BF16,
                                               /*local_num_k_heads=*/2,
                                               /*local_num_v_heads=*/4,
                                               /*head_k_dim=*/128,
                                               /*head_v_dim=*/64,
                                               /*conv_kernel_dim=*/4,
                                               /*seq_size_per_block=*/8);
    ASSERT_NE(mamba2, nullptr);
    EXPECT_EQ(mamba2->type, KVCacheSpecType::LinearAttention);
    // ssm elems stay local_v_heads * key_dim * value_dim.
    EXPECT_EQ(mamba2->local_kv_head_num, 4u);
    EXPECT_EQ(mamba2->k_block_size(), 4u * 128u * 64u);
    // conv elems stay (kernel - 1) * (2 * k_heads * key_dim + v_heads * value_dim).
    EXPECT_EQ(mamba2->v_block_size(), 3u * (2u * 2u * 128u + 4u * 64u));
    EXPECT_EQ(mamba2->block_size(), mamba2->k_block_size() + mamba2->v_block_size());
    EXPECT_EQ(mamba2->block_size_bytes(),
              mamba2->k_block_size() * getTypeSize(DataType::TYPE_BF16)
                  + mamba2->v_block_size() * getTypeSize(DataType::TYPE_BF16));

    // Legacy equal-dim geometry is unchanged (qwen3-next / kimi-linear shapes).
    const auto equal_dims = makeResolvedLinearSpec(DataType::TYPE_BF16,
                                                   /*local_num_k_heads=*/2,
                                                   /*local_num_v_heads=*/4,
                                                   /*head_k_dim=*/64,
                                                   /*head_v_dim=*/64,
                                                   /*conv_kernel_dim=*/4,
                                                   /*seq_size_per_block=*/8);
    ASSERT_NE(equal_dims, nullptr);
    EXPECT_EQ(equal_dims->k_block_size(), 4u * 64u * 64u);
    EXPECT_EQ(equal_dims->v_block_size(), 3u * (2u * 2u * 64u + 4u * 64u));
}

TEST(CacheConfigRelaxationTest, BuildLayerSpecsSkipsEmptyDescRows) {
    const auto  mc    = makeHybridEmptyLayerModelConfig();
    const auto& descs = mc.kv_cache_spec_descs;

    AttentionConfigs attn;
    attn.kv_head_num      = mc.attn_config.kv_head_num;
    attn.size_per_head    = mc.attn_config.size_per_head;
    attn.tokens_per_block = mc.attn_config.tokens_per_block;
    ParallelismConfig parallelism;
    SpecBuildContext   ctx;
    ctx.dtype                   = DataType::TYPE_FP16;
    ctx.seq_size_per_block      = 8;
    ctx.kernel_tokens_per_block = 8;
    ctx.attn_config             = &attn;
    ctx.linear_attention_config = &mc.linear_attention_config;
    ctx.parallelism_config      = &parallelism;

    const auto specs = CacheConfigCreator::buildLayerSpecsFromDescs(descs, ctx, mc.num_layers);
    ASSERT_EQ(specs.size(), 4u);
    ASSERT_EQ(specs[0].size(), 1u);
    EXPECT_TRUE(specs[1].empty());
    ASSERT_EQ(specs[2].size(), 1u);
    EXPECT_TRUE(specs[3].empty());
    EXPECT_EQ(specs[0].front()->tag, "full");
    EXPECT_EQ(specs[2].front()->tag, "linear");
}

// ============================================================
// Relaxation 2: attention-free layers (empty desc rows)
// ============================================================

TEST(CacheConfigRelaxationTest, TopologyAcceptsAttentionFreeLayers) {
    const auto topology = CacheTopology::create(
        {makeRelaxationGroup("full", CacheGroupType::FULL),
         makeRelaxationGroup("linear", CacheGroupType::LINEAR)},
        {{0, {"full"}}, {1, {}}, {2, {"linear"}}, {3, {}}});

    EXPECT_EQ(topology->layers().size(), 4u);
    EXPECT_EQ(topology->layer(0).group_tags, (std::vector<std::string>{"full"}));
    EXPECT_TRUE(topology->layer(1).group_tags.empty());
    EXPECT_EQ(topology->layer(2).group_tags, (std::vector<std::string>{"linear"}));
    EXPECT_TRUE(topology->layer(3).group_tags.empty());

    EXPECT_EQ(topology->layerIdsForGroup("full"), (std::vector<int>{0}));
    EXPECT_EQ(topology->layerIdsForGroup("linear"), (std::vector<int>{2}));
    EXPECT_TRUE(topology->groupsForLayer(1).empty());
    EXPECT_TRUE(topology->groupsForLayer(3).empty());
    EXPECT_ANY_THROW(topology->soleGroupForLayer(1));
    EXPECT_FALSE(topology->hasOneGroupPerLayer());
}

TEST(CacheConfigRelaxationTest, HybridConfigCreatesAttentionFreeLayers) {
    ParallelismConfig pc;
    const auto        config = CacheConfigCreator::createWarmupConfig(makeHybridEmptyLayerModelConfig(), pc, 0);

    EXPECT_EQ(config.layer_num, 4u);
    EXPECT_EQ(config.groupTags(), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(config.topology().layer(0).group_tags, (std::vector<std::string>{"full"}));
    EXPECT_TRUE(config.topology().layer(1).group_tags.empty());
    EXPECT_EQ(config.topology().layer(2).group_tags, (std::vector<std::string>{"linear"}));
    EXPECT_TRUE(config.topology().layer(3).group_tags.empty());

    EXPECT_EQ(config.layerIdsForGroup("full"), (std::vector<int>{0}));
    EXPECT_EQ(config.layerIdsForGroup("linear"), (std::vector<int>{2}));
    EXPECT_TRUE(config.groupsForLayer(1).empty());
    EXPECT_TRUE(config.groupsForLayer(3).empty());
    EXPECT_ANY_THROW(config.soleGroupForLayer(1));

    // Attention-free layers occupy no per-layer block bytes.
    EXPECT_EQ(config.layerBlockStrideBytes(1), 0u);
    EXPECT_EQ(config.layerBlockStrideBytes(3), 0u);
    EXPECT_GT(config.layerBlockStrideBytes(0), 0u);
    EXPECT_GT(config.layerBlockStrideBytes(2), 0u);

    // Warmup capacities are still published for the real cache groups.
    EXPECT_EQ(config.group("full").block_num, 2u);
    EXPECT_EQ(config.group("linear").block_num, 2u);
}

TEST(CacheConfigRelaxationTest, AttentionFreeLayersConsumeNoBlockBudget) {
    ParallelismConfig pc;
    const auto        hybrid = CacheConfigCreator::createWarmupConfig(makeHybridEmptyLayerModelConfig(), pc, 0);

    // Control: same two cached layers on a dense 2-layer model. Dropping the
    // cache from layers 1/3 must not change the per-group budget bytes.
    KVCacheSpecDesc full_desc;
    full_desc.tag        = "full";
    full_desc.cache_type = KVCacheSpecType::MultiHeadAttention;

    KVCacheSpecDesc linear_desc;
    linear_desc.tag        = "linear";
    linear_desc.cache_type = KVCacheSpecType::LinearAttention;

    auto dense_mc = makeHybridEmptyLayerModelConfig();
    dense_mc.num_layers = 2;
    dense_mc.hybrid_attention_config.hybrid_attention_types = {
        HybridAttentionType::NONE, HybridAttentionType::LINEAR};
    dense_mc.kv_cache_spec_descs = {{full_desc}, {linear_desc}};
    const auto dense = CacheConfigCreator::createWarmupConfig(dense_mc, pc, 0);

    EXPECT_EQ(hybrid.blockSizeBytesForGroup("full"), dense.blockSizeBytesForGroup("full"));
    EXPECT_EQ(hybrid.blockSizeBytesForGroup("linear"), dense.blockSizeBytesForGroup("linear"));

    // Mirror CacheConfigCreator's paged budget (FULL + LINEAR groups are paged)
    // and confirm the affordable block count ignores attention-free layers.
    KVCacheBlockBudget budget;
    budget.paged_block_bytes = hybrid.blockSizeBytesForGroup("full") + hybrid.blockSizeBytesForGroup("linear");
    EXPECT_GT(budget.paged_block_bytes, 0u);
    const auto total = budget.paged_block_bytes * 7;
    EXPECT_EQ(maxKVCacheBlockNumForBudget(total, budget, /*linear_step=*/1), 7u);
}

}  // namespace test
}  // namespace rtp_llm
