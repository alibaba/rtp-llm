#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {
namespace test {
namespace {

constexpr uint32_t kTokensPerBlock = 8;

constexpr uint32_t kSwaKvHeadNum    = 8;
constexpr uint32_t kSwaSizePerHead  = 256;
constexpr uint32_t kFullKvHeadNum   = 2;
constexpr uint32_t kFullSizePerHead = 512;

// Gemma4-style heterogeneous KV geometry: the global attn_config carries the
// sliding-layer geometry (8 kv-heads x 256), while the full-attention layers
// override to 2 kv-heads x 512 through per-layer KVCacheSpecDesc fields.
ModelConfig makeGemma4GeometryModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 2;
    mc.hidden_size                  = 256;
    mc.attn_config.head_num         = 16;
    mc.attn_config.kv_head_num      = kSwaKvHeadNum;
    mc.attn_config.size_per_head    = kSwaSizePerHead;
    mc.attn_config.sliding_window   = 1024;
    mc.attn_config.tokens_per_block = kTokensPerBlock;

    KVCacheSpecDesc swa_desc;
    swa_desc.tag           = "swa";
    swa_desc.cache_type    = KVCacheSpecType::MultiHeadAttention;
    swa_desc.group_type    = CacheGroupType::SWA;
    swa_desc.kv_head_num   = kSwaKvHeadNum;
    swa_desc.size_per_head = kSwaSizePerHead;
    CacheTailPolicyDesc tail;
    tail.active_tail_blocks = 129;
    swa_desc.tail           = tail;
    CacheCapacityPolicyDesc capacity;
    capacity.reservable             = false;
    swa_desc.capacity               = capacity;

    KVCacheSpecDesc full_desc;
    full_desc.tag           = "full";
    full_desc.cache_type    = KVCacheSpecType::MultiHeadAttention;
    full_desc.group_type    = CacheGroupType::FULL;
    full_desc.kv_head_num   = kFullKvHeadNum;
    full_desc.size_per_head = kFullSizePerHead;

    mc.kv_cache_spec_descs = {{swa_desc}, {full_desc}};
    return mc;
}

std::shared_ptr<MHAKVCacheSpec>
buildMhaSpec(const KVCacheSpecDesc& desc, const AttentionConfigs& attn, int64_t tp_size) {
    ParallelismConfig parallelism;
    parallelism.tp_size = tp_size;

    SpecBuildContext ctx;
    ctx.dtype                   = DataType::TYPE_FP16;
    ctx.seq_size_per_block      = kTokensPerBlock;
    ctx.kernel_tokens_per_block = kTokensPerBlock;
    ctx.attn_config             = &attn;
    ctx.parallelism_config      = &parallelism;
    return std::dynamic_pointer_cast<MHAKVCacheSpec>(SpecBuilder::build(desc, ctx));
}

KVCacheSpecDesc makeMhaDesc(const std::string& tag, CacheGroupType group_type) {
    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.group_type = group_type;
    return desc;
}

AttentionConfigs makeSwaGeometryAttnConfig() {
    AttentionConfigs attn;
    attn.kv_head_num      = kSwaKvHeadNum;
    attn.size_per_head    = kSwaSizePerHead;
    attn.tokens_per_block = kTokensPerBlock;
    return attn;
}

}  // namespace

TEST(KVCacheSpecGeometryOverrideTest, MhaBuildUsesDescGeometryOverride) {
    const auto attn = makeSwaGeometryAttnConfig();

    auto desc          = makeMhaDesc("full", CacheGroupType::FULL);
    desc.kv_head_num   = kFullKvHeadNum;
    desc.size_per_head = kFullSizePerHead;

    const auto spec = buildMhaSpec(desc, attn, /*tp_size=*/1);
    ASSERT_NE(spec, nullptr);
    EXPECT_EQ(spec->tag, "full");
    EXPECT_EQ(spec->local_kv_head_num, kFullKvHeadNum);
    EXPECT_EQ(spec->seq_size_per_block, kTokensPerBlock);
    // per-token K elems = local_kv_heads * overridden size_per_head = 2 * 512.
    EXPECT_EQ(spec->k_block_size(), static_cast<size_t>(kFullKvHeadNum) * kFullSizePerHead * kTokensPerBlock);
    EXPECT_EQ(spec->v_block_size(), static_cast<size_t>(kFullKvHeadNum) * kFullSizePerHead * kTokensPerBlock);
    EXPECT_EQ(spec->block_size(), 2 * static_cast<size_t>(kFullKvHeadNum) * kFullSizePerHead * kTokensPerBlock);
    EXPECT_EQ(spec->block_size_bytes(),
              2 * static_cast<size_t>(kFullKvHeadNum) * kFullSizePerHead * kTokensPerBlock
                  * getTypeSize(DataType::TYPE_FP16));
}

TEST(KVCacheSpecGeometryOverrideTest, MhaBuildFallsBackToAttnConfigGeometry) {
    const auto attn = makeSwaGeometryAttnConfig();

    // No overrides: the global attn_config geometry (8 x 256) must be used.
    const auto desc = makeMhaDesc("swa", CacheGroupType::SWA);
    const auto spec = buildMhaSpec(desc, attn, /*tp_size=*/1);
    ASSERT_NE(spec, nullptr);
    EXPECT_EQ(spec->local_kv_head_num, kSwaKvHeadNum);
    EXPECT_EQ(spec->block_size(), 2 * static_cast<size_t>(kSwaKvHeadNum) * kSwaSizePerHead * kTokensPerBlock);
}

TEST(KVCacheSpecGeometryOverrideTest, MhaBuildSplitsOverriddenHeadsAcrossTp) {
    const auto attn = makeSwaGeometryAttnConfig();

    auto swa_desc        = makeMhaDesc("swa", CacheGroupType::SWA);
    swa_desc.kv_head_num = kSwaKvHeadNum;
    const auto swa_spec  = buildMhaSpec(swa_desc, attn, /*tp_size=*/2);
    ASSERT_NE(swa_spec, nullptr);
    EXPECT_EQ(swa_spec->local_kv_head_num, kSwaKvHeadNum / 2);
    EXPECT_EQ(swa_spec->block_size(), 2 * static_cast<size_t>(kSwaKvHeadNum / 2) * kSwaSizePerHead * kTokensPerBlock);

    auto full_desc          = makeMhaDesc("full", CacheGroupType::FULL);
    full_desc.kv_head_num   = kFullKvHeadNum;
    full_desc.size_per_head = kFullSizePerHead;
    const auto full_spec    = buildMhaSpec(full_desc, attn, /*tp_size=*/2);
    ASSERT_NE(full_spec, nullptr);
    EXPECT_EQ(full_spec->local_kv_head_num, kFullKvHeadNum / 2);
    EXPECT_EQ(full_spec->block_size(),
              2 * static_cast<size_t>(kFullKvHeadNum / 2) * kFullSizePerHead * kTokensPerBlock);
}

TEST(KVCacheSpecGeometryOverrideTest, MhaBuildRejectsNonPositiveOverride) {
    const auto attn = makeSwaGeometryAttnConfig();

    auto zero_heads        = makeMhaDesc("full", CacheGroupType::FULL);
    zero_heads.kv_head_num = 0;
    EXPECT_THROW((void)buildMhaSpec(zero_heads, attn, /*tp_size=*/1), std::exception);

    auto zero_dim          = makeMhaDesc("full", CacheGroupType::FULL);
    zero_dim.size_per_head = 0;
    EXPECT_THROW((void)buildMhaSpec(zero_dim, attn, /*tp_size=*/1), std::exception);
}

TEST(KVCacheSpecGeometryOverrideTest, WarmupConfigPublishesPerGroupGeometry) {
    const ModelConfig mc = makeGemma4GeometryModelConfig();
    ParallelismConfig pc;

    // One FULL MHA group plus one SWA group must not trip the
    // "multiple FULL MHA/MLA cache groups" restriction.
    const auto config = CacheConfigCreator::createWarmupConfig(mc, pc, /*gen_num_per_cycle=*/0);

    EXPECT_EQ(config.layer_num, 2u);
    EXPECT_EQ(publishedGroupTags(config.topology()), (std::vector<std::string>{"swa", "full"}));
    EXPECT_EQ(publishedGroupTypes(config.topology()),
              (std::vector<CacheGroupType>{CacheGroupType::SWA, CacheGroupType::FULL}));

    const auto& groups = config.topology().groups();
    ASSERT_EQ(groups.size(), 2u);

    const auto& swa_group = groups[0];
    EXPECT_EQ(swa_group.tag, "swa");
    ASSERT_NE(swa_group.spec, nullptr);
    EXPECT_EQ(swa_group.spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(swa_group.spec->local_kv_head_num, kSwaKvHeadNum);
    EXPECT_EQ(swa_group.spec->seq_size_per_block, kTokensPerBlock);
    EXPECT_EQ(swa_group.spec->block_size_bytes(),
              2 * static_cast<size_t>(kSwaKvHeadNum) * kSwaSizePerHead * kTokensPerBlock
                  * getTypeSize(DataType::TYPE_FP16));
    EXPECT_EQ(swa_group.policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(swa_group.policy.sliding_window_size, 1024);
    EXPECT_EQ(swa_group.policy.active_tail_blocks, 129u);
    EXPECT_FALSE(swa_group.policy.reservable);
    EXPECT_EQ(config.layerIdsForGroup("swa"), (std::vector<int>{0}));

    const auto& full_group = groups[1];
    EXPECT_EQ(full_group.tag, "full");
    ASSERT_NE(full_group.spec, nullptr);
    EXPECT_EQ(full_group.spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(full_group.spec->local_kv_head_num, kFullKvHeadNum);
    EXPECT_EQ(full_group.spec->block_size_bytes(),
              2 * static_cast<size_t>(kFullKvHeadNum) * kFullSizePerHead * kTokensPerBlock
                  * getTypeSize(DataType::TYPE_FP16));
    EXPECT_EQ(full_group.policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.layerIdsForGroup("full"), (std::vector<int>{1}));

    size_t full_mha_groups = 0;
    for (const auto& group : config.topology().groups()) {
        if (group.policy.group_type == CacheGroupType::FULL && group.spec
            && (group.spec->type == KVCacheSpecType::MultiHeadAttention
                || group.spec->type == KVCacheSpecType::MultiHeadLatentAttention)) {
            ++full_mha_groups;
        }
    }
    EXPECT_EQ(full_mha_groups, 1u);
}

}  // namespace test
}  // namespace rtp_llm
