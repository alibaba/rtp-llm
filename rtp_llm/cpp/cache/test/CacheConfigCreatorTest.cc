#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/RequestPrefixResource.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm {
namespace {

CacheConfig makeCapacityConfig(const std::vector<std::pair<std::string, uint32_t>>& tagged_spans) {
    CacheConfig config;
    config.dtype         = DataType::TYPE_FP16;
    config.layer_num     = static_cast<uint32_t>(tagged_spans.size());
    config.layer_all_num = config.layer_num;
    std::vector<GroupBase> groups;
    groups.reserve(tagged_spans.size());
    for (size_t i = 0; i < tagged_spans.size(); ++i) {
        const auto& [tag, span] = tagged_spans[i];
        auto spec               = test::makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, span, tag);
        groups.push_back(
            test::makeTestGroupBase(spec, defaultCacheGroupPolicy(CacheGroupType::FULL), {static_cast<int>(i)}));
    }
    test::setTestTopology(config, std::move(groups));
    return config;
}

size_t capacityBytes(const CacheConfig& config, uint64_t tokens) {
    size_t bytes = 0;
    for (const auto& group : config.topology().groups()) {
        const auto blocks = group.policy.fixed_block_num > 0 ?
                                group.policy.fixed_block_num :
                                (tokens + group.seq_size_per_block - 1) / group.seq_size_per_block;
        if (group.policy.fixed_block_num == 0 || group.policy.charge_to_paged_budget) {
            bytes += blocks * config.blockSizeBytesForGroup(group.tag);
        }
    }
    return bytes;
}

ModelConfig makeSingleModelConfig(uint32_t layer_num = 2) {
    ModelConfig model_config;
    model_config.num_layers                   = layer_num;
    model_config.max_seq_len                  = 128;
    model_config.hidden_size                  = 128;
    model_config.attn_config.head_num         = 4;
    model_config.attn_config.kv_head_num      = 2;
    model_config.attn_config.size_per_head    = 32;
    model_config.attn_config.tokens_per_block = 8;
    model_config.kv_cache_spec_descs.resize(layer_num);
    for (auto& layer_descs : model_config.kv_cache_spec_descs) {
        layer_descs.push_back(KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention});
    }
    return model_config;
}

ModelConfig makeHybridModelConfig() {
    ModelConfig model_config                                     = makeSingleModelConfig(/*layer_num=*/4);
    model_config.hybrid_attention_config.enable_hybrid_attention = true;
    model_config.hybrid_attention_config.hybrid_attention_types  = {
        HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::LINEAR, HybridAttentionType::NONE};
    model_config.linear_attention_config.linear_conv_kernel_dim = 4;
    model_config.linear_attention_config.linear_key_head_dim    = 16;
    model_config.linear_attention_config.linear_value_head_dim  = 16;
    model_config.linear_attention_config.linear_num_key_heads   = 2;
    model_config.linear_attention_config.linear_num_value_heads = 2;
    for (size_t layer_id = 0; layer_id < model_config.kv_cache_spec_descs.size(); ++layer_id) {
        const bool is_linear =
            model_config.hybrid_attention_config.hybrid_attention_types[layer_id] == HybridAttentionType::LINEAR;
        model_config.kv_cache_spec_descs[layer_id] = {
            KVCacheSpecDesc{is_linear ? "linear" : "full",
                            is_linear ? KVCacheSpecType::LinearAttention : KVCacheSpecType::MultiHeadAttention}};
    }
    return model_config;
}

TEST(CacheConfigCreatorTest, BasicConfigPublishesDefaultKernelShape) {
    ParallelismConfig parallelism_config;
    auto              config = CacheConfigCreator::createBasicConfig(
        makeSingleModelConfig(), parallelism_config, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);

    EXPECT_EQ(config.seqSizePerBlockForGroup("full"), 8u);
    EXPECT_EQ(config.kernelSeqSizePerBlockForGroup("full"), 8u);
    EXPECT_EQ(config.kvBlockStrideBytesForGroup("full"), config.specForGroup("full")->block_size_bytes());
    EXPECT_EQ(config.kvScaleStrideBytesForGroup("full"), config.specForGroup("full")->scale_block_size_bytes());
}

TEST(CacheConfigCreatorTest, ModelKernelShapePopulatesSingleGroup) {
    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num                   = 2;
    auto model_config                                = makeSingleModelConfig();
    model_config.attn_config.kernel_tokens_per_block = 4;

    auto config = CacheConfigCreator::createConfig(model_config, parallelism_config, runtime_config, kv_cache_config);

    EXPECT_EQ(config.linear_step, 1);
    EXPECT_EQ(config.kernelSeqSizePerBlockForGroup("full"), 4u);
    EXPECT_EQ(config.kernelBlocksPerKvBlockForGroup("full"), 2u);
}

TEST(CacheConfigCreatorTest, ModelKernelShapeRejectsNonDivisibleShape) {
    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num                   = 2;
    auto model_config                                = makeSingleModelConfig();
    model_config.attn_config.kernel_tokens_per_block = 3;

    EXPECT_THROW(
        (void)CacheConfigCreator::createConfig(model_config, parallelism_config, runtime_config, kv_cache_config),
        std::exception);
}

TEST(CacheConfigCreatorTest, HybridAppliesModelKernelShapeOnlyToFullGroup) {
    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num                   = 2;
    auto model_config                                = makeHybridModelConfig();
    model_config.attn_config.kernel_tokens_per_block = 4;

    auto config = CacheConfigCreator::createConfig(model_config, parallelism_config, runtime_config, kv_cache_config);

    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_EQ(config.kernelSeqSizePerBlockForGroup("full"), 4u);
    EXPECT_EQ(config.kernelSeqSizePerBlockForGroup("linear"), config.seqSizePerBlockForGroup("linear"));
    EXPECT_EQ(config.kvBlockStrideBytesForGroup("full"), config.specForGroup("full")->block_size_bytes());
    EXPECT_EQ(config.kvBlockStrideBytesForGroup("linear"), config.specForGroup("linear")->block_size_bytes());
    EXPECT_NE(config.kvBlockStrideBytesForGroup("full"), config.kvBlockStrideBytesForGroup("linear"));
}

TEST(CacheConfigCreatorTest, HybridPoolPublishesCompleteGroupsFromModelShape) {
    auto model_config                                = makeHybridModelConfig();
    model_config.attn_config.kernel_tokens_per_block = 4;

    ParallelismConfig parallelism_config;
    auto config = CacheConfigCreator::createBasicConfig(model_config, parallelism_config, /*is_mtp=*/false, /*gen=*/0);

    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_EQ(config.kernelSeqSizePerBlockForGroup("full"), 4u);
    EXPECT_EQ(config.kernelSeqSizePerBlockForGroup("linear"), config.seqSizePerBlockForGroup("linear"));
    for (const auto& group : config.topology().groups()) {
        EXPECT_GT(group.seq_size_per_block, 0u) << group.tag;
        EXPECT_GT(group.kernel_seq_size_per_block, 0u) << group.tag;
        EXPECT_GT(group.kv_block_stride_bytes, 0u) << group.tag;
    }
}

TEST(CacheConfigCreatorTest, SetTopologyRejectsIncompleteGroupShape) {
    CacheConfig config;
    config.layer_num     = 1;
    config.layer_all_num = 1;

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = "full";
    spec->seq_size_per_block = 8;
    GroupBase group;
    group.tag       = spec->tag;
    group.spec      = spec;
    group.policy    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids = {0};

    EXPECT_THROW(config.setTopology({group}, {{0, {"full"}}}), std::exception);
}

TEST(CacheConfigCreatorTest, PolicyUpdatePreservesGroupShape) {
    ParallelismConfig parallelism_config;
    auto              model_config                   = makeSingleModelConfig();
    model_config.attn_config.kernel_tokens_per_block = 4;
    auto config = CacheConfigCreator::createBasicConfig(model_config, parallelism_config, /*is_mtp=*/false, /*gen=*/0);

    const auto                                        original = config.group("full");
    std::unordered_map<std::string, CacheGroupPolicy> policies{{original.tag, original.policy}};
    policies.at(original.tag).enable_prefix_reuse = !original.policy.enable_prefix_reuse;
    config.setGroupPolicies(policies);

    const auto& updated = config.group("full");
    EXPECT_EQ(updated.seq_size_per_block, original.seq_size_per_block);
    EXPECT_EQ(updated.kernel_seq_size_per_block, original.kernel_seq_size_per_block);
    EXPECT_EQ(updated.kv_block_stride_bytes, original.kv_block_stride_bytes);
    EXPECT_EQ(updated.kv_scale_stride_bytes, original.kv_scale_stride_bytes);
}

TEST(CacheConfigCreatorTest, TokenCapacitySupportsOneTwoAndSevenGroups) {
    for (const size_t group_num : {1u, 2u, 7u}) {
        std::vector<std::pair<std::string, uint32_t>> groups;
        for (size_t i = 0; i < group_num; ++i) {
            groups.emplace_back("group" + std::to_string(i), 4);
        }
        auto       config = makeCapacityConfig(groups);
        const auto budget = capacityBytes(config, 20);
        EXPECT_EQ(maxKVCacheTokenCapacityForBudget(budget, config), 20u);
        config.applyTokenCapacity(20);
        for (const auto& group : config.topology().groups()) {
            EXPECT_EQ(group.block_num, 5u);
        }
    }
}

TEST(CacheConfigCreatorTest, FixedBlockNumUsesZeroAsLogicalSentinel) {
    KVCacheSpecDesc desc;
    desc.tag = "full";

    const auto logical_policy = SpecBuilder::groupPolicy(desc);
    EXPECT_EQ(logical_policy.fixed_block_num, 0u);

    desc.capacity                  = CacheCapacityPolicyDesc{};
    desc.capacity->fixed_block_num = 3;
    const auto fixed_policy        = SpecBuilder::groupPolicy(desc);
    EXPECT_EQ(fixed_policy.fixed_block_num, 3u);

    desc.capacity->fixed_block_num = 0;
    EXPECT_THROW((void)SpecBuilder::groupPolicy(desc), std::exception);
}

TEST(CacheConfigCreatorTest, HeterogeneousPhysicalSpansAreOrderIndependent) {
    auto       first  = makeCapacityConfig({{"span4", 4}, {"span6", 6}});
    auto       second = makeCapacityConfig({{"span6", 6}, {"span4", 4}});
    const auto budget = capacityBytes(first, 24);

    EXPECT_EQ(maxKVCacheTokenCapacityForBudget(budget, first), 24u);
    EXPECT_EQ(maxKVCacheTokenCapacityForBudget(budget, second), 24u);
    first.applyTokenCapacity(24);
    second.applyTokenCapacity(24);
    EXPECT_EQ(first.blockNumForGroup("span4"), 6u);
    EXPECT_EQ(first.blockNumForGroup("span6"), 4u);
    EXPECT_EQ(second.blockNumForGroup("span4"), 6u);
    EXPECT_EQ(second.blockNumForGroup("span6"), 4u);
    RequestPrefixResource first_prefix;
    RequestPrefixResource second_prefix;
    first_prefix.configure(first.topology());
    second_prefix.configure(second.topology());
    EXPECT_EQ(first_prefix.matchSpanTokens(), 12u);
    EXPECT_EQ(second_prefix.matchSpanTokens(), 12u);
}

TEST(CacheConfigCreatorTest, FixedGroupChargesBudgetWithoutCappingLogicalTokens) {
    auto                                              config = makeCapacityConfig({{"logical", 4}, {"fixed", 10}});
    std::unordered_map<std::string, CacheGroupPolicy> policies;
    for (const auto& group : config.topology().groups()) {
        policies.emplace(group.tag, group.policy);
    }
    policies.at("fixed").fixed_block_num        = 3;
    policies.at("fixed").charge_to_paged_budget = true;
    config.setGroupPolicies(policies);

    const auto budget = capacityBytes(config, 50);
    EXPECT_EQ(maxKVCacheTokenCapacityForBudget(budget, config), 52u);
    config.applyTokenCapacity(52);
    EXPECT_EQ(config.blockNumForGroup("logical"), 13u);
    EXPECT_EQ(config.blockNumForGroup("fixed"), 3u);
    EXPECT_EQ(config.tokenCapacity(), 52u);
    EXPECT_THROW(maxKVCacheTokenCapacityForBudget(3 * config.blockSizeBytesForGroup("fixed") - 1, config),
                 std::exception);
}

TEST(CacheConfigCreatorTest, GroupBlockBytesFailFastBeforeStrideArithmeticWraps) {
    auto config                     = makeCapacityConfig({{"full", 4}});
    auto groups                     = config.topology().groups();
    groups[0].kv_block_stride_bytes = std::numeric_limits<size_t>::max();
    groups[0].kv_scale_stride_bytes = 1;
    config.setTopology(std::move(groups), config.topology().layers());

    EXPECT_THROW((void)config.blockSizeBytesForGroup("full"), std::exception);
}

TEST(CacheConfigCreatorTest, CapacityPlannerRejectsPathologicalBoundaryCount) {
    auto config = makeCapacityConfig({{"unit", 1}, {"large", (1u << 20) + 1}});
    EXPECT_THROW(maxKVCacheTokenCapacityForBudget(std::numeric_limits<size_t>::max(), config), std::exception);
}

TEST(CacheConfigCreatorTest, NonUnitLinearStepPropagatesToGroupOwnedAllocator) {
    auto              model_config = makeSingleModelConfig();
    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 2;
    kv_cache_config.linear_step    = 2;
    const auto config =
        CacheConfigCreator::createConfig(model_config, parallelism_config, runtime_config, kv_cache_config);
    EXPECT_EQ(config.linear_step, 2);
    EXPECT_EQ(config.blockNumForGroup("full"), 2u);
}

TEST(CacheConfigCreatorTest, SpeculativeConfigPropagatesKernelShapeToMainAndMtpModules) {
    auto score_model_config                                  = makeSingleModelConfig(/*layer_num=*/2);
    auto propose_model_config                                = makeSingleModelConfig(/*layer_num=*/1);
    score_model_config.attn_config.kernel_tokens_per_block   = 4;
    propose_model_config.attn_config.kernel_tokens_per_block = 4;

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 2;
    kv_cache_config.linear_step    = 2;

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 2;

    auto config = CacheConfigCreator::createSpConfig(score_model_config,
                                                     propose_model_config,
                                                     parallelism_config,
                                                     runtime_config,
                                                     kv_cache_config,
                                                     sp_config,
                                                     std::nullopt,
                                                     /*is_mtp=*/true,
                                                     /*is_eagle=*/false);

    EXPECT_EQ(config.linear_step, 2);
    EXPECT_EQ(config.kernelSeqSizePerBlockForGroup("full"), 4u);
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_NE(sub_config, nullptr);
        EXPECT_EQ(sub_config->linear_step, 2);
        EXPECT_EQ(sub_config->kernelSeqSizePerBlockForGroup("full"), 4u);
    }
}

TEST(CacheConfigCreatorTest, SpeculativeConfigRejectsSharedTagPhysicalSpanMismatch) {
    auto score_model_config                           = makeSingleModelConfig(/*layer_num=*/2);
    auto propose_model_config                         = makeSingleModelConfig(/*layer_num=*/1);
    propose_model_config.attn_config.tokens_per_block = 4;

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 2;
    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 1;

    EXPECT_THROW(CacheConfigCreator::createSpConfig(score_model_config,
                                                    propose_model_config,
                                                    parallelism_config,
                                                    runtime_config,
                                                    kv_cache_config,
                                                    sp_config,
                                                    std::nullopt,
                                                    /*is_mtp=*/true,
                                                    /*is_eagle=*/false),
                 std::exception);
}

}  // namespace
}  // namespace rtp_llm
