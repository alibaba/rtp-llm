#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

ModelConfig makeDomainModel(uint32_t layers, uint32_t head_size, const std::string& tag) {
    ModelConfig model;
    model.num_layers                          = layers;
    model.max_seq_len                         = 64;
    model.data_type                           = DataType::TYPE_FP16;
    model.attn_config.head_num                = 1;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = head_size;
    model.attn_config.tokens_per_block        = 4;
    model.attn_config.kernel_tokens_per_block = 4;
    test::setDefaultKvCacheSpec(model);
    for (auto& layer : model.kv_cache_spec_descs) {
        layer.front().tag = tag;
    }
    return model;
}

void addBudgetGroups(ModelConfig& model, bool host_state) {
    KVCacheSpecDesc state;
    state.tag                              = "state";
    state.cache_type                       = KVCacheSpecType::OpaqueState;
    state.is_state_cache                   = true;
    state.entry_elems                      = 3;
    state.entry_dtype                      = DataType::TYPE_FP16;
    state.explicit_entry_count             = 1;
    state.capacity                         = CacheCapacityPolicyDesc{};
    state.capacity->explicit_block_num     = 7;
    state.capacity->charge_to_paged_budget = !host_state;
    state.memory                           = CacheMemoryPolicyDesc{};
    state.memory->placement = host_state ? CacheMemoryPlacement::HOST_PINNED : CacheMemoryPlacement::DEVICE;
    for (auto& layer : model.kv_cache_spec_descs) {
        auto window       = layer.front();
        window.tag        = "window";
        window.group_type = CacheGroupType::SWA;
        layer.push_back(window);
        layer.push_back(state);
    }
}

}  // namespace

TEST(CacheDomainConfigCreatorTest, KeepsLocalLayersAndOriginalTagsWithoutPlaceholders) {
    const auto                 main  = makeDomainModel(2, 2, "target");
    const auto                 draft = makeDomainModel(3, 4, "default");
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_MTP;
    sp.gen_num_per_cycle = 3;
    auto domain          = CacheConfigCreator::createDomainConfig(main, {}, {}, sp, &draft, true);
    ASSERT_EQ(domain.moduleCount(), 4u);
    EXPECT_EQ(domain.main().layer_num, 2u);
    EXPECT_EQ(domain.main().groupTags(), (std::vector<std::string>{"target"}));
    for (size_t slot = 1; slot < domain.moduleCount(); ++slot) {
        const auto& leaf = domain.module(CacheModuleId(slot));
        EXPECT_EQ(leaf.layer_num, 3u);
        EXPECT_EQ(leaf.topology().layers().size(), 3u);
        EXPECT_EQ(leaf.groupTags(), (std::vector<std::string>{"default"}));
        EXPECT_EQ(leaf.layerIdsForGroup("default"), (std::vector<int>{0, 1, 2}));
        EXPECT_EQ(domain.groupForLayer(CacheModuleId(slot), 2, "default").tag, "default");
    }
    EXPECT_ANY_THROW(domain.groupForLayer(CacheModuleId::main(), 2, "target"));
    EXPECT_ANY_THROW(domain.groupForLayer(CacheModuleId(1), 0, "target"));
    domain.finalizeBlockNums(11, {});
    EXPECT_EQ(domain.main().group("target").block_num, 11u);
    EXPECT_EQ(domain.module(CacheModuleId(3)).group("default").block_num, 11u);
}

TEST(CacheDomainConfigCreatorTest, DecodeKeepsFullLocalStorageForShardedPrefillSource) {
    const auto model = makeDomainModel(2, 2, "default");
    ParallelismConfig parallelism;
    parallelism.role_type                          = RoleType::DECODE;
    parallelism.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    parallelism.tp_rank                            = 1;
    parallelism.tp_size                            = 2;

    const auto decode = CacheConfigCreator::createDomainConfig(model, parallelism, {});
    EXPECT_EQ(decode.main().storage_cp_size, 1);
    EXPECT_EQ(decode.main().storage_cp_rank, 0);
    EXPECT_FALSE(decode.main().request_projection_required);
    EXPECT_EQ(decode.main().cache_info_block_tokens.value(), 4u);

    parallelism.role_type = RoleType::PREFILL;
    const auto prefill = CacheConfigCreator::createDomainConfig(model, parallelism, {});
    EXPECT_EQ(prefill.main().storage_cp_size, 2);
    EXPECT_EQ(prefill.main().storage_cp_rank, 1);
    EXPECT_TRUE(prefill.main().request_projection_required);
    EXPECT_EQ(prefill.main().cache_info_block_tokens.value(), 8u);
}

TEST(CacheDomainConfigCreatorTest, DsparkBillsEachSlotAndEagleKeepsOneDraftSlot) {
    const auto main  = makeDomainModel(2, 2, "shared");
    const auto draft = makeDomainModel(3, 4, "shared");
    EXPECT_EQ(CacheConfigCreator::createDomainConfig(main, {}, {}).moduleCount(), 1u);
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_DSPARK;
    sp.gen_num_per_cycle = 3;
    const auto dspark    = CacheConfigCreator::createDomainConfig(main, {}, {}, sp, &draft, true);
    const auto eagle     = CacheConfigCreator::createDomainConfig(main, {}, {}, sp, &draft, true, true);
    ASSERT_EQ(dspark.moduleCount(), 4u);
    ASSERT_EQ(eagle.moduleCount(), 2u);
    EXPECT_EQ(dspark.activeModuleCount(), 2u);
    EXPECT_TRUE(dspark.isActive(CacheModuleId(1)));
    EXPECT_FALSE(dspark.isActive(CacheModuleId(2)));
    EXPECT_FALSE(dspark.isActive(CacheModuleId(3)));
    EXPECT_EQ(dspark.groupKeys(),
              (std::vector<ModuleGroupKey>{{CacheModuleId(0), "shared"},
                                           {CacheModuleId(1), "shared"},
                                           {CacheModuleId(2), "shared"},
                                           {CacheModuleId(3), "shared"}}));
    // FP16 MHA bytes: K+V * 4 tokens * 1 KV head * head size * 2 bytes * layers.
    EXPECT_EQ(CacheConfigCreator::blockBudget(dspark).paged_block_bytes, 64u + 3u * 192u);
    EXPECT_EQ(CacheConfigCreator::blockBudget(eagle).paged_block_bytes, 64u + 192u);
}

TEST(CacheDomainConfigCreatorTest, JointBudgetChargesPagedSwaAndDeviceReserveExactlyOnce) {
    for (bool host_state : {false, true}) {
        auto main = makeDomainModel(2, 2, "shared");
        addBudgetGroups(main, host_state);
        const auto                 draft = makeDomainModel(3, 4, "shared");
        SpeculativeExecutionConfig sp;
        sp.type              = SP_TYPE_DSPARK;
        sp.gen_num_per_cycle = 3;
        const auto   domain  = CacheConfigCreator::createDomainConfig(main, {}, {}, sp, &draft, true);
        const auto   budget  = CacheConfigCreator::blockBudget(domain);
        const size_t reserve = host_state ? 0 : 3u * 2u * 2u * 7u;
        EXPECT_EQ(budget.explicit_pool_reserve_bytes, reserve);
        EXPECT_EQ(budget.paged_block_bytes, 640u);
        EXPECT_EQ(budget.swa_block_bytes, 64u);
        const size_t bytes_for_seven_blocks = reserve + 7u * 640u + 3u * 64u;
        EXPECT_EQ(maxKVCacheBlockNumForBudget(bytes_for_seven_blocks, budget, 3), 7u);
        EXPECT_EQ(maxKVCacheBlockNumForBudget(bytes_for_seven_blocks - 1, budget, 3), 6u);
        const auto one_draft = CacheConfigCreator::createDomainConfig(main, {}, {}, sp, &draft, true, true);
        EXPECT_GT(maxKVCacheBlockNumForBudget(bytes_for_seven_blocks, CacheConfigCreator::blockBudget(one_draft), 3),
                  7u);
    }
}

TEST(CacheDomainConfigCreatorTest, RejectsMissingSpeculationAndIncompatibleBaseKeyGrid) {
    const auto main  = makeDomainModel(2, 2, "shared");
    auto       draft = makeDomainModel(3, 4, "shared");
    EXPECT_ANY_THROW(CacheConfigCreator::createDomainConfig(main, {}, {}, std::nullopt, &draft, true));
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_MTP;
    sp.gen_num_per_cycle = 0;
    EXPECT_ANY_THROW(CacheConfigCreator::createDomainConfig(main, {}, {}, sp, &draft, true));
    sp.gen_num_per_cycle                      = 2;
    draft.attn_config.tokens_per_block        = 8;
    draft.attn_config.kernel_tokens_per_block = 8;
    EXPECT_ANY_THROW(CacheConfigCreator::createDomainConfig(main, {}, {}, sp, &draft, true));
}

TEST(CacheDomainConfigCreatorTest, SharedCpKeyGridAllowsDifferentLeafReuseGranularity) {
    auto main                             = test::makeSimpleMhaCacheConfig(1, 8, 4, TYPE_FP16);
    auto draft                            = test::makeSimpleLinearCacheConfig(2, 8, 4, TYPE_FP16);
    auto main_groups                      = main.groups();
    main_groups.front().policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    main.setTopology(std::move(main_groups), main.topology().layers());
    ParallelismConfig parallelism;
    parallelism.tp_size                            = 2;
    parallelism.tp_rank                            = 0;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    main.finalizeStorageLayout(parallelism);
    draft.finalizeStorageLayout(parallelism);
    ASSERT_EQ(main.reuse_unit_tokens, 8u);
    ASSERT_EQ(draft.reuse_unit_tokens, 4u);
    ASSERT_EQ(main.group("default").cacheKeyTokenStride(), draft.group("linear").cacheKeyTokenStride());
    CacheDomainConfig domain(std::vector<CacheConfig>{main, draft});
    EXPECT_EQ(domain.main().reuse_unit_tokens, 8u);
    EXPECT_EQ(domain.module(CacheModuleId(1)).reuse_unit_tokens, 4u);
}

}  // namespace rtp_llm
