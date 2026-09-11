#include <gtest/gtest.h>

#include <numeric>
#include <stdexcept>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/cache/DSV41KVCacheSpec.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {
namespace {

ModelConfig modelConfig(bool draft = false) {
    ModelConfig model;
    model.num_layers                = draft ? 3 : 40;
    model.hidden_size               = 5120;
    model.max_seq_len               = 1048576;
    auto& attn                      = model.attn_config;
    attn.dsv41_cache_layout_version = 1;
    attn.head_num                   = 64;
    attn.kv_head_num                = 1;
    attn.size_per_head              = 512;
    attn.sliding_window             = 128;
    attn.indexer_head_dim           = 128;
    attn.indexer_head_num           = 32;
    attn.indexer_topk               = 512;
    for (int layer = 0; layer < model.num_layers; ++layer) {
        attn.layer_compress_ratios.push_back(draft || layer < 2 ? 0 : (layer < 20 ? 2 : 1));
    }
    return model;
}

KVCacheConfig kvConfig(uint32_t block = 128) {
    KVCacheConfig config;
    config.seq_size_per_block     = block;
    config.dsv4_fixed_pool_blocks = 8;
    config.test_block_num         = 4;
    return config;
}

ParallelismConfig cpConfig(RoleType role) {
    ParallelismConfig parallelism;
    parallelism.role_type                          = role;
    parallelism.tp_size                            = role == RoleType::PREFILL ? 8 : 1;
    parallelism.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    parallelism.prefill_cp_config.prefill_cp_size  = 8;
    return parallelism;
}

CacheConfig basicConfig(const ParallelismConfig& parallelism = ParallelismConfig(),
                        const KVCacheConfig&     kv          = kvConfig()) {
    return CacheConfigCreator::createBasicConfig(modelConfig(), parallelism, kv, false, 5);
}

CacheConfig dsparkConfig(const ParallelismConfig& parallelism = cpConfig(RoleType::DECODE)) {
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_DSPARK;
    speculative.gen_num_per_cycle = 5;
    return CacheConfigCreator::createSpConfig(modelConfig(),
                                              modelConfig(true),
                                              parallelism,
                                              RuntimeConfig(),
                                              kvConfig(),
                                              speculative,
                                              std::nullopt,
                                              true,
                                              false);
}

const DSV41KVCacheSpec& spec(const CacheConfig& config, size_t group) {
    auto typed = std::dynamic_pointer_cast<DSV41KVCacheSpec>(config.cache_specs[group]);
    if (!typed) {
        throw std::runtime_error("expected a V4.1 compact cache spec");
    }
    return *typed;
}

}  // namespace

TEST(DSV41CacheConfigTest, ExplicitVersionDispatchesToCompactOwnerPools) {
    auto config = basicConfig();
    ASSERT_EQ(config.dsv41_cache_layout_version, 1);
    ASSERT_EQ(config.cache_specs.size(), 6u);
    EXPECT_EQ(config.global_layer_ids[0], (std::vector<int>{2, 8, 14}));
    EXPECT_EQ(config.global_layer_ids[1], (std::vector<int>{20}));
    EXPECT_EQ(config.global_layer_ids[2], (std::vector<int>{2, 8, 14}));
    EXPECT_EQ(config.global_layer_ids[3], (std::vector<int>{20}));
    EXPECT_EQ(config.global_layer_ids[4], (std::vector<int>{2, 8, 14}));
    EXPECT_EQ(config.global_layer_ids[5].size(), 40u);
    EXPECT_EQ(config.full_group_num, 4);
    EXPECT_EQ(config.swa_group_num, 2);
    EXPECT_EQ(spec(config, 0).entry_bytes, 288u);
    EXPECT_EQ(spec(config, 1).ratio, 1u);
    EXPECT_EQ(spec(config, 2).entry_bytes, 68u);
    EXPECT_EQ(spec(config, 5).entry_bytes, 528u);
    EXPECT_EQ(spec(config, 0).encoding, DSV41KVEncoding::GLOBAL_FP4_E4M3);
    EXPECT_EQ(spec(config, 2).encoding, DSV41KVEncoding::INDEX_FP4_UE8M0);
    EXPECT_EQ(spec(config, 5).encoding, DSV41KVEncoding::SWA_FP8_UE8M0);
}

TEST(DSV41CacheConfigTest, ConsumersShareOwnerMappingsWithoutExtraPhysicalSlots) {
    auto config = basicConfig();
    for (int layer = 2; layer < 40; ++layer) {
        const int expected = layer < 20 ? 2 + ((layer - 2) / 6) * 6 : 20;
        EXPECT_EQ(config.physicalOwner(layer, KVCacheRegionName::DSV41_GLOBAL_KV), expected);
        EXPECT_EQ(config.physicalOwner(layer, KVCacheRegionName::DSV41_INDEX_KV), expected);
        EXPECT_EQ(config.physicalOwner(layer, KVCacheRegionName::SWA_KV), layer);
        EXPECT_EQ(config.dsv41_topk_owner[layer], layer < 20 ? expected : 20 + ((layer - 20) / 4) * 4);
    }
    EXPECT_THROW(config.physicalOwner(0, KVCacheRegionName::DSV41_GLOBAL_KV), std::exception);
    EXPECT_EQ(config.physicalOwner(39, KVCacheRegionName::DEFAULT), 39);
    EXPECT_EQ(config.global_layer_ids[0].size() + config.global_layer_ids[1].size(), 4u);
    EXPECT_EQ(config.global_layer_ids[2].size() + config.global_layer_ids[3].size(), 4u);
    EXPECT_TRUE(skipReuseCacheRegion(KVCacheRegionName::DSV41_PAIR_STATE));
    EXPECT_FALSE(skipReuseCacheRegion(KVCacheRegionName::SWA_KV));
}

TEST(DSV41CacheConfigTest, UntaggedRatioOneOrTwoCannotEnterLegacyHca) {
    for (int ratio : {1, 2}) {
        auto model                                   = modelConfig();
        model.attn_config.dsv41_cache_layout_version = 0;
        model.attn_config.layer_compress_ratios[2]   = ratio;
        EXPECT_THROW(CacheConfigCreator::createBasicConfig(model, ParallelismConfig(), kvConfig(), false, 5),
                     std::exception);
    }
}

TEST(DSV41CacheConfigTest, LegacyV4KeepsSevenPoolsAnd584ByteRows) {
    auto model                                   = modelConfig();
    model.attn_config.dsv41_cache_layout_version = 0;
    model.attn_config.kv_cache_dtype             = KvCacheDataType::FP8;
    for (int layer = 2; layer < 40; ++layer) {
        model.attn_config.layer_compress_ratios[layer] = layer % 2 ? 128 : 4;
    }
    auto config = CacheConfigCreator::createBasicConfig(model, ParallelismConfig(), kvConfig(), false, 5);
    EXPECT_EQ(config.dsv41_cache_layout_version, 0);
    EXPECT_EQ(config.cache_specs.size(), 7u);
    EXPECT_TRUE(config.layer_region_to_owner.empty());
    EXPECT_EQ(config.group_region_names[0], KVCacheRegionName::CSA_KV);
    EXPECT_EQ(config.group_region_names[1], KVCacheRegionName::HCA_KV);
    EXPECT_EQ(config.global_layer_ids[6].size(), 40u);
    auto main  = std::dynamic_pointer_cast<DSV4KVSpec>(config.cache_specs[0]);
    auto index = std::dynamic_pointer_cast<DSV4KVSpec>(config.cache_specs[2]);
    ASSERT_NE(main, nullptr);
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(main->entry_elems, 584u);
    EXPECT_EQ(index->entry_elems, 132u);
}

TEST(DSV41CacheConfigTest, Cp8UsesWholeGlobalPagesAndByteSlicedFixedState) {
    for (uint32_t block : {128u, 256u}) {
        auto prefill = basicConfig(cpConfig(RoleType::PREFILL), kvConfig(block));
        auto decode  = basicConfig(cpConfig(RoleType::DECODE), kvConfig(block));
        EXPECT_EQ(prefill.block_size_bytes, decode.block_size_bytes);
        EXPECT_EQ(decode.block_size_bytes, block == 128 ? 114688u : 227840u);
        for (size_t gid = 0; gid < 4; ++gid) {
            EXPECT_EQ(spec(prefill, gid).block_size_bytes(), spec(decode, gid).block_size_bytes());
            EXPECT_FALSE(spec(prefill, gid).prefill_byte_slice);
        }
        for (size_t gid : {4u, 5u}) {
            EXPECT_EQ(spec(prefill, gid).block_size_bytes() * 8, spec(decode, gid).block_size_bytes());
            EXPECT_TRUE(spec(prefill, gid).prefill_byte_slice);
            EXPECT_EQ(prefill.group_seq_size_per_block[gid], block * 8);
        }
        EXPECT_EQ(spec(decode, 5).entries_per_block, 136u);
        EXPECT_EQ(spec(decode, 5).block_size_bytes(), 72192u);
        EXPECT_EQ(spec(prefill, 5).block_size_bytes(), 9024u);
    }
}

TEST(DSV41CacheConfigTest, IndexPagesPreserve512ByteAlignment) {
    auto config = basicConfig(cpConfig(RoleType::DECODE));
    EXPECT_EQ(spec(config, 2).entries_per_block, 64u);
    EXPECT_EQ(spec(config, 2).block_size_bytes(), 4608u);
    EXPECT_EQ(spec(config, 3).entries_per_block, 128u);
    EXPECT_EQ(spec(config, 3).block_size_bytes(), 8704u);
    for (size_t gid = 0; gid < 4; ++gid) {
        EXPECT_EQ(spec(config, gid).block_size_bytes() % 512, 0u);
        EXPECT_EQ(config.group_kv_scale_stride_bytes[gid], 0u);
    }
}

TEST(DSV41CacheConfigTest, PairStateHasSevenCompleteTypedSnapshots) {
    auto config = basicConfig(cpConfig(RoleType::DECODE));
    EXPECT_EQ(spec(config, 4).entries_per_block, 7u);
    EXPECT_EQ(spec(config, 4).entry_bytes, 4112u);
    EXPECT_EQ(DSV41KVCacheSpec::kPairScoreOffset, 2048u);
    EXPECT_EQ(DSV41KVCacheSpec::kPairPositionOffset, 4096u);
    EXPECT_EQ(DSV41KVCacheSpec::kPairValidOffset, 4104u);
    EXPECT_EQ(spec(config, 4).block_size_bytes(), 29184u);
    EXPECT_EQ(config.state_block_size_bytes, 3u * 29184);
}

TEST(DSV41CacheConfigTest, RejectsWrongSchemaRatiosAndGeometry) {
    auto model                                   = modelConfig();
    model.attn_config.dsv41_cache_layout_version = 2;
    EXPECT_THROW(CacheConfigCreator::createBasicConfig(model, ParallelismConfig(), kvConfig(), false, 5),
                 std::exception);
    model                                       = modelConfig();
    model.attn_config.layer_compress_ratios[20] = 2;
    EXPECT_THROW(CacheConfigCreator::createBasicConfig(model, ParallelismConfig(), kvConfig(), false, 5),
                 std::exception);
    model                              = modelConfig();
    model.attn_config.indexer_head_num = 64;
    EXPECT_THROW(CacheConfigCreator::createBasicConfig(model, ParallelismConfig(), kvConfig(), false, 5),
                 std::exception);
    EXPECT_THROW(basicConfig(ParallelismConfig(), kvConfig(64)), std::exception);
    auto kv                      = kvConfig(256);
    kv.kernel_seq_size_per_block = 128;
    EXPECT_THROW(basicConfig(ParallelismConfig(), kv), std::exception);
}

TEST(DSV41CacheConfigTest, RejectsCp4AndMissingDecodeCpIdentity) {
    auto prefill    = cpConfig(RoleType::PREFILL);
    prefill.tp_size = 4;
    EXPECT_THROW(basicConfig(prefill), std::exception);
    auto decode                              = cpConfig(RoleType::DECODE);
    decode.prefill_cp_config.prefill_cp_size = 0;
    EXPECT_THROW(basicConfig(decode), std::exception);
    auto unsharded                               = cpConfig(RoleType::PREFILL);
    unsharded.prefill_cp_config.kv_cache_sharded = false;
    EXPECT_THROW(basicConfig(unsharded), std::exception);
}

TEST(DSV41CacheConfigTest, RejectsDiskIndependentEvictionAndDirectHostActivePools) {
    auto kv                     = kvConfig();
    kv.enable_memory_cache_disk = true;
    EXPECT_THROW(basicConfig(ParallelismConfig(), kv), std::exception);
    kv                                              = kvConfig();
    kv.enable_dsv4_state_block_independent_eviction = true;
    EXPECT_THROW(basicConfig(ParallelismConfig(), kv), std::exception);
    kv                            = kvConfig();
    kv.dsv4_fixed_pool_use_memory = true;
    EXPECT_THROW(basicConfig(ParallelismConfig(), kv), std::exception);
}

TEST(DSV41CacheConfigTest, DsparkAddsThreeSwaLayersWithoutExtraGlobalOwners) {
    auto config = dsparkConfig();
    EXPECT_EQ(config.layer_num, 40u);
    EXPECT_EQ(config.layer_all_num, 43u);
    EXPECT_EQ(config.global_layer_ids[5].size(), 43u);
    EXPECT_EQ(spec(config, 5).layer_num, 43u);
    EXPECT_EQ(config.block_size_bytes, 114688u);
    EXPECT_EQ(config.swa_block_size_bytes, 43u * 72192);
    EXPECT_EQ(config.group_block_size_bytes[5], 43u * 72192);
    EXPECT_EQ(config.mtp_sub_configs.size(), 1u);
    EXPECT_EQ(config.mtp_sub_configs[0]->block_size_bytes, 0u);
    EXPECT_EQ(config.mtp_sub_configs[0]->global_layer_ids[5], (std::vector<int>{40, 41, 42}));
    for (int layer = 40; layer < 43; ++layer) {
        EXPECT_EQ(config.physicalOwner(layer, KVCacheRegionName::SWA_KV), layer);
        EXPECT_EQ(config.dsv41_topk_owner[layer], -1);
        EXPECT_THROW(config.physicalOwner(layer, KVCacheRegionName::DSV41_GLOBAL_KV), std::exception);
    }
    size_t slots = 0;
    for (const auto& owners : config.global_layer_ids) {
        slots += owners.size();
    }
    EXPECT_EQ(slots, 54u);  // 43 SWA + 8 global/index + 3 active pair-state regions.
}

TEST(DSV41CacheConfigTest, PhysicalBudgetCountsEachOwnerOnlyOnce) {
    auto   config         = dsparkConfig();
    size_t physical_bytes = 0;
    for (size_t gid = 0; gid < config.cache_specs.size(); ++gid) {
        auto pool = BlockPoolConfigHelper::createConfigForGroup(config, gid);
        EXPECT_EQ(pool.memory_layouts.size(), 1u);
        EXPECT_EQ(pool.memory_layouts[0].layer_num, config.global_layer_ids[gid].size());
        physical_bytes += pool.total_size_bytes;
    }
    EXPECT_EQ(physical_bytes, 4u * 114688 + 8u * (43u * 72192 + 3u * 29184));
}

class DSV41AllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        initLogger();
        createDevice();
    }
};

TEST_F(DSV41AllocatorTest, SharedReadersReceiveTheSamePhysicalBuffer) {
    auto config    = dsparkConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    ASSERT_EQ(allocator->groupBlockPools().size(), 6u);
    const auto   layout = allocator->allLayerCacheBase();
    const size_t global = static_cast<size_t>(KVCacheRegionName::DSV41_GLOBAL_KV);
    const size_t index  = static_cast<size_t>(KVCacheRegionName::DSV41_INDEX_KV);
    const size_t swa    = static_cast<size_t>(KVCacheRegionName::SWA_KV);
    for (int layer = 2; layer < 40; ++layer) {
        const int owner = config.physicalOwner(layer, KVCacheRegionName::DSV41_GLOBAL_KV);
        ASSERT_TRUE(layout.layers_to_kv_buffer_ptrs_by_attn[layer][global].defined());
        EXPECT_EQ(layout.layers_to_kv_buffer_ptrs_by_attn[layer][global].data_ptr(),
                  layout.layers_to_kv_buffer_ptrs_by_attn[owner][global].data_ptr());
        EXPECT_EQ(layout.layers_to_kv_buffer_ptrs_by_attn[layer][index].data_ptr(),
                  layout.layers_to_kv_buffer_ptrs_by_attn[owner][index].data_ptr());
        EXPECT_EQ(allocator->convertIndexToAddr(layer, KVCacheRegionName::DSV41_GLOBAL_KV, 1).kv_addr,
                  allocator->convertIndexToAddr(owner, KVCacheRegionName::DSV41_GLOBAL_KV, 1).kv_addr);
    }
    for (size_t layer = 0; layer < 43; ++layer) {
        ASSERT_TRUE(layout.layers_to_kv_buffer_ptrs_by_attn[layer][swa].defined());
        if (layer > 0) {
            EXPECT_NE(layout.layers_to_kv_buffer_ptrs_by_attn[layer][swa].data_ptr(),
                      layout.layers_to_kv_buffer_ptrs_by_attn[layer - 1][swa].data_ptr());
        }
    }
}

TEST_F(DSV41AllocatorTest, BlockCopyPreservesOwnerKvAndAllTargetDraftSwa) {
    auto config    = dsparkConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    auto layout = allocator->allLayerCacheBase();
    for (size_t gid = 0; gid < config.global_layer_ids.size(); ++gid) {
        const auto region = static_cast<size_t>(config.group_region_names[gid]);
        for (int owner : config.global_layer_ids[gid]) {
            auto tensor = layout.layers_to_kv_buffer_ptrs_by_attn[owner][region];
            tensor.select(0, 1).fill_(17 + owner + gid);
            tensor.select(0, 2).zero_();
        }
    }
    const BlockIdPair copy{1, 2};
    allocator->blockBatchCopy(&copy, &copy + 1);
    for (size_t gid = 0; gid < config.global_layer_ids.size(); ++gid) {
        const auto region = static_cast<size_t>(config.group_region_names[gid]);
        for (int owner : config.global_layer_ids[gid]) {
            auto tensor = layout.layers_to_kv_buffer_ptrs_by_attn[owner][region];
            EXPECT_TRUE(torch::equal(tensor.select(0, 1), tensor.select(0, 2)));
        }
    }
}

}  // namespace test
}  // namespace rtp_llm
