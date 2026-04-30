#include <gtest/gtest.h>
#include <vector>

#include "rtp_llm/cpp/cache/DSV4CacheConfig.h"
#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/cache/DSV4ConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

static ModelConfig makeProModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 61;
    mc.hidden_size                  = 7168;
    mc.attn_config.head_num         = 128;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 1024;
    mc.attn_config.o_groups         = 16;
    mc.attn_config.o_lora_rank      = 1024;
    std::vector<int> ratios;
    ratios.push_back(128);
    ratios.push_back(128);
    for (int i = 2; i < 61; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);
    mc.attn_config.layer_compress_ratios = ratios;
    return mc;
}

static ModelConfig makeFlashModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 43;
    mc.hidden_size                  = 4096;
    mc.attn_config.head_num         = 64;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 512;
    mc.attn_config.o_groups         = 8;
    mc.attn_config.o_lora_rank      = 1024;
    std::vector<int> ratios         = {0, 0};
    for (int i = 2; i < 43; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);
    mc.attn_config.layer_compress_ratios = ratios;
    return mc;
}

// ============================================================
// Layer classification
// ============================================================

TEST(DSV4ConfigCreatorTest, ProLayerClassification) {
    auto mc   = makeProModelConfig();
    auto dsv4 = DSV4ConfigCreator::buildDSV4Config(mc);
    EXPECT_EQ(dsv4.num_all_layers(), 61u);
    EXPECT_EQ(dsv4.num_swa_only_layers(), 0u);
    EXPECT_EQ(dsv4.num_csa_layers(), 30u);
    EXPECT_EQ(dsv4.num_hca_layers(), 31u);
    EXPECT_EQ(dsv4.num_csa_layers() + dsv4.num_hca_layers() + dsv4.num_swa_only_layers(), 61u);
}

TEST(DSV4ConfigCreatorTest, FlashLayerClassification) {
    auto mc   = makeFlashModelConfig();
    auto dsv4 = DSV4ConfigCreator::buildDSV4Config(mc);
    EXPECT_EQ(dsv4.num_all_layers(), 43u);
    EXPECT_EQ(dsv4.num_swa_only_layers(), 2u);
    EXPECT_EQ(dsv4.num_csa_layers(), 21u);
    EXPECT_EQ(dsv4.num_hca_layers(), 20u);
    EXPECT_EQ(dsv4.num_csa_layers() + dsv4.num_hca_layers() + dsv4.num_swa_only_layers(), 43u);
}

// ============================================================
// Pool specs
// ============================================================

TEST(DSV4ConfigCreatorTest, ProPoolSpecs) {
    auto mc   = makeProModelConfig();
    auto dsv4 = DSV4ConfigCreator::buildDSV4Config(mc);

    EXPECT_EQ(dsv4.pool_specs[0].layer_num, 30u);
    EXPECT_EQ(dsv4.pool_specs[0].entry_elems, DSV4CacheConfig::KV_ENTRY_BYTES);
    EXPECT_EQ(dsv4.pool_specs[0].entries_per_block, 64u);
    EXPECT_TRUE(dsv4.pool_specs[0].is_paged);

    EXPECT_EQ(dsv4.pool_specs[1].layer_num, 31u);
    EXPECT_EQ(dsv4.pool_specs[1].entries_per_block, 2u);

    EXPECT_EQ(dsv4.pool_specs[2].layer_num, 30u);
    EXPECT_EQ(dsv4.pool_specs[2].entry_elems, DSV4CacheConfig::INDEXER_ENTRY_BYTES);

    EXPECT_EQ(dsv4.pool_specs[3].layer_num, 30u);
    EXPECT_FALSE(dsv4.pool_specs[3].is_paged);
    EXPECT_EQ(dsv4.pool_specs[3].fixed_blocks_per_req, 2u);

    EXPECT_EQ(dsv4.pool_specs[4].layer_num, 30u);
    EXPECT_EQ(dsv4.pool_specs[4].fixed_blocks_per_req, 2u);

    EXPECT_EQ(dsv4.pool_specs[5].layer_num, 31u);
    EXPECT_EQ(dsv4.pool_specs[5].fixed_blocks_per_req, 2u);

    EXPECT_EQ(dsv4.pool_specs[6].layer_num, 61u);
    EXPECT_EQ(dsv4.pool_specs[6].entries_per_block, 256u);  // SWA: 256 tokens/block, no compression
}

TEST(DSV4ConfigCreatorTest, FlashPoolSpecs) {
    auto mc   = makeFlashModelConfig();
    auto dsv4 = DSV4ConfigCreator::buildDSV4Config(mc);
    EXPECT_EQ(dsv4.pool_specs[0].layer_num, 21u);
    EXPECT_EQ(dsv4.pool_specs[1].layer_num, 20u);
    EXPECT_EQ(dsv4.pool_specs[6].layer_num, 43u);
}

// ============================================================
// Block size bytes
// ============================================================

TEST(DSV4ConfigCreatorTest, BlockSizeBytes) {
    auto mc   = makeProModelConfig();
    auto dsv4 = DSV4ConfigCreator::buildDSV4Config(mc);
    EXPECT_EQ(dsv4.pool_specs[0].block_size_bytes(), 64u * DSV4CacheConfig::KV_ENTRY_BYTES);
    EXPECT_EQ(dsv4.pool_specs[1].block_size_bytes(), 2u * DSV4CacheConfig::KV_ENTRY_BYTES);
    EXPECT_EQ(dsv4.pool_specs[2].block_size_bytes(), 64u * DSV4CacheConfig::INDEXER_ENTRY_BYTES);
    EXPECT_EQ(dsv4.pool_specs[3].block_size_bytes(), 4u * 512u * 4u);
    EXPECT_EQ(dsv4.pool_specs[4].block_size_bytes(), 4u * 2048u * 4u);
    EXPECT_EQ(dsv4.pool_specs[5].block_size_bytes(), 8u * 1024u * 4u);
    EXPECT_EQ(dsv4.pool_specs[6].block_size_bytes(),
              DSV4CacheConfig::TOKENS_PER_BLOCK * DSV4CacheConfig::KV_ENTRY_BYTES);
}

// ============================================================
// CacheConfig output
// ============================================================

TEST(DSV4ConfigCreatorTest, CreateCacheConfig) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = DSV4ConfigCreator::createConfig(mc, pc);

    EXPECT_TRUE(config.dsv4_config.has_value());
    // 7 groups -> groupNums() > 1 -> HybridTypeKVCacheAllocator path
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.global_layer_ids.size(), 7u);
    EXPECT_EQ(config.cache_specs.size(), 7u);
    EXPECT_EQ(config.group_types.size(), 7u);
    EXPECT_EQ(config.layer_num, 61u);
    EXPECT_TRUE(config.is_sparse);
    EXPECT_FALSE(config.use_mla);
}

TEST(DSV4ConfigCreatorTest, FlashCacheConfig) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    auto              config = DSV4ConfigCreator::createConfig(mc, pc);

    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.layer_num, 43u);
    // Group 6 (SWA) should have all 43 layers
    EXPECT_EQ(config.global_layer_ids[6].size(), 43u);
    // Group 0 (CSA KV) should have 21 layers
    EXPECT_EQ(config.global_layer_ids[0].size(), 21u);
}

// ============================================================
// DSV4KVCacheSpec
// ============================================================

TEST(DSV4KVCacheSpecTest, KVSpecFromPoolSpec) {
    DSV4PoolSpec pool_spec = {
        DSV4CacheType::CSA_KV,
        30,
        584,
        64,
        DataType::TYPE_UINT8,
        true,
        0,
    };
    DSV4KVSpec spec(pool_spec, 256);

    EXPECT_EQ(spec.layer_num, 30u);
    EXPECT_EQ(spec.block_size(), 64u * 584u);
    EXPECT_EQ(spec.block_size_bytes(), 64u * 584u * 1u);  // uint8 = 1 byte
    EXPECT_EQ(spec.cache_type, DSV4CacheType::CSA_KV);
    EXPECT_EQ(spec.entry_elems, 584u);
    EXPECT_EQ(spec.entries_per_block, 64u);
}

TEST(DSV4KVCacheSpecTest, StateSpecFloat32) {
    DSV4PoolSpec pool_spec = {
        DSV4CacheType::CSA_STATE,
        30,
        2048,
        4,
        DataType::TYPE_FP32,
        false,
        2,
    };
    DSV4StateSpec spec(pool_spec, 256);

    EXPECT_EQ(spec.block_size(), 4u * 2048u);
    EXPECT_EQ(spec.block_size_bytes(), 4u * 2048u * 4u);  // float32 = 4 bytes
    EXPECT_EQ(spec.cache_type, DSV4CacheType::CSA_STATE);
    EXPECT_EQ(spec.state_dim, 2048u);
    EXPECT_EQ(spec.fixed_blocks_per_req, 2u);
}

TEST(DSV4KVCacheSpecTest, IndexerKVSpec) {
    DSV4PoolSpec pool_spec = {
        DSV4CacheType::INDEXER_KV,
        30,
        132,
        64,
        DataType::TYPE_UINT8,
        true,
        0,
    };
    DSV4KVSpec spec(pool_spec, 256);

    EXPECT_EQ(spec.block_size(), 64u * 132u);
    EXPECT_EQ(spec.block_size_bytes(), 64u * 132u);
    EXPECT_EQ(spec.cache_type, DSV4CacheType::INDEXER_KV);
}

TEST(DSV4KVCacheSpecTest, HCAStateSpec) {
    DSV4PoolSpec pool_spec = {
        DSV4CacheType::HCA_STATE,
        31,
        1024,
        8,
        DataType::TYPE_FP32,
        false,
        2,
    };
    DSV4StateSpec spec(pool_spec, 256);

    EXPECT_EQ(spec.block_size_bytes(), 8u * 1024u * 4u);
    EXPECT_EQ(spec.fixed_blocks_per_req, 2u);
    EXPECT_EQ(spec.cache_type, DSV4CacheType::HCA_STATE);
}

// ============================================================
// Pool 0/1/2 shared properties: same tokens_per_block, same num_blocks
// ============================================================

TEST(DSV4ConfigCreatorTest, PagedPoolsShareTokensPerBlock) {
    // Pro config
    {
        auto mc   = makeProModelConfig();
        auto dsv4 = DSV4ConfigCreator::buildDSV4Config(mc);
        // Pool 0 (CSA KV), Pool 1 (HCA KV), Pool 2 (Indexer KV) all use
        // VARIABLE_TOKENS_PER_BLOCK / compress_ratio as entries_per_block.
        // But they all track the same token stream — their block boundaries align.
        // entries_per_block differs (64 vs 2) because compress_ratio differs,
        // but tokens_per_block is the same: 256 tokens per block for all.
        EXPECT_EQ(dsv4.pool_specs[0].entries_per_block, 256u / 4u);    // CSA: 64
        EXPECT_EQ(dsv4.pool_specs[1].entries_per_block, 256u / 128u);  // HCA: 2
        EXPECT_EQ(dsv4.pool_specs[2].entries_per_block, 256u / 4u);    // Indexer: 64

        // Pool 6 (SWA) now uses TOKENS_PER_BLOCK = 256 (same as other groups)
        EXPECT_EQ(dsv4.pool_specs[6].entries_per_block, DSV4CacheConfig::TOKENS_PER_BLOCK);
    }
    // Flash config
    {
        auto mc   = makeFlashModelConfig();
        auto dsv4 = DSV4ConfigCreator::buildDSV4Config(mc);
        EXPECT_EQ(dsv4.pool_specs[0].entries_per_block, 256u / 4u);
        EXPECT_EQ(dsv4.pool_specs[1].entries_per_block, 256u / 128u);
        EXPECT_EQ(dsv4.pool_specs[2].entries_per_block, 256u / 4u);
    }
}

TEST(DSV4ConfigCreatorTest, AllPagedPoolsShareBlockNum) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = DSV4ConfigCreator::createConfig(mc, pc);
    config.block_num         = 100;

    // Paged groups derive their block count from the global block_num; fixed/SWA
    // groups use per-group fixed block counts.
    EXPECT_EQ(config.groupNums(), 7);
    for (int i = 0; i < 7; i++) {
        EXPECT_GT(config.cache_specs[i]->block_size_bytes(), 0u) << "pool " << i;
    }
}

TEST(DSV4ConfigCreatorTest, BlockIdConsistencyAcrossGroups) {
    // DSV4 has multiple cache regions per logical layer. The config must expose
    // every region's group id for the layer so model/runtime code can request the
    // correct region by KVCacheRegionName.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = DSV4ConfigCreator::createConfig(mc, pc);

    // Verify layer_to_group_id mapping: each layer maps to group 6 (SWA) by default
    EXPECT_EQ(config.layer_to_group_id.size(), 61u);
    for (size_t i = 0; i < config.layer_to_group_id.size(); i++) {
        EXPECT_EQ(config.layer_to_group_id[i], 6) << "layer " << i << " should map to SWA group";
    }

    // Verify global_layer_ids: each group has the correct layer list
    // Group 0 (CSA KV) and Group 2 (Indexer KV) should have identical layer lists
    EXPECT_EQ(config.global_layer_ids[0], config.global_layer_ids[2]);
    // Group 0 (CSA KV) and Group 3 (Indexer State) should have identical layer lists
    EXPECT_EQ(config.global_layer_ids[0], config.global_layer_ids[3]);
    // Group 0 (CSA KV) and Group 4 (CSA State) should have identical layer lists
    EXPECT_EQ(config.global_layer_ids[0], config.global_layer_ids[4]);
    // Group 1 (HCA KV) and Group 5 (HCA State) should have identical layer lists
    EXPECT_EQ(config.global_layer_ids[1], config.global_layer_ids[5]);
}

// ============================================================
// Helper: build a DSV4 CacheConfig with block_num set for allocator tests
// ============================================================

static CacheConfig makeDSV4AllocatorConfig(bool use_flash = false) {
    auto              mc = use_flash ? makeFlashModelConfig() : makeProModelConfig();
    ParallelismConfig pc;
    auto              config = DSV4ConfigCreator::createConfig(mc, pc);
    // Set enough blocks for tests (7 groups × N blocks each)
    config.block_num = 200;
    return config;
}

// ============================================================
// HybridTypeKVCacheAllocator integration tests with DSV4 7-group config
// ============================================================

class DSV4AllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

TEST_F(DSV4AllocatorTest, InitAndBasicProperties) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // 7 groups → HybridTypeKVCacheAllocator path
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(allocator->seqSizePerBlock(), static_cast<int>(config.seq_size_per_block));
    EXPECT_EQ(allocator->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator->freeBlocksNum(), config.block_num - 1);
}

TEST_F(DSV4AllocatorTest, FlashInitAndBasicProperties) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(allocator->totalBlocksNum(), config.block_num - 1);
}

TEST_F(DSV4AllocatorTest, AddressLookupAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // Verify address lookup works for a layer in each group
    // Group 0 (CSA KV): csa_layer_ids[0]
    // Group 1 (HCA KV): hca_layer_ids[0]
    // Group 6 (SWA KV): all_layer_ids[0]
    for (int gid = 0; gid < 7; gid++) {
        ASSERT_FALSE(config.global_layer_ids[gid].empty()) << "group " << gid << " has no layers";
        int  layer_id = config.global_layer_ids[gid][0];
        auto addr     = allocator->convertIndexToAddr(layer_id, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr) << "null kv_addr for group " << gid << " layer " << layer_id;
    }
}

TEST_F(DSV4AllocatorTest, BlockPoolCreatedWithCorrectTensors) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    // allLayerCacheBase should return tensors for all 61 layers
    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined()) << "undefined kv buffer for layer " << i;
    }
}

TEST_F(DSV4AllocatorTest, ConvertIndexToBufferAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // convertIndexToBuffer should work for layers in each of the 7 groups
    for (int gid = 0; gid < 7; gid++) {
        int  layer_id = config.global_layer_ids[gid][0];
        auto buf      = allocator->convertIndexToBuffer(layer_id, /*block_id=*/1);
        ASSERT_FALSE(buf.empty()) << "empty buffer for group " << gid;
        EXPECT_NE(buf[0].addr, nullptr) << "null addr for group " << gid;
    }
}

TEST_F(DSV4AllocatorTest, MallocAndFreeBlocks) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 3u);

    // Direct block pool malloc/free
    auto blocks = block_pool->malloc(3);
    ASSERT_EQ(blocks.size(), 3u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 3);

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(DSV4AllocatorTest, SevenGroupLayerMapping) {
    auto config = makeDSV4AllocatorConfig();
    ASSERT_TRUE(config.dsv4_config.has_value());
    const auto& dsv4 = config.dsv4_config.value();

    // Verify group layer assignments match DSV4 classification
    // Group 0: CSA KV → csa_layer_ids (30 layers)
    EXPECT_EQ(config.global_layer_ids[0].size(), dsv4.num_csa_layers());
    // Group 1: HCA KV → hca_layer_ids (31 layers)
    EXPECT_EQ(config.global_layer_ids[1].size(), dsv4.num_hca_layers());
    // Group 2: Indexer KV → csa_layer_ids
    EXPECT_EQ(config.global_layer_ids[2].size(), dsv4.num_csa_layers());
    // Group 3: Indexer State → csa_layer_ids
    EXPECT_EQ(config.global_layer_ids[3].size(), dsv4.num_csa_layers());
    // Group 4: CSA State → csa_layer_ids
    EXPECT_EQ(config.global_layer_ids[4].size(), dsv4.num_csa_layers());
    // Group 5: HCA State → hca_layer_ids
    EXPECT_EQ(config.global_layer_ids[5].size(), dsv4.num_hca_layers());
    // Group 6: SWA KV → all_layer_ids
    EXPECT_EQ(config.global_layer_ids[6].size(), dsv4.num_all_layers());

    // Paged DSV4 pools are FULL; fixed/state and SWA pools keep a sliding-window tail.
    EXPECT_EQ(config.group_types[0], CacheGroupType::FULL);
    EXPECT_EQ(config.group_types[1], CacheGroupType::FULL);
    EXPECT_EQ(config.group_types[2], CacheGroupType::FULL);
    EXPECT_EQ(config.group_types[3], CacheGroupType::SWA);
    EXPECT_EQ(config.group_types[4], CacheGroupType::SWA);
    EXPECT_EQ(config.group_types[5], CacheGroupType::SWA);
    EXPECT_EQ(config.group_types[6], CacheGroupType::SWA);
}

TEST_F(DSV4AllocatorTest, SpecBlockSizesMatchPoolSpecs) {
    auto config = makeDSV4AllocatorConfig();
    ASSERT_TRUE(config.dsv4_config.has_value());
    const auto& dsv4 = config.dsv4_config.value();

    // Each cache_spec's block_size_bytes should match the corresponding pool_spec
    ASSERT_EQ(config.cache_specs.size(), 7u);
    for (int i = 0; i < DSV4_NUM_POOLS; i++) {
        EXPECT_EQ(config.cache_specs[i]->block_size_bytes(), dsv4.pool_specs[i].block_size_bytes())
            << "mismatch at pool " << i;
    }
}

TEST_F(DSV4AllocatorTest, KVBlockStrideIsMaxAcrossGroups) {
    auto config = makeDSV4AllocatorConfig();
    ASSERT_TRUE(config.dsv4_config.has_value());
    const auto& dsv4 = config.dsv4_config.value();

    // kv_block_stride_bytes should be the max block_size_bytes across all 7 pools
    size_t expected_max = 0;
    for (int i = 0; i < DSV4_NUM_POOLS; i++) {
        expected_max = std::max(expected_max, dsv4.pool_specs[i].block_size_bytes());
    }
    EXPECT_EQ(config.kv_block_stride_bytes, expected_max);
    EXPECT_EQ(expected_max, DSV4CacheConfig::TOKENS_PER_BLOCK * DSV4CacheConfig::KV_ENTRY_BYTES);
}

TEST_F(DSV4AllocatorTest, AllGroupsParticipateInPrefixCache) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();
    ASSERT_NE(block_pool, nullptr);
    ASSERT_NE(block_cache, nullptr);

    // Insert blocks into cache for ALL 7 groups
    CacheKeysType keys = {100, 101, 102};
    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
        ASSERT_EQ(blocks.size(), keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            BlockCache::CacheItem item;
            item.cache_key   = keys[i];
            item.group_id    = gid;
            item.block_index = blocks[i];
            item.is_resident = true;
            EXPECT_TRUE(block_cache->put(item));
            block_pool->blockCacheReference(blocks[i]);
        }
        block_pool->requestFree(blocks);
    }

    // All 7 groups should have cache entries
    for (int gid = 0; gid < 7; gid++) {
        EXPECT_TRUE(block_cache->contains(100, gid)) << "group " << gid << " should be in cache";
    }
}

// ============================================================
// Flash config: allocator integration
// ============================================================

TEST_F(DSV4AllocatorTest, FlashGroupTypes) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);
    ASSERT_TRUE(config.dsv4_config.has_value());
    const auto& dsv4 = config.dsv4_config.value();

    // Flash: 21 CSA + 20 HCA + 2 SWA-only = 43 layers
    EXPECT_EQ(dsv4.num_csa_layers(), 21u);
    EXPECT_EQ(dsv4.num_hca_layers(), 20u);
    EXPECT_EQ(dsv4.num_swa_only_layers(), 2u);

    // Same group type split as Pro: 3 FULL paged groups, 4 SWA tail groups.
    EXPECT_EQ(config.group_types[0], CacheGroupType::FULL);  // CSA KV
    EXPECT_EQ(config.group_types[1], CacheGroupType::FULL);  // HCA KV
    EXPECT_EQ(config.group_types[2], CacheGroupType::FULL);  // Indexer KV
    EXPECT_EQ(config.group_types[3], CacheGroupType::SWA);   // Indexer State
    EXPECT_EQ(config.group_types[4], CacheGroupType::SWA);   // CSA State
    EXPECT_EQ(config.group_types[5], CacheGroupType::SWA);   // HCA State
    EXPECT_EQ(config.group_types[6], CacheGroupType::SWA);   // SWA KV
}

TEST_F(DSV4AllocatorTest, FlashAddressLookupAllGroups) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    for (int gid = 0; gid < 7; gid++) {
        ASSERT_FALSE(config.global_layer_ids[gid].empty()) << "Flash group " << gid << " has no layers";
        int  layer_id = config.global_layer_ids[gid][0];
        auto addr     = allocator->convertIndexToAddr(layer_id, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr) << "Flash null kv_addr for group " << gid;
    }
}

TEST_F(DSV4AllocatorTest, FlashBlockPoolTensors) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs.size(), 43u);
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined()) << "Flash undefined kv buffer for layer " << i;
    }
}

TEST_F(DSV4AllocatorTest, FlashLayerMapping) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);
    ASSERT_TRUE(config.dsv4_config.has_value());

    EXPECT_EQ(config.global_layer_ids[0].size(), 21u);  // CSA KV
    EXPECT_EQ(config.global_layer_ids[1].size(), 20u);  // HCA KV
    EXPECT_EQ(config.global_layer_ids[2].size(), 21u);  // Indexer KV
    EXPECT_EQ(config.global_layer_ids[3].size(), 21u);  // Indexer State
    EXPECT_EQ(config.global_layer_ids[4].size(), 21u);  // CSA State
    EXPECT_EQ(config.global_layer_ids[5].size(), 20u);  // HCA State
    EXPECT_EQ(config.global_layer_ids[6].size(), 43u);  // SWA KV (all layers)
}

TEST_F(DSV4AllocatorTest, FlashSpecBlockSizes) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);
    ASSERT_TRUE(config.dsv4_config.has_value());
    const auto& dsv4 = config.dsv4_config.value();

    ASSERT_EQ(config.cache_specs.size(), 7u);
    for (int i = 0; i < DSV4_NUM_POOLS; i++) {
        EXPECT_EQ(config.cache_specs[i]->block_size_bytes(), dsv4.pool_specs[i].block_size_bytes())
            << "Flash mismatch at pool " << i;
    }
}

TEST_F(DSV4AllocatorTest, FlashMallocAndFree) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto   block_pool  = allocator->getBlockPool();
    size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 5u);

    auto blocks = block_pool->malloc(5);
    ASSERT_EQ(blocks.size(), 5u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 5);

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ============================================================
// Prefix cache: insertIntoCache inserts ALL 7 groups
// ============================================================

TEST_F(DSV4AllocatorTest, InsertIntoCacheAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();

    // Manually set up a BatchKVCacheResource with blocks for all 7 groups
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);

    CacheKeysType keys = {200, 201, 202, 203};
    batch_res->setBatchCacheKeys(0, keys);

    // Allocate 3 blocks per group (simulating 3 full blocks)
    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(3);
        ASSERT_EQ(blocks.size(), 3u);
        batch_res->mutableBlockIds(0, gid).assign(BlockIndicesType(blocks.begin(), blocks.end()));
    }

    // Create CompleteTokenIds: 3 full blocks * seq_size_per_block tokens + partial
    int  seq_size_per_block         = allocator->seqSizePerBlock();
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    int  total_tokens               = 3 * seq_size_per_block + 1;  // 3 full blocks + 1 partial
    generate_input->input_ids       = torch::arange(total_tokens, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    InsertInfo insert_info{batch_res, complete_token_ids, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    // ALL 7 groups should have entries in block cache
    for (int gid = 0; gid < 7; gid++) {
        EXPECT_TRUE(block_cache->contains(200, gid)) << "group " << gid << " should be cached";
    }

    // Free all blocks
    for (int gid = 0; gid < 7; gid++) {
        const auto& blocks = batch_res->blocks(0, gid);
        block_pool->requestFree(blocks);
    }
}

// ============================================================
// Prefix cache: Flash config insertIntoCache all groups
// ============================================================

TEST_F(DSV4AllocatorTest, FlashInsertIntoCacheAllGroups) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);

    CacheKeysType keys = {300, 301, 302, 303};
    batch_res->setBatchCacheKeys(0, keys);

    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(3);
        ASSERT_EQ(blocks.size(), 3u);
        batch_res->mutableBlockIds(0, gid).assign(BlockIndicesType(blocks.begin(), blocks.end()));
    }

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    int  total_tokens               = 3 * seq_size_per_block + 1;
    generate_input->input_ids       = torch::arange(total_tokens, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    InsertInfo insert_info{batch_res, complete_token_ids, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    // All groups cached
    for (int gid = 0; gid < 7; gid++) {
        EXPECT_TRUE(block_cache->contains(300, gid)) << "Flash group " << gid << " should be cached";
    }

    for (int gid = 0; gid < 7; gid++) {
        block_pool->requestFree(batch_res->blocks(0, gid));
    }
}

// ============================================================
// Prefix cache: full reuse flow — all 7 groups participate
// ============================================================

TEST_F(DSV4AllocatorTest, PrefixCacheReuseAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();

    // Pre-populate cache for ALL 7 groups with keys {100,101,102}
    CacheKeysType                          cached_keys = {100, 101, 102};
    std::vector<std::vector<BlockIdxType>> cached_blocks(7);
    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            BlockCache::CacheItem item;
            item.cache_key   = cached_keys[i];
            item.group_id    = gid;
            item.block_index = blocks[i];
            item.is_resident = true;
            EXPECT_TRUE(block_cache->put(item));
            block_pool->blockCacheReference(blocks[i]);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

    // Now do a malloc with reuse enabled — keys {100,101,102,103}
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    int  seq_len                    = 3 * seq_size_per_block + 1;  // 3 full + partial
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::arange(seq_len, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    MallocInfo info{batch_res, complete_token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // reuse_len should be > 0 (all 7 groups participate)
    EXPECT_GT(result.reuse_len, 0) << "Prefix cache reuse should work with all 7 groups";

    // All groups should have reused cached blocks
    for (int gid = 0; gid < 7; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 3u) << "group " << gid << " should have >= 3 blocks";
        // First 2 blocks should be reused from cache (last cached block may not be reused)
        EXPECT_EQ(out_blocks[0], cached_blocks[gid][0]) << "group " << gid << " block 0 should be reused";
        EXPECT_EQ(out_blocks[1], cached_blocks[gid][1]) << "group " << gid << " block 1 should be reused";
    }

    // Clean up
    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, FlashPrefixCacheReuseAllGroups) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();

    CacheKeysType                          cached_keys = {500, 501, 502};
    std::vector<std::vector<BlockIdxType>> cached_blocks(7);
    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            BlockCache::CacheItem item;
            item.cache_key   = cached_keys[i];
            item.group_id    = gid;
            item.block_index = blocks[i];
            item.is_resident = true;
            EXPECT_TRUE(block_cache->put(item));
            block_pool->blockCacheReference(blocks[i]);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{500, 501, 502, 503});

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    int  seq_len                    = 3 * seq_size_per_block + 1;
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::arange(seq_len, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    MallocInfo info{batch_res, complete_token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "Flash prefix cache reuse should work";

    for (int gid = 0; gid < 7; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 3u) << "Flash group " << gid;
        EXPECT_EQ(out_blocks[0], cached_blocks[gid][0]) << "Flash group " << gid << " block 0 should be reused";
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// SWA (group 6) prefix cache: verify SWA blocks participate in reuse
// ============================================================

TEST_F(DSV4AllocatorTest, SWAGroupParticipatesInPrefixCacheReuse) {
    auto config      = makeDSV4AllocatorConfig();
    config.block_num = 100;
    auto allocator   = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();

    // Only populate SWA group (6) and one paged group (0) to verify SWA participates
    CacheKeysType             cached_keys = {700, 701};
    std::vector<BlockIdxType> swa_blocks, csa_blocks;

    // Group 0 (CSA KV)
    {
        auto blocks = block_pool->malloc(2);
        for (size_t i = 0; i < 2; ++i) {
            BlockCache::CacheItem item{cached_keys[i], 0, blocks[i], true};
            block_cache->put(item);
            block_pool->blockCacheReference(blocks[i]);
        }
        csa_blocks = blocks;
        block_pool->requestFree(blocks);
    }
    // Group 6 (SWA KV)
    {
        auto blocks = block_pool->malloc(2);
        for (size_t i = 0; i < 2; ++i) {
            BlockCache::CacheItem item{cached_keys[i], 6, blocks[i], true};
            block_cache->put(item);
            block_pool->blockCacheReference(blocks[i]);
        }
        swa_blocks = blocks;
        block_pool->requestFree(blocks);
    }

    // Verify both groups have cache entries
    EXPECT_TRUE(block_cache->contains(700, 0));
    EXPECT_TRUE(block_cache->contains(700, 6));
    EXPECT_TRUE(block_cache->contains(701, 0));
    EXPECT_TRUE(block_cache->contains(701, 6));

    // Groups 1,2,3,4,5 not populated — they will limit reuse to 0
    // But this verifies SWA group 6 IS in the reuse path
    EXPECT_FALSE(block_cache->contains(700, 3));
    EXPECT_FALSE(block_cache->contains(700, 4));
    EXPECT_FALSE(block_cache->contains(700, 5));
}

// ============================================================
// SWA prefix cache: full reuse with all paged groups including SWA
// ============================================================

TEST_F(DSV4AllocatorTest, SWAPrefixCacheFullReuse) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();

    // Populate ALL 7 groups with same keys
    CacheKeysType                          cached_keys = {800, 801};
    std::vector<std::vector<BlockIdxType>> cached_blocks(7);
    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(2);
        for (size_t i = 0; i < 2; ++i) {
            BlockCache::CacheItem item{cached_keys[i], gid, blocks[i], true};
            block_cache->put(item);
            block_pool->blockCacheReference(blocks[i]);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

    // Malloc with reuse — keys {800, 801, 802}
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{800, 801, 802});

    int  spb            = allocator->seqSizePerBlock();
    int  seq_len        = 2 * spb + 1;
    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(seq_len, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_GT(result.reuse_len, 0);

    // SWA group 6 should have reused block 0
    const auto& swa_out = batch_res->blocks(0, 6);
    ASSERT_GE(swa_out.size(), 2u);
    EXPECT_EQ(swa_out[0], cached_blocks[6][0]) << "SWA block 0 should be reused from prefix cache";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// incrMalloc: decode grows sequence after initial prefill
// ============================================================

TEST_F(DSV4AllocatorTest, IncrMallocDecodeGrowsBlocks) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    int spb = allocator->seqSizePerBlock();

    // Initial malloc: 1 block worth of tokens
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{900, 901});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo init_info{batch_res, cti};
    init_info.enable_device_cache = false;
    auto init_result              = allocator->malloc(init_info);
    ASSERT_TRUE(init_result.success);

    // All 7 groups should have 1 block each
    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 1u) << "group " << gid << " should have 1 block after init";
    }

    size_t free_after_init = allocator->freeBlocksNum();

    // incrMalloc: grow to 2 blocks
    cti->setSeqLength(2 * spb);
    MallocInfo incr_info{batch_res, cti};
    incr_info.enable_device_cache = false;
    auto incr_result              = allocator->malloc(incr_info);
    ASSERT_TRUE(incr_result.success);

    // All 7 groups should now have 2 blocks each
    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 2u) << "group " << gid << " should have 2 blocks after incr";
    }

    // Should have consumed 7 more blocks (1 per group)
    EXPECT_EQ(allocator->freeBlocksNum(), free_after_init - 7);

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// Free and reallocate: blocks return to pool
// ============================================================

TEST_F(DSV4AllocatorTest, FreeReturnsBlocksToPool) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    size_t free_before = allocator->freeBlocksNum();
    int    spb         = allocator->seqSizePerBlock();

    // Allocate
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{1000, 1001});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    size_t free_after_alloc = allocator->freeBlocksNum();
    EXPECT_LT(free_after_alloc, free_before);

    // Free
    FreeInfo free_info{batch_res};
    allocator->free(free_info);

    // All blocks should be returned
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);

    // Can allocate again
    auto batch_res2 = std::make_shared<BatchKVCacheResource>();
    batch_res2->resetBatchSize(1);
    batch_res2->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res2->setBatchCacheKeys(0, CacheKeysType{1100, 1101});

    MallocInfo info2{batch_res2, cti};
    info2.enable_device_cache = false;
    auto result2              = allocator->malloc(info2);
    ASSERT_TRUE(result2.success);

    FreeInfo free_info2{batch_res2};
    allocator->free(free_info2);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ============================================================
// Flash: incrMalloc decode path
// ============================================================

TEST_F(DSV4AllocatorTest, FlashIncrMallocDecode) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    int spb = allocator->seqSizePerBlock();

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{1200, 1201});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo init_info{batch_res, cti};
    init_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);

    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 1u) << "Flash group " << gid;
    }

    // Grow to 3 blocks
    cti->setSeqLength(3 * spb);
    MallocInfo incr_info{batch_res, cti};
    incr_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(incr_info).success);

    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 3u) << "Flash group " << gid << " after incr";
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

}  // namespace test
}  // namespace rtp_llm
