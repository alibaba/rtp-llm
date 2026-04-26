#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/DSV4ConfigCreator.h"
#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
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
    for (int i = 2; i < 61; ++i) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);  // MTP tail marker stripped by DSV4ConfigCreator
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

    std::vector<int> ratios = {0, 0};
    for (int i = 2; i < 43; ++i) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);
    mc.attn_config.layer_compress_ratios = ratios;
    return mc;
}

static CacheConfig makeAllocatorConfig(bool use_flash = false) {
    auto              mc = use_flash ? makeFlashModelConfig() : makeProModelConfig();
    ParallelismConfig pc;
    auto              config = DSV4ConfigCreator::createConfig(mc, pc);
    config.block_num         = 32;
    config.group_block_nums.assign(config.groupNums(), config.block_num);
    return config;
}

static CompleteTokenIdsPtr makeCompleteTokenIds(int token_num, int seq_size_per_block) {
    auto complete_token_ids = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input     = std::make_shared<GenerateInput>();
    generate_input->input_ids = torch::arange(token_num, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);
    return complete_token_ids;
}

static BatchKVCacheResourcePtr makeBatchResource(const CacheConfig& config, int batch_size = 1) {
    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(static_cast<size_t>(batch_size));
    batch_resource->initGroups(config.groupNums(),
                               static_cast<int>(config.layer_all_num),
                               config.layer_to_group_id,
                               config.kernelBlocksPerKvBlock(),
                               config.group_types,
                               config.layer_attn_to_group_id);
    return batch_resource;
}

static size_t dtypeSize(DataType dtype) {
    return getTypeSize(dtype);
}

TEST(DSV4ConfigCreatorTest, ClassifiesProAndFlashLayers) {
    auto pro = DSV4ConfigCreator::buildDSV4Config(makeProModelConfig());
    EXPECT_EQ(pro.num_all_layers(), 61u);
    EXPECT_EQ(pro.num_csa_layers(), 30u);
    EXPECT_EQ(pro.num_hca_layers(), 31u);
    EXPECT_EQ(pro.num_swa_only_layers(), 0u);

    auto flash = DSV4ConfigCreator::buildDSV4Config(makeFlashModelConfig());
    EXPECT_EQ(flash.num_all_layers(), 43u);
    EXPECT_EQ(flash.num_csa_layers(), 21u);
    EXPECT_EQ(flash.num_hca_layers(), 20u);
    EXPECT_EQ(flash.num_swa_only_layers(), 2u);
}

TEST(DSV4ConfigCreatorTest, CreatesSevenGroupsWithExpectedPolicies) {
    auto config = makeAllocatorConfig();
    ASSERT_TRUE(config.dsv4_config.has_value());

    EXPECT_TRUE(config.use_independent_block_pools);
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.group_types[0], CacheGroupType::FULL);
    EXPECT_EQ(config.group_types[1], CacheGroupType::FULL);
    EXPECT_EQ(config.group_types[2], CacheGroupType::FULL);
    EXPECT_EQ(config.group_types[3], CacheGroupType::LINEAR);
    EXPECT_EQ(config.group_types[4], CacheGroupType::LINEAR);
    EXPECT_EQ(config.group_types[5], CacheGroupType::LINEAR);
    EXPECT_EQ(config.group_types[6], CacheGroupType::LINEAR);

    EXPECT_EQ(config.group_attn_types[0], KVCacheAttnType::CSA_KV);
    EXPECT_EQ(config.group_attn_types[1], KVCacheAttnType::HCA_KV);
    EXPECT_EQ(config.group_attn_types[2], KVCacheAttnType::INDEXER_KV);
    EXPECT_EQ(config.group_attn_types[3], KVCacheAttnType::INDEXER_STATE);
    EXPECT_EQ(config.group_attn_types[4], KVCacheAttnType::CSA_STATE);
    EXPECT_EQ(config.group_attn_types[5], KVCacheAttnType::HCA_STATE);
    EXPECT_EQ(config.group_attn_types[6], KVCacheAttnType::SWA_KV);

    ASSERT_EQ(config.group_seq_size_per_block.size(), static_cast<size_t>(config.groupNums()));
    for (auto group_seq_size : config.group_seq_size_per_block) {
        EXPECT_EQ(group_seq_size, config.seq_size_per_block);
    }
}

TEST(DSV4ConfigCreatorTest, LayerToGroupIsMultiMapping) {
    auto config = makeAllocatorConfig();
    const auto& dsv4 = config.dsv4_config.value();

    const int csa_layer = dsv4.csa_layer_ids.front();
    EXPECT_EQ(config.layer_attn_to_group_id[csa_layer][static_cast<size_t>(KVCacheAttnType::CSA_KV)], 0);
    EXPECT_EQ(config.layer_attn_to_group_id[csa_layer][static_cast<size_t>(KVCacheAttnType::INDEXER_KV)], 2);
    EXPECT_EQ(config.layer_attn_to_group_id[csa_layer][static_cast<size_t>(KVCacheAttnType::INDEXER_STATE)], 3);
    EXPECT_EQ(config.layer_attn_to_group_id[csa_layer][static_cast<size_t>(KVCacheAttnType::CSA_STATE)], 4);
    EXPECT_EQ(config.layer_attn_to_group_id[csa_layer][static_cast<size_t>(KVCacheAttnType::SWA_KV)], 6);
    EXPECT_EQ(config.layer_to_group_ids[csa_layer].size(), 5u);

    const int hca_layer = dsv4.hca_layer_ids.front();
    EXPECT_EQ(config.layer_attn_to_group_id[hca_layer][static_cast<size_t>(KVCacheAttnType::HCA_KV)], 1);
    EXPECT_EQ(config.layer_attn_to_group_id[hca_layer][static_cast<size_t>(KVCacheAttnType::HCA_STATE)], 5);
    EXPECT_EQ(config.layer_attn_to_group_id[hca_layer][static_cast<size_t>(KVCacheAttnType::SWA_KV)], 6);
    EXPECT_EQ(config.layer_to_group_ids[hca_layer].size(), 3u);
}

TEST(KVCacheResourceTest, LayerAttnTypeIndexesDifferentGroups) {
    auto config = makeAllocatorConfig();
    const int csa_layer = config.dsv4_config->csa_layer_ids.front();

    KVCacheResource resource;
    resource.initGroups(config.groupNums(),
                        static_cast<int>(config.layer_all_num),
                        config.layer_to_group_id,
                        config.kernelBlocksPerKvBlock(),
                        config.group_types,
                        config.layer_attn_to_group_id);

    resource.mutableBlockIds(0).assign({10, 11});
    resource.mutableBlockIds(2).assign({20, 21});
    resource.mutableBlockIds(6).assign({60, 61});

    EXPECT_EQ(resource.groupId(csa_layer, KVCacheAttnType::CSA_KV), 0);
    EXPECT_EQ(resource.groupId(csa_layer, KVCacheAttnType::INDEXER_KV), 2);
    EXPECT_EQ(resource.groupId(csa_layer, KVCacheAttnType::SWA_KV), 6);
    EXPECT_EQ(resource.blocks(csa_layer, KVCacheAttnType::CSA_KV)[0], 10);
    EXPECT_EQ(resource.blocks(csa_layer, KVCacheAttnType::INDEXER_KV)[0], 20);
    EXPECT_EQ(resource.blocks(csa_layer, KVCacheAttnType::SWA_KV)[0], 60);
}

class DSV4HybridPoolAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
    }
};

TEST_F(DSV4HybridPoolAllocatorTest, InitUsesIndependentPoolAccounting) {
    auto config    = makeAllocatorConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    size_t expected_total_blocks = 0;
    for (auto block_num : config.group_block_nums) {
        expected_total_blocks += block_num - 1;  // block 0 reserved in each pool
    }
    EXPECT_EQ(allocator->totalBlocksNum(), expected_total_blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), expected_total_blocks);
}

TEST_F(DSV4HybridPoolAllocatorTest, AvailableTokensUsesBottleneckGroupPool) {
    auto config = makeAllocatorConfig();
    config.group_block_nums.assign(config.groupNums(), 8);
    config.group_block_nums[static_cast<size_t>(DSV4CacheType::SWA_KV)] = 5;

    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    const size_t expected_total_blocks = 6 * (8 - 1) + (5 - 1);
    const size_t expected_tokens       = (5 - 1) * config.seq_size_per_block;
    EXPECT_EQ(allocator->totalBlocksNum(), expected_total_blocks);
    EXPECT_EQ(allocator->availableTokensNum(), expected_tokens);
    EXPECT_EQ(allocator->maxAvailableTokensNum(), expected_tokens);
}

TEST_F(DSV4HybridPoolAllocatorTest, TypedAddressLookupReturnsDifferentGroupBuffers) {
    auto config    = makeAllocatorConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    const int csa_layer = config.dsv4_config->csa_layer_ids.front();
    auto csa_buf = allocator->convertIndexToBuffer(csa_layer, KVCacheAttnType::CSA_KV, /*block_id=*/1);
    auto swa_buf = allocator->convertIndexToBuffer(csa_layer, KVCacheAttnType::SWA_KV, /*block_id=*/1);
    ASSERT_FALSE(csa_buf.empty());
    ASSERT_FALSE(swa_buf.empty());
    ASSERT_NE(csa_buf[0].addr, nullptr);
    ASSERT_NE(swa_buf[0].addr, nullptr);
    EXPECT_NE(csa_buf[0].addr, swa_buf[0].addr);
    EXPECT_EQ(csa_buf[0].size_bytes, config.cache_specs[0]->block_size_bytes());
    EXPECT_EQ(swa_buf[0].size_bytes, config.cache_specs[6]->block_size_bytes());
}

TEST_F(DSV4HybridPoolAllocatorTest, AllLayerCacheBaseExposesPerAttnRawGroupTensors) {
    auto config    = makeAllocatorConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs_by_attn.size(), static_cast<size_t>(config.layer_all_num));
    ASSERT_EQ(layout.layer_attn_to_group_id.size(), static_cast<size_t>(config.layer_all_num));

    const int csa_layer = config.dsv4_config->csa_layer_ids.front();
    const auto& csa_attn_tensors = layout.layers_to_kv_buffer_ptrs_by_attn[static_cast<size_t>(csa_layer)];
    ASSERT_GT(csa_attn_tensors.size(), static_cast<size_t>(KVCacheAttnType::SWA_KV));

    const auto& csa_kv = csa_attn_tensors[static_cast<size_t>(KVCacheAttnType::CSA_KV)];
    const auto& indexer_kv = csa_attn_tensors[static_cast<size_t>(KVCacheAttnType::INDEXER_KV)];
    const auto& csa_state = csa_attn_tensors[static_cast<size_t>(KVCacheAttnType::CSA_STATE)];
    const auto& swa_kv = csa_attn_tensors[static_cast<size_t>(KVCacheAttnType::SWA_KV)];

    ASSERT_TRUE(csa_kv.defined());
    ASSERT_TRUE(indexer_kv.defined());
    ASSERT_TRUE(csa_state.defined());
    ASSERT_TRUE(swa_kv.defined());

    EXPECT_EQ(csa_kv.scalar_type(), torch::kUInt8);
    EXPECT_EQ(indexer_kv.scalar_type(), torch::kUInt8);
    EXPECT_EQ(swa_kv.scalar_type(), torch::kUInt8);
    EXPECT_EQ(csa_state.scalar_type(), torch::kFloat32);

    EXPECT_EQ(csa_kv.dim(), 2);
    EXPECT_EQ(indexer_kv.dim(), 2);
    EXPECT_EQ(csa_state.dim(), 2);
    EXPECT_EQ(swa_kv.dim(), 2);
    EXPECT_EQ(csa_kv.size(0), config.block_num);
    EXPECT_EQ(indexer_kv.size(0), config.block_num);
    EXPECT_EQ(csa_state.size(0), config.block_num);
    EXPECT_EQ(swa_kv.size(0), config.block_num);

    auto expect_stride_elems = [&](size_t gid) {
        return static_cast<int64_t>(config.cache_specs[gid]->block_size_bytes() / dtypeSize(config.cache_specs[gid]->dtype));
    };
    EXPECT_EQ(csa_kv.size(1), expect_stride_elems(0));
    EXPECT_EQ(indexer_kv.size(1), expect_stride_elems(2));
    EXPECT_EQ(csa_state.size(1), expect_stride_elems(4));
    EXPECT_EQ(swa_kv.size(1), expect_stride_elems(6));

    ASSERT_TRUE(layout.layers_to_kv_buffer_ptrs[static_cast<size_t>(csa_layer)].defined());
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs[static_cast<size_t>(csa_layer)].data_ptr(), swa_kv.data_ptr());
}

TEST_F(DSV4HybridPoolAllocatorTest, HcaLayerHasHcaStateAndSwaButNoCsaKvTensor) {
    auto config    = makeAllocatorConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    const int hca_layer = config.dsv4_config->hca_layer_ids.front();
    const auto& hca_attn_tensors = layout.layers_to_kv_buffer_ptrs_by_attn[static_cast<size_t>(hca_layer)];

    EXPECT_FALSE(hca_attn_tensors[static_cast<size_t>(KVCacheAttnType::CSA_KV)].defined());
    EXPECT_TRUE(hca_attn_tensors[static_cast<size_t>(KVCacheAttnType::HCA_KV)].defined());
    EXPECT_TRUE(hca_attn_tensors[static_cast<size_t>(KVCacheAttnType::HCA_STATE)].defined());
    EXPECT_TRUE(hca_attn_tensors[static_cast<size_t>(KVCacheAttnType::SWA_KV)].defined());

    EXPECT_EQ(layout.layer_attn_to_group_id[static_cast<size_t>(hca_layer)]
                                      [static_cast<size_t>(KVCacheAttnType::HCA_KV)],
              1);
    EXPECT_EQ(layout.layer_attn_to_group_id[static_cast<size_t>(hca_layer)]
                                      [static_cast<size_t>(KVCacheAttnType::HCA_STATE)],
              5);
    EXPECT_EQ(layout.layer_attn_to_group_id[static_cast<size_t>(hca_layer)]
                                      [static_cast<size_t>(KVCacheAttnType::SWA_KV)],
              6);
}

TEST_F(DSV4HybridPoolAllocatorTest, MallocAndFreeAllGroups) {
    auto config    = makeAllocatorConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto batch_resource = makeBatchResource(config);

    auto complete_token_ids = makeCompleteTokenIds(/*token_num=*/300, static_cast<int>(config.seq_size_per_block));
    MallocInfo malloc_info{batch_resource, complete_token_ids};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;

    const size_t free_before = allocator->freeBlocksNum();
    auto         result      = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);

    for (int gid = 0; gid < config.groupNums(); ++gid) {
        EXPECT_GT(batch_resource->blocksNum(0, gid), 0) << "group " << gid;
    }
    EXPECT_LT(allocator->freeBlocksNum(), free_before);

    allocator->free(FreeInfo{batch_resource, complete_token_ids});
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(DSV4HybridPoolAllocatorTest, InsertThenDeviceReuseUsesFullGroupsAndLinearTail) {
    auto config    = makeAllocatorConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto cached_resource = makeBatchResource(config);
    cached_resource->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto cached_tokens = makeCompleteTokenIds(/*token_num=*/513, static_cast<int>(config.seq_size_per_block));
    MallocInfo cached_malloc{cached_resource, cached_tokens};
    cached_malloc.reuse_cache         = true;
    cached_malloc.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(cached_malloc).success);

    allocator->insertIntoCache(InsertInfo{cached_resource, cached_tokens, /*is_resident=*/false});
    EXPECT_GT(allocator->blockCacheRefBlocksNum(), 0);

    auto reuse_resource = makeBatchResource(config);
    reuse_resource->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto reuse_tokens = makeCompleteTokenIds(/*token_num=*/513, static_cast<int>(config.seq_size_per_block));
    MallocInfo reuse_malloc{reuse_resource, reuse_tokens};
    reuse_malloc.reuse_cache         = true;
    reuse_malloc.enable_device_cache = true;

    auto result = allocator->malloc(reuse_malloc);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 2 * static_cast<int>(config.seq_size_per_block));
    EXPECT_EQ(reuse_resource->cacheResource(0).deviceReuseBlockNum(), 2u);

    for (int gid : {0, 1, 2}) {
        ASSERT_GE(reuse_resource->blocksNum(0, gid), 2);
        EXPECT_EQ(reuse_resource->blocks(0, gid)[0], cached_resource->blocks(0, gid)[0]) << "group " << gid;
        EXPECT_EQ(reuse_resource->blocks(0, gid)[1], cached_resource->blocks(0, gid)[1]) << "group " << gid;
    }
    for (int gid : {3, 4, 5, 6}) {
        ASSERT_GE(reuse_resource->blocksNum(0, gid), 2);
        EXPECT_TRUE(isNullBlockIdx(reuse_resource->blocks(0, gid)[0])) << "group " << gid;
        EXPECT_EQ(reuse_resource->blocks(0, gid)[1], cached_resource->blocks(0, gid)[1]) << "group " << gid;
    }

    allocator->free(FreeInfo{reuse_resource, reuse_tokens});
    allocator->free(FreeInfo{cached_resource, cached_tokens});
}

TEST_F(DSV4HybridPoolAllocatorTest, BlockCacheFreeReleasesGroupLocalCacheRefs) {
    auto config    = makeAllocatorConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto cached_resource = makeBatchResource(config);
    cached_resource->setBatchCacheKeys(0, CacheKeysType{200, 201, 202});
    auto tokens = makeCompleteTokenIds(/*token_num=*/513, static_cast<int>(config.seq_size_per_block));
    MallocInfo malloc_info{cached_resource, tokens};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(malloc_info).success);
    allocator->insertIntoCache(InsertInfo{cached_resource, tokens, /*is_resident=*/false});

    const size_t block_cache_refs = allocator->blockCacheRefBlocksNum();
    ASSERT_GT(block_cache_refs, 0);

    auto evicted_resource = makeBatchResource(config);
    for (int gid = 0; gid < config.groupNums(); ++gid) {
        const auto& blocks = cached_resource->blocks(0, gid);
        ASSERT_GE(blocks.size(), 2u);
        evicted_resource->setBatchBlocks(0, gid, BlockIndicesType{blocks[0], blocks[1], blocks[1], NULL_BLOCK_IDX});
    }

    allocator->blockCacheFree(evicted_resource);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);

    allocator->free(FreeInfo{cached_resource, tokens});
}

TEST_F(DSV4HybridPoolAllocatorTest, IncrKVCacheRefAddsAndReleasesRequestRefs) {
    auto config    = makeAllocatorConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto batch_resource = makeBatchResource(config);
    batch_resource->setBatchCacheKeys(0, CacheKeysType{300, 301});
    auto tokens = makeCompleteTokenIds(/*token_num=*/300, static_cast<int>(config.seq_size_per_block));
    MallocInfo malloc_info{batch_resource, tokens};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(malloc_info).success);

    const size_t refs_before = allocator->requestRefBlocksNum();
    const size_t free_before = allocator->freeBlocksNum();
    {
        auto selected = allocator->incrKVCacheRef(batch_resource->cacheResource(0), CacheKeysType{300});
        ASSERT_NE(selected, nullptr);
        EXPECT_EQ(selected->cacheKeys().size(), 1u);
        ASSERT_EQ(selected->groupNums(), config.groupNums());

        for (int gid = 0; gid < config.groupNums(); ++gid) {
            ASSERT_EQ(selected->blocks(gid).size(), 1u) << "group " << gid;
            ASSERT_FALSE(batch_resource->blocks(0, gid).empty()) << "group " << gid;
            EXPECT_EQ(selected->blocks(gid)[0], batch_resource->blocks(0, gid)[0]) << "group " << gid;
        }
        EXPECT_EQ(allocator->requestRefBlocksNum(), refs_before);
        EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    }
    EXPECT_EQ(allocator->requestRefBlocksNum(), refs_before);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);

    allocator->free(FreeInfo{batch_resource, tokens});
}

}  // namespace test
}  // namespace rtp_llm
