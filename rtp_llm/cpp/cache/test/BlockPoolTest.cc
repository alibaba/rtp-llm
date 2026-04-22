#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <set>
#include <torch/torch.h>
#include <numeric>
#include <optional>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class BlockPoolTest: public ::testing::Test {
protected:
    void SetUp() override {
        createDevice();
    }

    void TearDown() override {
        block_pool_.reset();
    }

    std::shared_ptr<BlockPool> block_pool_;
};

namespace {

static rtp_llm::ModelConfig makeTestModelConfig(uint32_t num_layers) {
    rtp_llm::ModelConfig m;
    m.num_layers                   = static_cast<int>(num_layers);
    m.max_seq_len                  = 128;
    m.hidden_size                  = 1;
    m.vocab_size                   = 1;
    m.data_type                    = rtp_llm::DataType::TYPE_FP16;
    m.attn_config.use_mla          = false;
    m.attn_config.tokens_per_block = 4;
    m.attn_config.kv_head_num      = 2;
    m.attn_config.size_per_head    = 1;
    m.attn_config.kv_cache_dtype   = KvCacheDataType::INT8;  // enable kv-scale
    m.attn_config.kv_lora_rank     = 0;
    m.attn_config.rope_head_dim    = 0;
    m.attn_config.head_num         = 2;
    // keep other fields default
    return m;
}

static rtp_llm::CacheConfig
makeMtpCacheConfigByCreateSpConfig(uint32_t main_layers, int mtp_module_num, uint32_t block_num) {
    auto score_model_config   = makeTestModelConfig(main_layers);
    auto propose_model_config = makeTestModelConfig(/*num_layers=*/1);

    rtp_llm::ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    rtp_llm::RuntimeConfig runtime_config;

    rtp_llm::KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = static_cast<int>(block_num);

    rtp_llm::SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = mtp_module_num;

    // NOTE: createSpConfig will populate allocator_configs for main + MTP sub-models.
    auto cfg = rtp_llm::CacheConfigCreator::createSpConfig(score_model_config,
                                                           propose_model_config,
                                                           parallelism_config,
                                                           runtime_config,
                                                           kv_cache_config,
                                                           sp_config,
                                                           /*warm_up_result=*/std::nullopt,
                                                           /*is_mtp=*/true,
                                                           /*is_eagle=*/false);
    return cfg;
}

}  // namespace

// Initialization Test
TEST_F(BlockPoolTest, ConstructorAndInit) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    ASSERT_NE(block_pool_, nullptr);

    bool init_result = block_pool_->init();
    EXPECT_TRUE(init_result);

    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, MTPIndependentBlockPools) {
    // After global_layer_id removal, each model (main + MTP sub-models) has its own
    // independent BlockPool with model-local layer IDs starting from 0.
    // Verify that per-model BlockPools use model-local layer IDs correctly.
    auto cache_cfg = makeMtpCacheConfigByCreateSpConfig(/*main_layers=*/2, /*mtp_module_num=*/2, /*block_num=*/4);

    // allocator_configs[0] = main model, allocator_configs[1..] = MTP sub-models.
    ASSERT_EQ(cache_cfg.allocator_configs.size(), 3u);  // main + 2 MTP modules

    // Main model: BlockPool with 2 layers, using local layer IDs [0, 1]
    auto main_pool_cfg = rtp_llm::BlockPoolConfigHelper::createConfig(cache_cfg);
    // main_pool_cfg has memory_layouts[0].layer_num == 2 (plus MTP layouts)
    ASSERT_GE(main_pool_cfg.memory_layouts.size(), 1u);
    ASSERT_EQ(main_pool_cfg.memory_layouts[0].layer_num, 2u);

    block_pool_ = std::make_shared<BlockPool>(main_pool_cfg);
    ASSERT_TRUE(block_pool_->init());

    // model-local layer 0 and layer 1 should have distinct addresses
    auto addr0 = block_pool_->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    auto addr1 = block_pool_->convertIndexToAddr(/*layer_id=*/1, /*block_id=*/1);
    ASSERT_NE(addr0.kv_addr, nullptr);
    ASSERT_NE(addr1.kv_addr, nullptr);
    EXPECT_NE(addr0.kv_addr, addr1.kv_addr);

    // MTP sub-model: separate BlockPool with its own 1 layer, using local layer ID [0].
    // Build per-model CacheConfig from allocator_configs[1].
    CacheConfig mtp_cfg;
    {
        auto sub_alloc                      = cache_cfg.getAllocatorConfig(1);
        sub_alloc.model_id                  = 0;  // local model_id within this per-model config
        mtp_cfg.seq_size_per_block          = sub_alloc.seq_size_per_block;
        mtp_cfg.layer_all_num               = sub_alloc.layer_num;
        mtp_cfg.layer_to_group_id           = sub_alloc.layer_to_group_id;
        mtp_cfg.layer_to_block_stride_bytes = sub_alloc.layer_to_block_stride_bytes;
        mtp_cfg.allocator_configs           = {sub_alloc};
    }
    auto mtp_pool_cfg = rtp_llm::BlockPoolConfigHelper::createConfig(mtp_cfg);
    ASSERT_EQ(mtp_pool_cfg.memory_layouts.size(), 1u);
    ASSERT_EQ(mtp_pool_cfg.memory_layouts[0].layer_num, 1u);

    auto mtp_pool = std::make_shared<BlockPool>(mtp_pool_cfg);
    ASSERT_TRUE(mtp_pool->init());

    auto mtp_addr0 = mtp_pool->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    ASSERT_NE(mtp_addr0.kv_addr, nullptr);

    // Partitioned buffer correctness on MTP layer (heads=2, partition_count=2, partition_id=1)
    auto mtp_parts = mtp_pool->convertIndexToBuffer(
        /*layer_id=*/0, /*block_id=*/1, /*partition_count=*/2, /*partition_id=*/1);
    ASSERT_EQ(mtp_parts.size(), 4u);
    ASSERT_NE(mtp_parts[0].addr, nullptr);
    const auto& mtp_layout_cfg = mtp_pool_cfg.memory_layouts[0];
    EXPECT_EQ(mtp_parts[0].size_bytes, mtp_layout_cfg.k_block_stride_bytes / 2);
    EXPECT_EQ(mtp_parts[1].size_bytes, mtp_layout_cfg.v_block_stride_bytes / 2);
}
// Allocation Test
TEST_F(BlockPoolTest, AllocSingleBlock) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(1);
    EXPECT_EQ(blocks.size(), 1);
    EXPECT_GE(blocks[0], 0);
    EXPECT_LT(blocks[0], static_cast<BlockIdxType>(config.block_num));
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 2);
}

TEST_F(BlockPoolTest, AllocMultipleBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    int  alloc_count = 5;
    auto blocks      = block_pool_->malloc(alloc_count);
    EXPECT_EQ(blocks.size(), alloc_count);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - alloc_count - 1);

    std::set<BlockIdxType> unique_blocks(blocks.begin(), blocks.end());
    EXPECT_EQ(unique_blocks.size(), alloc_count);
}

TEST_F(BlockPoolTest, AllocAllBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(config.block_num - 1);
    EXPECT_EQ(blocks.size(), config.block_num - 1);
    EXPECT_EQ(block_pool_->freeBlocksNum(), 0);
}

TEST_F(BlockPoolTest, AllocMoreThanAvailable) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks1 = block_pool_->malloc(5);
    EXPECT_EQ(blocks1.size(), 5);

    auto blocks2 = block_pool_->malloc(10);
    EXPECT_EQ(blocks2.size(), 0);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);
}

// Free Test
TEST_F(BlockPoolTest, FreeBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(5);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, FreePartialBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(5);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);

    std::vector<BlockIdxType> partial_blocks(blocks.begin(), blocks.begin() + 3);
    block_pool_->requestFree(partial_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);
}

TEST_F(BlockPoolTest, ReferenceAndFree) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();
    auto total_blocks = block_pool_->freeBlocksNum();

    {
        auto blocks = block_pool_->malloc(3);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestReference(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);
    }

    // Blocks referred to by the block cache do not affect the freeblocks count.
    // Blocks referred to by the block cache do not affect the available blocks count.
    {
        auto blocks2 = block_pool_->malloc(3);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->blockCacheReference(blocks2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 3);

        block_pool_->requestFree(blocks2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 3);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);

        block_pool_->blockCacheFree(blocks2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);
    }

    {
        auto blocks = block_pool_->malloc(2);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 2);

        block_pool_->blockCacheReference(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 2);

        block_pool_->connectorReference(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->connectorRefBlocksNum(), 2);
        EXPECT_EQ(block_pool_->requestRefBlocksNum(), 2);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->connectorRefBlocksNum(), 2);
        EXPECT_EQ(block_pool_->requestRefBlocksNum(), 0);

        block_pool_->connectorFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks - 2);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);
        EXPECT_EQ(block_pool_->connectorRefBlocksNum(), 0);
        EXPECT_EQ(block_pool_->requestRefBlocksNum(), 0);

        block_pool_->blockCacheFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), total_blocks);
        EXPECT_EQ(block_pool_->availableBlocksNum(), total_blocks);
    }
}

TEST_F(BlockPoolTest, MultipleReferencesAndFrees) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(2);

    block_pool_->requestReference(blocks);
    block_pool_->requestReference(blocks);
    block_pool_->requestReference(blocks);

    // free for 4 times (1 + 3)
    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 3);

    block_pool_->requestFree(blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

// Convert Index to Addr Test
TEST_F(BlockPoolTest, ConvertIndexToAddr) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    const auto layer_num = static_cast<int>(config.memory_layouts[0].layer_num);
    for (int layer = 0; layer < layer_num; ++layer) {
        for (int block = 0; block < 3; ++block) {
            auto addr_info = block_pool_->convertIndexToAddr(layer, block);
            EXPECT_NE(addr_info.kv_addr, nullptr);
        }
    }
}

TEST_F(BlockPoolTest, ConvertIndexToBuffer) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    int layer = 0;
    int block = 0;

    auto buffer_info = block_pool_->convertIndexToBuffer(layer, block);
    ASSERT_EQ(buffer_info.size(), 1u);
    EXPECT_NE(buffer_info[0].addr, nullptr);
}

TEST_F(BlockPoolTest, ConvertIndexToAddrAndBufferWithScale) {
    // dtype=int8 will enable kv-scale pool automatically in BlockPoolConfigHelper.
    auto config = createTestConfig(
        /*k_block_stride_bytes=*/512,
        /*v_block_stride_bytes=*/512,
        /*k_scale_stride_bytes=*/128,
        /*v_scale_stride_bytes=*/128,
        /*dtype=*/rtp_llm::DataType::TYPE_INT8,
        /*local_head_num_kv=*/2,
        /*seq_size_per_block=*/4);

    block_pool_ = std::make_shared<BlockPool>(config);
    ASSERT_TRUE(block_pool_->init());

    const auto& layout_cfg = config.memory_layouts[0];
    const int   layer      = 0;
    const int   block      = 0;
    auto        addr       = block_pool_->convertIndexToAddr(layer, block);
    EXPECT_NE(addr.kv_addr, nullptr);
    EXPECT_NE(addr.kv_scale_addr, nullptr);

    auto buf = block_pool_->convertIndexToBuffer(layer, block);
    ASSERT_EQ(buf.size(), 2u);
    EXPECT_NE(buf[0].addr, nullptr);
    EXPECT_NE(buf[1].addr, nullptr);
    EXPECT_EQ(buf[1].size_bytes, layout_cfg.kv_scale_stride_bytes);
}

// LayerCache Base Test
TEST_F(BlockPoolTest, LayerCacheBase) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto layer_tensors = block_pool_->allLayerCacheBase();
    EXPECT_EQ(layer_tensors.size(), config.memory_layouts[0].layer_num);

    for (size_t i = 0; i < layer_tensors.size(); ++i) {
        EXPECT_TRUE(layer_tensors[i].defined());
        EXPECT_GT(layer_tensors[i].numel(), 0);
    }
}

// Boundary Condition Test
TEST_F(BlockPoolTest, AllocZeroBlocks) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    auto blocks = block_pool_->malloc(0);
    EXPECT_EQ(blocks.size(), 0);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, FreeEmptyVector) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    std::vector<BlockIdxType> empty_blocks;
    block_pool_->requestFree(empty_blocks);
    EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
}

TEST_F(BlockPoolTest, OutOfRangeLayerId) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    int invalid_layer = static_cast<int>(config.memory_layouts[0].layer_num) + 10;
    EXPECT_THROW((void)block_pool_->convertIndexToAddr(invalid_layer, 0), rtp_llm::RTPException);
}

TEST_F(BlockPoolTest, AllocFreeAllocCycle) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    for (int i = 0; i < 5; ++i) {
        auto blocks = block_pool_->malloc(5);
        EXPECT_EQ(blocks.size(), 5);
        EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 6);

        block_pool_->requestFree(blocks);
        EXPECT_EQ(block_pool_->freeBlocksNum(), config.block_num - 1);
    }
}

TEST_F(BlockPoolTest, MixedAllocFreeOperations) {
    auto config = createTestConfig();
    block_pool_ = std::make_shared<BlockPool>(config);
    block_pool_->init();

    std::vector<std::vector<BlockIdxType>> allocated_blocks;

    allocated_blocks.push_back(block_pool_->malloc(2));
    EXPECT_EQ(block_pool_->freeBlocksNum(), 7);

    allocated_blocks.push_back(block_pool_->malloc(3));
    EXPECT_EQ(block_pool_->freeBlocksNum(), 4);

    block_pool_->requestFree(allocated_blocks[0]);
    EXPECT_EQ(block_pool_->freeBlocksNum(), 6);

    allocated_blocks.push_back(block_pool_->malloc(4));
    EXPECT_EQ(block_pool_->freeBlocksNum(), 2);

    block_pool_->requestFree(allocated_blocks[1]);
    block_pool_->requestFree(allocated_blocks[2]);
    EXPECT_EQ(block_pool_->freeBlocksNum(), 9);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
