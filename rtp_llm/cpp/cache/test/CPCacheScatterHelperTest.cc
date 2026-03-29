#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "rtp_llm/cpp/cache/CPCacheScatterHelper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

class CPCacheScatterHelperTest : public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        device_ = createDevice();
        ASSERT_NE(device_, nullptr);
    }

    std::shared_ptr<KVCacheManager> createMlaCacheManager(int layer_num, int block_num, int block_size) {
        // MLA config: use MLAKVCacheSpec so convertIndexToBuffer returns fused kv
        CacheConfig config;
        config.dtype              = DataType::TYPE_FP16;
        config.layer_num          = static_cast<uint32_t>(layer_num);
        config.layer_all_num      = static_cast<uint32_t>(layer_num);
        config.block_num          = static_cast<uint32_t>(block_num);
        config.seq_size_per_block = static_cast<size_t>(block_size);
        config.use_mla            = true;

        // Element stride per token must be 16-byte aligned for the scatter kernel.
        // kv_lora_rank=8, FP16 => K per token = 8*2 = 16 bytes (aligned).
        const uint32_t kv_lora_rank = 8;
        const uint32_t rope_dim     = 8;

        auto spec                = std::make_shared<MLAKVCacheSpec>();
        spec->type               = KVCacheSpecType::MultiHeadLatentAttention;
        spec->dtype              = DataType::TYPE_FP16;
        spec->layer_num          = static_cast<uint32_t>(layer_num);
        spec->local_head_num_kv  = 1;
        spec->seq_size_per_block = static_cast<uint32_t>(block_size);
        spec->kv_lora_rank       = kv_lora_rank;
        spec->rope_head_dim      = rope_dim;
        config.cache_specs.push_back(spec);

        std::vector<int> layer_ids(layer_num);
        std::iota(layer_ids.begin(), layer_ids.end(), 0);
        config.layer_ids.push_back(layer_ids);
        config.global_layer_ids.push_back(layer_ids);
        config.layer_to_group_id.assign(layer_num, 0);

        config.kv_block_stride_bytes = spec->block_size_bytes();
        config.kv_block_size_bytes   = static_cast<size_t>(spec->block_size_bytes() * layer_num);
        config.block_size_bytes      = config.kv_block_size_bytes;

        const size_t per_layer_stride = config.kv_block_stride_bytes;
        config.layer_to_block_stride_bytes.assign(layer_num, static_cast<int>(per_layer_stride));

        auto mgr = std::make_shared<KVCacheManager>(config, device_);
        EXPECT_TRUE(mgr->init());
        return mgr;
    }

    DeviceBase* device_ = nullptr;
};

TEST_F(CPCacheScatterHelperTest, PrepareStagingPlanAllocatesCorrectBlocks) {
    const int layer_num  = 2;
    const int block_num  = 30;
    const int block_size = 4;
    auto      mgr        = createMlaCacheManager(layer_num, block_num, block_size);

    CPCacheScatterHelper helper(mgr.get(), device_);
    const int            vblock_count = 3;
    const int            cp_size      = 2;

    auto plan = helper.prepareStagingPlan(vblock_count, cp_size, layer_num);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->vblock_count, vblock_count);
    EXPECT_EQ(plan->cp_size, cp_size);
    EXPECT_EQ(static_cast<int>(plan->staging_block_ids.size()), vblock_count * cp_size);
    EXPECT_EQ(plan->layer_infos.size(), static_cast<size_t>(layer_num));

    // Each staging block should have non-null address and non-zero size
    for (int layer = 0; layer < layer_num; ++layer) {
        const auto& layer_info = plan->layer_infos[layer];
        ASSERT_EQ(static_cast<int>(layer_info.infos.size()), vblock_count * cp_size);
        for (int s = 0; s < vblock_count * cp_size; ++s) {
            EXPECT_NE(layer_info.infos[s].addr, nullptr) << "layer=" << layer << " s=" << s;
            EXPECT_GT(layer_info.infos[s].size_bytes, 0u) << "layer=" << layer << " s=" << s;
        }
    }
}

TEST_F(CPCacheScatterHelperTest, StagingPlanRAIIFreesBlocksOnDestruction) {
    const int layer_num  = 2;
    const int block_num  = 20;
    const int block_size = 4;
    auto      mgr        = createMlaCacheManager(layer_num, block_num, block_size);

    auto block_pool = mgr->getBlockPool();
    ASSERT_NE(block_pool, nullptr);
    size_t free_before = block_pool->freeBlocksNum();

    CPCacheScatterHelper helper(mgr.get(), device_);
    const int            staging_cnt = 6;  // 3 vblocks * cp_size 2

    {
        auto plan = helper.prepareStagingPlan(3, 2, layer_num);
        ASSERT_NE(plan, nullptr);
        // While plan is alive, staging blocks should be borrowed
        size_t free_during = block_pool->freeBlocksNum();
        EXPECT_EQ(free_during, free_before - staging_cnt);
    }
    // After plan destruction, blocks should be returned
    size_t free_after = block_pool->freeBlocksNum();
    EXPECT_EQ(free_after, free_before);
}

TEST_F(CPCacheScatterHelperTest, ScatterAndReleaseProducesContiguousLayout) {
    const int layer_num  = 1;
    const int block_num  = 30;
    const int block_size = 4;
    const int cp_size    = 2;
    const int vblock_count = 2;
    auto      mgr = createMlaCacheManager(layer_num, block_num, block_size);

    CPCacheScatterHelper helper(mgr.get(), device_);
    auto plan = helper.prepareStagingPlan(vblock_count, cp_size, layer_num);
    ASSERT_NE(plan, nullptr);

    const int tokens_per_vb = block_size * cp_size;
    const int total_tokens  = vblock_count * tokens_per_vb;
    const int decode_blocks = total_tokens / block_size;

    // Determine element stride from the first staging block
    auto sample_parts   = mgr->convertIndexToBuffer(plan->staging_block_ids[0], 0, 1, 0);
    int  elem_stride    = static_cast<int>(sample_parts[0].size_bytes / block_size);
    ASSERT_GT(elem_stride, 0);
    size_t block_bytes  = static_cast<size_t>(block_size) * elem_stride;

    // Fill staging blocks with known pattern:
    // For vblock v, peer p, slot s: tag = global_token & 0xFF
    for (int v = 0; v < vblock_count; ++v) {
        for (int p = 0; p < cp_size; ++p) {
            int staging_idx = v * cp_size + p;
            int bid         = plan->staging_block_ids[staging_idx];
            auto parts      = mgr->convertIndexToBuffer(bid, 0, 1, 0);

            std::vector<uint8_t> block_data(block_bytes, 0);
            for (int s = 0; s < block_size; ++s) {
                int global_token = v * tokens_per_vb + s * cp_size + p;
                uint8_t tag = static_cast<uint8_t>(global_token & 0xFF);
                std::memset(block_data.data() + s * elem_stride, tag, elem_stride);
            }
            device_->copy({Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_BYTES, {block_bytes}, parts[0].addr),
                           Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_BYTES, {block_bytes}, block_data.data())});
        }
    }
    device_->syncAndCheck();

    // Allocate decode blocks
    auto decode_block_ids = mgr->getBlockPool()->malloc(decode_blocks);
    ASSERT_EQ(static_cast<int>(decode_block_ids.size()), decode_blocks);

    auto block_ids_holder = std::make_shared<BlockIds>();
    block_ids_holder->blocks() = decode_block_ids;
    GroupBlockIds block_ids_by_group = {block_ids_holder};

    // Run scatter
    helper.scatterAndRelease(std::move(plan), block_ids_by_group, mgr->cacheConfig(), layer_num);

    // Verify: each decode block should contain contiguous tokens
    for (int t = 0; t < total_tokens; ++t) {
        int blk_idx = t / block_size;
        int slot    = t % block_size;
        uint8_t expected = static_cast<uint8_t>(t & 0xFF);

        auto parts = mgr->convertIndexToBuffer(decode_block_ids[blk_idx], 0, 1, 0);
        std::vector<uint8_t> block_data(block_bytes);
        device_->copy({Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_BYTES, {block_bytes}, block_data.data()),
                       Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_BYTES, {block_bytes}, parts[0].addr)});
        device_->syncAndCheck();

        const uint8_t* ptr = block_data.data() + slot * elem_stride;
        for (int b = 0; b < elem_stride; ++b) {
            ASSERT_EQ(ptr[b], expected) << "token=" << t << " blk=" << blk_idx << " slot=" << slot << " byte=" << b;
        }
    }

    // Cleanup decode blocks
    mgr->getBlockPool()->requestFree(decode_block_ids);
}

TEST_F(CPCacheScatterHelperTest, ScatterReleasesBlocksAfterCompletion) {
    const int layer_num  = 1;
    const int block_num  = 30;
    const int block_size = 4;
    auto      mgr = createMlaCacheManager(layer_num, block_num, block_size);

    auto   block_pool  = mgr->getBlockPool();
    size_t free_before = block_pool->freeBlocksNum();

    CPCacheScatterHelper helper(mgr.get(), device_);
    auto plan = helper.prepareStagingPlan(2, 2, layer_num);
    ASSERT_NE(plan, nullptr);

    // Allocate decode blocks
    auto decode_block_ids = block_pool->malloc(4);  // 2 vblocks * cp_size 2 = 8 tokens, block_size=4 => 2 decode blocks ... wait
    // Actually: 2 vblocks, cp_size 2, block_size 4 => total_tokens = 2*4*2=16, decode_blocks = 16/4 = 4
    ASSERT_EQ(static_cast<int>(decode_block_ids.size()), 4);

    auto block_ids_holder = std::make_shared<BlockIds>();
    block_ids_holder->blocks() = decode_block_ids;
    GroupBlockIds block_ids_by_group = {block_ids_holder};

    helper.scatterAndRelease(std::move(plan), block_ids_by_group, mgr->cacheConfig(), layer_num);

    // After scatter, staging blocks should be freed
    // (decode blocks are still held, so subtract those)
    size_t free_after = block_pool->freeBlocksNum();
    EXPECT_EQ(free_after, free_before - 4 /* staging */ - 4 /* decode */ + 4 /* staging freed */);

    block_pool->requestFree(decode_block_ids);
}

}  // namespace test
}  // namespace rtp_llm
