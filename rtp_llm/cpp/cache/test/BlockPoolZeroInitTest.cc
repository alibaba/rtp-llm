#include <gtest/gtest.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"

namespace rtp_llm {

TEST(BlockPoolZeroInitTest, ZeroBackingAndPreserveFiniteValuesAcrossReuse) {
#if USING_ROCM
    DeviceBlockPoolConfig config;
    config.physical_block_count = 4;
    config.total_size_bytes = 256;
    MemoryLayoutConfig layout;
    layout.layer_num = 1;
    layout.block_num = 4;
    layout.dtype = TYPE_FP16;
    layout.kv_block_pool_size_bytes = 256;
    layout.total_size_bytes = 256;
    layout.kv_block_stride_bytes = 64;
    layout.block_stride_bytes = 64;
    layout.k_block_stride_bytes = 32;
    layout.v_block_stride_bytes = 32;
    layout.local_head_num_kv = 1;
    layout.seq_size_per_block = 4;
    config.memory_layouts = {layout};
    for (int backing = 0; backing < 3; ++backing) {
        config.use_pinned_cpu_backing = backing == 0;
        config.use_device_malloc_backing = backing == 2;
        DeviceBlockPool pool(std::make_shared<DeviceBlockPoolConfig>(config));
        ASSERT_TRUE(pool.init());
        for (auto& tensor : pool.allLayerCacheBase()) {
            EXPECT_EQ(torch::count_nonzero(tensor).item<int64_t>(), 0);
            tensor.fill_(1);
        }
        auto blocks = pool.malloc(1);
        ASSERT_TRUE(blocks.has_value());
        ASSERT_EQ(blocks->size(), 1);
        pool.incRef(*blocks);
        pool.decRef(*blocks);
        auto reused = pool.malloc(1);
        ASSERT_TRUE(reused.has_value());
        ASSERT_EQ(reused->size(), 1);
        for (auto& tensor : pool.allLayerCacheBase()) {
            EXPECT_TRUE(torch::all(tensor == 1).item<bool>());
        }
        pool.incRef(*reused);
        pool.decRef(*reused);
    }
#else
    GTEST_SKIP() << "ROCm-specific pool initialization policy";
#endif
}

}  // namespace rtp_llm
