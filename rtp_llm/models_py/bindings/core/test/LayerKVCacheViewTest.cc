#include "rtp_llm/models_py/bindings/OpDefs.h"

#include <gtest/gtest.h>

namespace torch_ext {
namespace {

TEST(LayerKVCacheViewTest, DefaultLinearLayerUsesItsGroupTokenSpan) {
    KVCache cache;
    cache.seq_size_per_block        = 256;
    cache.kernel_seq_size_per_block = 128;
    cache.layer_group_types         = {rtp_llm::CacheGroupType::FULL, rtp_llm::CacheGroupType::LINEAR};
    cache.group_seq_size_per_block  = {256, 1024};
    cache.kv_cache_base_by_layer    = {torch::empty({1}), torch::empty({1})};

    constexpr size_t region_count  = static_cast<size_t>(rtp_llm::KVCacheRegionName::REGION_COUNT);
    cache.layer_region_to_group_id = std::vector<std::vector<int>>(2, std::vector<int>(region_count, -1));
    cache.layer_region_to_group_id[0][static_cast<size_t>(rtp_llm::KVCacheRegionName::DEFAULT)] = 0;
    cache.layer_region_to_group_id[1][static_cast<size_t>(rtp_llm::KVCacheRegionName::DEFAULT)] = 1;

    const auto full   = cache.getLayerCache(0);
    const auto linear = cache.getLayerCache(1);

    EXPECT_EQ(full.group_id, 0);
    EXPECT_EQ(full.seq_size_per_block, 128);
    EXPECT_EQ(linear.group_id, 1);
    EXPECT_EQ(linear.seq_size_per_block, 1024);
}

TEST(LayerKVCacheViewTest, SlidingWindowMlaUsesKernelPagesWithoutChangingItsStoragePolicy) {
    KVCache cache;
    cache.use_mla                   = true;
    cache.kv_lora_rank              = 4;
    cache.rope_head_dim             = 2;
    cache.seq_size_per_block        = 256;
    cache.kernel_seq_size_per_block = 128;
    cache.layer_group_types         = {rtp_llm::CacheGroupType::SWA};
    cache.kv_cache_base_by_layer    = {torch::arange(2 * 256 * 6, torch::kFloat32).reshape({2, 256, 6})};

    const auto layer = cache.getLayerCache(0);
    EXPECT_EQ(layer.seq_size_per_block, 128);
    ASSERT_EQ(layer.kv_cache_base.sizes(), torch::IntArrayRef({4, 128, 6}));
    EXPECT_EQ(layer.kv_cache_base.data_ptr(), cache.kv_cache_base_by_layer[0].data_ptr());
    EXPECT_EQ(layer.kv_cache_base[2][0][0].item<float>(), 256 * 6);
    EXPECT_EQ(cache.layer_group_types[0], rtp_llm::CacheGroupType::SWA);
}

}  // namespace
}  // namespace torch_ext
