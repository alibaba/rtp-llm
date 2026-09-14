#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

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

TEST(LayerKVCacheViewTest, DenseGroupKernelIdsAddressTheirOwnPhysicalMlaBlock) {
    using namespace rtp_llm;
    for (const int physical_tokens : {128, 256, 512, 1024, 8192}) {
        for (const auto type : {CacheGroupType::FULL, CacheGroupType::SWA}) {
            SCOPED_TRACE(::testing::Message() << "physical_tokens=" << physical_tokens << " group=" << int(type));
            KVCacheResource resource;
            resource.initGroups(1, 1, {0}, physical_tokens / 128, {type});
            resource.mutableBlockIds().add({1});

            KVCache cache;
            cache.use_mla                   = true;
            cache.kv_lora_rank              = 4;
            cache.rope_head_dim             = 2;
            cache.seq_size_per_block        = physical_tokens;
            cache.kernel_seq_size_per_block = 128;
            cache.layer_group_types         = {type};
            auto physical = torch::arange(3 * physical_tokens * 6, torch::kInt64).reshape({3, physical_tokens, 6});
            cache.kv_cache_base_by_layer = {physical};
            const auto  layer            = cache.getLayerCache(0);
            const auto& ids              = resource.kernelBlocks();
            ASSERT_EQ(ids.size(), physical_tokens / 128);
            for (int pos = 0; pos < physical_tokens; ++pos) {
                EXPECT_EQ(layer.kv_cache_base[ids[pos / 128]][pos % 128][0].item<int64_t>(),
                          physical[1][pos][0].item<int64_t>());
            }
            EXPECT_EQ(resource.blocks(), (BlockIndicesType{1}));
            EXPECT_EQ(cache.layer_group_types[0], type);
        }
    }
}

TEST(LayerKVCacheViewTest, DenseSwaKernelIdsStayAlignedAfterBlockMutations) {
    using namespace rtp_llm;
    KVCacheResource resource;
    resource.initGroups(2, 2, {0, 1}, 2, {CacheGroupType::SWA, CacheGroupType::LINEAR});
    auto& ids = resource.mutableBlockIds(0);
    ids.assign({1, NULL_BLOCK_IDX, 3});
    EXPECT_EQ(ids.kernelBlocks(), (BlockIndicesType{2, 3, -1, -1, 6, 7}));
    ids.swap(0, 2);
    ids.setAt(1, 2);
    ids.remove({2});
    ids.resize(4, 4);
    EXPECT_EQ(ids.kernelBlocks(), (BlockIndicesType{6, 7, 4, 5, -1, -1, 8, 9}));
    EXPECT_EQ(ids.popBack(), 4);
    EXPECT_EQ(ids.blocks(), (BlockIndicesType{3, 2, -1}));
    EXPECT_EQ(ids.kernelBlocks(), (BlockIndicesType{6, 7, 4, 5, -1, -1}));
    resource.mutableBlockIds(1).add({1, NULL_BLOCK_IDX});
    EXPECT_EQ(resource.kernelBlocks(1), (BlockIndicesType{1, -1}));
}

TEST(LayerKVCacheViewTest, TypedFixedRegionsKeepPhysicalIdsAlongsideDenseDraftSwa) {
    using namespace rtp_llm;
    constexpr int                 region_count = static_cast<int>(KVCacheRegionName::REGION_COUNT);
    std::vector<std::vector<int>> regions(2, std::vector<int>(region_count, -1));
    regions[0][static_cast<int>(KVCacheRegionName::CSA_KV)]    = 0;
    regions[0][static_cast<int>(KVCacheRegionName::SWA_KV)]    = 1;
    regions[0][static_cast<int>(KVCacheRegionName::CSA_STATE)] = 2;
    regions[1][static_cast<int>(KVCacheRegionName::DEFAULT)]   = 3;
    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(4,
                     2,
                     {1, 3},
                     4,
                     {CacheGroupType::FULL, CacheGroupType::SWA, CacheGroupType::SWA, CacheGroupType::SWA},
                     regions);
    for (int b = 0; b < 2; ++b) {
        for (int g = 0; g < 4; ++g) {
            batch.mutableBlockIds(b, g).add({1});
            EXPECT_EQ(batch.blocks(b, g), (BlockIndicesType{1}));
        }
        EXPECT_EQ(batch.kernelBlocks(b, 0), (BlockIndicesType{4, 5, 6, 7}));
        EXPECT_EQ(batch.kernelBlocks(b, 1), (BlockIndicesType{1}));
        EXPECT_EQ(batch.kernelBlocks(b, 2), (BlockIndicesType{1}));
        EXPECT_EQ(batch.kernelBlocks(b, 3), (BlockIndicesType{4, 5, 6, 7}));
    }
}

TEST(LayerKVCacheViewTest, UnusedTypedFixedGroupKeepsPhysicalIds) {
    using namespace rtp_llm;
    // DSV4 emits fixed STATE pools even when no layer has that region.
    std::vector<std::vector<int>> regions(1, std::vector<int>(8, -1));
    regions[0][static_cast<int>(KVCacheRegionName::SWA_KV)] = 0;
    KVCacheResource resource;
    resource.initGroups(2, 1, {0}, 4, {CacheGroupType::SWA, CacheGroupType::SWA}, regions);
    resource.mutableBlockIds(1).add({1});
    EXPECT_EQ(resource.kernelBlocks(1), (BlockIndicesType{1}));
}

}  // namespace
}  // namespace torch_ext
