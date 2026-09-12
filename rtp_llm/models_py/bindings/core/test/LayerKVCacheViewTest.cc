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

TEST(LayerKVCacheViewTest, DefaultLinearReplayViewRetainsPhysicalGroupSpanAndSharedTensors) {
    KVCache cache;
    cache.seq_size_per_block        = 256;
    cache.kernel_seq_size_per_block = 128;
    cache.layer_group_types         = {rtp_llm::CacheGroupType::LINEAR, rtp_llm::CacheGroupType::FULL};
    cache.group_seq_size_per_block  = {256, 512, 1024, 2048};
    cache.kv_cache_base_by_layer    = {torch::empty({2, 64}), torch::empty({1})};

    constexpr size_t region_count  = static_cast<size_t>(rtp_llm::KVCacheRegionName::REGION_COUNT);
    constexpr auto default_region = rtp_llm::KVCacheRegionName::DEFAULT;
    cache.layer_region_to_group_id = std::vector<std::vector<int>>(2, std::vector<int>(region_count, -1));
    cache.layer_region_to_group_id[0][static_cast<size_t>(default_region)] = 3;
    cache.layer_region_to_group_id[1][static_cast<size_t>(default_region)] = 0;

    LinearReplayLayerCache replay;
    replay.k                = torch::empty({2, 4, 1, 2});
    replay.u                = torch::empty_like(replay.k);
    replay.g                = torch::empty_like(replay.k);
    replay.conv_inputs      = torch::empty({2, 4, 6}, torch::kBFloat16);
    replay.slot_generations = torch::tensor({int64_t{1} << 40, int64_t{2} << 40}, torch::kInt64);
    replay.log_epochs       = torch::tensor({int64_t{3} << 40, int64_t{4} << 40}, torch::kInt64);
    replay.valid_counts     = torch::tensor({4, 3}, torch::kInt32);
    replay.error_flags      = torch::zeros({2}, torch::kInt32);
    cache.linear_replay_by_layer.resize(2);
    cache.linear_replay_by_layer[0] = replay;

    for (const auto& linear : {cache.getLayerCache(0), cache.getLayerCache(0, default_region)}) {
        EXPECT_EQ(linear.layer_id, 0);
        EXPECT_EQ(linear.group_id, 3);
        EXPECT_EQ(linear.seq_size_per_block, 2048);
        EXPECT_EQ(linear.kv_cache_base.data_ptr(), cache.kv_cache_base_by_layer[0].data_ptr());
        ASSERT_TRUE(linear.linear_replay.has_value());
        const auto& view = *linear.linear_replay;
        EXPECT_EQ(view.k.data_ptr(), replay.k.data_ptr());
        EXPECT_EQ(view.u.data_ptr(), replay.u.data_ptr());
        EXPECT_EQ(view.g.data_ptr(), replay.g.data_ptr());
        EXPECT_EQ(view.conv_inputs.data_ptr(), replay.conv_inputs.data_ptr());
        EXPECT_EQ(view.slot_generations.data_ptr(), replay.slot_generations.data_ptr());
        EXPECT_EQ(view.log_epochs.data_ptr(), replay.log_epochs.data_ptr());
        EXPECT_EQ(view.valid_counts.data_ptr(), replay.valid_counts.data_ptr());
        EXPECT_EQ(view.error_flags.data_ptr(), replay.error_flags.data_ptr());
    }
    EXPECT_FALSE(cache.getLayerCache(1).linear_replay.has_value());
}

TEST(LinearReplayInputsTest, PaddingPreservesLogicalOwnersAndMasksEveryDummyRow) {
    auto backing = rtp_llm::LinearReplayInputs::allocate(7, 3, torch::kCPU);
    int64_t value = 11;
    for (auto* tensor : backing.tensors()) {
        tensor->fill_(tensor->scalar_type() == torch::kInt64 ? (int64_t{1} << 40) + value : value);
        ++value;
    }
    // Narrowed group tables have a non-contiguous outer stride.
    auto source = backing.slice(1, 3);
    ASSERT_FALSE(source.active_block_ids.is_contiguous());
    auto originals = source.tensors();
    for (const int64_t physical : {3, 4, 8}) {
        auto padded = source.padToBatch(physical);
        auto fields = padded.tensors();
        EXPECT_EQ(physical * 4 / padded.slot_ids.numel(), 4);
        for (size_t i = 0; i < fields.size(); ++i) {
            const auto& original = *originals[i];
            const auto& actual = *fields[i];
            const auto dim = original.dim() - 1;
            EXPECT_EQ(actual.size(dim), physical);
            EXPECT_EQ(actual.scalar_type(), original.scalar_type());
            EXPECT_EQ(actual.device(), original.device());
            EXPECT_TRUE(torch::equal(actual.narrow(dim, 0, 3), original));
            if (physical == 3) {
                EXPECT_EQ(actual.data_ptr(), original.data_ptr());
            } else {
                const int64_t fill = i == 0 || i == 2 || i == 8 ? -1 : 0;
                const auto tail = actual.narrow(dim, 3, physical - 3);
                EXPECT_TRUE(torch::equal(tail, torch::full_like(tail, fill)));
                // Neither padding nor a later graph wave may mutate logical metadata.
                EXPECT_NE(actual.data_ptr(), original.data_ptr());
            }
            EXPECT_EQ(original.size(dim), 3);
            EXPECT_EQ(original.data_ptr(), source.tensors()[i]->data_ptr());
        }
    }
    EXPECT_THROW(source.padToBatch(2), c10::Error);
    auto malformed = source;
    malformed.history_epochs = torch::zeros({2}, torch::kInt64);
    EXPECT_THROW(malformed.padToBatch(4), c10::Error);
}

}  // namespace
}  // namespace torch_ext
