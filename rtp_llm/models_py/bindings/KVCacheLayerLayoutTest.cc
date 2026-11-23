#include "rtp_llm/models_py/bindings/OpDefs.h"

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"

namespace torch_ext {
namespace {

TEST(KVCacheLayerLayoutTest, DecodeHostBlockCountDoesNotDependOnDeviceScratch) {
    constexpr int64_t physical_tokens       = 512;
    constexpr int64_t kernel_tokens         = 64;
    constexpr int64_t logical_device_blocks = 5;
    constexpr int64_t device_scratch_blocks = 2;
    constexpr int64_t host_blocks           = 12;
    constexpr int64_t elements_per_token    = 656;

    auto spec     = std::make_shared<rtp_llm::MLAKVCacheSpec>("default", physical_tokens, kernel_tokens);
    auto policy   = rtp_llm::defaultCacheGroupPolicy(rtp_llm::CacheGroupType::FULL);
    auto topology = rtp_llm::CacheTopology::create(
        {{"default", std::move(spec), policy, logical_device_blocks + host_blocks}}, {{0, {"default"}}});

    // The view reshaping is independent of CUDA allocation. Use separate tensors
    // so the host suffix cannot accidentally be inferred from the device extent.
    const auto                  bytes = torch::TensorOptions().dtype(torch::kUInt8);
    rtp_llm::BlockBufferPtrInfo buffers;
    buffers.kv_addr =
        torch::empty({logical_device_blocks + device_scratch_blocks, physical_tokens * elements_per_token}, bytes);
    buffers.kv_host_addr          = torch::empty({host_blocks, physical_tokens * elements_per_token}, bytes);
    buffers.logical_device_blocks = logical_device_blocks;
    rtp_llm::GroupedCacheLayerLayout::GroupLayouts views;
    views.emplace("default", rtp_llm::CacheLayerLayout({std::move(buffers)}));
    const KVCache cache(rtp_llm::GroupedCacheLayerLayout(std::move(topology), std::move(views)));

    const auto layer = cache.getLayerCache(0, "default");
    EXPECT_EQ(layer.logical_device_blocks, logical_device_blocks);
    EXPECT_EQ(layer.seq_size_per_block, kernel_tokens);
    EXPECT_EQ(layer.kv_cache_base.size(0),
              (logical_device_blocks + device_scratch_blocks) * physical_tokens / kernel_tokens);
    EXPECT_EQ(layer.kv_host_base.size(0), host_blocks * physical_tokens / kernel_tokens);
    EXPECT_EQ(layer.kv_host_base.size(2), elements_per_token);
}

}  // namespace
}  // namespace torch_ext
