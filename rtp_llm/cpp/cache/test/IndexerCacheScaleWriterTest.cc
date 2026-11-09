#include "rtp_llm/cpp/cache/IndexerCacheScale.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/mla_quant_kernel.h"
#include <gtest/gtest.h>
#include <cstring>

namespace rtp_llm {
TEST(IndexerCacheScaleWriterTest, LegacyUnroundedWriterRequantizesOnlyOnReceive) {
    auto device = at::Device(at::kCUDA, 0);
    auto key    = at::full({1, 128}, 3.0, at::TensorOptions().dtype(at::kBFloat16).device(device));
    auto slots  = at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(device));
    auto cache  = at::zeros({1, 128, 132}, at::TensorOptions().dtype(at::kByte).device(device));
    // Actual MlaAttention -> Indexer default: scale_fmt="none". Execute its real writer.
    indexer_k_quant_and_cache(key, cache, slots, 128, "none");
    auto  before = cache.cpu();
    float scale;
    std::memcpy(&scale, before.data_ptr<uint8_t>() + 128 * 128, 4);
    EXPECT_FLOAT_EQ(scale, 3.0f / 448);
    EXPECT_EQ(before.data_ptr<uint8_t>()[0], 0x7e);  // 448
    BlockInfo target;
    target.addr           = cache.data_ptr();
    target.size_bytes     = cache.nbytes();
    target.is_cuda        = true;
    target.indexer_layout = {IndexerScaleFormat::FP32, 128, 128 * 132};
    convertIndexerCacheScales({{target, target.indexer_layout}});
    EXPECT_TRUE(cache.cpu().equal(before));
    target.indexer_layout.format = IndexerScaleFormat::UE8M0;
    convertIndexerCacheScales({{target, {IndexerScaleFormat::FP32, 128, 128 * 132}}});
    auto converted = cache.cpu();
    EXPECT_EQ(converted.data_ptr<uint8_t>()[0], 0x7c);  // 384 * (1/128) == 3
    uint32_t packed;
    std::memcpy(&packed, converted.data_ptr<uint8_t>() + 128 * 128, 4);
    EXPECT_EQ(packed, 0x78787878u);
    EXPECT_TRUE(key.equal(at::full_like(key, 3.0)));
}
}  // namespace rtp_llm
