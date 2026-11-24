#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace rtp_llm {

TEST(PpuKVPoolZeroTest, RejectsEmptyPoolAtThePublicBoundary) {
    MemoryLayoutConfig cfg;
    cfg.dtype                  = TYPE_FP16;
    auto                 empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    torch::Tensor        scale;
    MemoryLayoutStrategy strategy;
    EXPECT_THROW(strategy.init(cfg, empty, scale, empty.data_ptr()), RTPException);
}

TEST(PpuKVPoolZeroTest, ClearsOnlyTheWarmupViewOnTheCurrentStream) {
    const auto                       stream = c10::cuda::getStreamFromPool();
    const c10::cuda::CUDAStreamGuard guard(stream);
    constexpr int64_t                bytes = 196608, margin = 256;
    auto backing = torch::empty({bytes + 2 * margin}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto slice   = backing.narrow(0, margin, bytes);
    torch::Tensor      scale;
    MemoryLayoutConfig cfg;
    cfg.layer_num                = 24;
    cfg.block_num                = 2;
    cfg.dtype                    = TYPE_FP16;
    cfg.kv_block_stride_bytes    = bytes / (cfg.layer_num * cfg.block_num);
    cfg.kv_block_pool_size_bytes = bytes;
    for (bool mla : {false, true}) {
        cfg.use_mla            = mla;
        cfg.seq_size_per_block = 8;
        C10_CUDA_CHECK(cudaMemsetAsync(backing.data_ptr(), 0xa5, backing.nbytes(), stream.stream()));
        MemoryLayoutStrategy strategy;
        ASSERT_TRUE(strategy.init(cfg, slice, scale, backing.data_ptr()));
        const auto layers = strategy.getLayerCacheTensors();
        ASSERT_EQ(layers.size(), cfg.layer_num);
        ASSERT_EQ(layers.front().data_ptr(), slice.data_ptr());
        stream.synchronize();
        const auto  host = backing.cpu();
        const auto* p    = host.data_ptr<uint8_t>();
        EXPECT_TRUE(std::all_of(p, p + margin, [](uint8_t v) { return v == 0xa5; }));
        EXPECT_TRUE(std::all_of(p + margin, p + margin + bytes, [](uint8_t v) { return v == 0; }));
        EXPECT_TRUE(std::all_of(p + margin + bytes, p + host.numel(), [](uint8_t v) { return v == 0xa5; }));
    }
}

}  // namespace rtp_llm
