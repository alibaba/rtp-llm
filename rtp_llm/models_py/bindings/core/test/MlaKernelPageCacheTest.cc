#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/mla_quant_kernel.h"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>

namespace torch_ext {
namespace {

// FULL covers the replicated MTP destination; SWA covers Eagle3 on both sides
// of PD. Use real resource-generated IDs, not a hand-expanded kernel table.
class MlaKernelPageCacheTest: public ::testing::TestWithParam<std::tuple<int, rtp_llm::CacheGroupType, bool, bool>> {};

TEST_P(MlaKernelPageCacheTest, WritesOnlyAllocatedPhysicalBlocks) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }
    const auto [physical_tokens, group_type, fp8, use_graph] = GetParam();
    constexpr int                    kernel_tokens           = 128;
    constexpr int                    width                   = 576;
    constexpr int                    physical_blocks         = 5;
    constexpr int                    sentinel                = -96;
    const int                        ratio                   = physical_tokens / kernel_tokens;
    const int                        tokens                  = physical_tokens + 1;
    const auto                       stream                  = c10::cuda::getStreamFromPool();
    const c10::cuda::CUDAStreamGuard guard(stream);
    const auto                       device_options = torch::TensorOptions().device(torch::kCUDA);
    const auto                       cache_dtype    = fp8 ? at::kFloat8_e4m3fn : torch::kBFloat16;
    auto                             physical =
        torch::full({physical_blocks, physical_tokens, width}, sentinel, device_options.dtype(torch::kBFloat16))
            .to(cache_dtype);
    KVCache cache;
    cache.use_mla                   = true;
    cache.kv_lora_rank              = 512;
    cache.rope_head_dim             = 64;
    cache.seq_size_per_block        = physical_tokens;
    cache.kernel_seq_size_per_block = kernel_tokens;
    cache.layer_group_types         = {group_type};
    cache.kv_cache_base_by_layer    = {physical};
    auto layer                      = cache.getLayerCache(0);

    rtp_llm::KVCacheResource resource;
    resource.initGroups(1, 1, {0}, ratio, {group_type});
    // Match the gatherer's zero-padded row, including before the SWA fix.
    auto table     = torch::zeros({2 * ratio}, device_options.dtype(torch::kInt32));
    auto positions = torch::arange(tokens, device_options.dtype(torch::kInt64));
    auto values    = (torch::arange(tokens * width, device_options.dtype(torch::kInt64)) % 13 - 6)
                      .reshape({tokens, width})
                      .to(torch::kBFloat16);
    auto       latent = values.slice(1, 0, 512).contiguous();
    auto       rope   = values.slice(1, 512, width).contiguous();
    auto       scale  = torch::ones({}, device_options.dtype(torch::kFloat32));
    const auto write  = [&]() {
        auto columns = torch::floor_divide(positions, kernel_tokens);
        auto slots =
            table.index_select(0, columns).to(torch::kInt64) * kernel_tokens + positions.remainder(kernel_tokens);
        rtp_llm::concat_and_cache_mla(latent, rope, layer.kv_cache_base, slots, fp8 ? "fp8" : "auto", scale, false);
    };
    at::cuda::CUDAGraph graph;
    for (int iteration = 0; iteration < 2; ++iteration) {
        const int first  = iteration + 1;
        const int second = iteration + 3;
        resource.mutableBlockIds().assign({first, second});
        const auto& ids      = resource.kernelBlocks();
        auto        host_ids = torch::tensor(ids, torch::kInt32);
        table.zero_();
        table.slice(0, 0, host_ids.numel()).copy_(host_ids);
        physical.fill_(sentinel);
        if (use_graph) {
            if (iteration == 0) {
                write();  // Warm up allocation and CUDA dispatch before capture.
                physical.fill_(sentinel);
                stream.synchronize();
                graph.capture_begin();
                write();
                graph.capture_end();
            }
            graph.replay();  // The second replay must consume the new block IDs.
        } else {
            write();
        }
        stream.synchronize();
        auto expected    = torch::full({physical_blocks, physical_tokens, width}, sentinel, torch::kFloat32);
        auto host_values = values.to(torch::kCPU).to(torch::kFloat32);
        expected[first].copy_(host_values.slice(0, 0, physical_tokens));
        expected[second][0].copy_(host_values[physical_tokens]);
        EXPECT_TRUE(torch::equal(physical.to(torch::kCPU).to(torch::kFloat32), expected)) << "iteration=" << iteration;
        EXPECT_EQ(resource.blocks(), (rtp_llm::BlockIndicesType{first, second}));
        EXPECT_EQ(cache.layer_group_types[0], group_type);
    }
}

INSTANTIATE_TEST_SUITE_P(DensePhysicalPages,
                         MlaKernelPageCacheTest,
                         ::testing::Combine(::testing::Values(128, 256, 512, 1024, 8192),
                                            ::testing::Values(rtp_llm::CacheGroupType::FULL,
                                                              rtp_llm::CacheGroupType::SWA),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
}  // namespace torch_ext
