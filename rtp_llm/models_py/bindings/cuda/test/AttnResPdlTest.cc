#include <cstdlib>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/attn_res.h"

TEST(AttnResPdlTest, DependentLaunchesMatchSerializedExecution) {
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
    ASSERT_EQ(properties.major, 10);
    rtp_llm::StaticConfig::user_disable_pdl = std::getenv("ATTN_RES_DISABLE_PDL") != nullptr;
    const auto                 stream       = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    torch::manual_seed(719);
    const auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    const auto source  = torch::randn({2048, 7168}, options);
    const auto anchors = torch::randn({2048, 3, 7168}, options);
    const auto delta   = torch::randn_like(source) * 0.1;
    const auto norm    = torch::ones({7168}, options);
    const auto score   = torch::zeros({7168}, options);

    for (bool has_delta : {false, true}) {
        auto run = [&](bool serialized) {
            auto prefix = source.clone();
            auto blocks = anchors.clone();
            for (int step = 0; step < 8; ++step) {
                auto output = torch::empty_like(prefix);
                attn_res(prefix,
                         has_delta ? std::make_optional(delta) : std::nullopt,
                         blocks,
                         norm,
                         score,
                         norm,
                         output,
                         3,
                         -1,
                         1e-6,
                         1e-6);
                if (serialized) {
                    EXPECT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
                }
                // The successor reads this output as its prefix, including
                // when delta is present. PDL must order both input operands.
                prefix = std::move(output);
            }
            return prefix;
        };
        const auto expected = run(true);
        const auto actual   = run(false);
        ASSERT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
        EXPECT_TRUE(torch::equal(actual, expected)) << "has_delta=" << has_delta;
    }
}
