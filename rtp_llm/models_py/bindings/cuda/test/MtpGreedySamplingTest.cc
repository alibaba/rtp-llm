#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <limits>

#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_greedy_sampling.h"

namespace rtp_llm {
namespace {

void checkGreedy(const torch::Tensor& logits, const torch::Tensor& map = {}) {
    auto expected_ids   = torch::argmax(logits, -1, true);
    auto expected_probs = torch::zeros_like(logits).scatter_(-1, expected_ids, 1);
    if (map.defined() && map.numel() > 0) {
        auto valid  = expected_ids < map.numel();
        auto mapped = map.index_select(0, expected_ids.clamp_max(map.numel() - 1).flatten()).reshape_as(expected_ids);
        expected_ids = torch::where(valid, mapped, expected_ids);
    }
    torch::Tensor ids, probabilities;
    invokeMtpGreedySampling(logits, map, ids, probabilities, at::cuda::getCurrentCUDAStream().stream());
    EXPECT_EQ(ids.scalar_type(), torch::kInt32);
    EXPECT_EQ(probabilities.scalar_type(), logits.scalar_type());
    EXPECT_TRUE(torch::equal(ids, expected_ids.to(torch::kInt32)));
    EXPECT_TRUE(torch::equal(probabilities, expected_probs));
}

TEST(MtpGreedySamplingTest, PreservesTypesAndSupportsStridedLogits) {
    c10::InferenceMode inference;
    torch::manual_seed(587);
    for (auto dtype : {torch::kFloat32, torch::kFloat64, torch::kFloat16, torch::kBFloat16}) {
        for (const auto& shape : std::vector<std::vector<int64_t>>{{1, 1}, {7, 127}, {31, 8191}, {128, 131072}}) {
            const auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
            auto logits = torch::randn(shape, options);
            checkGreedy(logits);
            checkGreedy(logits.t());
            auto padded = torch::randn({shape[0], shape[1] + 3}, options);
            padded.slice(1, shape[1]).fill_(std::numeric_limits<float>::quiet_NaN());
            checkGreedy(padded.narrow(1, 0, shape[1]));
        }
    }
    checkGreedy(torch::empty({0, 127}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)));
}

TEST(MtpGreedySamplingTest, TiesNaNsInfinitiesSignedZerosAndSubnormals) {
    c10::InferenceMode inference;
    auto dense = torch::full({7, 127}, -1.0f, torch::kFloat32);
    auto values = dense.accessor<float, 2>();
    values[0][5] = values[0][100] = 5.0f;
    values[1][4] = values[1][44] = std::numeric_limits<float>::quiet_NaN();
    values[2][22] = values[2][99] = std::numeric_limits<float>::infinity();
    dense[3].fill_(-std::numeric_limits<float>::infinity());
    values[4][6] = -0.0f;
    values[4][48] = 0.0f;
    values[5][7] = std::numeric_limits<float>::denorm_min();
    values[5][63] = 2.0f * std::numeric_limits<float>::denorm_min();
    values[6][126] = 7.0f;
    checkGreedy(dense.cuda());
}

TEST(MtpGreedySamplingTest, MappingChangesOnlyTokenIdsAndHonorsTheCurrentStream) {
    c10::InferenceMode inference;
    const auto stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto logits = torch::randn({7, 127}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    for (int64_t map_size : {31, 127}) {
        auto map = torch::arange(map_size * 2, logits.options().dtype(torch::kInt64)).slice(0, 0, map_size * 2, 2);
        checkGreedy(logits, map + 200);
        checkGreedy(logits, map);
    }
}

TEST(MtpGreedySamplingTest, PreservesMappedInt32TokenValues) {
    c10::InferenceMode inference;
    auto logits = torch::tensor({{2.0f, 1.0f}, {1.0f, 2.0f}}, torch::kFloat32).cuda();
    auto map =
        torch::tensor({static_cast<int64_t>(std::numeric_limits<int32_t>::max()), int64_t{163839}}, torch::kInt64)
            .cuda();
    checkGreedy(logits, map);
}

}  // namespace
}  // namespace rtp_llm
