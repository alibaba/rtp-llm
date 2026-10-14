#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <limits>

#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_greedy_sampling.h"

namespace rtp_llm {
namespace {

struct Result {
    torch::Tensor tokens;
    torch::Tensor probabilities;
    torch::Tensor sequences;
    torch::Tensor positions;
};

Result reference(const torch::Tensor& rank_major, int64_t vocab, const torch::Tensor& map = {}) {
    auto dense = rank_major.permute({1, 0, 2})
                     .reshape({rank_major.size(1), rank_major.size(0) * rank_major.size(2)})
                     .narrow(1, 0, vocab)
                     .contiguous();
    auto ids   = torch::argmax(dense, -1, true);
    auto probs = torch::zeros_like(dense).scatter_(-1, ids, 1.0);
    if (map.defined() && map.numel() > 0) {
        auto valid = ids < map.numel();
        ids =
            torch::where(valid, map.index_select(0, ids.clamp_max(map.numel() - 1).reshape({-1})).reshape_as(ids), ids);
    }
    return {ids.to(torch::kInt32), probs, {}, {}};
}

Result candidate(const torch::Tensor& rank_major,
                 int64_t              vocab,
                 const torch::Tensor& map       = {},
                 const torch::Tensor& sequences = {},
                 const torch::Tensor& positions = {}) {
    Result output;
    TORCH_CHECK(tryInvokeMtpGreedyRankMajor(rank_major,
                                            map,
                                            vocab,
                                            sequences,
                                            positions,
                                            output.tokens,
                                            output.probabilities,
                                            output.sequences,
                                            output.positions,
                                            at::cuda::getCurrentCUDAStream()),
                "unexpected rank-major greedy fallback");
    return output;
}

TEST(MtpGreedyRankMajorTest, ModelAndPrimeShapesPreserveExactTokensAndDenseProbabilities) {
    c10::InferenceMode inference;
    torch::manual_seed(587);
    for (const auto& shape : std::vector<std::vector<int64_t>>{
             {1, 1, 31}, {7, 3, 127}, {31, 7, 8191}, {111, 16, 163840}, {112, 16, 163840}, {113, 16, 163840}}) {
        const int64_t batch = shape[0], tp = shape[1], vocab = shape[2];
        const int64_t local_vocab = (vocab + tp - 1) / tp;
        auto          logits =
            torch::randn({tp, batch, local_vocab}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        // Scores beyond the actual vocabulary must never win, including NaNs.
        if (vocab % tp != 0) {
            logits.select(0, tp - 1)
                .narrow(1, vocab - (tp - 1) * local_vocab, tp * local_vocab - vocab)
                .fill_(std::numeric_limits<float>::quiet_NaN());
        }
        auto expected = reference(logits, vocab);
        auto actual   = candidate(logits, vocab);
        EXPECT_EQ(actual.tokens.scalar_type(), torch::kInt32);
        EXPECT_TRUE(torch::equal(actual.tokens, expected.tokens)) << "B=" << batch << " TP=" << tp;
        EXPECT_TRUE(torch::equal(actual.probabilities, expected.probabilities));
    }
}

TEST(MtpGreedyRankMajorTest, TiesNaNsInfinitiesSignedZerosAndSubnormals) {
    c10::InferenceMode inference;
    constexpr int64_t  batch = 7, tp = 3, local_vocab = 43, vocab = 127;
    auto               dense = torch::full({batch, tp * local_vocab}, -1.0f, torch::kFloat32);
    auto               a     = dense.accessor<float, 2>();
    a[0][5]                  = 5.0f;
    a[0][100]                = 5.0f;
    a[1][4]                  = std::numeric_limits<float>::quiet_NaN();
    a[1][44]                 = std::numeric_limits<float>::quiet_NaN();
    a[2][22]                 = std::numeric_limits<float>::infinity();
    a[2][99]                 = std::numeric_limits<float>::infinity();
    dense[3].fill_(-std::numeric_limits<float>::infinity());
    a[4][6]   = -0.0f;
    a[4][48]  = 0.0f;
    a[5][7]   = std::numeric_limits<float>::denorm_min();
    a[5][63]  = 2.0f * std::numeric_limits<float>::denorm_min();
    a[6][126] = 7.0f;
    // Both padded tail entries are NaNs and must be ignored.
    dense.narrow(1, vocab, tp * local_vocab - vocab).fill_(std::numeric_limits<float>::quiet_NaN());
    auto logits   = dense.reshape({batch, tp, local_vocab}).permute({1, 0, 2}).contiguous().cuda();
    auto expected = reference(logits, vocab);
    auto actual   = candidate(logits, vocab);
    EXPECT_TRUE(torch::equal(actual.tokens, expected.tokens));
    EXPECT_TRUE(
        torch::equal(actual.tokens.cpu(), torch::tensor({5, 4, 22, 0, 6, 63, 126}, torch::kInt32).reshape({batch, 1})));
    EXPECT_TRUE(torch::equal(actual.probabilities, expected.probabilities));
}

TEST(MtpGreedyRankMajorTest, MappingChangesTokensButNotDraftProbabilityCoordinates) {
    c10::InferenceMode inference;
    constexpr int64_t  tp = 3, batch = 5, vocab = 127;
    auto logits = torch::randn({tp, batch, 43}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    for (int64_t map_size : {31, 127}) {
        auto map      = torch::arange(map_size - 1, -1, -1, logits.options().dtype(torch::kInt64)) + 200;
        auto expected = reference(logits, vocab, map);
        auto actual   = candidate(logits, vocab, map);
        EXPECT_TRUE(torch::equal(actual.tokens, expected.tokens));
        EXPECT_TRUE(torch::equal(actual.probabilities, expected.probabilities));
    }
}

TEST(MtpGreedyRankMajorTest, UpdatesUseCurrentStreamWithoutMutatingSourceMetadata) {
    c10::InferenceMode         inference;
    const auto                 stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto logits    = torch::randn({3, 7, 43}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto sequences = torch::arange(7, logits.options().dtype(torch::kInt32)) + 101;
    auto positions = torch::arange(21, sequences.options()).reshape({7, 3}) + 201;
    auto first     = candidate(logits, 127, {}, sequences, positions);
    sequences.add_(100);
    positions.add_(100);
    auto second = candidate(logits, 127, {}, sequences, positions);
    EXPECT_TRUE(torch::equal(first.sequences, sequences - 99));
    EXPECT_TRUE(torch::equal(first.positions, positions - 99));
    EXPECT_TRUE(torch::equal(second.sequences, sequences + 1));
    EXPECT_TRUE(torch::equal(second.positions, positions + 1));
    EXPECT_EQ(first.positions.sizes(), positions.sizes());
    EXPECT_NE(first.positions.data_ptr(), positions.data_ptr());
    EXPECT_NE(first.sequences.data_ptr(), sequences.data_ptr());
    EXPECT_TRUE(torch::equal(first.tokens, reference(logits, 127).tokens));
}

TEST(MtpGreedyRankMajorTest, UnsupportedInputsLeaveAllOutputsUnchanged) {
    c10::InferenceMode inference;
    auto logits   = torch::randn({3, 7, 43}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto sentinel = torch::full({1}, 17, logits.options());
    auto check    = [&](const torch::Tensor& input,
                     const torch::Tensor& map,
                     const torch::Tensor& sequence,
                     const torch::Tensor& position) {
        Result output{sentinel, sentinel, sentinel, sentinel};
        EXPECT_FALSE(tryInvokeMtpGreedyRankMajor(input,
                                                 map,
                                                 127,
                                                 sequence,
                                                 position,
                                                 output.tokens,
                                                 output.probabilities,
                                                 output.sequences,
                                                 output.positions,
                                                 at::cuda::getCurrentCUDAStream()));
        EXPECT_EQ(output.tokens.data_ptr(), sentinel.data_ptr());
        EXPECT_EQ(output.probabilities.data_ptr(), sentinel.data_ptr());
        EXPECT_EQ(output.sequences.data_ptr(), sentinel.data_ptr());
        EXPECT_EQ(output.positions.data_ptr(), sentinel.data_ptr());
    };
    check(logits.cpu(), {}, {}, {});
    check(logits.to(torch::kFloat16), {}, {}, {});
    check(logits.transpose(0, 1), {}, {}, {});
    check(logits.reshape({21, 43}), {}, {}, {});
    check(logits, torch::arange(127, logits.options().dtype(torch::kInt32)), {}, {});
    check(logits, {}, torch::ones({7}, logits.options().dtype(torch::kInt64)), {});
    check(logits, {}, {}, torch::ones({8}, logits.options().dtype(torch::kInt32)));
}

}  // namespace
}  // namespace rtp_llm
