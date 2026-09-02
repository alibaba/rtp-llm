#include <gtest/gtest.h>
#include <torch/torch.h>

namespace rtp_llm {
torch::Tensor sampleFromProbs(const torch::Tensor& probabilities);
}

namespace {

TEST(SampleFromProbsTest, HandlesSingleAndMultiBlockVocabulary) {
    auto forced_probs  = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).to(torch::kCUDA);
    auto forced_output = rtp_llm::sampleFromProbs(forced_probs);
    EXPECT_TRUE(forced_output.is_cuda());
    EXPECT_EQ(forced_output.scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::equal(forced_output.cpu(), torch::arange(4, torch::kInt32)));

    auto multi_block_probs     = torch::zeros({2, 2051}, torch::kFloat32);
    multi_block_probs[0][2048] = 1.0f;
    multi_block_probs[1][1024] = 1.0f;
    auto multi_block_output    = rtp_llm::sampleFromProbs(multi_block_probs.to(torch::kCUDA));
    EXPECT_TRUE(torch::equal(multi_block_output.cpu(), torch::tensor({2048, 1024}, torch::kInt32)));
}

TEST(SampleFromProbsTest, SamplesTheInputDistribution) {
    constexpr int64_t row_count = 4096;
    auto              probabilities =
        torch::softmax(torch::tensor({1.0f, 0.0f, -1.0f}, torch::kFloat32), -1).repeat({row_count, 1}).to(torch::kCUDA);
    auto output = rtp_llm::sampleFromProbs(probabilities);
    auto frequencies =
        torch::bincount(output.to(torch::kLong), {}, 3).to(torch::kFloat32) / static_cast<float>(row_count);
    auto expected = torch::softmax(torch::tensor({1.0f, 0.0f, -1.0f}), -1);
    EXPECT_TRUE(torch::allclose(frequencies.cpu(), expected, 0.05, 0.02)) << frequencies.cpu();
}

}  // namespace
