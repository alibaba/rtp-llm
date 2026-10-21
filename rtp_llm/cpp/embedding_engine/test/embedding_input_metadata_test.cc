#include "rtp_llm/cpp/embedding_engine/EmbeddingInputMetadata.h"

#include <gtest/gtest.h>
#include <torch/cuda.h>

namespace rtp_llm {
namespace {

TEST(EmbeddingInputMetadataTest, PreservesHostPlanningBuffers) {
    const auto options = torch::TensorOptions().dtype(torch::kInt32);
    auto tokens = torch::tensor({11, 17, 21, 37}, options);
    auto lengths = torch::tensor({3, 1}, options);
    auto sequence = torch::empty({0}, options);
    auto prefix = torch::zeros({2}, options);
    const auto original_tokens = tokens;
    const auto original_lengths = lengths;
    prepareEmbeddingInputMetadata(tokens, lengths, sequence, prefix);
    for (const auto& tensor : {tokens, lengths, sequence, prefix}) {
        EXPECT_FALSE(tensor.is_cuda());
        EXPECT_EQ(tensor.scalar_type(), torch::kInt32);
        EXPECT_TRUE(tensor.is_contiguous());
    }
    EXPECT_EQ(tokens.data_ptr(), original_tokens.data_ptr());
    EXPECT_EQ(lengths.data_ptr(), original_lengths.data_ptr());
    EXPECT_TRUE(torch::equal(lengths, original_lengths));
}

TEST(EmbeddingInputMetadataTest, NormalizesIntegerTypeAndStrides) {
    auto tokens = torch::arange(8, torch::kInt64).slice(0, 0, 8, 2);
    auto lengths = torch::tensor({4}, torch::kInt64);
    auto sequence = torch::Tensor();
    auto prefix = torch::zeros({1}, torch::kInt64);
    prepareEmbeddingInputMetadata(tokens, lengths, sequence, prefix);
    EXPECT_TRUE(tokens.is_contiguous());
    EXPECT_EQ(tokens.scalar_type(), torch::kInt32);
    EXPECT_TRUE(torch::equal(tokens, torch::tensor({0, 2, 4, 6}, torch::kInt32)));
    EXPECT_FALSE(sequence.defined());
}

}  // namespace
}  // namespace rtp_llm
