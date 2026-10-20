#include "rtp_llm/cpp/embedding_engine/EmbeddingInputMetadata.h"

#include <gtest/gtest.h>
#include <torch/cuda.h>

namespace rtp_llm {
namespace {

TEST(EmbeddingInputMetadataTest, ExecutorConversionRespectsUqiRegardlessOfTwoPass) {
#if USING_CUDA
    // Separate Bazel targets supply separate processes because production
    // caches its environment decision on the first request.
    const char* mask = std::getenv("USE_VISION_BERT_UQI_BLOCK_MASK");
    const char* two_pass = std::getenv("VISION_BERT_UQI_TWO_PASS");
    ASSERT_NE(mask, nullptr);
    ASSERT_NE(two_pass, nullptr);
    ASSERT_EQ(std::string(two_pass), "0");
    const bool uqi = std::string(mask) == "1";
    ASSERT_TRUE(torch::cuda::is_available());

    const auto options = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    // Includes UQI marker ID 2 and a trailing A token after SEP.
    auto tokens = torch::tensor({101, 17, 102, 2, 37, 102, 21}, options);
    auto lengths = torch::tensor({7}, options);
    auto sequence = torch::empty({0}, options);
    auto prefix = torch::zeros({1}, options);
    const auto original_tokens = tokens;
    const auto original_lengths = lengths;
    const auto original_sequence = sequence;
    const auto original_prefix = prefix;

    prepareEmbeddingInputMetadata(tokens, lengths, sequence, prefix);

    for (const auto& tensor : {tokens, lengths, sequence, prefix}) {
        EXPECT_EQ(tensor.is_cuda(), !uqi);
        EXPECT_EQ(tensor.scalar_type(), torch::kInt32);
    }
    EXPECT_TRUE(torch::equal(tokens.cpu(), original_tokens));
    EXPECT_TRUE(torch::equal(lengths.cpu(), original_lengths));
    EXPECT_TRUE(torch::equal(sequence.cpu(), original_sequence));
    EXPECT_TRUE(torch::equal(prefix.cpu(), original_prefix));
    if (uqi) {
        // Preserve the original host allocations, rather than round-tripping
        // through a device copy that hides the metadata pipeline regression.
        EXPECT_EQ(tokens.data_ptr(), original_tokens.data_ptr());
        EXPECT_EQ(lengths.data_ptr(), original_lengths.data_ptr());
        EXPECT_EQ(prefix.data_ptr(), original_prefix.data_ptr());
        EXPECT_TRUE(tokens.is_pinned());
        EXPECT_TRUE(lengths.is_pinned());
    }
#else
    GTEST_SKIP() << "CUDA model-facing metadata conversion only";
#endif
}

}  // namespace
}  // namespace rtp_llm
