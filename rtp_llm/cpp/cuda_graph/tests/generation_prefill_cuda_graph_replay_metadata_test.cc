#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "rtp_llm/cpp/cuda_graph/generation_prefill_cuda_graph_replay_metadata.h"

namespace rtp_llm {
namespace {

TEST(GenerationPrefillCudaGraphReplayMetadataTest, BuildsCanonicalLayout) {
    std::vector<int32_t> input_lengths{56, -1, -1, -1, -1};
    std::vector<int32_t> cu_seqlens(6, -1);
    std::vector<int32_t> padding_offset(64, -1);

    ASSERT_TRUE(prepareGenerationPrefillCudaGraphReplayMetadata(input_lengths.data(),
                                                                input_lengths.size(),
                                                                cu_seqlens.data(),
                                                                cu_seqlens.size(),
                                                                padding_offset.data(),
                                                                padding_offset.size(),
                                                                1,
                                                                4,
                                                                56,
                                                                64));
    EXPECT_EQ(input_lengths, (std::vector<int32_t>{64, 0, 0, 0, 0}));
    EXPECT_EQ(cu_seqlens, (std::vector<int32_t>{0, 64, 64, 64, 64, 64}));
    EXPECT_TRUE(std::all_of(padding_offset.begin(), padding_offset.end(), [](int32_t value) { return value == 0; }));
}

TEST(GenerationPrefillCudaGraphReplayMetadataTest, BuildsCanonicalLayoutAtExactTokenCapacity) {
    std::vector<int32_t> input_lengths{64, -1, -1};
    std::vector<int32_t> cu_seqlens(4, -1);
    std::vector<int32_t> padding_offset(64, -1);

    ASSERT_TRUE(prepareGenerationPrefillCudaGraphReplayMetadata(input_lengths.data(),
                                                                input_lengths.size(),
                                                                cu_seqlens.data(),
                                                                cu_seqlens.size(),
                                                                padding_offset.data(),
                                                                padding_offset.size(),
                                                                1,
                                                                2,
                                                                64,
                                                                64));
    EXPECT_EQ(input_lengths, (std::vector<int32_t>{64, 0, 0}));
    EXPECT_EQ(cu_seqlens, (std::vector<int32_t>{0, 64, 64, 64}));
}

TEST(GenerationPrefillCudaGraphReplayMetadataTest, RejectsMultiRequestUntilPerRequestSemanticsExist) {
    std::vector<int32_t> input_lengths{24, 32, -1, -1};
    std::vector<int32_t> cu_seqlens(5, -1);
    std::vector<int32_t> padding_offset(64, -1);

    EXPECT_FALSE(prepareGenerationPrefillCudaGraphReplayMetadata(input_lengths.data(),
                                                                 input_lengths.size(),
                                                                 cu_seqlens.data(),
                                                                 cu_seqlens.size(),
                                                                 padding_offset.data(),
                                                                 padding_offset.size(),
                                                                 2,
                                                                 4,
                                                                 56,
                                                                 64));
}

TEST(GenerationPrefillCudaGraphReplayMetadataTest, RejectsInvalidMetadata) {
    std::vector<int32_t> input_lengths{56, -1};
    std::vector<int32_t> cu_seqlens(4, -1);
    std::vector<int32_t> padding_offset(64, -1);

    auto prepare = [&](int requests, int max_requests, int real_tokens) {
        return prepareGenerationPrefillCudaGraphReplayMetadata(input_lengths.data(),
                                                               input_lengths.size(),
                                                               cu_seqlens.data(),
                                                               cu_seqlens.size(),
                                                               padding_offset.data(),
                                                               padding_offset.size(),
                                                               requests,
                                                               max_requests,
                                                               real_tokens,
                                                               64);
    };
    EXPECT_FALSE(prepare(0, 2, 56));
    EXPECT_FALSE(prepare(3, 2, 56));
    EXPECT_FALSE(prepare(1, 2, 55));

    input_lengths = {0, -1};
    EXPECT_FALSE(prepare(1, 2, 0));

    // Reject invalid lengths at the metadata owner, before copy kernels run.
    input_lengths = {-1, -1};
    EXPECT_FALSE(prepare(1, 2, 55));
    input_lengths = {65, -1};
    EXPECT_FALSE(prepare(1, 2, 64));
    input_lengths = {56, -1};
    EXPECT_FALSE(prepare(1, 2, 57));
}

TEST(GenerationPrefillCudaGraphReplayMetadataTest, RejectsInsufficientCapacity) {
    std::vector<int32_t> input_lengths{56, -1};
    std::vector<int32_t> cu_seqlens(4, -1);
    std::vector<int32_t> padding_offset(64, -1);

    const auto prepare = [&](size_t input_capacity, size_t cu_capacity, size_t padding_capacity) {
        return prepareGenerationPrefillCudaGraphReplayMetadata(input_lengths.data(),
                                                               input_capacity,
                                                               cu_seqlens.data(),
                                                               cu_capacity,
                                                               padding_offset.data(),
                                                               padding_capacity,
                                                               1,
                                                               2,
                                                               56,
                                                               64);
    };
    EXPECT_FALSE(prepare(2, cu_seqlens.size(), padding_offset.size()));
    EXPECT_FALSE(prepare(input_lengths.size(), 3, padding_offset.size()));
    EXPECT_FALSE(prepare(input_lengths.size(), cu_seqlens.size(), 63));
}

}  // namespace
}  // namespace rtp_llm
