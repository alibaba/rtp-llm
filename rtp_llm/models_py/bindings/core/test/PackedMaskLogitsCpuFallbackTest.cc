#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

using namespace rtp_llm;

TEST(PackedMaskLogitsCpuFallbackTest, UsesCompactRowMapping) {
    constexpr int64_t vocab_size        = 35;
    constexpr int64_t logits_columns    = 40;
    auto              packed_allow_mask = torch::tensor({1, 4, 2, 2}, torch::kInt32).reshape({2, 2});
    auto              row_indices       = torch::tensor({1, 3}, torch::kInt32);

    for (const auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
        auto logits = torch::ones({4, logits_columns}, torch::TensorOptions(dtype).device(torch::kCPU));
        ASSERT_NO_THROW(runtimeApplyPackedMaskLogits(logits, packed_allow_mask, row_indices, vocab_size));

        auto result = logits.to(torch::kFloat32).contiguous();
        for (int64_t row = 0; row < result.size(0); ++row) {
            for (int64_t token = 0; token < result.size(1); ++token) {
                const bool allowed = row == 0 || row == 2 || token >= vocab_size
                                     || (row == 1 && (token == 0 || token == 34))
                                     || (row == 3 && (token == 1 || token == 33));
                if (allowed) {
                    EXPECT_FLOAT_EQ(result[row][token].item<float>(), 1.0f);
                } else if (dtype == torch::kFloat32) {
                    EXPECT_FLOAT_EQ(result[row][token].item<float>(), -std::numeric_limits<float>::max());
                } else {
                    EXPECT_TRUE(std::isinf(result[row][token].item<float>()));
                    EXPECT_LT(result[row][token].item<float>(), 0.0f);
                }
            }
        }
    }
}

TEST(PackedMaskLogitsCpuFallbackTest, SupportsSingleRowIdentityMapping) {
    constexpr int64_t vocab_size = 35;
    auto              logits     = torch::ones({vocab_size}, torch::TensorOptions(torch::kFloat32).device(torch::kCPU));
    auto              packed_allow_mask = torch::tensor({1, 4}, torch::kInt32).reshape({1, 2});

    ASSERT_NO_THROW(runtimeApplyPackedMaskLogits(logits, packed_allow_mask, vocab_size));

    auto result = logits.contiguous();
    for (int64_t token = 0; token < vocab_size; ++token) {
        if (token == 0 || token == 34) {
            EXPECT_FLOAT_EQ(result[token].item<float>(), 1.0f);
        } else {
            EXPECT_FLOAT_EQ(result[token].item<float>(), -std::numeric_limits<float>::max());
        }
    }
}

TEST(PackedMaskLogitsCpuFallbackTest, SkipsOutOfRangeRows) {
    constexpr int64_t vocab_size = 4;
    auto              logits = torch::ones({3, vocab_size}, torch::TensorOptions(torch::kFloat32).device(torch::kCPU));
    auto              packed_allow_mask = torch::zeros({3, 1}, torch::kInt32);
    auto              row_indices       = torch::tensor({-1, 1, 3}, torch::kInt32);

    ASSERT_NO_THROW(runtimeApplyPackedMaskLogits(logits, packed_allow_mask, row_indices, vocab_size));

    auto result = logits.contiguous();
    for (int64_t token = 0; token < vocab_size; ++token) {
        EXPECT_FLOAT_EQ(result[0][token].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(result[1][token].item<float>(), -std::numeric_limits<float>::max());
        EXPECT_FLOAT_EQ(result[2][token].item<float>(), 1.0f);
    }
}

TEST(PackedMaskLogitsCpuFallbackTest, CopiesBackToNonContiguousInput) {
    constexpr int64_t vocab_size = 4;
    auto backing = torch::ones({2, vocab_size + 2}, torch::TensorOptions(torch::kFloat32).device(torch::kCPU));
    auto logits  = backing.narrow(/*dim=*/1, /*start=*/1, /*length=*/vocab_size);
    ASSERT_FALSE(logits.is_contiguous());
    ASSERT_EQ(logits.stride(0), vocab_size + 2);

    auto packed_allow_mask = torch::tensor({1, 8}, torch::kInt32).reshape({2, 1});

    ASSERT_NO_THROW(runtimeApplyPackedMaskLogits(logits, packed_allow_mask, vocab_size));

    auto result = backing.contiguous();
    for (int64_t row = 0; row < result.size(0); ++row) {
        EXPECT_FLOAT_EQ(result[row][0].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(result[row][vocab_size + 1].item<float>(), 1.0f);
        for (int64_t token = 0; token < vocab_size; ++token) {
            const bool allowed = (row == 0 && token == 0) || (row == 1 && token == 3);
            const auto value   = result[row][token + 1].item<float>();
            if (allowed) {
                EXPECT_FLOAT_EQ(value, 1.0f);
            } else {
                EXPECT_FLOAT_EQ(value, -std::numeric_limits<float>::max());
            }
        }
    }
}
