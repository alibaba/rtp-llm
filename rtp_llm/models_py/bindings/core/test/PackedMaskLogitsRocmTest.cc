#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

using namespace rtp_llm;

TEST(PackedMaskLogitsRocmTest, CompactRowsAndStridedLogits) {
    constexpr int64_t vocab_size = 35;
    auto backing = torch::ones({4, 42}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    auto logits = backing.narrow(1, 1, 40);
    auto mask = torch::tensor({1, 4, 2, 2}, torch::kInt32).reshape({2, 2}).to(torch::kCUDA);
    auto rows = torch::tensor({1, 3}, torch::kInt32).to(torch::kCUDA);

    runtimeApplyPackedMaskLogits(logits, mask, rows, vocab_size);
    auto result = backing.cpu();
    for (int64_t row = 0; row < 4; ++row) {
        for (int64_t token = 0; token < 42; ++token) {
            const bool allowed = row == 0 || row == 2 || token == 0 || token >= 36
                                 || (row == 1 && (token == 1 || token == 35))
                                 || (row == 3 && (token == 2 || token == 34));
            EXPECT_FLOAT_EQ(result[row][token].item<float>(),
                            allowed ? 1.0f : -std::numeric_limits<float>::max());
        }
    }
}

TEST(PackedMaskLogitsRocmTest, HalfTypesAndInvalidRows) {
    auto mask = torch::zeros({3, 1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    auto rows = torch::tensor({-1, 1, 3}, torch::kInt32).to(torch::kCUDA);
    for (const auto dtype : {torch::kFloat16, torch::kBFloat16}) {
        auto logits = torch::ones({3, 4}, torch::TensorOptions(dtype).device(torch::kCUDA));
        runtimeApplyPackedMaskLogits(logits, mask, rows, 4);
        auto result = logits.to(torch::kFloat32).cpu();
        for (int64_t row = 0; row < 3; ++row) {
            for (int64_t token = 0; token < 4; ++token) {
                const auto value = result[row][token].item<float>();
                if (row == 1) {
                    EXPECT_TRUE(std::isinf(value));
                    EXPECT_LT(value, 0.0f);
                } else {
                    EXPECT_FLOAT_EQ(value, 1.0f);
                }
            }
        }
    }
}

TEST(PackedMaskLogitsRocmTest, IdentityRow) {
    auto logits = torch::ones({35}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    auto mask = torch::tensor({1, 4}, torch::kInt32).reshape({1, 2}).to(torch::kCUDA);
    runtimeApplyPackedMaskLogits(logits, mask, 35);
    auto result = logits.cpu();
    for (int64_t token = 0; token < 35; ++token) {
        EXPECT_FLOAT_EQ(result[token].item<float>(),
                        (token == 0 || token == 34) ? 1.0f : -std::numeric_limits<float>::max());
    }
}
