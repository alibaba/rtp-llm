#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/split_softmax.h"

using namespace rtp_llm;

TEST(SplitSoftmaxTest, testSplitSoftmaxMatchesFloat64ReferenceAndNonfiniteContract) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(20261019);
    constexpr int64_t vocab = 248320;
    for (int64_t rows : {1, 3, 4, 8, 17, 32}) {
        for (float gain : {0.0f, 1.0f, 20.0f, 100.0f}) {
            auto logits = torch::randn({rows, vocab}, options) * gain;
            logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
            auto expected      = torch::softmax(logits.to(torch::kFloat64), -1).to(torch::kFloat32);
            auto original_fp32 = torch::softmax(logits, -1);
            rtp_llm::splitSoftmaxInplace(logits);
            ASSERT_TRUE(torch::equal(logits, original_fp32));
            ASSERT_TRUE(torch::allclose(logits, expected, 3e-6, 1e-8));
            ASSERT_TRUE(torch::allclose(logits.sum(-1), torch::ones({rows}, options), 3e-6, 1e-8));
        }
    }
    // Contiguous does not imply a 16-byte aligned storage offset.
    auto storage         = torch::randn({3 * 248320 + 1}, options);
    auto sentinel        = storage.slice(0, 0, 1).clone();
    auto unaligned       = storage.slice(0, 1).reshape({3, 248320});
    auto exact_unaligned = torch::softmax(unaligned, -1);
    rtp_llm::splitSoftmaxInplace(unaligned);
    ASSERT_TRUE(torch::equal(unaligned, exact_unaligned));
    ASSERT_TRUE(torch::equal(storage.slice(0, 0, 1), sentinel));
    for (float value : {-std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        auto logits = torch::zeros({3, 248320}, options);
        if (value < 0)
            logits.fill_(value);
        else
            logits.select(1, 1040).fill_(value);
        auto expected = torch::softmax(logits, -1);
        rtp_llm::splitSoftmaxInplace(logits);
        ASSERT_TRUE(torch::allclose(logits, expected, 3e-6, 1e-8, true));
    }
}

TEST(SplitSoftmaxTest, testSplitSoftmaxRejectsUnsupportedVocabulary) {
    auto logits = torch::zeros({3, 4093}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    EXPECT_THROW(rtp_llm::splitSoftmaxInplace(logits), c10::Error);
    ASSERT_EQ(logits.count_nonzero().item<int64_t>(), 0);
}
