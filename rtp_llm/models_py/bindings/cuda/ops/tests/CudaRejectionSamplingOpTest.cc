#include "rtp_llm/cpp/testing/RejectionSamplingOpTest.hpp"

class CudaRejectionSamplingOpTest: public RejectionSamplingOpTest {};

TEST_F(CudaRejectionSamplingOpTest, referenceCases) {
    runReferenceCases();
}

TEST_F(CudaRejectionSamplingOpTest, zeroAndOneSpeculativeTokenCases) {
    runZeroAndOneSpeculativeTokenCases();
}

TEST_F(CudaRejectionSamplingOpTest, rejectsInvalidTensorMetadata) {
    runRejectsInvalidTensorMetadata();
}

TEST_F(CudaRejectionSamplingOpTest, qwen35LargeVocabResidualSamplingIsBitwiseStable) {
    constexpr int batch_size             = 32;
    constexpr int num_speculative_tokens = 4;
    constexpr int vocab_size             = 248320;
    constexpr int target_token_stride    = 1;
    constexpr int repeats                = 64;

    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto int_options   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto bool_options  = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    auto draft_probs = torch::full(
        {batch_size, num_speculative_tokens, vocab_size}, 1.0f / static_cast<float>(vocab_size), float_options);
    auto target_probs = torch::zeros({batch_size, num_speculative_tokens + 1, vocab_size}, float_options);
    target_probs.slice(2, vocab_size / 2, vocab_size).fill_(2.0f / static_cast<float>(vocab_size));
    auto draft_token_ids = torch::zeros({batch_size, num_speculative_tokens}, int_options);
    auto uniform_samples = torch::full({batch_size, num_speculative_tokens + 1}, 0.73125f, float_options);
    uniform_samples.select(1, 0).fill_(0.5f);
    auto target_token_ids =
        torch::full({batch_size * (num_speculative_tokens + 1), target_token_stride}, vocab_size / 2, int_options);
    auto output_token_ids          = torch::empty({batch_size, num_speculative_tokens + 1}, int_options);
    auto output_accepted_token_num = torch::empty({batch_size}, int_options);
    auto do_sample                 = torch::ones({batch_size}, bool_options);

    RejectionSamplingParams params{draft_probs,
                                   draft_token_ids,
                                   uniform_samples,
                                   target_probs,
                                   target_token_ids,
                                   output_token_ids,
                                   output_accepted_token_num,
                                   do_sample};

    torch::Tensor baseline_token_ids;
    torch::Tensor baseline_accepted_token_num;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        output_token_ids.fill_(-7);
        output_accepted_token_num.fill_(-7);
        rejectionSampling(params);

        auto token_ids          = output_token_ids.cpu().contiguous();
        auto accepted_token_num = output_accepted_token_num.cpu().contiguous();
        if (iteration == 0) {
            baseline_token_ids          = token_ids.clone();
            baseline_accepted_token_num = accepted_token_num.clone();
            continue;
        }
        ASSERT_TRUE(torch::equal(token_ids, baseline_token_ids))
            << "rejection output token IDs differ at iteration " << iteration;
        ASSERT_TRUE(torch::equal(accepted_token_num, baseline_accepted_token_num))
            << "rejection accepted lengths differ at iteration " << iteration;
    }

    ASSERT_TRUE(baseline_token_ids.defined());
    ASSERT_EQ(baseline_accepted_token_num.min().item<int>(), 1);
    ASSERT_EQ(baseline_accepted_token_num.max().item<int>(), 1);
    auto sampled_tokens = baseline_token_ids.select(1, 0);
    ASSERT_GE(sampled_tokens.min().item<int>(), vocab_size / 2);
    ASSERT_LT(sampled_tokens.max().item<int>(), vocab_size);
    ASSERT_TRUE(torch::equal(baseline_token_ids.slice(1, 1),
                             torch::full({batch_size, num_speculative_tokens}, -1, torch::kInt32)));
    for (int row = 1; row < batch_size; ++row) {
        ASSERT_TRUE(torch::equal(baseline_token_ids[row], baseline_token_ids[0]))
            << "identical rejection inputs differ between batch rows 0 and " << row;
    }
}
