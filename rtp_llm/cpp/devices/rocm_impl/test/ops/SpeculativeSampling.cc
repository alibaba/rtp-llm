#include <gtest/gtest.h>
#include "rtp_llm/cpp/rocm/speculative_sampling/sampling.cuh"
#include <iostream>

class SpeculativeSamplingTest: public ::testing::Test {};

TEST_F(SpeculativeSamplingTest, basic_test) {
    int batch_size = 4;
    int vocab_size = 10000;
    int num_speculate_tokens = 8;

    auto float_tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCUDA));
    auto bf16_tensor_options  = torch::TensorOptions(torch::kBFloat16).device(torch::Device(torch::kCUDA));
    auto int_tensor_options   = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCUDA));

    auto pre_norm_draft_prob = torch::rand({batch_size, num_speculate_tokens, vocab_size}, float_tensor_options);
    auto normalized_draft_prob = pre_norm_draft_prob / pre_norm_draft_prob.sum(
        -1, true
    );
    auto draft_token_ids = torch::randint(
        vocab_size, {batch_size, num_speculate_tokens}, int_tensor_options
    );
    torch::Tensor uniform_samples_d  = torch::rand({(long)batch_size, (long)num_speculate_tokens + 1},
                                                  torch::TensorOptions().device(torch::Device(torch::kCUDA)).dtype(torch::kFloat));
    auto pre_norm_target_prob = torch::rand({batch_size, num_speculate_tokens + 1, vocab_size}, float_tensor_options);
    auto target_onehot_prob = pre_norm_target_prob / pre_norm_target_prob.sum(
        -1, true
    );
    auto accepted_num = torch::zeros({batch_size}, int_tensor_options);
    auto emitted_num = torch::zeros({batch_size}, int_tensor_options);

    auto output_accepted_token_num = torch::zeros({batch_size}, int_tensor_options);
    auto output_emitted_draft_token_num = torch::zeros({batch_size}, int_tensor_options);
    auto output_token_ids = torch::zeros({batch_size, num_speculate_tokens + 1}, int_tensor_options);
    chain_speculative_sampling(
        normalized_draft_prob,
        draft_token_ids,
        uniform_samples_d,
        target_onehot_prob,
        output_token_ids,
        output_accepted_token_num,
        output_emitted_draft_token_num,
        false,
        0
    );
}
