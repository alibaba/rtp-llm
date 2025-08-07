#include <gtest/gtest.h>
#include "rtp_llm/cpp/devices/torch_impl/BeamSearchOp.h"

namespace rtp_llm {

class TorchBeamSearchOpTest: public ::testing::Test {
public:
    torch::TensorOptions bool_options  = torch::TensorOptions(torch::kBool).device(torch::Device(torch::kCPU));
    torch::TensorOptions int_options   = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));
    torch::TensorOptions float_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
};

TEST_F(TorchBeamSearchOpTest, simpleTest) {
    int batch_size         = 1;
    int beam_width         = 4;
    int vocab_size         = 10;
    int max_seq_len        = 8;
    int sequence_lengths[] = {4, 4, 4, 4};
    int input_lengths[]    = {3, 3, 3, 3};

    float logits[] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
                      1.8, 1.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    int token_ids[] = {1, 1, 1, 5, 0, 0, 0, 0, 2, 2, 2, 6, 0, 0, 0, 0, 3, 3, 3, 7, 0, 0, 0, 0, 4, 4, 4, 8, 0, 0, 0, 0};

    float cum_log_probs[] = {-1, -1, -1, -1};

    auto logtis_t           = torch::from_blob(logits, {batch_size, beam_width, vocab_size}, float_options);
    auto cum_log_probs_t    = torch::from_blob(cum_log_probs, {batch_size, beam_width}, float_options);
    auto input_lengths_t    = torch::from_blob(input_lengths, {batch_size, beam_width}, int_options);
    auto sequence_lengths_t = torch::from_blob(sequence_lengths, {batch_size, beam_width}, int_options);
    auto token_ids_t        = torch::from_blob(token_ids, {batch_size, beam_width, max_seq_len}, int_options);

    torch_impl::BeamSearchOp beam_search;
    beam_search.ptr()->to(torch::Device(torch::kCPU));
    auto result =
        beam_search->forward({logtis_t, token_ids_t, input_lengths_t, sequence_lengths_t, cum_log_probs_t, beam_width});

    int expected_token_ids[] = {1, 1, 1, 5, 9, 0, 0, 0, 3, 3, 3, 7, 9, 0, 0, 0,
                                3, 3, 3, 7, 8, 0, 0, 0, 1, 1, 1, 5, 8, 0, 0, 0};

    float expected_cum_log_probs[] = {-2.8935, -2.8935, -2.9935, -2.9935};
    int   expected_beam_index[]    = {0, 2, 2, 0};
    auto  expected_token_ids_t =
        torch::from_blob(expected_token_ids, {batch_size, beam_width, max_seq_len}, int_options);
    auto expected_cum_log_probs_t = torch::from_blob(expected_cum_log_probs, {batch_size, beam_width}, float_options);
    auto expected_beam_indices    = torch::from_blob(expected_beam_index, {batch_size, beam_width}, int_options);

    ASSERT_TRUE(torch::allclose(result.token_ids, expected_token_ids_t));
    ASSERT_TRUE(torch::allclose(result.cum_log_probs, expected_cum_log_probs_t));
    ASSERT_TRUE(torch::allclose(result.beam_indices, expected_beam_indices));
}

TEST_F(TorchBeamSearchOpTest, FirstBeamSearchTest) {
    int batch_size         = 1;
    int beam_width         = 4;
    int vocab_size         = 10;
    int max_seq_len        = 8;
    int sequence_lengths[] = {4, 4, 4, 4};
    int input_lengths[]    = {4, 4, 4, 4};

    float logits[] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.0, 1.1, 1.2, 1.3,
                      1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
                      1.8, 1.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};

    int token_ids[] = {1, 1, 1, 1, 0, 0, 0, 0};

    float cum_log_probs[] = {-1};

    auto logtis_t           = torch::from_blob(logits, {batch_size, 1, vocab_size}, float_options);
    auto cum_log_probs_t    = torch::from_blob(cum_log_probs, {batch_size, 1}, float_options);
    auto input_lengths_t    = torch::from_blob(input_lengths, {batch_size, 1}, int_options);
    auto sequence_lengths_t = torch::from_blob(sequence_lengths, {batch_size, 1}, int_options);
    auto token_ids_t        = torch::from_blob(token_ids, {batch_size, 1, max_seq_len}, int_options);

    torch_impl::BeamSearchOp beam_search;
    beam_search.ptr()->to(torch::Device(torch::kCPU));
    auto result =
        beam_search->forward({logtis_t, token_ids_t, input_lengths_t, sequence_lengths_t, cum_log_probs_t, beam_width});

    int expected_token_ids[] = {1, 1, 1, 1, 9, 0, 0, 0, 1, 1, 1, 1, 8, 0, 0, 0,
                                1, 1, 1, 1, 7, 0, 0, 0, 1, 1, 1, 1, 6, 0, 0, 0};

    float expected_cum_log_probs[] = {-2.8935, -2.9935, -3.0935, -3.1935};
    int   expected_beam_index[]    = {0, 0, 0, 0};

    auto expected_token_ids_t =
        torch::from_blob(expected_token_ids, {batch_size, beam_width, max_seq_len}, int_options);
    auto expected_cum_log_probs_t = torch::from_blob(expected_cum_log_probs, {batch_size, beam_width}, float_options);
    auto expected_beam_indices    = torch::from_blob(expected_beam_index, {batch_size, beam_width}, int_options);

    ASSERT_TRUE(torch::allclose(result.token_ids, expected_token_ids_t));
    ASSERT_TRUE(torch::allclose(result.cum_log_probs, expected_cum_log_probs_t));
    ASSERT_TRUE(torch::allclose(result.beam_indices, expected_beam_indices));
}

};  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}