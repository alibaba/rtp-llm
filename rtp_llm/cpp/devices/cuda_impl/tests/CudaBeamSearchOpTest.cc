#include "rtp_llm/cpp/devices/base_tests/BeamSearchOpTest.hpp"
#include <limits>
using namespace std;
using namespace rtp_llm;

class CudaBeamSearchOpTest: public BeamSearchOpTest {};

TEST_F(CudaBeamSearchOpTest, maskedVariableBeamRegression) {
    // Exercise both one-block and multi-block TopK, including the production vocabulary.
    // Most per-parent candidates are -inf, but enough finite candidates exist globally.
    for (int vocab_size : {8192, 217216}) {
        for (auto widths : {std::pair<int, int>{1, 4}, {4, 7}, {7, 3}, {512, 3500}}) {
            SCOPED_TRACE(::testing::Message() << vocab_size << ": " << widths.first << " -> " << widths.second);
            auto input = prepareInput(1, widths.first, widths.second, vocab_size, 4);
            input.input_lengths.fill_(1);
            input.sequence_lengths.fill_(2);
            input.logits.fill_(-std::numeric_limits<float>::infinity());
            for (int beam = 0; beam < widths.first; ++beam) {
                input.cum_log_probs.index_put_({0, beam}, -10.0f * beam);
                for (int candidate = 0; candidate < 8; ++candidate) {
                    input.logits.index_put_({0, beam, 17 + 101 * candidate}, -float(candidate));
                }
            }
            auto expected = torchRef(input);
            for (int repeat = 0; repeat < 3; ++repeat) {
                auto actual = opRun(input);
                ASSERT_TRUE(torch::isfinite(actual.cum_log_probs).all().item<bool>());
                assertTensorClose(actual.cum_log_probs, expected.cum_log_probs);
                assertTensorClose(actual.beam_indices, expected.beam_indices);
                assertTensorClose(actual.token_ids, expected.token_ids);
            }
        }
    }
    // The optimization must not change ordinary (unmasked) beam search.
    variableBeamWidthTest(2, 5, 7, 512, 8);
    variableBeamWidthTest(2, 7, 3, 512, 8);
}

TEST_F(CudaBeamSearchOpTest, maskedFiniteTiesRemainStable) {
    for (int vocab_size : {5000, 217216}) {
        auto input = prepareInput(1, 4, 7, vocab_size, 4);
        input.input_lengths.fill_(1);
        input.sequence_lengths.fill_(2);
        input.cum_log_probs.zero_();
        input.logits.fill_(-std::numeric_limits<float>::infinity());
        for (int candidate = 0; candidate < 8; ++candidate) {
            input.logits.index_put_({0, torch::indexing::Slice(), 17 + 101 * candidate}, 0.0f);
        }
        auto expected = opRun(input);
        ASSERT_TRUE(torch::isfinite(expected.cum_log_probs).all().item<bool>());
        for (int repeat = 0; repeat < 5; ++repeat) {
            auto actual = opRun(input);
            assertTensorClose(actual.cum_log_probs, expected.cum_log_probs);
            assertTensorClose(actual.beam_indices, expected.beam_indices);
            assertTensorClose(actual.token_ids, expected.token_ids);
        }
    }
}

TEST_F(CudaBeamSearchOpTest, simpleTest) {
    std::vector<int> batch_sizes = {1, 2, 15, 32};
    std::vector<int> beam_widths = {1, 2, 4, 5, 8, 16, 32, 64, 70, 128, 256, 500, 1024, 2500};
    std::vector<int> vocab_sizes = {3000, 7000};
    std::vector<int> max_seq_len = {10, 100, 1000};

    for (auto batch_size : batch_sizes) {
        for (auto beam_width : beam_widths) {
            auto vocab_size = *std::lower_bound(vocab_sizes.begin(), vocab_sizes.end(), 2 * beam_width);

            for (auto seq_len : max_seq_len) {
                std::cout << "batch_size: " << batch_size << ", beam_width: " << beam_width
                          << ", vocab_size: " << vocab_size << ", seq_len: " << seq_len << std::endl;
                simpleTest(batch_size, beam_width, vocab_size, seq_len);
            }
        }
    }
}

TEST_F(CudaBeamSearchOpTest, variableBeamWidthTest) {
    std::vector<int> batch_sizes = {1, 2, 15, 32};
    std::vector<int> beam_widths = {1, 5, 70, 500, 1000, 3000};
    std::vector<int> vocab_sizes = {3000, 7000};
    std::vector<int> max_seq_len = {10, 500};

    for (auto batch_size : batch_sizes) {
        for (auto beam_width_in : beam_widths) {
            for (auto beam_width_out : beam_widths) {
                if (beam_width_in == beam_width_out)
                    continue;
                auto vocab_size = *std::lower_bound(
                    vocab_sizes.begin(), vocab_sizes.end(), 2 * std::max(beam_width_in, beam_width_out));

                for (auto seq_len : max_seq_len) {
                    std::cout << "batch_size: " << batch_size << ", beam_width_in: " << beam_width_in
                              << ", beam_width_out: " << beam_width_out << ", vocab_size: " << vocab_size
                              << ", seq_len: " << seq_len << std::endl;
                    variableBeamWidthTest(batch_size, beam_width_in, beam_width_out, vocab_size, seq_len);
                }
            }
        }
    }
}
