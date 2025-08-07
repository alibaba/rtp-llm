#include "rtp_llm/cpp/devices/base_tests/BeamSearchOpTest.hpp"
using namespace std;
using namespace rtp_llm;

class CudaBeamSearchOpTest: public BeamSearchOpTest {};

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
