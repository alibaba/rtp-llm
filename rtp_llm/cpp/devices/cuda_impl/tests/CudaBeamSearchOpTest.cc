#include "rtp_llm/cpp/devices/base_tests/BeamSearchOpTest.hpp"
using namespace std;
using namespace rtp_llm;

class CudaBeamSearchOpTest: public BeamSearchOpTest {};

TEST_F(CudaBeamSearchOpTest, simpleTest) {
    std::vector<int> batch_sizes = {1, 2, 15, 32};
    std::vector<int> beam_widths = {1, 2, 4, 5, 64, 70, 128, 500, 1024, 2500};
    std::vector<int> max_seq_len = {10, 100, 1000};
    const int vocab_size = 7000;

    for (auto batch_size : batch_sizes) {
        for (auto beam_width : beam_widths) {
            for (auto seq_len : max_seq_len) {
                std::cout << "batch_size: " << batch_size << ", beam_width: " << beam_width
                          << ", vocab_size: " << vocab_size << ", seq_len: " << seq_len << std::endl;
                simpleTest(batch_size, beam_width, vocab_size, seq_len);
            }
        }
    }
}

TEST_F(CudaBeamSearchOpTest, variableBeamWidthTest) {
    std::vector<int> batch_sizes = {1, 2, 31};
    std::vector<int> beam_widths = {1, 5, 70, 500, 2500};
    std::vector<int> max_seq_len = {10, 500};
    const int vocab_size = 7000;

    for (auto batch_size : batch_sizes) {
        for (auto beam_width_in : beam_widths) {
            for (auto beam_width_out : beam_widths) {
                if (beam_width_in == beam_width_out)
                    continue;
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
