#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/base_tests/BeamSearchOpTest.hpp"
using namespace std;
using namespace fastertransformer;
class CudaBeamSearchOpTest: public BeamSearchOpTest {};
TEST_F(CudaBeamSearchOpTest, simpleTestWithEndIds) {
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    std::vector<int> beam_widths = {2, 4, 8, 16};
    std::vector<int> vocab_sizes = {100};
    std::vector<int> max_seq_len = {10, 100, 1000};
    for (auto batch_size : batch_sizes) {
        for (auto beam_width: beam_widths) {
            for (auto vocab_size : vocab_sizes) {
                for (auto seq_len : max_seq_len) {
                    simpleTest(batch_size, beam_width, vocab_size, seq_len);
                }
            }
        }
    }
}
