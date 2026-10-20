#include "rtp_llm/cpp/testing/BeamSearchOpTest.hpp"

using namespace std;
using namespace rtp_llm;

class CudaBeamSearchOpTest: public BeamSearchOpTest {};

TEST_F(CudaBeamSearchOpTest, simpleTest) {
    runSimpleTests();
}

TEST_F(CudaBeamSearchOpTest, variableBeamWidthTest) {
    runVariableBeamWidthTests();
}

TEST_F(CudaBeamSearchOpTest, prunedVocabV2Test) {
    runPrunedVocabV2Tests();
}

TEST_F(CudaBeamSearchOpTest, unsupportedBeamWidthTest) {
    runUnsupportedBeamWidthTests();
}

TEST_F(CudaBeamSearchOpTest, selectAllTopkTest) {
    // A separate target sets FORCE_PATH before the cached routing switch is read.
    // Unique scores make exact token and parent-beam comparisons meaningful.
    for (const auto& shape : {std::vector<int>{1, 3000, 3000}, {2, 4096, 2400}, {1, 2500, 7000}}) {
        const int beam_in = shape[0], beam_out = shape[1], vocab = shape[2];
        auto      input = prepareInput(1, beam_in, beam_out, vocab, 4);
        input.logits = (torch::arange(vocab, float_options) * 0.0625f).reshape({1, 1, vocab}).repeat({1, beam_in, 1});
        input.cum_log_probs = (torch::arange(beam_in, float_options) * 0.03125f).reshape({1, beam_in});
        input.input_lengths.fill_(1);
        input.sequence_lengths.fill_(2);
        auto result = opRun(input);
        auto ref    = torchRef(input);
        assertTensorClose(result.cum_log_probs, ref.cum_log_probs);
        EXPECT_TRUE(torch::equal(result.token_ids, ref.token_ids));
        EXPECT_TRUE(torch::equal(result.beam_indices, ref.beam_indices));
        EXPECT_TRUE(torch::equal(result.sequence_lengths, ref.sequence_lengths));
        EXPECT_TRUE(torch::equal(result.input_lengths, ref.input_lengths));
    }
}
