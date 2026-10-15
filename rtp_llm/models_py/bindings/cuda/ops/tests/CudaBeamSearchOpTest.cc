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

TEST_F(CudaBeamSearchOpTest, selectAllVocabV2Test) {
    const int vocab_size = 3000;
    for (const int batch_size : {1, 2}) {
        TestBeamSearchInput input;
        input.logits =
            (torch::arange(vocab_size, float_options) * 0.01f).reshape({1, 1, vocab_size}).repeat({batch_size, 1, 1});
        input.token_ids = torch::tensor({7, 8, 0, 0}, int_options).reshape({1, 1, 4}).repeat({batch_size, 1, 1});
        input.token_ids.select(2, 0).copy_(torch::arange(batch_size, int_options).reshape({batch_size, 1}) + 7);
        input.input_lengths    = torch::full({batch_size, 1}, 2, int_options);
        input.sequence_lengths = torch::full({batch_size, 1}, 2, int_options);
        input.cum_log_probs    = torch::zeros({batch_size, 1}, float_options);
        input.beam_width_out   = vocab_size;

        auto expected_ids = input.token_ids.repeat({1, vocab_size, 1});
        expected_ids.select(2, 2).copy_(
            torch::arange(vocab_size - 1, -1, -1, int_options).reshape({1, vocab_size}).repeat({batch_size, 1}));
        auto expected_probs = torch::log_softmax(input.logits, -1).flip({2}).reshape({batch_size, vocab_size});
        auto result         = opRun(input);
        ASSERT_TRUE(torch::equal(result.token_ids, expected_ids));
        ASSERT_TRUE(torch::equal(result.beam_indices, torch::zeros({batch_size, vocab_size}, int_options)));
        assertTensorClose(result.cum_log_probs, expected_probs, 1e-5, 1e-5);
        ASSERT_TRUE(torch::equal(result.input_lengths, torch::full({batch_size, vocab_size}, 2, int_options)));
        ASSERT_TRUE(torch::equal(result.sequence_lengths, torch::full({batch_size, vocab_size}, 3, int_options)));
    }
}
