#pragma once
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ops/BeamSearchOp.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <torch/torch.h>
#include <limits>

using namespace rtp_llm;

class BeamSearchOpTest: public DeviceTestBase {
public:
    torch::TensorOptions bool_options  = torch::TensorOptions(torch::kBool).device(torch::Device(torch::kCPU));
    torch::TensorOptions int_options   = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));
    torch::TensorOptions float_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));

    // BS: batch_size, MSL: max_seq_len
    // BMI: beam_width_in, BMO: beam_width_out

    struct TestBeamSearchInput {
        // float [BS, BMI]
        torch::Tensor logits;
        // int [BS, MBI, MSL]
        torch::Tensor token_ids;
        // int [BS, MBI] context + decoder
        torch::Tensor input_lengths;
        // int [BS, MBI] context + decoder
        torch::Tensor sequence_lengths;
        // float [BS, MBI]
        torch::Tensor cum_log_probs;
        // int
        int beam_width_out;
    };

    struct TestBeamSearchOutput {
        // int [BS, BMO, MSL]
        torch::Tensor token_ids;
        // int [BS, BMO] context + decoder
        torch::Tensor input_lengths;
        // int [BS, BMO] context + decoder
        torch::Tensor sequence_lengths;
        // float [BS, BMO]
        torch::Tensor cum_log_probs;
        // int [BS, BMO]
        torch::Tensor beam_indices;
    };

    TestBeamSearchInput
    prepareInput(int batch_size, int beam_width_in, int beam_width_out, int vocab_size, int max_seq_len) {
        int rand_input_length    = torch::randint(1, max_seq_len, {1}, int_options).item<int>();
        int rand_length          = torch::randint(0, max_seq_len - rand_input_length, {1}, int_options).item<int>();
        int rand_sequence_length = rand_length + rand_input_length;

        std::cout << "rand_input_length = " << rand_input_length << ", rand_sequence_length = " << rand_sequence_length
                  << ", max_seq_len = " << max_seq_len << std::endl;

        auto       input_lengths    = torch::full({batch_size, beam_width_in}, rand_input_length, int_options);
        auto       sequence_lengths = torch::full({batch_size, beam_width_in}, rand_sequence_length, int_options);
        at::Tensor cum_log_probs, token_ids, logits;
        if (beam_width_in == beam_width_out && rand_sequence_length == rand_input_length) {
            logits = torch::randn({batch_size, 1, vocab_size}, float_options).repeat({1, beam_width_in, 1});
            token_ids =
                torch::randint(0, vocab_size, {batch_size, 1, max_seq_len}, int_options).repeat({1, beam_width_in, 1});
            cum_log_probs = torch::full({batch_size, beam_width_in}, -1e9);
            cum_log_probs.index_put_({"...", 0}, 0.0);
        } else {
            logits        = torch::randn({batch_size, beam_width_in, vocab_size}, float_options);
            token_ids     = torch::randint(0, vocab_size, {batch_size, beam_width_in, max_seq_len}, int_options);
            cum_log_probs = torch::randn({batch_size, beam_width_in}, float_options);
        }

        return TestBeamSearchInput({logits, token_ids, input_lengths, sequence_lengths, cum_log_probs, beam_width_out});
    };

    void runMaskedEqualSmallBeamV2Tests() {
        for (const int valid_tokens : {1, 2}) {
            for (const int vocab_size : {3, 7, 65536}) {
                SCOPED_TRACE(vocab_size);
                TestBeamSearchInput input;
                input.logits = torch::full({1, 2, vocab_size}, -std::numeric_limits<float>::infinity(), float_options);
                input.logits[0][0][0] = 3.0f;
                if (valid_tokens > 1)
                    input.logits[0][0][1] = 1.0f;
                input.logits[0][1][0] = 2.0f;
                if (valid_tokens > 1)
                    input.logits[0][1][1] = 0.0f;
                input.token_ids        = torch::tensor({10, 0, 11, 0}, int_options).reshape({1, 2, 2});
                input.input_lengths    = torch::ones({1, 2}, int_options);
                input.sequence_lengths = torch::ones({1, 2}, int_options);
                input.cum_log_probs    = torch::tensor({0.0f, -10.0f}, float_options).reshape({1, 2});
                input.beam_width_out   = 2;
                auto ref               = torchRef(input);
                auto result            = opRun(input, true);
                ASSERT_TRUE(torch::equal(result.token_ids, ref.token_ids));
                ASSERT_TRUE(torch::equal(result.beam_indices, ref.beam_indices));
                assertTensorClose(result.cum_log_probs, ref.cum_log_probs);
            }
        }
    }

    TestBeamSearchOutput opRun(TestBeamSearchInput& input, bool force_v2 = false) {
        auto logits     = input.logits.to(torch::kCUDA);
        auto cuda_input = BeamSearchParams({logits,
                                            input.token_ids.to(torch::kCUDA),
                                            input.input_lengths.to(torch::kCUDA),
                                            input.sequence_lengths.to(torch::kCUDA),
                                            input.cum_log_probs.to(torch::kCUDA),
                                            static_cast<size_t>(input.beam_width_out),
                                            force_v2});

        auto result = execSampleBeamSearch(std::move(cuda_input));

        return TestBeamSearchOutput({result.token_ids.cpu(),
                                     result.input_lengths.cpu(),
                                     result.sequence_lengths.cpu(),
                                     result.cum_log_probs.cpu(),
                                     result.beam_indices.cpu()});
    };

    TestBeamSearchOutput torchRef(TestBeamSearchInput& input) {

        torch_impl::BeamSearchOp beam_search;
        beam_search.ptr()->to(torch::Device(torch::kCPU));

        auto result = beam_search->forward({input.logits,
                                            input.token_ids,
                                            input.input_lengths,
                                            input.sequence_lengths,
                                            input.cum_log_probs,
                                            input.beam_width_out});

        return TestBeamSearchOutput({result.token_ids,
                                     result.input_lengths,
                                     result.sequence_lengths,
                                     result.cum_log_probs,
                                     result.beam_indices});
    };

    void simpleTest(int batch_size, int beam_width, int vocab_size, int max_seq_len) {
        variableBeamWidthTest(batch_size, beam_width, beam_width, vocab_size, max_seq_len);
    }

    void variableBeamWidthTest(int batch_size, int beam_width_in, int beam_width_out, int vocab_size, int max_seq_len) {
        auto input  = prepareInput(batch_size, beam_width_in, beam_width_out, vocab_size, max_seq_len);
        auto result = opRun(input);
        auto ref    = torchRef(input);

        assertTensorClose(result.cum_log_probs, ref.cum_log_probs);
        assertTensorClose(result.sequence_lengths, ref.sequence_lengths);
        assertTensorClose(result.input_lengths, ref.input_lengths);

        // FIXME(zhangjianning.zjn): It is likely that the reference result and the op result mismatch
        // due to extremely close cum_log_probs of two beams at the boundary of the beam_width_out window.
        // Hence the following comparison is disabled for now. It would be better to figure a way out to check the
        // result.

        // assertTensorClose(result.token_ids, ref.token_ids);
        // assertTensorClose(result.beam_indices, ref.beam_indices);
    }

    void runSimpleTests() {
        std::vector<int> batch_sizes  = {1, 2, 15, 32};
        std::vector<int> beam_widths  = {1, 2, 4, 5, 64, 70, 128, 500, 1024, 2500};
        std::vector<int> max_seq_lens = {10, 100, 1000};
        const int        vocab_size   = 7000;
        for (auto batch_size : batch_sizes) {
            for (auto beam_width : beam_widths) {
                for (auto seq_len : max_seq_lens) {
                    std::cout << "batch_size: " << batch_size << ", beam_width: " << beam_width
                              << ", vocab_size: " << vocab_size << ", seq_len: " << seq_len << std::endl;
                    simpleTest(batch_size, beam_width, vocab_size, seq_len);
                }
            }
        }
    }

    void runVariableBeamWidthTests() {
        std::vector<int> batch_sizes  = {1, 2, 31};
        std::vector<int> beam_widths  = {1, 5, 70, 500, 2500};
        std::vector<int> max_seq_lens = {10, 500};
        const int        vocab_size   = 7000;
        for (auto batch_size : batch_sizes) {
            for (auto beam_width_in : beam_widths) {
                for (auto beam_width_out : beam_widths) {
                    if (beam_width_in == beam_width_out)
                        continue;
                    for (auto seq_len : max_seq_lens) {
                        std::cout << "batch_size: " << batch_size << ", beam_width_in: " << beam_width_in
                                  << ", beam_width_out: " << beam_width_out << ", vocab_size: " << vocab_size
                                  << ", seq_len: " << seq_len << std::endl;
                        variableBeamWidthTest(batch_size, beam_width_in, beam_width_out, vocab_size, seq_len);
                    }
                }
            }
        }
    }

    void runPrunedVocabV2Tests() {
        const int batch_size  = 1;
        const int beam_width  = 1500;
        const int vocab_size  = 2048;
        const int max_seq_len = 16;

        variableBeamWidthTest(batch_size, 1, beam_width, vocab_size, max_seq_len);
        simpleTest(batch_size, beam_width, vocab_size, max_seq_len);
        // Exercise the production wrapper, including the fixed-width tiny-vocab
        // V2 route. A flat CPU top-k is independent of its stage decomposition.
        for (int batches : {1, 2}) {
            prunedVocabMappingTest(batches, 2, 128, 67);
            prunedVocabMappingTest(batches, 4, 4, 7);
        }
        for (int unsupported : {4097, 5000}) {
            auto wide_output = prepareInput(1, 1, unsupported, 8192, 6);
            EXPECT_ANY_THROW(opRun(wide_output));
            auto wide_input = prepareInput(1, unsupported, 4, 2, 6);
            EXPECT_ANY_THROW(opRun(wide_input));
        }
        auto insufficient = prepareInput(1, 2, 135, 67, 6);
        EXPECT_ANY_THROW(opRun(insufficient));
    }

    void prunedVocabMappingTest(int batches, int beams_in, int beams_out, int vocab) {
        constexpr int       max_seq_len = 6;
        TestBeamSearchInput input{
            torch::arange(vocab, float_options).mul(0.125f).repeat({batches, beams_in, 1}),
            torch::arange(batches * beams_in * max_seq_len, int_options)
                .remainder(vocab)
                .reshape({batches, beams_in, max_seq_len}),
            torch::ones({batches, beams_in}, int_options),
            torch::full({batches, beams_in}, 3, int_options),
            torch::arange(beams_in, float_options).roll({1}, {0}).mul(-0.03125f).repeat({batches, 1}),
            beams_out};
        for (int parent = 0; parent < beams_in; ++parent) {
            input.input_lengths.select(1, parent).fill_(1 + parent % 2);
            input.sequence_lengths.select(1, parent).fill_(3 + parent % 2);
        }
        auto scores                     = input.logits.log_softmax(-1) + input.cum_log_probs.unsqueeze(-1);
        auto [expected_scores, indices] = scores.flatten(1).topk(beams_out, -1);
        auto parents                    = torch::div(indices, vocab, "floor").to(torch::kInt32);
        auto tokens                     = indices.remainder(vocab).to(torch::kInt32);
        auto expected_input_lengths     = input.input_lengths.gather(1, parents.to(torch::kLong));
        auto expected_lengths           = input.sequence_lengths.gather(1, parents.to(torch::kLong)) + 1;
        auto expected_ids =
            input.token_ids.gather(1, parents.to(torch::kLong).unsqueeze(-1).expand({batches, beams_out, max_seq_len}));
        for (int batch = 0; batch < batches; ++batch) {
            for (int beam = 0; beam < beams_out; ++beam) {
                expected_ids.index_put_({batch, beam, expected_lengths[batch][beam].item<int>() - 1},
                                        tokens[batch][beam]);
            }
        }
        auto actual = opRun(input);
        ASSERT_TRUE(torch::equal(actual.token_ids, expected_ids));
        ASSERT_TRUE(torch::equal(actual.beam_indices, parents));
        ASSERT_TRUE(torch::equal(actual.sequence_lengths, expected_lengths));
        ASSERT_TRUE(torch::equal(actual.input_lengths, expected_input_lengths));
        assertTensorClose(actual.cum_log_probs, expected_scores);
    }
};
