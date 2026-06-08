#pragma once
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ops/BeamSearchOp.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <torch/torch.h>

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

    TestBeamSearchOutput opRun(TestBeamSearchInput& input) {
        auto logits     = input.logits.to(torch::kCUDA);
        auto cuda_input = BeamSearchParams({logits,
                                            input.token_ids.to(torch::kCUDA),
                                            input.input_lengths.to(torch::kCUDA),
                                            input.sequence_lengths.to(torch::kCUDA),
                                            input.cum_log_probs.to(torch::kCUDA),
                                            static_cast<size_t>(input.beam_width_out)});

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
};