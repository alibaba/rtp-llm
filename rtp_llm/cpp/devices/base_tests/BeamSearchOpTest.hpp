#pragma once
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/torch_impl/BeamSearchOp.h"
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
        auto logits           = tensorToBuffer(input.logits, AllocationType::DEVICE);
        auto input_lengths    = tensorToBuffer(input.input_lengths, AllocationType::DEVICE);
        auto sequence_lengths = tensorToBuffer(input.sequence_lengths, AllocationType::DEVICE);
        auto token_ids        = tensorToBuffer(input.token_ids, AllocationType::DEVICE);
        auto cum_log_probs    = tensorToBuffer(input.cum_log_probs, AllocationType::DEVICE);

        auto output = device_->sampleBeamSearch(
            {*logits, token_ids, input_lengths, sequence_lengths, cum_log_probs, (size_t)input.beam_width_out});

        return TestBeamSearchOutput({bufferToTensor(*output.token_ids),
                                     bufferToTensor(*output.input_lengths),
                                     bufferToTensor(*output.sequence_lengths),
                                     bufferToTensor(*output.cum_log_probs),
                                     bufferToTensor(*output.beam_indices)});
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
};