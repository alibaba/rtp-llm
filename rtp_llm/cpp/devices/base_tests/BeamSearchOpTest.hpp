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

        // sort output beams besed on beam indices and token ids
        TestBeamSearchOutput sort() {
            auto latest_token_indices = (sequence_lengths - 1).unsqueeze(2).to(torch::kInt64);
            auto latest_tokens        = token_ids.gather(2, latest_token_indices).squeeze(2);

            at::Tensor indices, temp_indices;
            indices = latest_tokens.argsort(true, 1, true);

            temp_indices = beam_indices.gather(1, indices).argsort(true, 1, true);
            indices      = indices.gather(1, temp_indices);

            return TestBeamSearchOutput({
                token_ids.gather(1, indices.unsqueeze(2).expand_as(token_ids)),
                input_lengths.gather(1, indices),
                sequence_lengths.gather(1, indices),
                cum_log_probs.gather(1, indices),
                beam_indices.gather(1, indices),
            });
        }
    };

    TestBeamSearchInput
    prepareInput(int batch_size, int beam_width_in, int beam_width_out, int vocab_size, int max_seq_len) {
        int rand_input_length    = torch::randint(1, max_seq_len, {1}, int_options).item<int>();
        int rand_length          = torch::randint(0, max_seq_len - rand_input_length, {1}, int_options).item<int>();
        int rand_sequence_length = rand_length + rand_input_length;

        auto       input_lengths    = torch::full({batch_size, beam_width_in}, rand_input_length, int_options);
        auto       sequence_lengths = torch::full({batch_size, beam_width_in}, rand_sequence_length, int_options);
        at::Tensor cum_log_probs, token_ids, logits;
        if (rand_sequence_length == rand_input_length) {
            logits = torch::randn({batch_size, 1, vocab_size}, float_options).repeat({1, beam_width_in, 1});
            token_ids =
                torch::randint(0, vocab_size, {batch_size, 1, max_seq_len}, int_options).repeat({1, beam_width_in, 1});
            cum_log_probs = torch::zeros({batch_size, beam_width_in}, float_options);
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
            {*logits, token_ids, input_lengths, sequence_lengths, cum_log_probs, input.beam_width_out});

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

        // numerical error may ruin the order of results, sort them before comparison
        auto sorted_result = result.sort();
        auto sorted_ref    = ref.sort();

        assertTensorClose(sorted_result.token_ids, sorted_ref.token_ids);
        assertTensorClose(sorted_result.sequence_lengths, sorted_ref.sequence_lengths);
        assertTensorClose(sorted_result.input_lengths, sorted_ref.input_lengths);
        assertTensorClose(sorted_result.beam_indices, sorted_ref.beam_indices);
    }
};