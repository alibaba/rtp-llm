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

    struct TestBeamSearchInput {
        // float [BS, BM]
        torch::Tensor logits;
        // int [BS, BM, MSL]
        torch::Tensor token_ids;
        // int [BS, BM] context + decoder
        torch::Tensor input_lengths;
        // int [BS, BM] context + decoder
        torch::Tensor sequence_lengths;
        // float [BS, BM]
        torch::Tensor cum_log_probs;
        // int [BS, BM]
        torch::Tensor beam_index;
    };

    struct TestBeamSearchOutput {
        // int [BS, BM, MSL]
        torch::Tensor token_ids;
        // float [BS, BM]
        torch::Tensor cum_log_probs;
        // int [BS, BM]
        torch::Tensor beam_index;
    };

    TestBeamSearchInput prepareInput(int batch_size, int beam_width, int vocab_size, int max_seq_len) {
        int rand_input_length    = 4;
        int rand_length          = std::rand() % (max_seq_len - rand_input_length);
        int rand_sequence_length = rand_length + rand_input_length;

        auto input_lengths    = torch::ones({batch_size, beam_width}, int_options) * rand_input_length;
        auto sequence_lengths = torch::ones({batch_size, beam_width}, int_options) * rand_sequence_length;
        auto cum_log_probs    = torch::rand({batch_size, beam_width}, float_options);
        auto token_ids        = torch::randint(0, vocab_size - 1, {batch_size, beam_width, max_seq_len}, int_options);
        auto logits           = torch::rand({batch_size, beam_width, vocab_size}, float_options);
        if (rand_sequence_length == rand_input_length) {
            logits = torch::rand({batch_size, 1, vocab_size}, float_options).repeat({1, beam_width, 1});
            token_ids =
                torch::randint(0, vocab_size - 1, {batch_size, 1, max_seq_len}, int_options).repeat({1, beam_width, 1});
            cum_log_probs = torch::zeros({batch_size, beam_width}, float_options);
        }
        auto beam_index = torch::zeros({batch_size, beam_width}, int_options);

        return TestBeamSearchInput({logits, token_ids, input_lengths, sequence_lengths, cum_log_probs, beam_index});
    };

    TestBeamSearchOutput opRun(TestBeamSearchInput& input) {

        auto logits           = tensorToBuffer(input.logits, AllocationType::DEVICE);
        auto input_lengths    = tensorToBuffer(input.input_lengths, AllocationType::DEVICE);
        auto sequence_lengths = tensorToBuffer(input.sequence_lengths, AllocationType::DEVICE);
        auto token_ids        = tensorToBuffer(input.token_ids, AllocationType::DEVICE);
        auto cum_log_probs    = tensorToBuffer(input.cum_log_probs, AllocationType::DEVICE);
        auto beam_index       = tensorToBuffer(input.beam_index, AllocationType::DEVICE);

        device_->sampleBeamSearch(
            {*logits, *token_ids, *input_lengths, *sequence_lengths, *cum_log_probs, *beam_index});

        return TestBeamSearchOutput(
            {bufferToTensor(*token_ids), bufferToTensor(*cum_log_probs), bufferToTensor(*beam_index)});
    };

    TestBeamSearchOutput torchRef(TestBeamSearchInput& input) {

        torch_impl::BeamSearchOp beam_search;
        beam_search.ptr()->to(torch::Device(torch::kCPU));

        auto result = beam_search->forward({input.logits,
                                            input.token_ids,
                                            input.input_lengths,
                                            input.sequence_lengths,
                                            input.cum_log_probs,
                                            input.beam_index});

        return TestBeamSearchOutput({result.token_ids, result.cum_log_probs, result.beam_index});
    };

    void simpleTest(int batch_size, int beam_width, int vocab_size, int max_seq_len) {
        auto input  = prepareInput(batch_size, beam_width, vocab_size, max_seq_len);
        auto result = opRun(input);
        auto ref    = torchRef(input);
        assertTensorClose(result.token_ids, ref.token_ids);
        assertTensorClose(result.cum_log_probs, ref.cum_log_probs);
        assertTensorClose(result.beam_index, ref.beam_index);
    }
};