#pragma once
#include <torch/torch.h>

namespace rtp_llm {

namespace torch_impl {

struct BeamSearchOpInput {
    // [batch_size, beam_width_in, vocab_size] float
    torch::Tensor logits;
    // [batch_size, beam_width_in, max_seq_len] int
    torch::Tensor token_ids;
    // [batch_size, beam_width_in] int
    torch::Tensor input_lengths;
    // [batch_size, beam_width_in] int
    torch::Tensor sequence_lengths;
    // [batch_size, beam_width_in] float
    torch::Tensor cum_log_probs;
    int           beam_width_out;

    // only indexing on the first dim, split a batch computation into multiple single computations.
    BeamSearchOpInput operator[](int index) {
        return BeamSearchOpInput({logits[index],
                                  token_ids[index],
                                  input_lengths[index],
                                  sequence_lengths[index],
                                  cum_log_probs[index],
                                  beam_width_out});
    }
};

struct BeamSearchOpOutput {
    // [batch_size, beam_width_out, max_seq_len] int
    torch::Tensor token_ids;
    // [batch_size, beam_width_out] int
    torch::Tensor input_lengths;
    // [batch_size, beam_width_out] int
    torch::Tensor sequence_lengths;
    // [batch_size, beam_width_out] float
    torch::Tensor cum_log_probs;
    // [batch_size, beam_width_out] int
    torch::Tensor beam_indices;

    void zeros(int batch_size, int beam_width_out, int max_seq_len) {
        token_ids        = torch::zeros({batch_size, beam_width_out, max_seq_len},
                                 torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));
        input_lengths    = torch::zeros({batch_size, beam_width_out},
                                     torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));
        sequence_lengths = torch::zeros({batch_size, beam_width_out},
                                        torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));
        cum_log_probs    = torch::zeros({batch_size, beam_width_out},
                                     torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU)));
        beam_indices     = torch::zeros({batch_size, beam_width_out},
                                    torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));
    };

    // note output must has no batch dim.
    void set(int index, BeamSearchOpOutput output) {
        token_ids.index_put_({index}, output.token_ids);
        input_lengths.index_put_({index}, output.input_lengths);
        sequence_lengths.index_put_({index}, output.sequence_lengths);
        cum_log_probs.index_put_({index}, output.cum_log_probs);
        beam_indices.index_put_({index}, output.beam_indices);
    }
};

struct BeamSearchOpImpl: torch::nn::Module {

    BeamSearchOpOutput singleForward(BeamSearchOpInput input) {
        // logits float[beam_width_in, vocab_size]
        auto beam_width_out = input.beam_width_out;
        // token_ids int[beam_width_in, max_seq_len]
        auto max_seq_len = input.token_ids.size(1);

        // first topk from log softmax logits
        // log_probs: [beam_width_in, beam_width_out]
        auto [log_probs, index] = input.logits.log_softmax(-1).topk(beam_width_out, -1);

        // add cum_log_probs
        log_probs = log_probs + input.cum_log_probs.reshape({-1, 1});

        auto input_lengths =
            torch::zeros({beam_width_out}, torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));
        auto sequence_lengths =
            torch::zeros({beam_width_out}, torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));
        auto token_ids = torch::zeros({beam_width_out, max_seq_len},
                                      torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));

        // start beam search
        at::Tensor beam_indices;
        std::tie(log_probs, beam_indices) = log_probs.flatten(0, -1).topk(beam_width_out, -1);
        auto new_token_ids                = torch::gather(index.flatten(0, -1), 0, beam_indices);

        // keep track of beam indices
        auto beam_indices_in = torch::div(beam_indices, beam_width_out, "floor").squeeze();
        // in case only one beam is selected
        if (beam_indices_in.sizes().empty()) {
            beam_indices_in.unsqueeze_(0);
        }

        // according beam index to update input token ids
        for (int beam_idx_out = 0; beam_idx_out < beam_width_out; ++beam_idx_out) {
            auto beam_idx_in = beam_indices_in[beam_idx_out].item<int>();

            // keep track of input lengths
            auto input_len = input.input_lengths[beam_idx_in].item<int>();
            input_lengths.index_put_({beam_idx_out}, input_len);

            // keep track of sequence lengths
            auto seq_len = input.sequence_lengths[beam_idx_in].item<int>();
            sequence_lengths.index_put_({beam_idx_out}, seq_len + 1);

            // update token_ids
            token_ids.index_put_({beam_idx_out}, input.token_ids[beam_idx_in]);
            token_ids.index_put_({beam_idx_out, seq_len}, new_token_ids[beam_idx_out]);
        }
        return BeamSearchOpOutput({token_ids, input_lengths, sequence_lengths, log_probs, beam_indices_in});
    };

    BeamSearchOpOutput forward(BeamSearchOpInput input) {
        int                batch_size     = input.logits.size(0);
        int                beam_width_out = input.beam_width_out;
        int                max_seq_len    = input.token_ids.size(2);
        BeamSearchOpOutput output;
        output.zeros(batch_size, beam_width_out, max_seq_len);
        for (int i = 0; i < batch_size; i++) {
            auto result = singleForward(input[i]);
            output.set(i, result);
        }
        return output;
    };
};

TORCH_MODULE(BeamSearchOp);

};  // namespace torch_impl

};  // namespace rtp_llm
