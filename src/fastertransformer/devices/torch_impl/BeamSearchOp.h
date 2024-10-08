#pragma once
#include <torch/torch.h>

namespace fastertransformer {

namespace torch_impl {

struct BeamSearchOpInput {
    // [batch_size, beam_num, vocab_size] float
    torch::Tensor logits;
    // [batch_size, beam_num, max_seq_len] int
    torch::Tensor token_ids;
    // [batch_size, beam_num] int
    torch::Tensor input_lengths;
    // [batch_size, beam_num] int
    torch::Tensor sequence_lengths;
    // [batch_size, beam_num] float
    torch::Tensor cum_log_probs;
    // [batch_size, beam_num] int
    torch::Tensor beam_index;

    // only index in first dim, split batch compute to multi single compute.
    BeamSearchOpInput operator[](int index) {
        return BeamSearchOpInput({logits[index],
                                  token_ids[index],
                                  input_lengths[index],
                                  sequence_lengths[index],
                                  cum_log_probs[index],
                                  beam_index[index]});
    }
};

struct BeamSearchOpOutput {
    // [batch_size, beam_num, max_seq_len] int
    torch::Tensor token_ids;
    // [batch_size, beam_num] float
    torch::Tensor cum_log_probs;
    // [batch_size, beam_num] int
    torch::Tensor beam_index;

    void zeros(int batch_size, int beam_num, int max_seq_len) {
        token_ids = torch::zeros({batch_size, beam_num, max_seq_len},
            torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));
        cum_log_probs = torch::zeros({batch_size, beam_num},
            torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU)));
        beam_index = torch::zeros({batch_size, beam_num},
            torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));
    };

    // note output must has no batch dim.
    void set(int index, BeamSearchOpOutput output) {
        token_ids.index_put_({index}, output.token_ids);
        cum_log_probs.index_put_({index}, output.cum_log_probs);
        beam_index.index_put_({index}, output.beam_index);
    }
};

struct BeamSearchOpImpl : torch::nn::Module {

    BeamSearchOpOutput oneBatchForward(BeamSearchOpInput input) {
        // logits float[beam_width, vocab_size]
        auto beam_width = input.logits.size(0);
        // first topk from log softmax logits
        // log_probs: [beam_width, beam_width]
        auto [log_probs, index] = input.logits.log_softmax(-1).topk(beam_width, -1);
        // add cum_log_probs
        log_probs = log_probs + input.cum_log_probs.reshape({-1, 1});

        auto new_token_ids = torch::zeros({beam_width}, torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));
        auto beam_index    = torch::zeros({beam_width}, torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU)));
        // first beam search
        if (input.input_lengths.equal(input.sequence_lengths)) {
            new_token_ids = index[0];
            log_probs     = log_probs[0];
        } else {
            // second topk from probs
            // log_probs: [beam_width]
            std::tie(log_probs, beam_index) = log_probs.flatten(0, -1).topk(beam_width, -1);
            new_token_ids = torch::gather(index.flatten(0, -1), 0, beam_index);
            beam_index = torch::div(beam_index, beam_width, "floor").squeeze();
        }

        // according beam index to update input token ids
        auto token_ids = input.token_ids.clone();
        for (int i = 0; i < beam_width; i++) {
            // update sequence lengths
            auto sequence_lengths_index = input.sequence_lengths[i].item<int>() + 1;
            auto select_beam_index = beam_index[i].item<int>();
            auto tmp = input.token_ids[select_beam_index];
            token_ids.index_put_({i}, tmp);
            token_ids.index_put_({i, sequence_lengths_index - 1}, new_token_ids[i]);

        }
        return BeamSearchOpOutput({token_ids, log_probs, beam_index});
    };

    BeamSearchOpOutput forward(BeamSearchOpInput input) {
        int batch_size = input.logits.size(0);
        int beam_num = input.logits.size(1);
        int max_seq_len = input.token_ids.size(2);
        BeamSearchOpOutput output;
        output.zeros(batch_size, beam_num, max_seq_len);
        for (int i = 0; i < batch_size; i++) {
            auto result = oneBatchForward(input[i]);
            output.set(i, result);
        }
        return output;
    };
};


TORCH_MODULE(BeamSearchOp);

}; // namespace torch_impl

}; // namespace fastertransformer

