#pragma once

#include <ATen/Generator.h>
#include <cstddef>
#include <optional>
#include <vector>
#include <torch/torch.h>

namespace rtp_llm {

struct GreedySamplingBuffers {
    torch::Tensor seed_host;
    torch::Tensor offset_host;
    torch::Tensor output_ids_ptrs_host;
    size_t        max_batch_size = 0;
};

struct GreedyParams {
    torch::Tensor logits;            // [batch_size, vocab_size_padded], modified by penalties.
    torch::Tensor input_lengths;     // [batch_size]
    torch::Tensor sequence_lengths;  // [batch_size]
    torch::Tensor token_ids;         // [batch_size, max_input_length + 1]
    size_t        step;

    torch::Tensor top_k;
    torch::Tensor top_p;
    torch::Tensor temperature;

    std::optional<torch::Tensor> repetition_penalty;
    std::optional<torch::Tensor> no_repeat_ngram_size;
    std::optional<torch::Tensor> cum_log_probs;
    std::optional<torch::Tensor> output_log_probs;
    bool                         return_original_all_probs = false;
    std::optional<torch::Tensor> output_all_probs;
    std::optional<torch::Tensor> presence_penalty;
    std::optional<torch::Tensor> frequency_penalty;
    std::optional<torch::Tensor> do_sample;
    std::vector<at::Generator>   generator;
    GreedySamplingBuffers*       sampling_buffers = nullptr;
};

struct GreedyOutput {
    torch::Tensor success;
};

struct BeamSearchParams {
    // Modified in place; callers must not reuse logits after the call.
    torch::Tensor logits;            // [batch_size, num_beams_in, vocab_size]
    torch::Tensor token_ids;         // [batch_size, num_beams_in, max_seq_len]
    torch::Tensor input_lengths;     // [batch_size, num_beams_in]
    torch::Tensor sequence_lengths;  // [batch_size, num_beams_in]
    torch::Tensor cum_log_probs;     // [batch_size, num_beams_in]
    size_t        num_beams_out = 0;
};

struct BeamSearchOutput {
    torch::Tensor token_ids;         // [batch_size, num_beams_out, max_seq_len]
    torch::Tensor input_lengths;     // [batch_size, num_beams_out]
    torch::Tensor sequence_lengths;  // [batch_size, num_beams_out]
    torch::Tensor cum_log_probs;     // [batch_size, num_beams_out]
    torch::Tensor beam_indices;      // [batch_size, num_beams_out]
};

struct RejectionSamplingParams {
    torch::Tensor draft_probs_d;
    torch::Tensor draft_token_ids_d;
    torch::Tensor uniform_samples_d;
    torch::Tensor target_probs_d;
    torch::Tensor target_token_ids_d;
    torch::Tensor output_token_ids_d;
    torch::Tensor output_accepted_token_num_d;
    torch::Tensor do_sample_d;
    // In-model proposers such as DSpARK emit tokens without per-vocab draft probabilities.
    bool draft_probs_point_mass = false;
};

struct MappingDraft2TargetParams {
    torch::Tensor tokens;
    torch::Tensor d2t_map;
    int           batch_size;
    int           token_offset;
    int           token_stride;
};

}  // namespace rtp_llm
