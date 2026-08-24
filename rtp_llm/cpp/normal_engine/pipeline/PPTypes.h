#pragma once

#include <cstddef>
#include <map>
#include <string>

#include <torch/torch.h>

#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

// Transport-safe state reserved for PP sampling. Request objects, generators
// and logits-processor state are currently unsupported across the PP boundary.
struct PPSamplingPlan {
    torch::Tensor token_ids;
    torch::Tensor input_lengths;
    torch::Tensor sequence_lengths;

    torch::Tensor top_k;
    torch::Tensor top_p;
    torch::Tensor temperature;
    torch::Tensor repetition_penalty;
    torch::Tensor presence_penalty;
    torch::Tensor frequency_penalty;
    torch::Tensor no_repeat_ngram_size;
    torch::Tensor do_sample;
    torch::Tensor finished_mask;
    torch::Tensor cum_log_probs;

    torch::Tensor random_seeds;
    torch::Tensor random_offsets;

    size_t batch_size = 0;
    size_t step       = 0;
};

// Carries the execution payload along a PP lane. TP rank 0 carries the payload;
// every other TP lane carries an empty plan so its corresponding lane in the
// next stage advances by one batch.
struct PPExecutionPlan {
    GptModelInputs logical_model_input;
    PPSamplingPlan sampling_plan;
};

// Carries model-defined named tensors between adjacent PP stages. PP execution
// and transport treat the tensor keys as opaque.
struct PPIntermediateTensors {
    std::map<std::string, torch::Tensor> tensors;
};

struct PPSampleResult {
    torch::Tensor new_token_ids;  // [batch_size, 1]
    torch::Tensor success;
    torch::Tensor cum_log_probs;
    torch::Tensor next_random_offsets;
};

}  // namespace rtp_llm
