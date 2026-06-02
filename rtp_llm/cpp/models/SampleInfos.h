#pragma once

#include <torch/all.h>
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"

namespace rtp_llm {

class LogitsProcessorStates;
typedef std::shared_ptr<LogitsProcessorStates> LogitsProcessorStatesPtr;

struct SamplerInitParams {};

struct SamplerInputs {
public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "SamplerInputs { "
                     << "batch_size: " << batch_size << ", step: " << step
                     << ", logits: " << tensorDebugStringWithData<int32_t>(logits)
                     << ", token_ids: " << tensorDebugStringWithData<int32_t>(token_ids)
                     << ", input_lengths: " << tensorDebugStringWithData<int32_t>(input_lengths)
                     << ", sequence_lengths: " << tensorDebugStringWithData<int32_t>(sequence_lengths)
                     << ", cum_log_probs: " << tensorDebugStringWithData<float>(cum_log_probs) << "}";
        return debug_string.str();
    }

public:
    torch::Tensor         logits;         // shape: [batch_size, vocab_size]
    mutable torch::Tensor token_ids;      // shape: [batch_size, max_length]
    torch::Tensor         input_lengths;  // shape: [batch_size]
    // shape: [decoder_batch_size]
    torch::Tensor            sequence_lengths;
    LogitsProcessorStatesPtr logits_processor_states_ptr;

    size_t vocab_size;
    size_t step;  // typically largest sequence length in the batch

    size_t        batch_size;            // sum of all num_beams_in of all streams
    size_t        batch_size_out;        // sum of all num_beams_out of all streams
    torch::Tensor num_beams_in;          // shape: [batch_size], dtype via Buffer (UINT64)
    torch::Tensor num_beams_out;         // shape: [batch_size], dtype via Buffer (UINT64)
    torch::Tensor top_k;                 // shape: [batch_size], dtype via Buffer (UINT32)
    torch::Tensor top_p;                 // shape: [batch_size], dtype: float
    torch::Tensor temperature;           // shape: [batch_size], dtype: float
    torch::Tensor repetition_penalty;    // shape: [batch_size], dtype: float
    torch::Tensor presence_penalty;      // shape: [batch_size], dtype: float
    torch::Tensor frequency_penalty;     // shape: [batch_size], dtype: float
    torch::Tensor no_repeat_ngram_size;  // shape: [batch_size], dtype via Buffer (INT32)
    torch::Tensor do_sample;             // shape: [batch_size], dtype via Buffer (BOOL)
    torch::Tensor finished_mask;         // shape: [batch_size], dtype via Buffer (BOOL)

    mutable torch::Tensor cum_log_probs;  // shape: [batch_size]
    mutable torch::Tensor all_probs;      // shape: [batch_size, vocab_size]

    std::vector<at::Generator> generator;
};

struct SamplerOutput {
public:
    torch::Tensor token_ids;
    torch::Tensor cum_log_probs;
    torch::Tensor all_probs;
    torch::Tensor beam_index;
    torch::Tensor success;
};

struct MergedOutput {
public:
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
};

}  // namespace rtp_llm
