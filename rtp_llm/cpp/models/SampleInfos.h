#pragma once

#include <optional>
#include <vector>
#include <torch/all.h>
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"

namespace rtp_llm {

class LogitsProcessorStates;
typedef std::shared_ptr<LogitsProcessorStates> LogitsProcessorStatesPtr;

struct SamplerInitParams {
    // max_batch_size is an initial capacity. Set fixed_max_batch_size=false when callers can legally
    // fan out beyond that capacity, e.g. num_return_sequences or variable beam requests. When
    // max_batch_size == 0, fixed_max_batch_size is ignored and buffers grow dynamically.
    size_t max_batch_size       = 0;
    bool   fixed_max_batch_size = true;
};

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

    bool return_original_all_probs = false;

    // When true no writer touches the logits buffer for the rest of this
    // sampling round (it is either a private per-round clone or a read-only
    // view whose owners provably never read it back): the sampler may skip
    // writing softmax probabilities back into it. Set only by the speculative
    // (MTP) gather; every other path keeps the legacy in-place contract.
    bool skip_logits_probs_restore = false;

    // When true every consumer of all_probs treats the renormalized
    // distribution as a point mass at the sampled token, so the sampler may
    // skip materializing the one-hot [batch_size, vocab_size] matrix (and the
    // gather may leave all_probs uninitialized). Set only by the speculative
    // (MTP) verify gather when it is locally provable that the fused greedy
    // top-1 token-only path handles the whole batch; every other path keeps
    // the legacy fully-materialized all_probs contract.
    bool all_probs_point_mass = false;

    mutable torch::Tensor cum_log_probs;  // shape: [batch_size]
    mutable torch::Tensor all_probs;      // shape: [batch_size, vocab_size]

    std::vector<at::Generator> generator;
};

struct SamplerOutput {
public:
    torch::Tensor                         token_ids;
    torch::Tensor                         cum_log_probs;
    torch::Tensor                         all_probs;
    torch::Tensor                         beam_index;
    torch::Tensor                         success;
    std::vector<std::optional<ErrorInfo>> processor_errors;
    // The draft distribution is a one-hot at token_ids and need not be
    // materialized as [batch, speculative_steps, vocab].
    bool token_ids_are_point_mass = false;
};

struct MergedOutput {
public:
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
};

}  // namespace rtp_llm
