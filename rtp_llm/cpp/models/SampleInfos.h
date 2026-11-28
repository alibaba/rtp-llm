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
class RecordedBatch;
typedef std::shared_ptr<LogitsProcessorStates> LogitsProcessorStatesPtr;

struct SamplerInitParams {
    // max_batch_size is an initial capacity. Set fixed_max_batch_size=false when callers can legally
    // fan out beyond that capacity, e.g. num_return_sequences or variable beam requests. When
    // max_batch_size == 0, fixed_max_batch_size is ignored and buffers grow dynamically.
    size_t max_batch_size       = 0;
    bool   fixed_max_batch_size = true;
};

enum class SamplerProbabilityOutput {
    NONE,
    DENSE,         // Materialize [batch, vocab] probabilities.
    DISTRIBUTION,  // Consumers also accept point-mass tokens or probabilities with candidate IDs.
};

struct SamplingRowMapping {
    // Stable selected logical rows (int64) and their inverse (int32, -1 for omitted rows).
    // Logits already use the selected rows; request metadata still uses all logical rows.
    // Single-beam only. Every request must retain a row so its dispatch mode is represented.
    torch::Tensor packed_to_dense;
    torch::Tensor dense_to_packed;
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
    mutable torch::Tensor token_ids;      // [batch_size, step + 1] history; undefined when no history is consumed.
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

    // Logits are disposable for this round; no consumer requires softmax
    // probabilities written back into their storage.
    bool skip_logits_probs_restore = false;
    int  force_topk                = 0;  // DSpARK fixed-width sampling; 0 keeps full vocabulary.

    // Legacy PP callers request dense probabilities by providing this tensor.
    mutable torch::Tensor all_probs;
    bool all_probs_point_mass = false;
    SamplerProbabilityOutput probability_output = SamplerProbabilityOutput::NONE;
    mutable torch::Tensor    cum_log_probs;  // shape: [batch_size]

    // Retain the processed sampling distribution for generation logprobs.
    bool return_logprobs = false;

    std::vector<at::Generator>        generator;
    std::optional<SamplingRowMapping> row_mapping;
    torch::Tensor raw_logprobs_logits;
    torch::Tensor raw_logprobs_row_indices;
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
    bool          token_ids_are_point_mass = false;
    torch::Tensor raw_logprobs_logits;
    torch::Tensor raw_logprobs_row_indices;
    torch::Tensor all_probs_indices;  // original vocabulary IDs when all_probs is compact
    // Normalized probabilities after temperature, penalties, masks and top-k/top-p.
    // Uses sampler-input row coordinates (including tiled rows); beam_index maps outputs back.
    torch::Tensor sampling_probs;
    // DSpARK per-position acceptance-rate prediction from the draft model
    // confidence head: [batch, propose_step] FP32 in [0, 1].  Undefined when
    // the sampler ran with confidence disabled. Enabled V4.1 requires its head.
    // Column i is the predicted probability that draft position i survives
    // target verification, conditioned on all earlier positions surviving.
    torch::Tensor confidence_rates;
    // Optional CUDA int32 [B] valid draft prefix lengths. Undefined keeps the
    // full-width rejection path. The physical draft/target strides stay gamma
    // and gamma+1; confidence never changes the sampled draft distribution.
    torch::Tensor valid_draft_lengths;
};

struct SamplingLogprobs {
    torch::Tensor token_logprobs;         // FP32 [rows]
    torch::Tensor top_logprob_token_ids;  // INT32 [rows, min(requested_k, support_width)]
    torch::Tensor top_logprobs;           // FP32, same shape as top_logprob_token_ids
};

// Read only the emitted token and requested top entries from a normalized distribution.
// probs/candidate_ids are [rows, width]; emitted IDs and optional point-mass IDs are
// [rows] on the same device. Candidate IDs are unique vocabulary IDs per row.
// Point-mass rows do not read probs. Zero probability is reported as -inf.
SamplingLogprobs computeSamplingLogprobs(const torch::Tensor& probs,
                                         const torch::Tensor& candidate_ids,
                                         const torch::Tensor& point_mass_token_ids,
                                         const torch::Tensor& emitted_token_ids,
                                         int64_t              max_top_logprobs);

struct MergedOutput {
public:
    GptModelOutputs                model_output;
    SamplerOutput                  sampler_output;
    std::shared_ptr<RecordedBatch> recorded_batch;
};

}  // namespace rtp_llm
