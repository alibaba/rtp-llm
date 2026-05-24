#pragma once

#include <torch/all.h>
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"

namespace rtp_llm {

class LogitsProcessorStates;
typedef std::shared_ptr<LogitsProcessorStates> LogitsProcessorStatesPtr;

// Stage 2 of the XGrammar verify-only MTP async plan: phase metadata that
// travels with the sampler input batch so logits processors (XGrammar in
// particular) can dispatch on which speculative-decoding phase produced the
// batch. Default is NORMAL_DECODE; MTP target-verify batches are tagged
// MTP_VERIFY by MtpBatchStreamProcessor::gatherSpecSamplerInput, draft
// sampling stays as DRAFT_SAMPLE on the few paths that route through a
// processor at all (the FastTopKSampler path bypasses processors today).
enum class LogitsProcessorPhase : uint8_t {
    NORMAL_DECODE = 0,
    MTP_VERIFY    = 1,
    DRAFT_SAMPLE  = 2,
};

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

    LogitsProcessorPhase phase = LogitsProcessorPhase::NORMAL_DECODE;

    // ─── MTP_VERIFY-only fields, populated by SpecGrammarVerifyHelper ───
    // Bitmask block produced by the helper for the score-batch:
    //   shape [total_streams * (propose_step + 1), bitmask_word_count] CUDA int32,
    //   bit=1 means allow. Rows for streams that opted out (or for offsets past
    //   each stream's grammar cap) are filled with allow-all (0xFFFFFFFF).
    torch::Tensor                 bitmask_gpu;
    // Per-stream grammar cap in [0, propose_step], shape [total_streams] CUDA int32.
    // The executor truncates verifier accept_len with min(accept_len, cap + 1).
    torch::Tensor                 grammar_cap_gpu;
    // Main stream must wait this event before consuming bitmask_gpu / grammar_cap_gpu.
    std::shared_ptr<torch::Event> bitmask_ready_event;
    // Number of speculative draft tokens per stream (P).
    int                           propose_step = 0;
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
