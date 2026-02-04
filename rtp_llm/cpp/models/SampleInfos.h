#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"

namespace rtp_llm {

class LogitsProcessorStates;
typedef std::shared_ptr<LogitsProcessorStates> LogitsProcessorStatesPtr;

struct SamplerInitParams {
    rtp_llm::DeviceBase* device;
};

struct SamplerInputs {
public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "SamplerInputs { "
                     << "batch_size: " << batch_size << ", step: " << step
                     << ", logits: " << (logits ? logits->debugStringWithData<int32_t>() : "null")
                     << ", token_ids: " << (token_ids ? token_ids->debugStringWithData<int32_t>() : "null")
                     << ", input_lengths: " << (input_lengths ? input_lengths->debugStringWithData<int32_t>() : "null")
                     << ", sequence_lengths: "
                     << (sequence_lengths ? sequence_lengths->debugStringWithData<int32_t>() : "null")
                     << ", cum_log_probs: " << (cum_log_probs ? cum_log_probs->debugStringWithData<float>() : "null")
                     << "}";
        return debug_string.str();
    }

public:
    rtp_llm::BufferPtr         logits;         // shape: [batch_size, vocab_size]
    mutable rtp_llm::BufferPtr token_ids;      // shape: [batch_size, max_length]
    rtp_llm::BufferPtr         input_lengths;  // shape: [batch_size]
    // shape: [decoder_batch_size]
    rtp_llm::BufferPtr       sequence_lengths;
    LogitsProcessorStatesPtr logits_processor_states_ptr;

    size_t vocab_size;
    size_t step;  // typically largest sequence length in the batch

    size_t             batch_size;            // sum of all num_beams_in of all streams
    size_t             batch_size_out;        // sum of all num_beams_out of all streams
    rtp_llm::BufferPtr num_beams_in;          // shape: [batch_size]
    rtp_llm::BufferPtr num_beams_out;         // shape: [batch_size]
    rtp_llm::BufferPtr top_k;                 // shape: [batch_size]
    rtp_llm::BufferPtr top_p;                 // shape: [batch_size]
    rtp_llm::BufferPtr temperature;           // shape: [batch_size]
    rtp_llm::BufferPtr repetition_penalty;    // shape: [batch_size]
    rtp_llm::BufferPtr presence_penalty;      // shape: [batch_size]
    rtp_llm::BufferPtr frequency_penalty;     // shape: [batch_size]
    rtp_llm::BufferPtr no_repeat_ngram_size;  // shape: [batch_size]
    rtp_llm::BufferPtr do_sample;             // shape: [batch_size]
    rtp_llm::BufferPtr finished_mask;         // shape: [batch_size]

    bool return_original_all_probs = false;

    mutable rtp_llm::BufferPtr cum_log_probs;  // shape: [batch_size]
    mutable rtp_llm::BufferPtr all_probs;      // shape: [batch_size, vocab_size]

    std::vector<at::Generator> generator;
};

struct SamplerOutput {
public:
    rtp_llm::BufferPtr token_ids;
    rtp_llm::BufferPtr cum_log_probs;
    rtp_llm::BufferPtr all_probs;
    rtp_llm::BufferPtr beam_index;
    rtp_llm::BufferPtr success;
};

struct MergedOutput {
public:
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
};

}  // namespace rtp_llm
