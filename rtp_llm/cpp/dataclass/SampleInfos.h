#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/DFAUtil.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

class LogitsProcessorStates;
typedef std::shared_ptr<LogitsProcessorStates> LogitsProcessorStatesPtr;

struct SamplerInitParams {
    rtp_llm::DeviceBase* device;
    int32_t eos_id;
    size_t max_batch_size = 256; // default max batch size
};

struct SamplerInputs {
public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "SamplerInputs { "
                     << "batch_size: " << batch_size
                     << ", step: " << step
                     << ", logits: " << logits->debugStringWithData<int32_t>()
                     << ", token_ids: " << token_ids->debugStringWithData<int32_t>()
                     << ", input_lengths: " << input_lengths->debugStringWithData<int32_t>()
                     << ", sequence_lengths: " << sequence_lengths->debugStringWithData<int32_t>()
                     << ", cum_log_probs: " << cum_log_probs->debugStringWithData<float>()
                     << "}";
        return debug_string.str();
    }

public:
    rtp_llm::BufferPtr logits;            // shape: [batch_size * num_beams, vocab_size]
    mutable rtp_llm::BufferPtr token_ids; // shape: [batch_size * num_beams, max_length]
    rtp_llm::BufferPtr input_lengths;     // shape: [batch_size]
    // shape: [decoder_batch_size]
    rtp_llm::BufferPtr sequence_lengths;
    LogitsProcessorStatesPtr logits_processor_states_ptr;

    size_t    vocab_size;
    size_t    step;                  // typically largest sequence length in the batch

    size_t    batch_size;
    rtp_llm::BufferPtr num_beams;           // shape: [batch_size]
    rtp_llm::BufferPtr top_k;               // shape: [batch_size]
    rtp_llm::BufferPtr top_p;               // shape: [batch_size]
    rtp_llm::BufferPtr temperature;         // shape: [batch_size]
    rtp_llm::BufferPtr random_seeds;        // shape: [batch_size]
    rtp_llm::BufferPtr repetition_penalty;  // shape: [batch_size]
    rtp_llm::BufferPtr min_lengths;         // shape: [batch_size]
    rtp_llm::BufferPtr no_repeat_ngram_size;    // shape: [batch_size]

    mutable rtp_llm::BufferPtr cum_log_probs; // shape: [batch_size * num_beams]
    mutable rtp_llm::BufferPtr all_probs;     // shape: [batch_size * num_beams, vocab_size]

    // for beam search
    rtp_llm::BufferPtr beam_search_sequence_lengths;
    rtp_llm::BufferPtr beam_index;
};

struct SamplerOutput {
public:
    rtp_llm::BufferPtr token_ids;
    rtp_llm::BufferPtr cum_log_probs;
    rtp_llm::BufferPtr all_probs;
    rtp_llm::BufferPtr beam_index;
    rtp_llm::BufferPtr success;
};


}  // namespace rtp_llm
