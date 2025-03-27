#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/utils/DFAUtil.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class BaseLogitsProcessor;
typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

struct SamplerInitParams {
    ft::DeviceBase* device;
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
    ft::BufferPtr logits;            // shape: [batch_size * num_beams, vocab_size]
    mutable ft::BufferPtr token_ids; // shape: [batch_size * num_beams, max_length]
    ft::BufferPtr input_lengths;     // shape: [batch_size]
    // shape: [decoder_batch_size]
    ft::BufferPtr sequence_lengths;
    std::vector<BaseLogitsProcessorPtr> grammars;

    size_t    vocab_size;
    size_t    step;                  // typically largest sequence length in the batch

    size_t    batch_size;
    ft::BufferPtr num_beams;           // shape: [batch_size]
    ft::BufferPtr top_k;               // shape: [batch_size]
    ft::BufferPtr top_p;               // shape: [batch_size]
    ft::BufferPtr temperature;         // shape: [batch_size]
    ft::BufferPtr random_seeds;        // shape: [batch_size]
    ft::BufferPtr repetition_penalty;  // shape: [batch_size]
    ft::BufferPtr min_lengths;         // shape: [batch_size]
    ft::BufferPtr no_repeat_ngram_size;    // shape: [batch_size]

    mutable ft::BufferPtr cum_log_probs; // shape: [batch_size * num_beams]
    mutable ft::BufferPtr all_probs;     // shape: [batch_size * num_beams, vocab_size]

    // for beam search
    ft::BufferPtr beam_search_sequence_lengths;
    ft::BufferPtr beam_index;
};

struct SamplerOutput {
public:
    ft::BufferPtr token_ids;
    ft::BufferPtr cum_log_probs;
    ft::BufferPtr all_probs;
    ft::BufferPtr beam_index;
    ft::BufferPtr success;
};


}  // namespace rtp_llm
