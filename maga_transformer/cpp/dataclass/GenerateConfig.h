#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "maga_transformer/cpp/utils/StringUtil.h"
#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

// NOTE: The params in generate config should be splitted into two parts:
//       1. The params that can be different for a single sampler.
//       e.g. top_k, top_p, temperature, repetition_penalty, etc.
//       2. The params that must be the same for a single sampler.
//       e.g. beam_size, max_seq_len, etc.
//       For the second part, different samplers should be created for different params.
//       So they can not be batched together for now.

class GenerateConfig : public autil::legacy::Jsonizable {
public:
    int max_new_tokens     = 8192;
    int min_new_tokens     = 0;
    int num_validate_token = 0;  // for speculative decoding validation.

    int                  num_beams            = 1;
    int                  num_return_sequences = 1;
    int                  top_k                = 1;
    float                top_p                = 0.0;
    float                temperature          = 1.0;
    float                repetition_penalty   = 1.0;
    std::optional<int>   no_repeat_ngram_size;
    std::optional<int>   random_seed;
    std::optional<float> top_p_decay;
    std::optional<float> top_p_min;
    std::optional<int>   top_p_reset_ids;
    std::optional<std::string>   task_id;
    std::string          adapter_name = "";

    std::vector<int>    select_tokens_id;
    int                 calculate_loss       = 0;
    bool                return_logits        = false;
    bool                return_incremental   = false;
    bool                return_hidden_states = false;
    bool                is_streaming         = false;
    int                 timeout_ms           = -1;
    std::vector<std::vector<int>> stop_words_list;
    bool                sp_edit              = false;
    std::vector<int>    sp_advice_prompt_token_ids;

    bool top1() {
        return top_k == 1;
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GenerateConfig {"
                     << "max_new_tokens:" << max_new_tokens
                     << ", min_new_tokens:" << min_new_tokens << ", num_beams:" << num_beams
                     << ", num_return_sequences:" << num_return_sequences << ", calculate_loss:" << calculate_loss
                     << ", return_logits:" << return_logits << ", return_incremental: " << return_incremental
                     << ", return_hidden_states:" << return_hidden_states
                     << ", is_streaming:" << is_streaming
                     << ", timeout_ms:" << timeout_ms
                     << ", stop_words_list:" << vectorsToString(stop_words_list) << "}";
        return debug_string.str();
    }

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
#define JSONIZE(field) json.Jsonize(#field, field, field)
#define JSONIZE_OPTIONAL(field) if (field.has_value()) json.Jsonize(#field, field.value(), field.value())
        JSONIZE(max_new_tokens);
        JSONIZE(min_new_tokens);
        JSONIZE(num_validate_token);
        JSONIZE(num_beams);
        JSONIZE(num_return_sequences);
        JSONIZE(top_k);
        JSONIZE(top_p);
        JSONIZE(temperature);
        JSONIZE(repetition_penalty);
        JSONIZE_OPTIONAL(no_repeat_ngram_size);
        JSONIZE_OPTIONAL(random_seed);
        JSONIZE_OPTIONAL(top_p_decay);
        JSONIZE_OPTIONAL(top_p_min);
        JSONIZE_OPTIONAL(top_p_reset_ids);
        JSONIZE_OPTIONAL(task_id);
        JSONIZE(adapter_name);
        JSONIZE(select_tokens_id);
        JSONIZE(calculate_loss);
        JSONIZE(return_logits);
        JSONIZE(return_incremental);
        JSONIZE(return_hidden_states);
        JSONIZE(is_streaming);
        JSONIZE(timeout_ms);
        JSONIZE(stop_words_list);
        JSONIZE(sp_edit);
        JSONIZE(sp_advice_prompt_token_ids);
#undef JSONIZE
#undef JSONIZE_OPTIONAL
    }

};

}  // namespace rtp_llm
