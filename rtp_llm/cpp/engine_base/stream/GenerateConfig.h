#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

// NOTE: The params in generate config should be splitted into two parts:
//       1. The params that can be different for a single sampler.
//       e.g. top_k, top_p, temperature, repetition_penalty, etc.
//       2. The params that must be the same for a single sampler.
//       e.g. beam_size, max_seq_len, etc.
//       For the second part, different samplers should be created for different params.
//       So they can not be batched together for now.

class GenerateConfig: public autil::legacy::Jsonizable {
public:
    int global_request_id  = -1;
    int max_new_tokens     = 8192;
    int min_new_tokens     = 0;
    int num_validate_token = 0;  // for speculative decoding validation.

    int                        num_beams = 1;
    std::vector<int>           variable_num_beams;
    int                        num_return_sequences = 1;
    int                        top_k                = 0;
    float                      top_p                = 1.0;
    float                      temperature          = 1.0;
    float                      repetition_penalty   = 1.0;
    float                      presence_penalty     = 0.0;
    float                      frequency_penalty    = 0.0;
    std::optional<int>         no_repeat_ngram_size;
    std::optional<int>         random_seed;
    std::optional<float>       top_p_decay;
    std::optional<float>       top_p_min;
    std::optional<int>         top_p_reset_ids;
    std::optional<std::string> task_id;
    std::string                adapter_name = "";
    std::vector<std::string>   adapter_names;

    std::vector<int>              select_tokens_id;
    std::vector<std::string>      select_tokens_str;
    int                           calculate_loss           = 0;
    int                           hidden_states_cut_dim    = 0;
    bool                          return_logits            = false;
    bool                          return_cum_log_probs     = false;
    bool                          return_incremental       = false;
    bool                          return_hidden_states     = false;
    bool                          return_all_hidden_states = false;
    bool                          normalized_hidden_states = false;
    bool                          return_output_ids        = false;
    bool                          return_input_ids         = false;
    bool                          is_streaming             = false;
    int                           timeout_ms               = -1;
    bool                          sp_edit                  = false;
    bool                          force_disable_sp_run     = false;
    bool                          force_sp_accept          = false;
    bool                          return_all_probs         = false;
    bool                          return_softmax_probs     = false;
    std::vector<std::vector<int>> stop_words_list;
    std::vector<std::string>      stop_words_str;
    bool                          print_stop_words = false;
    std::string                   sp_advice_prompt;
    std::vector<int>              sp_advice_prompt_token_ids;

    bool do_sample             = true;
    bool can_use_pd_separation = true;
    bool pd_separation         = false;

    bool             in_think_mode       = false;
    int              max_thinking_tokens = 0;
    std::vector<int> end_think_token_ids;
    bool             gen_timeline              = false;
    int              profile_step              = 3;
    bool             ignore_eos                = false;
    bool             reuse_cache               = true;
    bool             enable_3fs                = true;
    bool             enable_memory_block_cache = true;
    std::string      trace_id;
    std::string      pd_sepration_unique_key;

    bool top1() {
        return top_k == 1;
    }

    std::vector<RoleAddr> role_addrs;
    int64_t               inter_request_id = -1;  // used for master scheduling

    int maxNumBeams() {
        if (variable_num_beams.size() > 0) {
            return *std::max_element(variable_num_beams.begin(), variable_num_beams.end());
        } else {
            return num_beams;
        }
    }

    bool hasNumBeams() {
        return maxNumBeams() > 1;
    }

    void addSpecialTokens(const rtp_llm::SpecialTokens& special_tokens) {
        for (const auto& vec : special_tokens.stop_words_id_list_) {
            std::vector<int> tmpVec;
            for (int64_t val : vec) {
                tmpVec.push_back(static_cast<int>(val));
            }
            stop_words_list.push_back(tmpVec);
        }
        const auto& vec = special_tokens.stop_words_str_list_;
        stop_words_str.insert(stop_words_str.begin(), vec.begin(), vec.end());
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GenerateConfig {"
                     << "max_new_tokens:" << max_new_tokens << ", min_new_tokens:" << min_new_tokens
                     << ", num_beams:" << num_beams << ", variable_num_beams:" << vectorToString(variable_num_beams)
                     << ", num_return_sequences:" << num_return_sequences << ", calculate_loss:" << calculate_loss
                     << ", return_logits:" << return_logits << ", return_incremental: " << return_incremental
                     << ", return_hidden_states:" << return_hidden_states
                     << ", return_all_hidden_states:" << return_all_hidden_states
                     << ", hidden_states_cut_dim:" << hidden_states_cut_dim
                     << ", normalized_hidden_states:" << normalized_hidden_states
                     << ", return_output_ids:" << return_output_ids << ", return_input_ids:" << return_input_ids
                     << ", is_streaming:" << is_streaming << ", timeout_ms:" << timeout_ms << ", top_k:" << top_k
                     << ", top_p:" << top_p << ", force_disable_sp_run: " << force_disable_sp_run
                     << ", force_sp_accept: " << force_sp_accept << ", return_all_probs: " << return_all_probs
                     << ", stop_words_list:" << vectorsToString(stop_words_list)
                     << ", can_use_pd_separation: " << can_use_pd_separation << ", pd_separation: " << pd_separation
                     << ", in_think_mode: " << in_think_mode << ", max_thinking_tokens: " << max_thinking_tokens
                     << ", end_think_token_ids: " << vectorToString(end_think_token_ids)
                     << ", gen_timeline: " << gen_timeline << ", profile_step: " << profile_step
                     << ", reuse_cache: " << reuse_cache << ", enable_3fs: " << enable_3fs
                     << ", enable_memory_block_cache: " << enable_memory_block_cache
                     << ", pd_sepration_unique_key: " << pd_sepration_unique_key << "}";
        return debug_string.str();
    }

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
#define JSONIZE(field) json.Jsonize(#field, field, field)
// used for de-serialization
#define JSONIZE_OPTIONAL(field)                                                                                        \
    try {                                                                                                              \
        using Type = decltype(field)::value_type;                                                                      \
        Type field##Tmp;                                                                                               \
        json.Jsonize(#field, field##Tmp);                                                                              \
        field = field##Tmp;                                                                                            \
    } catch (autil::legacy::ExceptionBase & e) {                                                                       \
        if (field.has_value() == false) {                                                                              \
            field = std::nullopt;                                                                                      \
        }                                                                                                              \
    }
        JSONIZE(max_new_tokens);
        JSONIZE(min_new_tokens);
        JSONIZE(num_validate_token);
        JSONIZE(num_beams);
        JSONIZE(variable_num_beams);
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
        try {
            std::string adapter_name_;
            json.Jsonize("adapter_name", adapter_name_);
            adapter_name = adapter_name_;
        } catch (autil::legacy::ExceptionBase& e) {
            try {
                std::vector<std::string> adapter_names_;
                json.Jsonize("adapter_name", adapter_names_);
                adapter_names = adapter_names_;
            } catch (autil::legacy::ExceptionBase& e) {
                // noop
            }
        }
        JSONIZE(select_tokens_id);
        JSONIZE(select_tokens_str);
        JSONIZE(calculate_loss);
        JSONIZE(return_logits);
        JSONIZE(return_incremental);
        JSONIZE(return_hidden_states);
        JSONIZE(return_all_hidden_states);
        JSONIZE(hidden_states_cut_dim);
        JSONIZE(normalized_hidden_states);
        JSONIZE(return_output_ids);
        JSONIZE(return_input_ids);
        JSONIZE(is_streaming);
        JSONIZE(timeout_ms);
        JSONIZE(stop_words_list);
        JSONIZE(stop_words_str);
        JSONIZE(print_stop_words);
        JSONIZE(sp_edit);
        JSONIZE(force_disable_sp_run);
        JSONIZE(force_sp_accept);
        JSONIZE(return_all_probs);
        JSONIZE(sp_advice_prompt);
        JSONIZE(sp_advice_prompt_token_ids);
        JSONIZE(in_think_mode);
        JSONIZE(max_thinking_tokens);
        JSONIZE(end_think_token_ids);
        JSONIZE(gen_timeline);
        JSONIZE(profile_step);
        JSONIZE(reuse_cache);
        JSONIZE(enable_3fs);
        JSONIZE(enable_memory_block_cache);
        JSONIZE(pd_sepration_unique_key);
#undef JSONIZE
#undef JSONIZE_OPTIONAL
    }
};

}  // namespace rtp_llm
