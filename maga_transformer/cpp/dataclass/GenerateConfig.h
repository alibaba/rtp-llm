#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "maga_transformer/cpp/utils/StringUtil.h"
#include "maga_transformer/cpp/tokenizer/Tokenizer.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "autil/legacy/jsonizable.h"

namespace ft = fastertransformer;

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

    std::vector<int>         select_tokens_id;
    std::vector<std::string> select_tokens_str;
    int                 calculate_loss       = 0;
    bool                return_logits        = false;
    bool                return_incremental   = false;
    bool                return_hidden_states = false;
    bool                return_output_ids    = false;
    bool                return_input_ids    = false;
    bool                is_streaming         = false;
    int                 timeout_ms           = -1;
    bool                sp_edit              = false;
    bool                force_disable_sp_run = false;
    bool                return_all_probs     = false;
    bool                return_softmax_probs = false;
    std::vector<std::vector<int>> stop_words_list;
    std::vector<std::string>      stop_words_str;
    bool                          print_stop_words = false;
    std::string         sp_advice_prompt;
    std::vector<int>    sp_advice_prompt_token_ids;

    bool can_use_pd_separation = true;
    bool pd_separation  = false;

    bool top1() {
        return top_k == 1;
    }

    void addSpecialTokens(const ft::SpecialTokens& special_tokens) {
        for (const auto& vec : special_tokens.stop_words_id_list_) {
            std::vector<int> tmpVec;
            for (int64_t val: vec) {
                tmpVec.push_back(static_cast<int>(val));
            }
            stop_words_list.push_back(tmpVec);
        }
        const auto& vec = special_tokens.stop_words_str_list_;
        stop_words_str.insert(stop_words_str.begin(), vec.begin(), vec.end());
    }

    void convertSelectTokens(int vocab_size, std::shared_ptr<Tokenizer> tokenizer) {
        for (const auto& token_str: select_tokens_str) {
            auto vec = tokenizer->encode(token_str);
            select_tokens_id.insert(select_tokens_id.begin(), vec.begin(), vec.end());
        }

        auto areTokensValid = [](const std::vector<int>& select_tokens_id, int vocab_size) {
            return std::all_of(select_tokens_id.begin(), select_tokens_id.end(), [vocab_size](int token_id) {
                return token_id < vocab_size && token_id >= 0;
            });
        };
        if (!areTokensValid(select_tokens_id, vocab_size)) {
            throw std::runtime_error("token_id should be less than vocab_size");
        }
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GenerateConfig {"
                     << "max_new_tokens:" << max_new_tokens
                     << ", min_new_tokens:" << min_new_tokens << ", num_beams:" << num_beams
                     << ", num_return_sequences:" << num_return_sequences << ", calculate_loss:" << calculate_loss
                     << ", return_logits:" << return_logits << ", return_incremental: " << return_incremental
                     << ", return_hidden_states:" << return_hidden_states
                     << ", return_output_ids:" << return_output_ids
                     << ", return_input_ids:" << return_input_ids
                     << ", is_streaming:" << is_streaming
                     << ", timeout_ms:" << timeout_ms
                     << ", top_k:" << top_k
                     << ", top_p:" << top_p
                     << ", force_disable_sp_run: " << force_disable_sp_run
                     << ", return_all_probs: " << return_all_probs
                     << ", stop_words_list:" << vectorsToString(stop_words_list)
                     << ", can_use_pd_separation: " << can_use_pd_separation
                     << ", pd_separation: " << pd_separation
                     << "}";
        return debug_string.str();
    }

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
#define JSONIZE(field) json.Jsonize(#field, field, field)
// used for de-serialization
#define JSONIZE_OPTIONAL(field) try { \
                                    using Type = decltype(field)::value_type; \
                                    Type field##Tmp; \
                                    json.Jsonize(#field, field##Tmp); \
                                    field = field##Tmp; \
                                } catch (autil::legacy::ExceptionBase &e) { \
                                    if (field.has_value() == false) { \
                                        field = std::nullopt; \
                                    } \
                                }
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
        JSONIZE(select_tokens_str);
        JSONIZE(calculate_loss);
        JSONIZE(return_logits);
        JSONIZE(return_incremental);
        JSONIZE(return_hidden_states);
        JSONIZE(return_output_ids);
        JSONIZE(return_input_ids);
        JSONIZE(is_streaming);
        JSONIZE(timeout_ms);
        JSONIZE(stop_words_list);
        JSONIZE(stop_words_str);
        JSONIZE(print_stop_words);
        JSONIZE(sp_edit);
        JSONIZE(force_disable_sp_run);
        JSONIZE(return_all_probs);
        JSONIZE(sp_advice_prompt);
        JSONIZE(sp_advice_prompt_token_ids);
#undef JSONIZE
#undef JSONIZE_OPTIONAL
    }

};

}  // namespace rtp_llm
