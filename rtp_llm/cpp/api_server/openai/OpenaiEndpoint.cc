#include "autil/StringUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/api_server/openai/OpenaiEndpoint.h"

namespace rtp_llm {

OpenaiEndpoint::OpenaiEndpoint(const std::shared_ptr<Tokenizer>&  tokenizer,
                               const std::shared_ptr<ChatRender>& chat_render,
                               const ModelConfig&                 model_config):
    tokenizer_(tokenizer), chat_render_(chat_render), model_config_(model_config) {

    max_seq_len_ = model_config_.max_seq_len;

    std::optional<int> res;
    if (tokenizer_ && tokenizer_->isPreTrainedTokenizer()) {
        res = tokenizer_->getEosTokenId();
    }
    eos_token_id_ = res.value_or(model_config_.special_tokens.eos_token_id);

    for (const auto& vec : model_config_.special_tokens.stop_words_id_list) {
        std::vector<int> tmpVec;
        for (int64_t val : vec) {
            tmpVec.push_back(static_cast<int>(val));
        }
        stop_word_ids_list_.push_back(tmpVec);
    }

    if (chat_render_) {
        auto extra_stop_word_ids_list = chat_render_->get_all_extra_stop_word_ids_list();
        stop_word_ids_list_.insert(
            stop_word_ids_list_.begin(), extra_stop_word_ids_list.begin(), extra_stop_word_ids_list.end());
    } else {
        RTP_LLM_LOG_WARNING("chat render is null");
    }

    for (const auto& stop_word_ids : stop_word_ids_list_) {
        if (tokenizer_) {
            auto word = tokenizer_->decode(stop_word_ids);
            if (!word.empty()) {
                stop_words_list_.push_back(word);
            }
        }
    }
    RTP_LLM_LOG_INFO("use stop_words_list [%s]", autil::StringUtil::join(stop_words_list_, ",").c_str());
}

std::shared_ptr<GenerateConfig> OpenaiEndpoint::extract_generation_config(const ChatCompletionRequest& req) {
    GenerateConfig config = req.extra_configs.value_or(GenerateConfig());
    config.is_streaming   = true;
    if (req.temperature.has_value()) {
        config.temperature = req.temperature.value();
    }
    if (req.top_p.has_value()) {
        config.top_p = req.top_p.value();
    }
    if (req.max_tokens.has_value()) {
        config.max_new_tokens = req.max_tokens.value();
    }
    config.num_return_sequences = req.n.value_or(1);
    std::vector<std::string> request_stop_words_list;
    if (req.stop.has_value()) {
        auto stop = req.stop.value();
        if (std::holds_alternative<std::string>(stop)) {
            auto stop_str = std::get<std::string>(stop);
            request_stop_words_list.push_back(stop_str);
        }
        if (std::holds_alternative<std::vector<std::string>>(stop)) {
            auto stop_vec = std::get<std::vector<std::string>>(stop);
            request_stop_words_list.insert(request_stop_words_list.begin(), stop_vec.begin(), stop_vec.end());
        }
    }
    config.stop_words_str.insert(
        config.stop_words_str.begin(), request_stop_words_list.begin(), request_stop_words_list.end());
    config.stop_words_str.insert(config.stop_words_str.begin(), stop_words_list_.begin(), stop_words_list_.end());
    config.stop_words_list.insert(
        config.stop_words_list.begin(), stop_word_ids_list_.begin(), stop_word_ids_list_.end());
    auto request_stop_words_list_ids = chat_render_->tokenize_words(request_stop_words_list);
    config.stop_words_list.insert(
        config.stop_words_list.begin(), request_stop_words_list_ids.begin(), request_stop_words_list_ids.end());
    // if (req.chat_id.has_value()) {
    //     config.chat_id = req.chat_id.value();
    // }
    if (req.seed.has_value()) {
        config.random_seed = req.seed.value();
    }
    if (req.logprobs.has_value()) {
        config.return_all_probs = req.logprobs.value();
    }
    config.addSpecialTokens(model_config_.special_tokens);

    auto select_tokens_id = tokenizer_->convertSelectTokens(config.select_tokens_str, model_config_.vocab_size);
    config.select_tokens_id.insert(config.select_tokens_id.begin(), select_tokens_id.begin(), select_tokens_id.end());
    if (config.sp_advice_prompt.empty() == false) {
        config.sp_advice_prompt_token_ids = tokenizer_->encode(config.sp_advice_prompt);
    }
    return std::make_shared<GenerateConfig>(config);
}

std::string OpenaiEndpoint::getDebugInfo(const ChatCompletionRequest& chat_request,
                                         const RenderedInputs&        rendered_input) {
    std::vector<std::string> input_urls;
    const auto&              mm_inputs = rendered_input.multimodal_inputs;
    std::transform(mm_inputs.begin(), mm_inputs.end(), std::back_inserter(input_urls), [](const auto& mm_input) {
        return mm_input.url;
    });

    std::string prompt;
    if (rendered_input.rendered_prompt.empty()) {
        prompt = tokenizer_->decode(rendered_input.input_ids);
    } else {
        prompt = rendered_input.rendered_prompt;
    }

    DebugInfo debug_info;
    debug_info.input_prompt       = prompt;
    debug_info.input_ids          = rendered_input.input_ids;
    debug_info.input_urls         = input_urls;
    debug_info.tokenizer_info     = tokenizer_->toString();
    debug_info.max_seq_len        = max_seq_len_;
    debug_info.eos_token_id       = eos_token_id_;
    debug_info.stop_word_ids_list = stop_word_ids_list_;
    debug_info.stop_words_list    = stop_words_list_;
    debug_info.renderer_info      = chat_render_->toString();
    debug_info.generate_config    = *(extract_generation_config(chat_request));

    return ToJsonString(debug_info, true);
}

}  // namespace rtp_llm
