#include "rtp_llm/cpp/api_server/openai/ApiDataType.h"

namespace rtp_llm {

using namespace autil::legacy;
using namespace autil::legacy::json;

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
#define JSONIZE(field) json.Jsonize(#field, field, field)

void FunctionCall::Jsonize(Jsonizable::JsonWrapper& json) {
    JSONIZE_OPTIONAL(name);
    JSONIZE_OPTIONAL(arguments);
}

void ToolCall::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("index", index, index);
    JSONIZE_OPTIONAL(id);
    json.Jsonize("type", type, type);
    json.Jsonize("function", function, function);
}

bool RoleEnum::contains(const std::string& role) {
    if (role == RoleEnum::user)
        return true;
    if (role == RoleEnum::assistant)
        return true;
    if (role == RoleEnum::system)
        return true;
    if (role == RoleEnum::function)
        return true;
    if (role == RoleEnum::tool)
        return true;
    if (role == RoleEnum::observation)
        return true;
    return false;
}

bool ContentPartTypeEnum::contains(const std::string& type) {
    if (type == ContentPartTypeEnum::text)
        return true;
    if (type == ContentPartTypeEnum::image_url)
        return true;
    if (type == ContentPartTypeEnum::video_url)
        return true;
    if (type == ContentPartTypeEnum::audio_url)
        return true;
    return false;
}

void ImageURL::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("url", url, url);
    JSONIZE_OPTIONAL(detail);
}

void AudioURL::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("url", url, url);
}

void ContentPart::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("type", type, type);
    if (ContentPartTypeEnum::contains(type) == false) {
        throw std::runtime_error("unknown content type");
    }

    JSONIZE_OPTIONAL(text);
    JSONIZE_OPTIONAL(image_url);
    JSONIZE_OPTIONAL(video_url);
    JSONIZE_OPTIONAL(audio_url);
}

void ChatMessage::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("role", role, role);
    if (RoleEnum::contains(role) == false) {
        throw std::runtime_error("unknown role");
    }

    try {
        std::string content_str;
        json.Jsonize("content", content_str, content_str);
        content = content_str;
    } catch (autil::legacy::ExceptionBase& e) {
        std::vector<ContentPart> content_vec;
        json.Jsonize("content", content_vec, content_vec);
        content = content_vec;
    }

    JSONIZE_OPTIONAL(reasoning_content);
    JSONIZE_OPTIONAL(function_call);
    JSONIZE_OPTIONAL(tool_calls);
}

void GPTFunctionDefinition::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("name", name, name);
    json.Jsonize("description", description, description);
    json.Jsonize("parameters", parameters, parameters);
    JSONIZE_OPTIONAL(name_for_model);
    JSONIZE_OPTIONAL(name_for_human);
    JSONIZE_OPTIONAL(description_for_model);
}

void ChatCompletionRequest::Jsonize(Jsonizable::JsonWrapper& json) {
    JSONIZE_OPTIONAL(model);
    json.Jsonize("messages", messages, messages);
    JSONIZE_OPTIONAL(functions);
    JSONIZE_OPTIONAL(temperature);
    JSONIZE_OPTIONAL(top_p);
    JSONIZE_OPTIONAL(max_tokens);

    std::variant<std::string, std::vector<std::string>> stop_;
    try {
        std::string stop1_;
        json.Jsonize("stop", stop1_);
        stop_ = stop1_;
        stop  = stop_;
    } catch (autil::legacy::ExceptionBase& e) {
        try {
            std::vector<std::string> stop2_;
            json.Jsonize("stop", stop2_);
            stop_ = stop2_;
            stop  = stop_;
        } catch (autil::legacy::ExceptionBase& e) {
            stop = std::nullopt;
        }
    }

    JSONIZE_OPTIONAL(stream);
    JSONIZE_OPTIONAL(user);
    JSONIZE_OPTIONAL(seed);
    JSONIZE_OPTIONAL(n);
    JSONIZE_OPTIONAL(logprobs);
    JSONIZE_OPTIONAL(top_logprobs);
    JSONIZE_OPTIONAL(extra_configs);
    JSONIZE_OPTIONAL(private_request);
    JSONIZE_OPTIONAL(trace_id);
    JSONIZE_OPTIONAL(chat_id);
    JSONIZE_OPTIONAL(template_key);
    JSONIZE_OPTIONAL(user_template);
    JSONIZE_OPTIONAL(source);
    JSONIZE_OPTIONAL(debug_info);
    JSONIZE_OPTIONAL(aux_info);
    JSONIZE_OPTIONAL(extend_fields);
}

void DebugInfo::Jsonize(Jsonizable::JsonWrapper& json) {
    JSONIZE(input_prompt);
    JSONIZE(input_ids);
    JSONIZE(input_urls);
    JSONIZE(tokenizer_info);
    JSONIZE(max_seq_len);
    JSONIZE(eos_token_id);
    JSONIZE(stop_word_ids_list);
    JSONIZE(stop_words_list);
    JSONIZE(renderer_info);
    JSONIZE(generate_config);
}

#undef JSONIZE
#undef JSONIZE_OPTIONAL

}  // namespace rtp_llm
