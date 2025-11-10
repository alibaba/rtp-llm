#pragma once

#include <variant>

#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

class FunctionCall: public autil::legacy::Jsonizable {
public:
    void                       Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    std::optional<std::string> name;
    std::optional<std::string> arguments;
};

class ToolCall: public autil::legacy::Jsonizable {
public:
    void                       Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    int                        index;
    std::optional<std::string> id;
    std::string                type;
    FunctionCall               function;
};

class RoleEnum {
public:
    static bool                  contains(const std::string& role);
    static constexpr const char* user        = "user";
    static constexpr const char* assistant   = "assistant";
    static constexpr const char* system      = "system";
    static constexpr const char* function    = "function";
    static constexpr const char* tool        = "tool";
    static constexpr const char* observation = "observation";
};

class ContentPartTypeEnum {
public:
    static bool                  contains(const std::string& type);
    static constexpr const char* text      = "text";
    static constexpr const char* image_url = "image_url";
    static constexpr const char* video_url = "video_url";
    static constexpr const char* audio_url = "audio_url";
};

class ImageURL: public autil::legacy::Jsonizable {
public:
    void                       Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    std::string                url;
    std::optional<std::string> detail = "auto";
};

class AudioURL: public autil::legacy::Jsonizable {
public:
    void        Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    std::string url;
};

class ContentPart: public autil::legacy::Jsonizable {
public:
    void                       Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    std::string                type;
    std::optional<std::string> text;
    std::optional<ImageURL>    image_url;
    std::optional<ImageURL>    video_url;
    std::optional<AudioURL>    audio_url;
};

class ChatMessage: public autil::legacy::Jsonizable {
public:
    void                                                Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    std::string                                         role;
    std::variant<std::string, std::vector<ContentPart>> content;
    std::optional<std::string>                          reasoning_content;
    std::optional<FunctionCall>                         function_call;
    std::optional<std::vector<ToolCall>>                tool_calls;
};

class GPTFunctionDefinition: public autil::legacy::Jsonizable {
public:
    void                               Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    std::string                        name;
    std::string                        description;
    std::map<std::string, std::string> parameters;

    // These parameters are for qwen style function.
    std::optional<std::string> name_for_model;
    std::optional<std::string> name_for_human;
    std::optional<std::string> description_for_model;
};

class ChatCompletionRequest: public autil::legacy::Jsonizable {
public:
    void                                              Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    std::optional<std::string>                        model;
    std::vector<ChatMessage>                          messages;
    std::optional<std::vector<GPTFunctionDefinition>> functions;
    std::optional<float>                              temperature = 0.7;
    std::optional<float>                              top_p       = 1.0;
    std::optional<int>                                max_tokens;
    std::optional<std::variant<std::string, std::vector<std::string>>> stop;
    std::optional<bool>                                                stream;
    std::optional<std::string>                                         user;
    std::optional<int>                                                 seed;
    std::optional<int>                                                 n;
    std::optional<bool>                                                logprobs;
    std::optional<int>                                                 top_logprobs;

    // These params are hacked for our framework, not standard.
    std::optional<GenerateConfig>                     extra_configs;
    std::optional<bool>                               private_request;
    std::optional<std::string>                        trace_id;
    std::optional<std::string>                        chat_id;
    std::optional<std::string>                        template_key;
    std::optional<std::string>                        user_template;
    std::optional<std::string>                        source;
    std::optional<bool>                               debug_info;
    std::optional<bool>                               aux_info;
    std::optional<std::map<std::string, std::string>> extend_fields;
};

class DebugInfo: public autil::legacy::Jsonizable {
public:
    void                          Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    std::string                   input_prompt;
    std::vector<int>              input_ids;
    std::vector<std::string>      input_urls;
    std::string                   tokenizer_info;
    int                           max_seq_len;
    int                           eos_token_id;
    std::vector<std::vector<int>> stop_word_ids_list;
    std::vector<std::string>      stop_words_list;
    std::string                   renderer_info;
    GenerateConfig                generate_config;
};

}  // namespace rtp_llm
