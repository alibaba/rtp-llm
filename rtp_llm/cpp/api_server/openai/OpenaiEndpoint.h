#pragma once

#include <memory>

#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/api_server/openai/ChatRender.h"
#include "rtp_llm/cpp/api_server/tokenizer/Tokenizer.h"

namespace th = torch;

namespace rtp_llm {

class OpenaiEndpoint {
public:
    OpenaiEndpoint(const std::shared_ptr<Tokenizer>&  tokenizer,
                   const std::shared_ptr<ChatRender>& chat_render,
                   const ModelConfig&                 model_config);
    virtual ~OpenaiEndpoint() {}

public:
    // `virtual` for test
    virtual std::shared_ptr<GenerateConfig> extract_generation_config(const ChatCompletionRequest& req);
    std::shared_ptr<ChatRender>             getChatRender() {
        return chat_render_;
    }
    std::string getDebugInfo(const ChatCompletionRequest& chat_request, const RenderedInputs& rendered_input);

private:
    int                           max_seq_len_;
    int                           eos_token_id_;
    std::vector<std::vector<int>> stop_word_ids_list_;
    std::vector<std::string>      stop_words_list_;

    std::shared_ptr<Tokenizer>  tokenizer_;
    std::shared_ptr<ChatRender> chat_render_;
    ModelConfig                 model_config_;
};

}  // namespace rtp_llm
