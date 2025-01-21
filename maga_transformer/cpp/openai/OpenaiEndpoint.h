#pragma once

#include <memory>

#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "maga_transformer/cpp/openai/ChatRender.h"
#include "maga_transformer/cpp/tokenizer/Tokenizer.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace rtp_llm {

class OpenaiEndpoint {
public:
    OpenaiEndpoint(const std::shared_ptr<Tokenizer>&  tokenizer,
                   const std::shared_ptr<ChatRender>& chat_render,
                   const ft::GptInitParameter&        params);
    virtual ~OpenaiEndpoint() {}

public:
    // `virtual` for test
    virtual std::shared_ptr<GenerateConfig> extract_generation_config(const ChatCompletionRequest& req);
    std::shared_ptr<ChatRender>             getChatRender() {
        return chat_render_;
    }
    std::string getDebugInfo(const ChatCompletionRequest& chat_request,
                             const RenderedInputs& rendered_input);

private:
    int                           max_seq_len_;
    int                           eos_token_id_;
    std::vector<std::vector<int>> stop_word_ids_list_;
    std::vector<std::string>      stop_words_list_;

    std::shared_ptr<Tokenizer>  tokenizer_;
    std::shared_ptr<ChatRender> chat_render_;
    ft::GptInitParameter        model_config_;
};

}  // namespace rtp_llm
