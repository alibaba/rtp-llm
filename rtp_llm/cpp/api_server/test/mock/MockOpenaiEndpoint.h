#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/openai/OpenaiEndpoint.h"

namespace rtp_llm {

class MockOpenaiEndpoint: public OpenaiEndpoint {
public:
    MockOpenaiEndpoint(const std::shared_ptr<Tokenizer>&  tokenizer,
                       const std::shared_ptr<ChatRender>& chat_render,
                       const ModelConfig&                 model_config):
        OpenaiEndpoint(tokenizer, chat_render, model_config) {}
    ~MockOpenaiEndpoint() override = default;

public:
    MOCK_METHOD1(extract_generation_config, std::shared_ptr<GenerateConfig>(const ChatCompletionRequest&));
};

}  // namespace rtp_llm