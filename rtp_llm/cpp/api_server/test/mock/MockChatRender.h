#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/openai/ChatRender.h"

namespace rtp_llm {

class MockChatRender: public ChatRender {
public:
    MockChatRender(): ChatRender(py::none()) {}
    ~MockChatRender() override = default;

public:
    MOCK_METHOD1(tokenize_words, std::vector<std::vector<int>>(const std::vector<std::string>&));
    MOCK_METHOD0(get_all_extra_stop_word_ids_list, std::vector<std::vector<int>>());
    MOCK_METHOD1(render_chat_request, RenderedInputs(const std::string&));
    MOCK_METHOD0(getRenderContext, std::shared_ptr<RenderContext>());
};

class MockRenderContext: public RenderContext {
public:
    MockRenderContext(): RenderContext() {}
    ~MockRenderContext() override = default;

public:
    MOCK_METHOD(void, init, (int n, std::string body, std::shared_ptr<ChatRender> chat_render), (override));
    MOCK_METHOD(std::string, render_stream_response_first, (int n, std::string debug_info), (override));
    MOCK_METHOD(std::string,
                render_stream_response,
                (const GenerateOutputs& outputs, const std::shared_ptr<GenerateConfig>& config, bool is_streaming),
                (override));
    MOCK_METHOD(std::string,
                render_stream_response_flush,
                (const GenerateOutputs& outputs, const std::shared_ptr<GenerateConfig>& config, bool is_streaming),
                (override));
    MOCK_METHOD(std::string, render_stream_response_final, (const GenerateOutputs& outputs), (override));
};

}  // namespace rtp_llm
