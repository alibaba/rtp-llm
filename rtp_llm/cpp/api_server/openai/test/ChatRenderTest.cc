#include <gtest/gtest.h>
#include <pybind11/eval.h>

#include "rtp_llm/cpp/api_server/openai/ChatRender.h"

namespace py = pybind11;

namespace rtp_llm {

TEST(ChatRenderTest, AppliesRendererConstraintPatchWithoutReplacingNativeConfig) {
    py::dict globals;
    py::exec(R"(
class Renderer:
    cpp_http_constraints_enabled = True

    def apply_chat_completion_constraints_from_json(self, request_json, config_json):
        assert 'messages' in request_json
        return (
            '{"temperature":1.0,"top_p":0.95,"in_think_mode":false,'
            '"max_thinking_tokens":0,"structural_tag":"{\\"type\\":\\"structural_tag\\"}"}',
            ['json_schema'],
        )
)",
             globals);

    auto renderer    = globals["Renderer"]();
    auto chat_render = std::make_shared<ChatRender>(renderer);
    auto config      = std::make_shared<GenerateConfig>();
    config->temperature         = 0.7f;
    config->top_p               = 1.0f;
    config->in_think_mode       = true;
    config->max_thinking_tokens = 123;
    config->json_schema         = R"({"type":"object"})";
    config->trace_id            = "trace-native-only";

    chat_render->apply_chat_completion_constraints(R"({"messages":[]})", config);

    EXPECT_FLOAT_EQ(config->temperature, 1.0f);
    EXPECT_FLOAT_EQ(config->top_p, 0.95f);
    EXPECT_FALSE(config->in_think_mode);
    EXPECT_EQ(config->max_thinking_tokens, 0);
    ASSERT_TRUE(config->structural_tag.has_value());
    EXPECT_EQ(config->structural_tag.value(), R"({"type":"structural_tag"})");
    EXPECT_FALSE(config->json_schema.has_value());
    EXPECT_EQ(config->trace_id, "trace-native-only");
}

TEST(ChatRenderTest, SkipsConstraintBridgeUnlessRendererOptsIn) {
    py::dict globals;
    py::exec(R"(
class Renderer:
    cpp_http_constraints_enabled = False

    def apply_chat_completion_constraints_from_json(self, request_json, config_json):
        raise AssertionError('disabled renderer constraint bridge must not be called')
)",
             globals);

    auto chat_render = std::make_shared<ChatRender>(globals["Renderer"]());
    auto config      = std::make_shared<GenerateConfig>();
    config->temperature = 0.7f;

    chat_render->apply_chat_completion_constraints(R"({"messages":[]})", config);

    EXPECT_FLOAT_EQ(config->temperature, 0.7f);
}

}  // namespace rtp_llm
