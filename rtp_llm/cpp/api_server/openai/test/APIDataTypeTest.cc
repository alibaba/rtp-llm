
#include "gtest/gtest.h"
#include "rtp_llm/cpp/api_server/openai/ApiDataType.h"

namespace rtp_llm {

using namespace autil::legacy;
using namespace autil::legacy::json;

class APIDataTypeTest: public ::testing::Test {
public:
    std::string jsonStr;
};

TEST_F(APIDataTypeTest, testFunctionCall) {
    FunctionCall obj;
    jsonStr = R"({"name": "tom", "arguments": "jerry"})";
    FromJsonString(obj, jsonStr);
    ASSERT_EQ(obj.name, "tom");
    ASSERT_EQ(obj.arguments, "jerry");

    FunctionCall obj1;
    jsonStr = R"({"name": "tom"})";
    FromJsonString(obj1, jsonStr);
    ASSERT_FALSE(obj1.arguments.has_value());

    FunctionCall obj2;
    jsonStr = R"({"arguments": "jerry"})";
    FromJsonString(obj2, jsonStr);
    ASSERT_FALSE(obj2.name.has_value());
}

TEST_F(APIDataTypeTest, testToolCall) {
    ToolCall obj;
    jsonStr = R"({"index": 1,
                 "id": "someId",
                 "type": "someType",
                 "function": {"name": "tom", "arguments": "jerry"}})";
    FromJsonString(obj, jsonStr);

    ASSERT_EQ(obj.index, 1);
    ASSERT_EQ(obj.id, "someId");
    ASSERT_EQ(obj.type, "someType");

    ASSERT_EQ(obj.function.name, "tom");
    ASSERT_EQ(obj.function.arguments, "jerry");
}

TEST_F(APIDataTypeTest, testRoleEnum) {
    ASSERT_TRUE(RoleEnum::contains("user"));
    ASSERT_TRUE(RoleEnum::contains("assistant"));
    ASSERT_TRUE(RoleEnum::contains("system"));
    ASSERT_TRUE(RoleEnum::contains("function"));
    ASSERT_TRUE(RoleEnum::contains("tool"));
    ASSERT_TRUE(RoleEnum::contains("observation"));

    ASSERT_FALSE(RoleEnum::contains("invalid_role"));
}

TEST_F(APIDataTypeTest, testContentPartTypeEnum) {
    ASSERT_TRUE(ContentPartTypeEnum::contains("text"));
    ASSERT_TRUE(ContentPartTypeEnum::contains("image_url"));
    ASSERT_TRUE(ContentPartTypeEnum::contains("video_url"));
    ASSERT_TRUE(ContentPartTypeEnum::contains("audio_url"));

    ASSERT_FALSE(ContentPartTypeEnum::contains("invalid_content_part_type"));
}

TEST_F(APIDataTypeTest, testImageURL) {
    ImageURL obj;
    jsonStr = R"({"url": "tom"})";
    FromJsonString(obj, jsonStr);
    ASSERT_EQ(obj.detail, "auto");

    ImageURL obj1;
    jsonStr = R"({"url": "tom", "detail": "jerry"})";
    FromJsonString(obj1, jsonStr);
    ASSERT_EQ(obj1.url, "tom");
    ASSERT_EQ(obj1.detail, "jerry");
}

TEST_F(APIDataTypeTest, testAudioURL) {
    AudioURL obj;
    jsonStr = R"({"url": "tom"})";
    FromJsonString(obj, jsonStr);
    ASSERT_EQ(obj.url, "tom");
}

TEST_F(APIDataTypeTest, testContentPart) {
    ContentPart obj;
    jsonStr = R"({"type": "text", "text": "someText"})";
    FromJsonString(obj, jsonStr);
    ASSERT_EQ(obj.type, "text");
    ASSERT_EQ(obj.text, "someText");

    ContentPart obj1;
    jsonStr = R"({"type": "image_url", "image_url": {"url": "tom", "detail": "jerry"}})";
    FromJsonString(obj1, jsonStr);
    ASSERT_TRUE(obj1.image_url.has_value());
    ASSERT_EQ(obj1.image_url.value().url, "tom");
    ASSERT_EQ(obj1.image_url.value().detail, "jerry");

    ContentPart obj2;
    jsonStr = R"({"type": "invalid_type"})";
    try {
        FromJsonString(obj2, jsonStr);
    } catch (const std::exception& e) {
        ASSERT_EQ(std::string(e.what()), "unknown content type");
    }
}

TEST_F(APIDataTypeTest, testChatMessage) {
    ChatMessage obj;
    jsonStr = R"({"role": "user", "content": "hello world"})";
    FromJsonString(obj, jsonStr);
    ASSERT_EQ(obj.role, "user");
    ASSERT_EQ(std::get<std::string>(obj.content), "hello world");

    ChatMessage obj1;
    jsonStr = R"({"role": "user", "content": [{"type": "text", "text": "someText"},
                                              {"type": "text", "text": "someText"}]})";
    FromJsonString(obj1, jsonStr);
    ASSERT_EQ(std::get<std::vector<ContentPart>>(obj1.content)[0].type, "text");
    ASSERT_EQ(std::get<std::vector<ContentPart>>(obj1.content)[0].text, "someText");
    ASSERT_EQ(std::get<std::vector<ContentPart>>(obj1.content)[1].type, "text");
    ASSERT_EQ(std::get<std::vector<ContentPart>>(obj1.content)[1].text, "someText");

    ChatMessage obj2;
    jsonStr = R"({"role": "user", "content": "hello world",
                  "tool_calls": [{"index": 1,
                                 "id": "someId",
                                 "type": "someType",
                                 "function": {"name": "tom", "arguments": "jerry"}},
                                {"index": 2,
                                 "id": "someId",
                                 "type": "someType",
                                 "function": {"name": "tom", "arguments": "jerry"}}]})";
    FromJsonString(obj2, jsonStr);
    ASSERT_TRUE(obj2.tool_calls.has_value());
    ASSERT_EQ(obj2.tool_calls.value()[0].index, 1);
    ASSERT_EQ(obj2.tool_calls.value()[1].index, 2);
}

TEST_F(APIDataTypeTest, testGPTFunctionDefinition) {
    GPTFunctionDefinition obj;
    jsonStr = R"({"name": "tom",
                  "description": "hello world",
                  "parameters": {"foo": "bar",
                                 "bar": "foo"},
                  "name_for_model": "tom",
                  "name_for_human": "tom",
                  "description_for_model": "hello world"})";
    FromJsonString(obj, jsonStr);

    ASSERT_EQ(obj.name, "tom");
    ASSERT_EQ(obj.description, "hello world");

    auto it = obj.parameters.find("foo");
    ASSERT_NE(it, obj.parameters.end());
    ASSERT_EQ(it->second, "bar");
    it = obj.parameters.find("bar");
    ASSERT_NE(it, obj.parameters.end());
    ASSERT_EQ(it->second, "foo");

    ASSERT_TRUE(obj.name_for_model.has_value());
    ASSERT_EQ(obj.name_for_model.value(), "tom");

    ASSERT_TRUE(obj.name_for_human.has_value());
    ASSERT_EQ(obj.name_for_human.value(), "tom");

    ASSERT_TRUE(obj.description_for_model.has_value());
    ASSERT_EQ(obj.description_for_model.value(), "hello world");
}

TEST_F(APIDataTypeTest, testChatCompletionRequest) {
    ChatCompletionRequest obj;
    jsonStr = R"({"model": "gpt",
                  "messages": [{"role": "user", "content": "who are you"},
                               {"role": "system", "content": "you are a English teacher"}],
                  "functions": [{"name": "tom", "description": "hello world", "parameters": {"foo": "bar", "bar": "foo"}},
                                {"name": "tom", "description": "hello world", "parameters": {"foo": "bar", "bar": "foo"}}],
                  "temperature": 0.8,
                  "top_p": 0.9,
                  "max_tokens": 100,
                  "stop": "xxx",
                  "stream": true,
                  "user": "tom",
                  "seed": "spaceX",
                  "extra_configs": {"is_streaming": true},
                  "private_request": true,
                  "trace_id": "traceXXX",
                  "chat_id": "chatXXX",
                  "template_key": "templateXXX",
                  "user_template": "templateYYY",
                  "debug_info": false,
                  "aux_info": true,
                  "extend_fields": {"foo": "bar", "bar": "foo"}})";
    FromJsonString(obj, jsonStr);

    ASSERT_EQ(obj.model, "gpt");
    ASSERT_EQ(obj.messages[0].role, "user");
    ASSERT_EQ(std::get<std::string>(obj.messages[0].content), "who are you");
    ASSERT_EQ(obj.messages[1].role, "system");
    ASSERT_EQ(std::get<std::string>(obj.messages[1].content), "you are a English teacher");
}

}  // namespace rtp_llm
