
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/RequestExtractor.h"

namespace rtp_llm {

class RequestExtractorTest: public ::testing::Test {};

TEST(RequestExtractorTest, Constructor_PromptBatch) {
    std::string jsonStr = R"({"prompt_batch": ["prompt1", "prompt2", "prompt3"]})";
    RawRequest  rq;
    FromJsonString(rq, jsonStr);

    RequestExtractor req(rq);
    ASSERT_EQ(req.batch_infer, true);
    ASSERT_EQ(req.is_streaming, false);
    ASSERT_EQ(req.input_texts.size(), 3);
    ASSERT_EQ(req.input_urls.size(), 3);
    ASSERT_EQ(req.generate_configs.size(), 3);
}

TEST(RequestExtractorTest, Constructor_Prompt) {
    std::string jsonStr = R"({"prompt": "prompt1"})";
    RawRequest  rq;
    FromJsonString(rq, jsonStr);

    RequestExtractor req(rq);
    ASSERT_EQ(req.batch_infer, false);
    ASSERT_EQ(req.is_streaming, false);
    ASSERT_EQ(req.input_texts.size(), 1);
    ASSERT_EQ(req.input_urls.size(), 1);
    ASSERT_EQ(req.generate_configs.size(), 1);
}

}  // namespace rtp_llm
