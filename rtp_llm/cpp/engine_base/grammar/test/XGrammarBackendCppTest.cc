#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>
#include <xgrammar/tokenizer_info.h>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"

namespace rtp_llm {
namespace {

constexpr int32_t kPrefixToken = 0;
constexpr int32_t kBodyToken   = 1;
constexpr int32_t kCloseToken  = 2;
constexpr int32_t kStopToken   = 3;

std::string makeTokenizerInfoJson() {
    std::vector<std::string> vocab = {
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n"
        "<｜DSML｜parameter name=\"text\" string=\"true\">",
        "a",
        "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        "<eos>",
    };
    xgrammar::TokenizerInfo info(vocab,
                                 xgrammar::VocabType::RAW,
                                 /*vocab_size=*/vocab.size(),
                                 /*stop_token_ids=*/std::vector<int32_t>{kStopToken});
    return info.SerializeJSON();
}

const char* kDeepSeekXMLSchema = R"json(
{
  "type": "structural_tag",
  "format": {
    "type": "tag",
    "begin": "<｜DSML｜tool_calls>\n",
    "content": {
      "type": "tags_with_separator",
      "tags": [
        {
          "type": "tag",
          "begin": "<｜DSML｜invoke name=\"f\">\n",
          "content": {
            "type": "json_schema",
            "json_schema": {
              "type": "object",
              "properties": {
                "text": {
                  "type": "string",
                  "minLength": 129,
                  "maxLength": 130
                }
              },
              "required": ["text"],
              "additionalProperties": false
            },
            "style": "deepseek_xml",
            "any_order": false
          },
          "end": "</｜DSML｜invoke>\n"
        }
      ],
      "separator": "",
      "at_least_one": true,
      "stop_after_first": false
    },
    "end": "</｜DSML｜tool_calls>"
  }
}
)json";

std::vector<int32_t> nextTokenBitmask(const std::shared_ptr<RtpGrammarMatcher>& matcher) {
    const int32_t        word_count = (matcher->vocabSize() + 31) / 32;
    std::vector<int32_t> bitmask(word_count, -1);
    int64_t              shape[2] = {1, word_count};
    DLTensor             tensor{};
    tensor.data   = bitmask.data();
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim   = 2;
    tensor.dtype  = DLDataType{kDLInt, 32, 1};
    tensor.shape  = shape;
    EXPECT_TRUE(matcher->fillBitmask(&tensor, 0));
    return bitmask;
}

bool allowsToken(const std::vector<int32_t>& bitmask, int32_t token_id) {
    return (static_cast<uint32_t>(bitmask[token_id / 32]) & (1u << (token_id % 32))) != 0;
}

TEST(XGrammarBackendCppTest, DeepSeekXMLStringLengthUsesCounterBoundaries) {
    XGrammarBackendOptions options;
    options.max_compiler_threads = 1;
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), options);

    auto compile_result = backend.compileNow({"structural_tag", kDeepSeekXMLSchema});
    ASSERT_NE(compile_result.compiled, nullptr) << compile_result.error_message;
    auto matcher = backend.createMatcher(compile_result.compiled, /*require_reasoning=*/false, std::nullopt);

    ASSERT_TRUE(matcher->acceptToken(kPrefixToken));
    for (int i = 0; i < 128; ++i) {
        ASSERT_TRUE(matcher->acceptToken(kBodyToken)) << "body token " << i;
    }
    EXPECT_FALSE(allowsToken(nextTokenBitmask(matcher), kCloseToken));

    ASSERT_TRUE(matcher->acceptToken(kBodyToken));
    auto at_minimum = nextTokenBitmask(matcher);
    EXPECT_TRUE(allowsToken(at_minimum, kCloseToken));
    EXPECT_TRUE(allowsToken(at_minimum, kBodyToken));

    ASSERT_TRUE(matcher->acceptToken(kBodyToken));
    auto at_maximum = nextTokenBitmask(matcher);
    EXPECT_TRUE(allowsToken(at_maximum, kCloseToken));
    EXPECT_FALSE(allowsToken(at_maximum, kBodyToken));
    ASSERT_TRUE(matcher->acceptToken(kCloseToken));
    EXPECT_TRUE(allowsToken(nextTokenBitmask(matcher), kStopToken));
    ASSERT_TRUE(matcher->acceptToken(kStopToken));
    EXPECT_TRUE(matcher->isTerminated());
}

}  // namespace
}  // namespace rtp_llm
