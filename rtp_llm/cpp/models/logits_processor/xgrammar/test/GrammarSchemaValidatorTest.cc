#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarSchemaValidator.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

GrammarKeyCpp jsonKey(const std::string& schema) {
    return {"json", schema};
}

}  // namespace

TEST(GrammarSchemaValidatorTest, AcceptsMinimalObjectSchema) {
    auto result = validateGrammarKey(jsonKey(R"({"type":"object"})"));
    EXPECT_EQ(result.status, GrammarValidateStatus::Ok);
}

TEST(GrammarSchemaValidatorTest, RejectsUnsupportedMultipleOf) {
    auto result = validateGrammarKey(jsonKey(R"({"type":"number","multipleOf":2})"));
    EXPECT_EQ(result.status, GrammarValidateStatus::UnsupportedFeature);
}

TEST(GrammarSchemaValidatorTest, RejectsEmptyRegex) {
    auto result = validateGrammarKey({"regex", ""});
    EXPECT_EQ(result.status, GrammarValidateStatus::InvalidSyntax);
}

TEST(GrammarSchemaValidatorTest, AcceptsSimpleRegexWithoutSyncCompile) {
    // Syntax validation is deferred to the async compile worker.
    auto result = validateGrammarKey({"regex", "ab"});
    EXPECT_EQ(result.status, GrammarValidateStatus::Ok);
}

}  // namespace rtp_llm
