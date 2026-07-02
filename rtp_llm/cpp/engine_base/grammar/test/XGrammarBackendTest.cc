// XGrammarBackend + RtpGrammarMatcher unit tests (native-C++ path, no Python).

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

namespace rtp_llm {
namespace {

// 128-char ASCII fixture vocab — enough to construct TokenizerInfo + trie.
std::string makeTokenizerInfoJson() {
    std::vector<std::string> vocab;
    vocab.reserve(128);
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    xgrammar::TokenizerInfo info(vocab,
                                 xgrammar::VocabType::RAW,
                                 /*vocab_size=*/128,
                                 /*stop_token_ids=*/std::vector<int32_t>{0});
    return info.SerializeJSON();
}

XGrammarBackendOptions defaultOptions() {
    XGrammarBackendOptions opts;
    opts.any_whitespace            = true;
    opts.strict_mode               = true;
    opts.max_compiler_threads      = 2;
    opts.compiler_cache_bytes      = -1;
    return opts;
}

TEST(XGrammarBackendTest, ConstructFromTokenizerInfoJson) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    EXPECT_TRUE(backend.isEnabled());
}

TEST(XGrammarBackendTest, CompileBuiltinJSONViaSentinel) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());

    GrammarKeyCpp key{"json", "$$ANY$$"};
    auto          result = backend.getOrCompile(key);
    ASSERT_TRUE(result.compiled) << "$$ANY$$ should map to builtin JSON grammar";
    EXPECT_FALSE(result.is_invalid);
    EXPECT_GT(result.compiled->MemorySizeBytes(), 0u);
}

TEST(XGrammarBackendTest, CompileSimpleJsonSchema) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    GrammarKeyCpp   key{"json", R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"};

    auto result = backend.getOrCompile(key);
    ASSERT_TRUE(result.compiled);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_TRUE(result.error_message.empty());
}

TEST(XGrammarBackendTest, CompileMalformedJsonSchemaIsInvalid) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    // Malformed JSON must surface as cacheable is_invalid, not throw.
    GrammarKeyCpp key{"json", "{this is not json at all"};

    auto result = backend.getOrCompile(key);
    EXPECT_FALSE(result.compiled);
    EXPECT_TRUE(result.is_invalid);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(XGrammarBackendTest, CreateMatcherProducesUsableObject) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    auto            result = backend.getOrCompile({"json", "$$ANY$$"});
    ASSERT_TRUE(result.compiled);

    auto matcher =
        backend.createMatcher(result.compiled, /*require_reasoning=*/false, /*think_end_token_ids=*/std::nullopt);
    ASSERT_TRUE(matcher);
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
    EXPECT_FALSE(matcher->isTerminated());
}

// ---- RtpGrammarMatcher rollback ----------------------------------------

TEST(RtpGrammarMatcherTest, RollbackRestoresAcceptedCount) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    auto            result = backend.getOrCompile({"regex", "a"});
    ASSERT_TRUE(result.compiled);

    auto matcher =
        backend.createMatcher(result.compiled, /*require_reasoning=*/false, /*think_end_token_ids=*/std::nullopt);
    constexpr int kA = 'a';
    EXPECT_TRUE(matcher->acceptToken(kA));
    EXPECT_EQ(matcher->numAcceptedTokens(), 1);
    matcher->rollback(1);
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
}

}  // namespace
}  // namespace rtp_llm
