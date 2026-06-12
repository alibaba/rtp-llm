// Phase 1 sanity test: prove xgrammar's C++ headers + libxgrammar.a are
// linkable from a Bazel cc_test under RTP-LLM, with no Python in sight.
//
// If this builds and the basic constructions don't crash, Phase 1 is done.
// Functional behavior is exercised in Phase 2 unit tests.

#include <xgrammar/compiler.h>
#include <xgrammar/grammar.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace rtp_llm {

// Minimal vocab — just enough to construct a TokenizerInfo. Real uses pull
// from the model tokenizer; the sanity test only proves the C++ path is wired.
TEST(XGrammarSanityTest, ConstructTokenizerInfo) {
    std::vector<std::string> vocab = {"<s>", "</s>", "{", "}", ":", "\"", "a", "1"};
    xgrammar::TokenizerInfo  info(vocab);
    EXPECT_EQ(info.GetVocabSize(), static_cast<int>(vocab.size()));
}

TEST(XGrammarSanityTest, CompileBuiltinJSON) {
    std::vector<std::string>  vocab = {"<s>", "</s>", "{", "}", ":", "\"", "a", "1", "true", "false", "null"};
    xgrammar::TokenizerInfo   info(vocab);
    xgrammar::GrammarCompiler compiler(info);

    xgrammar::CompiledGrammar cg = compiler.CompileBuiltinJSONGrammar();
    EXPECT_GT(cg.MemorySizeBytes(), 0u);
}

TEST(XGrammarSanityTest, ConstructMatcher) {
    std::vector<std::string>  vocab = {"<s>", "</s>", "{", "}", ":", "\"", "a", "1", "true", "false", "null"};
    xgrammar::TokenizerInfo   info(vocab);
    xgrammar::GrammarCompiler compiler(info);
    xgrammar::CompiledGrammar cg = compiler.CompileBuiltinJSONGrammar();

    xgrammar::GrammarMatcher matcher(cg);
    EXPECT_FALSE(matcher.IsTerminated());
    EXPECT_GT(xgrammar::GetBitmaskSize(static_cast<int>(vocab.size())), 0);
}

}  // namespace rtp_llm
