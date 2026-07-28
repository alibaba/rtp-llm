// XGrammarBackend + RtpGrammarMatcher unit tests (native-C++ path, no Python).

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfo.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

#include "absl/status/status.h"

namespace rtp_llm {
namespace {

// 128-char ASCII fixture vocab — enough to construct TokenizerInfo + trie.
xgrammar::TokenizerInfo makeTokenizerInfo() {
    std::vector<std::string> vocab;
    vocab.reserve(128);
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    return xgrammar::TokenizerInfo(vocab,
                                   xgrammar::VocabType::RAW,
                                   /*vocab_size=*/128,
                                   /*stop_token_ids=*/std::vector<int32_t>{0});
}

static_assert(!std::is_move_constructible_v<RtpGrammarMatcher>);
static_assert(!std::is_move_assignable_v<RtpGrammarMatcher>);
static_assert(!std::is_move_constructible_v<XGrammarBackend>);
static_assert(!std::is_move_assignable_v<XGrammarBackend>);

GrammarConfig defaultGrammarConfig() {
    GrammarConfig cfg;
    cfg.num_workers          = 2;
    cfg.compiler_cache_bytes = -1;
    return cfg;
}

std::shared_ptr<XGrammarBackend> makeBackend() {
    return XGrammarBackend::create(makeTokenizerInfo().SerializeJSON(), defaultGrammarConfig());
}

TEST(XGrammarBackendTest, CreateFromSerializedTokenizerInfo) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);

    auto matcher_or = backend->createMatcherFromKey({"json", R"({"type":"object"})"});
    ASSERT_TRUE(matcher_or.ok()) << matcher_or.status().ToString();
    EXPECT_EQ(matcher_or.value()->numAcceptedTokens(), 0);
}

TEST(XGrammarTokenizerInfoTest, SerializesTokenizerInfoFromPreparedHFData) {
    const std::vector<std::string> encoded_vocab{"A", "<0x20>", "B", "", ""};
    const std::string              tokenizer_metadata_json =
        R"({"vocab_size":5,"stop_token_ids":[2],"hf_tokenizer_json":"{\"decoder\":{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"ByteFallback\"}]},\"normalizer\":{\"type\":\"Prepend\",\"prepend\":\"\\u2581\"}}"})";
    const std::string opaque = xgrammar_impl::serializeTokenizerInfo(encoded_vocab, tokenizer_metadata_json);
    auto result = xgrammar::TokenizerInfo::DeserializeJSON(opaque);
    ASSERT_TRUE(std::holds_alternative<xgrammar::TokenizerInfo>(result));

    const auto& tokenizer_info = std::get<xgrammar::TokenizerInfo>(result);
    EXPECT_EQ(tokenizer_info.GetVocabType(), xgrammar::VocabType::BYTE_FALLBACK);
    EXPECT_EQ(tokenizer_info.GetVocabSize(), 5);
    EXPECT_TRUE(tokenizer_info.GetAddPrefixSpace());
    EXPECT_EQ(tokenizer_info.GetStopTokenIds(), std::vector<int32_t>{2});
    EXPECT_EQ(tokenizer_info.GetDecodedVocab()[1], " ");

    const auto& special_token_ids = tokenizer_info.GetSpecialTokenIds();
    EXPECT_NE(std::find(special_token_ids.begin(), special_token_ids.end(), 3), special_token_ids.end());
    EXPECT_NE(std::find(special_token_ids.begin(), special_token_ids.end(), 4), special_token_ids.end());
}

TEST(XGrammarTokenizerInfoTest, SerializesTokenizerInfoFromExplicitParams) {
    const std::vector<std::string> encoded_vocab{"A", "B", ""};
    const std::string              tokenizer_metadata_json =
        R"({"vocab_size":3,"stop_token_ids":[1],"vocab_type":"RAW","add_prefix_space":false})";
    const std::string              opaque =
        xgrammar_impl::serializeTokenizerInfo(encoded_vocab, tokenizer_metadata_json);
    auto result = xgrammar::TokenizerInfo::DeserializeJSON(opaque);
    ASSERT_TRUE(std::holds_alternative<xgrammar::TokenizerInfo>(result));

    const auto& tokenizer_info = std::get<xgrammar::TokenizerInfo>(result);
    EXPECT_EQ(tokenizer_info.GetVocabType(), xgrammar::VocabType::RAW);
    EXPECT_EQ(tokenizer_info.GetVocabSize(), 3);
    EXPECT_FALSE(tokenizer_info.GetAddPrefixSpace());
    EXPECT_EQ(tokenizer_info.GetStopTokenIds(), std::vector<int32_t>{1});
}

TEST(XGrammarBackendTest, CreateMatcherFromSimpleJsonSchema) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    GrammarKeyCpp   key{"json", R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"};

    auto result = backend->createMatcherFromKey(key);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
}

TEST(XGrammarBackendTest, CompileMalformedJsonSchemaIsInvalid) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    // Malformed JSON must surface as a user-input status, not throw.
    GrammarKeyCpp key{"json", "{this is not json at all"};

    auto result = backend->createMatcherFromKey(key);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_FALSE(result.status().message().empty());
}

TEST(XGrammarBackendTest, CreateMatcherFromStructuralTagWithBoundedAnyText) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    GrammarKeyCpp   key{"structural_tag",
                      R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
                        R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":1},"end":"z"},)"
                        R"({"type":"regex","pattern":"a"}]}})"};

    auto result = backend->createMatcherFromKey(key);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
}

TEST(XGrammarBackendTest, CreateMatcherFromStructuralTagWithBoundedAnyTextTokenEnd) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    GrammarKeyCpp   key{"structural_tag",
                      R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
                        R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":1},)"
                        R"("end":{"type":"token","token":122}},)"
                        R"({"type":"regex","pattern":"a"}]}})"};

    auto result = backend->createMatcherFromKey(key);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
}

TEST(XGrammarBackendTest, CompileStructuralTagRejectsMultipleBoundedRegions) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    GrammarKeyCpp   key{"structural_tag",
                      R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
                        R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":1},"end":"z"},)"
                        R"({"type":"any_text","max_tokens":1}]}})"};

    auto result = backend->createMatcherFromKey(key);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_FALSE(result.status().message().empty());
}

TEST(XGrammarBackendTest, CreateMatcherProducesUsableObject) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);

    auto matcher_or = backend->createMatcherFromKey({"json", R"({"type":"object"})"});
    ASSERT_TRUE(matcher_or.ok()) << matcher_or.status().ToString();
    auto matcher = matcher_or.value();
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
    auto terminated = matcher->isTerminated();
    ASSERT_TRUE(terminated.ok());
    EXPECT_FALSE(terminated.value());
}

// ---- RtpGrammarMatcher rollback ----------------------------------------

TEST(RtpGrammarMatcherTest, RollbackRestoresAcceptedCount) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);

    auto matcher_or = backend->createMatcherFromKey({"regex", "a"});
    ASSERT_TRUE(matcher_or.ok()) << matcher_or.status().ToString();
    auto          matcher  = matcher_or.value();
    constexpr int kA       = 'a';
    auto          accepted = matcher->acceptToken(kA);
    ASSERT_TRUE(accepted.ok());
    EXPECT_TRUE(accepted.value());
    EXPECT_EQ(matcher->numAcceptedTokens(), 1);
    EXPECT_FALSE(matcher->rollback(1).hasError());
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
}

}  // namespace
}  // namespace rtp_llm
