// XGrammarBackend + RtpGrammarMatcher unit tests (native-C++ path, no Python).

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfo.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
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

GrammarConfig grammarConfig(bool terminate_without_stop_token = false) {
    GrammarConfig cfg;
    cfg.num_workers                  = 2;
    cfg.compiler_cache_bytes         = -1;
    cfg.terminate_without_stop_token = terminate_without_stop_token;
    return cfg;
}

std::shared_ptr<XGrammarBackend> makeBackend(bool terminate_without_stop_token = false) {
    return XGrammarBackend::create(makeTokenizerInfo().SerializeJSON(), grammarConfig(terminate_without_stop_token));
}

std::shared_ptr<XGrammarBackend> makeWhitespaceBackend(bool disable_any_whitespace) {
    auto cfg                                    = grammarConfig();
    cfg.constrained_json_disable_any_whitespace = disable_any_whitespace;
    return XGrammarBackend::create(makeTokenizerInfo().SerializeJSON(), cfg);
}

bool acceptsCompleteText(XGrammarBackend& backend, const GrammarKeyCpp& key, const std::string& text) {
    auto result = backend.createMatcherFromKey(key);
    EXPECT_TRUE(result.ok()) << result.status().ToString();
    if (!result.ok()) {
        return false;
    }
    for (const unsigned char token : text) {
        const auto accepted = result.value()->acceptToken(token);
        EXPECT_TRUE(accepted.ok()) << accepted.status().ToString();
        if (!accepted.ok() || !accepted.value()) {
            return false;
        }
    }
    const auto stop = result.value()->acceptToken(0);
    EXPECT_TRUE(stop.ok()) << stop.status().ToString();
    return stop.ok() && stop.value();
}

TEST(XGrammarBackendTest, CreateFromSerializedTokenizerInfo) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);

    auto matcher_or = backend->createMatcherFromKey({"json", R"({"type":"object"})"});
    ASSERT_TRUE(matcher_or.ok()) << matcher_or.status().ToString();
    EXPECT_EQ(matcher_or.value()->numAcceptedTokens(), 0);
}

TEST(XGrammarBackendTest, EmptyTokenizerInfoIntentionallyDisablesBackend) {
    EXPECT_EQ(XGrammarBackend::create("", grammarConfig()), nullptr);
}

TEST(XGrammarBackendTest, NonEmptyMalformedTokenizerInfoFailsFast) {
    EXPECT_THROW(XGrammarBackend::create("not-json", grammarConfig()), std::runtime_error);
}

TEST(XGrammarTokenizerInfoTest, SerializesTokenizerInfoFromPreparedHFData) {
    const std::vector<std::string> encoded_vocab{"A", "<0x20>", "B", "", ""};
    const std::string              tokenizer_metadata_json =
        R"({"vocab_size":5,"stop_token_ids":[2],"hf_tokenizer_json":"{\"decoder\":{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"ByteFallback\"}]},\"normalizer\":{\"type\":\"Prepend\",\"prepend\":\"\\u2581\"}}"})";
    const std::string opaque = xgrammar_impl::serializeTokenizerInfo(encoded_vocab, tokenizer_metadata_json);
    auto              result = xgrammar::TokenizerInfo::DeserializeJSON(opaque);
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
    const std::string opaque = xgrammar_impl::serializeTokenizerInfo(encoded_vocab, tokenizer_metadata_json);
    auto              result = xgrammar::TokenizerInfo::DeserializeJSON(opaque);
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
    GrammarKeyCpp key{"json", R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"};

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

TEST(XGrammarBackendTest, WrappedJsonUsesTheServiceWhitespacePolicy) {
    const std::string schema =
        R"({"type":"object","properties":{"items":{"type":"array","items":{"type":"integer"}}},"required":["items"]})";
    const GrammarKeyCpp direct{"json", schema};
    const GrammarKeyCpp wrapped{
        "structural_tag", R"({"type":"structural_tag","format":{"type":"json_schema","json_schema":)" + schema + "}}"};
    for (const bool disable : {false, true}) {
        auto backend = makeWhitespaceBackend(disable);
        ASSERT_TRUE(backend);
        for (const auto& key : {direct, wrapped}) {
            EXPECT_TRUE(acceptsCompleteText(*backend, key, R"({"items": [1, 2]})"));
            EXPECT_EQ(acceptsCompleteText(*backend, key, "{\"items\": [\n \t\t1, 2]}"), !disable);
            EXPECT_EQ(acceptsCompleteText(*backend, key, "{\"items\": [" + std::string(100, ' ') + "]}"), !disable);
        }
    }
}

TEST(XGrammarBackendTest, FixedWhitespaceReachesJsonInsideReasoningBranches) {
    auto backend = makeWhitespaceBackend(true);
    ASSERT_TRUE(backend);
    const GrammarKeyCpp key{"structural_tag", R"({"type":"structural_tag","format":{"type":"or","elements":[
        {"type":"sequence","elements":[
            {"type":"tag","begin":"<think>","content":{"type":"any_text"},"end":"</think>"},
            {"type":"json_schema","json_schema":{"type":"array","items":{"type":"integer"}}}]},
        {"type":"json_schema","json_schema":{"type":"array","items":{"type":"integer"}}}]}})"};
    EXPECT_TRUE(acceptsCompleteText(*backend, key, "[1, 2]"));
    EXPECT_TRUE(acceptsCompleteText(*backend, key, "<think>  x\n  y </think>[1, 2]"));
    EXPECT_FALSE(acceptsCompleteText(*backend, key, "[  1, 2]"));
    EXPECT_FALSE(acceptsCompleteText(*backend, key, "<think>  x\n  y </think>[  1, 2]"));
}

TEST(XGrammarBackendTest, FixedWhitespacePreservesSchemaDataAndStringSpaces) {
    auto backend = makeWhitespaceBackend(true);
    ASSERT_TRUE(backend);
    const GrammarKeyCpp key{"structural_tag", R"({"type":"structural_tag","format":{"type":"json_schema","json_schema":{
        "type":"object","properties":{
            "type":{"const":"json_schema"},
            "json_schema":{"const":{"type":"json_schema","json_schema":false}},
            "text":{"const":"two  spaces"}},"required":["type","json_schema","text"]}}})"};
    EXPECT_TRUE(acceptsCompleteText(
        *backend,
        key,
        R"({"type": "json_schema", "json_schema": {"type":"json_schema","json_schema":false}, "text": "two  spaces"})"));
}

TEST(XGrammarBackendTest, FixedWhitespacePreservesExplicitNodeCaps) {
    auto backend = makeWhitespaceBackend(true);
    ASSERT_TRUE(backend);
    for (const int cap : {1, 2}) {
        const GrammarKeyCpp key{"structural_tag",
                                R"({"type":"structural_tag","format":{"type":"json_schema","max_whitespace_cnt":)"
                                    + std::to_string(cap)
                                    + R"(,"json_schema":{"type":"array","items":{"type":"integer"}}}})"};
        EXPECT_TRUE(acceptsCompleteText(*backend, key, "[1,2]"));
        EXPECT_TRUE(acceptsCompleteText(*backend, key, "[ 1, 2]"));
        EXPECT_EQ(acceptsCompleteText(*backend, key, "[  1,2]"), cap == 2);
        EXPECT_FALSE(acceptsCompleteText(*backend, key, "[   1,2]"));
    }
}

TEST(XGrammarBackendTest, FixedWhitespacePreservesAnyOrder) {
    auto backend = makeWhitespaceBackend(true);
    ASSERT_TRUE(backend);
    const GrammarKeyCpp key{"structural_tag", R"({"type":"structural_tag","format":{"type":"json_schema",
        "style":"json","any_order":true,"max_whitespace_cnt":null,"json_schema":{
            "type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}})"};
    EXPECT_TRUE(acceptsCompleteText(*backend, key, R"({"a": 1, "b": 2})"));
    EXPECT_TRUE(acceptsCompleteText(*backend, key, R"({"b": 2, "a": 1})"));
    EXPECT_FALSE(acceptsCompleteText(*backend, key, "{\"b\":\n2, \"a\": 1}"));
}

TEST(XGrammarBackendTest, FixedWhitespacePreservesXmlToolFormats) {
    const GrammarKeyCpp key{"structural_tag", R"({"type":"structural_tag","format":{"type":"json_schema",
        "style":"qwen_xml","json_schema":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}})"};
    auto                normal_backend = makeWhitespaceBackend(false);
    auto                fixed_backend  = makeWhitespaceBackend(true);
    ASSERT_TRUE(normal_backend);
    ASSERT_TRUE(fixed_backend);
    const auto normal = normal_backend->compile(key);
    const auto fixed  = fixed_backend->compile(key);
    ASSERT_TRUE(normal.ok()) << normal.status().ToString();
    ASSERT_TRUE(fixed.ok()) << fixed.status().ToString();
    EXPECT_EQ(normal.value()->GetGrammar().ToString(), fixed.value()->GetGrammar().ToString());
}

TEST(XGrammarBackendTest, FixedWhitespaceDoesNotHideMalformedStructuralTags) {
    auto backend = makeWhitespaceBackend(true);
    ASSERT_TRUE(backend);
    for (
        const std::string input :
        {"not-json",
         R"({"type":"structural_tag","format":{"type":"json_schema","json_schema":"bad"}})",
         R"({"type":"structural_tag","format":{"type":"json_schema","json_schema":true,"any_order":"bad"}})",
         R"({"type":"structural_tag","format":{"type":"json_schema","json_schema":true,"style":"bad"}})",
         R"({"type":"structural_tag","format":{"type":"json_schema","json_schema":true,"max_whitespace_cnt":0}})",
         R"({"type":"structural_tag","format":{"type":"json_schema","json_schema":true,"max_whitespace_cnt":"bad"}})"}) {
        const auto result = backend->createMatcherFromKey({"structural_tag", input});
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    }
}

TEST(XGrammarBackendTest, CreateMatcherFromStructuralTagWithBoundedAnyText) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    GrammarKeyCpp key{"structural_tag",
                      R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
                      R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":1},"end":"z"},)"
                      R"({"type":"regex","pattern":"a"}]}})"};

    auto result = backend->createMatcherFromKey(key);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
}

TEST(XGrammarBackendTest, CreateMatcherFromStructuralTagWithBoundedAnyTextTokenEnd) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    GrammarKeyCpp key{"structural_tag",
                      R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
                      R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":1},)"
                      R"("end":{"type":"token","token":122}},)"
                      R"({"type":"regex","pattern":"a"}]}})"};

    auto result = backend->createMatcherFromKey(key);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
}

TEST(XGrammarBackendTest, CompileStructuralTagSupportsMultipleBoundedRegions) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    GrammarKeyCpp key{"structural_tag",
                      R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
                      R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":1},"end":"z"},)"
                      R"({"type":"any_text","max_tokens":1}]}})"};

    auto result = backend->createMatcherFromKey(key);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
}

TEST(XGrammarBackendTest, CreateMatcherFromAdaptiveReasoningStructuralTag) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    GrammarKeyCpp key{"structural_tag",
                      R"({"type":"structural_tag","format":{"type":"or","elements":[)"
                      R"({"type":"sequence","elements":[)"
                      R"({"type":"tag","begin":"<think>","content":{"type":"any_text","max_tokens":2},)"
                      R"("end":"</think>"},{"type":"regex","pattern":"a"}]},)"
                      R"({"type":"any_text","excludes":["<think>","</think>"]}]}})"};

    const auto accepts = [&](const std::string& text) {
        auto matcher_or = backend->createMatcherFromKey(key);
        EXPECT_TRUE(matcher_or.ok()) << matcher_or.status().ToString();
        if (!matcher_or.ok()) {
            return false;
        }
        for (const unsigned char token : text) {
            auto accepted = matcher_or.value()->acceptToken(token);
            EXPECT_TRUE(accepted.ok()) << accepted.status().ToString();
            EXPECT_TRUE(accepted.ok() && accepted.value()) << "rejected token " << token;
        }
        return true;
    };

    EXPECT_TRUE(accepts("<think>x</think>a"));
    EXPECT_TRUE(accepts("plain answer"));
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

TEST(XGrammarBackendTest, TerminationBehaviorComesFromServiceGrammarConfig) {
    auto auto_terminate_backend = makeBackend(/*terminate_without_stop_token=*/true);
    ASSERT_TRUE(auto_terminate_backend);

    auto matcher_or = auto_terminate_backend->createMatcherFromKey({"regex", "a"});
    ASSERT_TRUE(matcher_or.ok()) << matcher_or.status().ToString();
    auto matcher = matcher_or.value();

    auto accepted = matcher->acceptToken('a');
    ASSERT_TRUE(accepted.ok());
    EXPECT_TRUE(accepted.value());
    auto terminated = matcher->isTerminated();
    ASSERT_TRUE(terminated.ok());
    EXPECT_TRUE(terminated.value());

    auto require_stop_backend = makeBackend(/*terminate_without_stop_token=*/false);
    ASSERT_TRUE(require_stop_backend);
    matcher_or = require_stop_backend->createMatcherFromKey({"regex", "a"});
    ASSERT_TRUE(matcher_or.ok()) << matcher_or.status().ToString();
    matcher  = matcher_or.value();
    accepted = matcher->acceptToken('a');
    ASSERT_TRUE(accepted.ok());
    EXPECT_TRUE(accepted.value());
    terminated = matcher->isTerminated();
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
