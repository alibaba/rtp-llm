// XGrammarBackendCpp + RtpGrammarMatcher unit tests.
//
// These tests exercise the new native-C++ grammar path end-to-end without
// any Python/pybind dependency. They prove:
//   * The backend can deserialize a TokenizerInfo, compile a JSON schema,
//     and produce a usable matcher.
//   * Cache hits are reported correctly via Stats.
//   * An invalid schema is reported as is_invalid (cacheable failure),
//     not as an exception.
//   * RtpGrammarMatcher's reasoning gate suppresses the bitmask while
//     in <think>, then enables it once think_end_id is observed.
//   * Rolling back across the think_end boundary re-enters passthrough.
//
// Run: bazel test //rtp_llm/cpp/engine_base/grammar/test:xgrammar_backend_cpp_test --config=cuda12_9

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

namespace rtp_llm {
namespace {

// Tiny fixture vocab — just enough to construct a TokenizerInfo and let
// xgrammar build the trie. Real tests below use simple JSON fragments that
// only require these characters; we don't depend on the parser actually
// having a meaningful vocabulary for matching beyond construction.
std::string makeTokenizerInfoJson() {
    std::vector<std::string> vocab;
    vocab.reserve(128);
    for (int i = 0; i < 128; ++i) {
        // RAW vocab: each "token" is one ASCII char as a single-byte string.
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
    opts.any_whitespace        = true;
    opts.strict_mode           = true;
    opts.max_compiler_threads  = 2;
    opts.enable_compiler_cache = true;
    opts.compiler_cache_bytes  = -1;
    return opts;
}

TEST(XGrammarBackendCppTest, ConstructFromTokenizerInfoJson) {
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), defaultOptions());
    EXPECT_FALSE(backend.hasReasoner());
}

TEST(XGrammarBackendCppTest, CompileBuiltinJSONViaSentinel) {
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), defaultOptions());

    GrammarKeyCpp key{"json", "$$ANY$$"};
    auto          result = backend.compileNow(key);
    ASSERT_TRUE(result.compiled) << "$$ANY$$ should map to builtin JSON grammar";
    EXPECT_FALSE(result.is_invalid);
    EXPECT_GT(result.compiled->MemorySizeBytes(), 0u);
}

TEST(XGrammarBackendCppTest, CompileSimpleJsonSchema) {
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), defaultOptions());
    GrammarKeyCpp      key{"json", R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"};

    auto result = backend.compileNow(key);
    ASSERT_TRUE(result.compiled);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_TRUE(result.error_message.empty());
}

TEST(XGrammarBackendCppTest, CompileMalformedJsonSchemaIsInvalid) {
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), defaultOptions());
    // Garbage that cannot parse as JSON. Must surface as is_invalid (cacheable
    // failure) rather than a thrown exception — that's the contract the
    // GrammarManager slow path relies on.
    GrammarKeyCpp key{"json", "{this is not json at all"};

    auto result = backend.compileNow(key);
    EXPECT_FALSE(result.compiled);
    EXPECT_TRUE(result.is_invalid);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(XGrammarBackendCppTest, CacheGetAndSet) {
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), defaultOptions());
    GrammarKeyCpp      key{"json", R"({"type":"integer"})"};

    EXPECT_FALSE(backend.getCached(key));

    auto compiled = backend.compileNow(key).compiled;
    ASSERT_TRUE(compiled);
    backend.setCache(key, compiled);

    auto cached = backend.getCached(key);
    ASSERT_TRUE(cached);
    EXPECT_EQ(cached.get(), compiled.get()) << "cache must hand back the same shared_ptr";
}

TEST(XGrammarBackendCppTest, InvalidCacheShortCircuitsRepeat) {
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), defaultOptions());
    GrammarKeyCpp      key{"json", "{not valid"};

    EXPECT_TRUE(backend.getCachedInvalid(key).empty());

    auto first = backend.compileNow(key);
    ASSERT_TRUE(first.is_invalid);
    backend.setCacheInvalid(key, first.error_message);

    const auto cached_err = backend.getCachedInvalid(key);
    EXPECT_FALSE(cached_err.empty());
    EXPECT_EQ(cached_err, first.error_message);
}

TEST(XGrammarBackendCppTest, ClearDropsCaches) {
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), defaultOptions());
    GrammarKeyCpp      key{"json", R"({"type":"integer"})"};

    auto compiled = backend.compileNow(key).compiled;
    ASSERT_TRUE(compiled);
    backend.setCache(key, compiled);
    ASSERT_TRUE(backend.getCached(key));

    backend.clear();
    EXPECT_FALSE(backend.getCached(key));
}

TEST(XGrammarBackendCppTest, CreateMatcherProducesUsableObject) {
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), defaultOptions());
    auto compiled = backend.compileNow({"json", "$$ANY$$"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled, /*require_reasoning=*/false);
    ASSERT_TRUE(matcher);
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
    EXPECT_FALSE(matcher->isPassthroughForMask());
    EXPECT_FALSE(matcher->isTerminated());
}

// ---- sanitizeStructuralTag --------------------------------------------
//
// Mirrors the Python xgrammar_backend.py sanitize behavior. -fno-access-control
// (set in BUILD) lets us reach the private static directly without exposing it.

TEST(XGrammarBackendCppTest, SanitizeLegacyStructuresFillsMissingSchema) {
    const std::string in = R"({"structures":[{"begin":"<a>","end":"</a>"},{"begin":"<b>","schema":{"type":"object"},"end":"</b>"}],"triggers":["<"]})";
    const std::string out = XGrammarBackendCpp::sanitizeStructuralTag(in);
    // First structure had no schema → must now contain "schema":{}.
    EXPECT_NE(out.find(R"("begin":"<a>")"), std::string::npos);
    EXPECT_NE(out.find(R"("schema":{})"), std::string::npos);
    // Second structure's existing schema must survive.
    EXPECT_NE(out.find(R"("type":"object")"), std::string::npos);
}

TEST(XGrammarBackendCppTest, SanitizeNewFormatJsonSchemaFillsMissingField) {
    const std::string in = R"({"format":{"type":"json_schema"}})";
    const std::string out = XGrammarBackendCpp::sanitizeStructuralTag(in);
    EXPECT_NE(out.find(R"("json_schema":{})"), std::string::npos);
}

TEST(XGrammarBackendCppTest, SanitizeNewFormatRecursesIntoSequenceElements) {
    const std::string in = R"({"format":{"type":"sequence","elements":[{"type":"json_schema"},{"type":"qwen_xml_parameter"}]}})";
    const std::string out = XGrammarBackendCpp::sanitizeStructuralTag(in);
    // Both inner nodes should now carry an empty json_schema.
    size_t first = out.find(R"("json_schema":{})");
    ASSERT_NE(first, std::string::npos);
    size_t second = out.find(R"("json_schema":{})", first + 1);
    EXPECT_NE(second, std::string::npos);
}

TEST(XGrammarBackendCppTest, SanitizeLeavesPresentJsonSchemaUntouched) {
    const std::string in = R"({"format":{"type":"json_schema","json_schema":{"type":"integer"}}})";
    const std::string out = XGrammarBackendCpp::sanitizeStructuralTag(in);
    EXPECT_NE(out.find(R"("type":"integer")"), std::string::npos);
    // No spurious empty-dict insertion.
    EXPECT_EQ(out.find(R"("json_schema":{})"), std::string::npos);
}

TEST(XGrammarBackendCppTest, SanitizeMalformedJsonReturnsInputUnchanged) {
    const std::string in = "this is not json";
    EXPECT_EQ(XGrammarBackendCpp::sanitizeStructuralTag(in), in);
}

// ---- RtpGrammarMatcher reasoning gate ----------------------------------

TEST(RtpGrammarMatcherTest, ReasoningPassthroughSuppressesMask) {
    XGrammarBackendOptions opts = defaultOptions();
    opts.think_end_id           = 7;  // arbitrary id from our tiny vocab
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), opts);

    auto compiled = backend.compileNow({"json", "$$ANY$$"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled, /*require_reasoning=*/true);
    matcher->initReasoning(/*in_think_body=*/true);

    EXPECT_TRUE(matcher->isPassthroughForMask())
        << "reasoning matcher must start in passthrough phase";

    // Accepting non-think_end tokens should NOT advance the parser nor exit
    // passthrough. Use ASCII 'a' (id 97) — definitely not the think_end id.
    EXPECT_TRUE(matcher->acceptToken(97));
    EXPECT_TRUE(matcher->isPassthroughForMask());
    EXPECT_EQ(matcher->numAcceptedTokens(), 1);

    // Now feed the think_end_id. The state machine flips out of passthrough.
    EXPECT_TRUE(matcher->acceptToken(7));
    EXPECT_FALSE(matcher->isPassthroughForMask())
        << "after think_end_id the bitmask must apply";
}

TEST(RtpGrammarMatcherTest, NoReasoningNeverEntersPassthrough) {
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), defaultOptions());  // no think_end_id

    auto compiled = backend.compileNow({"json", "$$ANY$$"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled, /*require_reasoning=*/false);
    EXPECT_FALSE(matcher->isPassthroughForMask());
    matcher->initReasoning(true);  // should be a no-op when require_reasoning is false
    EXPECT_FALSE(matcher->isPassthroughForMask());
}

TEST(RtpGrammarMatcherTest, RollbackAcrossThinkEndReentersPassthrough) {
    XGrammarBackendOptions opts = defaultOptions();
    opts.think_end_id           = 7;
    XGrammarBackendCpp backend(makeTokenizerInfoJson(), opts);

    auto compiled = backend.compileNow({"json", "$$ANY$$"}).compiled;
    ASSERT_TRUE(compiled);
    auto matcher = backend.createMatcher(compiled, /*require_reasoning=*/true);
    matcher->initReasoning(/*in_think_body=*/true);

    EXPECT_TRUE(matcher->acceptToken(7));  // think_end → exit passthrough
    EXPECT_FALSE(matcher->isPassthroughForMask());

    // Rolling back the think_end token should re-enter passthrough.
    matcher->rollback(1);
    EXPECT_TRUE(matcher->isPassthroughForMask());
}

}  // namespace
}  // namespace rtp_llm
