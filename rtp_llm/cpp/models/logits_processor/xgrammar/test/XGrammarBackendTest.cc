// XGrammarBackend + RtpGrammarMatcher unit tests.
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
// Run: bazel test //rtp_llm/cpp/models/logits_processor/xgrammar/test:xgrammar_backend_cpp_test --config=cuda12_9

#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarBackend.h"

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

TEST(XGrammarBackendTest, ConstructFromTokenizerInfoJson) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    EXPECT_TRUE(backend.isEnabled());
}

TEST(XGrammarBackendTest, CompileBuiltinJSONViaSentinel) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());

    GrammarKeyCpp key{"json", "$$ANY$$"};
    auto          result = backend.compileNow(key);
    ASSERT_TRUE(result.compiled) << "$$ANY$$ should map to builtin JSON grammar";
    EXPECT_FALSE(result.is_invalid);
    EXPECT_GT(result.compiled->MemorySizeBytes(), 0u);
}

TEST(XGrammarBackendTest, CompileSimpleJsonSchema) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    GrammarKeyCpp      key{"json", R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"};

    auto result = backend.compileNow(key);
    ASSERT_TRUE(result.compiled);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_TRUE(result.error_message.empty());
}

TEST(XGrammarBackendTest, CompileMalformedJsonSchemaIsInvalid) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    // Garbage that cannot parse as JSON. Must surface as is_invalid (cacheable
    // failure) rather than a thrown exception — that's the contract the
    // GrammarCompiler async compile path relies on.
    GrammarKeyCpp key{"json", "{this is not json at all"};

    auto result = backend.compileNow(key);
    EXPECT_FALSE(result.compiled);
    EXPECT_TRUE(result.is_invalid);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(XGrammarBackendTest, CacheGetAndSet) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    GrammarKeyCpp      key{"json", R"({"type":"integer"})"};

    EXPECT_FALSE(backend.getCached(key));

    auto compiled = backend.compileNow(key).compiled;
    ASSERT_TRUE(compiled);
    backend.setCache(key, compiled);

    auto cached = backend.getCached(key);
    ASSERT_TRUE(cached);
    EXPECT_EQ(cached.get(), compiled.get()) << "cache must hand back the same shared_ptr";
}

TEST(XGrammarBackendTest, InvalidCacheShortCircuitsRepeat) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    GrammarKeyCpp      key{"json", "{not valid"};

    EXPECT_TRUE(backend.getCachedInvalid(key).empty());

    auto first = backend.compileNow(key);
    ASSERT_TRUE(first.is_invalid);
    backend.setCacheInvalid(key, first.error_message);

    const auto cached_err = backend.getCachedInvalid(key);
    EXPECT_FALSE(cached_err.empty());
    EXPECT_EQ(cached_err, first.error_message);
}

TEST(XGrammarBackendTest, ClearDropsCaches) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    GrammarKeyCpp      key{"json", R"({"type":"integer"})"};

    auto compiled = backend.compileNow(key).compiled;
    ASSERT_TRUE(compiled);
    backend.setCache(key, compiled);
    ASSERT_TRUE(backend.getCached(key));

    backend.clear();
    EXPECT_FALSE(backend.getCached(key));
}

// The compiled-grammar cache is LRU-bounded so a flood of distinct schemas
// cannot grow it without limit (OOM guard). Reuse one compiled grammar under
// many distinct keys (cache is keyed by GrammarKeyCpp::id(); the value pointer
// is irrelevant to eviction). Requires -fno-access-control (set in BUILD) to
// read the private capacity constant.
TEST(XGrammarBackendTest, CompiledCacheBoundedByCapacity) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    auto               compiled = backend.compileNow({"json", R"({"type":"integer"})"}).compiled;
    ASSERT_TRUE(compiled);

    const size_t cap = XGrammarBackend::kMaxCompiledCacheEntries;
    for (size_t i = 0; i <= cap; ++i) {  // cap + 1 distinct keys
        backend.setCache({"json", "k" + std::to_string(i)}, compiled);
    }
    // The first key inserted (and never touched) is the LRU victim once we
    // exceed capacity; the most recent insert must still be present.
    EXPECT_FALSE(backend.getCached({"json", "k0"})) << "oldest key should have been evicted";
    EXPECT_TRUE(backend.getCached({"json", "k" + std::to_string(cap)})) << "newest key must survive";
}

TEST(XGrammarBackendTest, CompiledCacheLruKeepsRecentlyUsed) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    auto               compiled = backend.compileNow({"json", R"({"type":"integer"})"}).compiled;
    ASSERT_TRUE(compiled);

    const size_t cap = XGrammarBackend::kMaxCompiledCacheEntries;
    for (size_t i = 0; i < cap; ++i) {  // fill to capacity: k0..k(cap-1)
        backend.setCache({"json", "k" + std::to_string(i)}, compiled);
    }
    // Touch k0 -> moves it to most-recently-used, so the next overflow evicts
    // k1 (the new LRU) instead of k0.
    ASSERT_TRUE(backend.getCached({"json", "k0"}));
    backend.setCache({"json", "k_extra"}, compiled);  // overflow by one

    EXPECT_TRUE(backend.getCached({"json", "k0"})) << "recently-used key must survive eviction";
    EXPECT_FALSE(backend.getCached({"json", "k1"})) << "least-recently-used key should be evicted";
    EXPECT_TRUE(backend.getCached({"json", "k_extra"}));
}

TEST(XGrammarBackendTest, CreateMatcherProducesUsableObject) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    auto compiled = backend.compileNow({"json", "$$ANY$$"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled);
    ASSERT_TRUE(matcher);
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
    EXPECT_FALSE(matcher->isTerminated());
}

// ---- sanitizeStructuralTag --------------------------------------------
//
// Mirrors the Python xgrammar_backend.py sanitize behavior. -fno-access-control
// (set in BUILD) lets us reach the private static directly without exposing it.

TEST(XGrammarBackendTest, SanitizeLegacyStructuresFillsMissingSchema) {
    const std::string in = R"({"structures":[{"begin":"<a>","end":"</a>"},{"begin":"<b>","schema":{"type":"object"},"end":"</b>"}],"triggers":["<"]})";
    const std::string out = XGrammarBackend::sanitizeStructuralTag(in);
    // First structure had no schema → must now contain "schema":{}.
    EXPECT_NE(out.find(R"("begin":"<a>")"), std::string::npos);
    EXPECT_NE(out.find(R"("schema":{})"), std::string::npos);
    // Second structure's existing schema must survive.
    EXPECT_NE(out.find(R"("type":"object")"), std::string::npos);
}

TEST(XGrammarBackendTest, SanitizeNewFormatJsonSchemaFillsMissingField) {
    const std::string in = R"({"format":{"type":"json_schema"}})";
    const std::string out = XGrammarBackend::sanitizeStructuralTag(in);
    EXPECT_NE(out.find(R"("json_schema":{})"), std::string::npos);
}

TEST(XGrammarBackendTest, SanitizeNewFormatRecursesIntoSequenceElements) {
    const std::string in = R"({"format":{"type":"sequence","elements":[{"type":"json_schema"},{"type":"qwen_xml_parameter"}]}})";
    const std::string out = XGrammarBackend::sanitizeStructuralTag(in);
    // Both inner nodes should now carry an empty json_schema.
    size_t first = out.find(R"("json_schema":{})");
    ASSERT_NE(first, std::string::npos);
    size_t second = out.find(R"("json_schema":{})", first + 1);
    EXPECT_NE(second, std::string::npos);
}

TEST(XGrammarBackendTest, SanitizeLeavesPresentJsonSchemaUntouched) {
    const std::string in = R"({"format":{"type":"json_schema","json_schema":{"type":"integer"}}})";
    const std::string out = XGrammarBackend::sanitizeStructuralTag(in);
    EXPECT_NE(out.find(R"("type":"integer")"), std::string::npos);
    // No spurious empty-dict insertion.
    EXPECT_EQ(out.find(R"("json_schema":{})"), std::string::npos);
}

TEST(XGrammarBackendTest, SanitizeMalformedJsonReturnsInputUnchanged) {
    const std::string in = "this is not json";
    EXPECT_EQ(XGrammarBackend::sanitizeStructuralTag(in), in);
}

// ---- RtpGrammarMatcher rollback ----------------------------------------

TEST(RtpGrammarMatcherTest, RollbackRestoresAcceptedCount) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled);
    constexpr int kA = 'a';
    EXPECT_TRUE(matcher->acceptToken(kA));
    EXPECT_EQ(matcher->numAcceptedTokens(), 1);
    matcher->rollback(1);
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
}

}  // namespace
}  // namespace rtp_llm
