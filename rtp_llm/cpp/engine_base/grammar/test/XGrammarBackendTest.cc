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
    GrammarKeyCpp   key{"json", R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"};

    auto result = backend.compileNow(key);
    ASSERT_TRUE(result.compiled);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_TRUE(result.error_message.empty());
}

TEST(XGrammarBackendTest, CompileMalformedJsonSchemaIsInvalid) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    // Malformed JSON must surface as cacheable is_invalid, not throw.
    GrammarKeyCpp key{"json", "{this is not json at all"};

    auto result = backend.compileNow(key);
    EXPECT_FALSE(result.compiled);
    EXPECT_TRUE(result.is_invalid);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(XGrammarBackendTest, OversizeKeyStringRejectedAtEntry) {
    // Caller-controlled payload above kMaxKeyStringBytes must be rejected as
    // is_invalid without entering either cache, so an adversary cannot amplify
    // memory by submitting many distinct large blobs.
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    std::string     huge(64 * 1024 + 1, 'x');
    GrammarKeyCpp   key{"json", huge};

    auto result = backend.getOrCompile(key);
    EXPECT_FALSE(result.compiled);
    EXPECT_TRUE(result.is_invalid);
    EXPECT_NE(result.error_message.find("too large"), std::string::npos);
    EXPECT_TRUE(backend.getCachedInvalid(key).empty()) << "oversize keys must not populate invalid_cache_";
    EXPECT_FALSE(backend.getCached(key)) << "oversize keys must not populate cache_";
}

TEST(XGrammarBackendTest, CacheGetAndSet) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    GrammarKeyCpp   key{"json", R"({"type":"integer"})"};

    EXPECT_FALSE(backend.getCached(key));

    auto compiled = backend.compileNow(key).compiled;
    ASSERT_TRUE(compiled);
    backend.setCache(key, compiled);

    auto cached = backend.getCached(key);
    ASSERT_TRUE(cached);
    EXPECT_EQ(cached.get(), compiled.get()) << "cache must hand back the same shared_ptr";
}

// LRU-bounded compiled-grammar cache; needs -fno-access-control for private capacity.
TEST(XGrammarBackendTest, CompiledCacheBoundedByCapacity) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    auto            compiled = backend.compileNow({"json", R"({"type":"integer"})"}).compiled;
    ASSERT_TRUE(compiled);

    const size_t cap = XGrammarBackend::kMaxCompiledCacheEntries;
    for (size_t i = 0; i <= cap; ++i) {  // cap + 1 distinct keys
        backend.setCache({"json", "k" + std::to_string(i)}, compiled);
    }
    EXPECT_FALSE(backend.getCached({"json", "k0"})) << "oldest key should have been evicted";
    EXPECT_TRUE(backend.getCached({"json", "k" + std::to_string(cap)})) << "newest key must survive";
}

TEST(XGrammarBackendTest, CompiledCacheLruKeepsRecentlyUsed) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    auto            compiled = backend.compileNow({"json", R"({"type":"integer"})"}).compiled;
    ASSERT_TRUE(compiled);

    const size_t cap = XGrammarBackend::kMaxCompiledCacheEntries;
    for (size_t i = 0; i < cap; ++i) {  // fill to capacity: k0..k(cap-1)
        backend.setCache({"json", "k" + std::to_string(i)}, compiled);
    }
    // Touching k0 promotes it; the next overflow now evicts k1 instead.
    ASSERT_TRUE(backend.getCached({"json", "k0"}));
    backend.setCache({"json", "k_extra"}, compiled);  // overflow by one

    EXPECT_TRUE(backend.getCached({"json", "k0"})) << "recently-used key must survive eviction";
    EXPECT_FALSE(backend.getCached({"json", "k1"})) << "least-recently-used key should be evicted";
    EXPECT_TRUE(backend.getCached({"json", "k_extra"}));
}

// Invalid-cache byte budget pins worst-case memory below kMaxInvalidCacheBytes
// even when the entry-count cap alone would admit much more.
TEST(XGrammarBackendTest, InvalidCacheRespectsByteBudget) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());

    // Each kid contributes ~kMaxKeyStringBytes; flooding well past the byte
    // budget must be absorbed by LRU eviction, not by unbounded growth.
    const size_t      payload_size = XGrammarBackend::kMaxKeyStringBytes - 64;
    const std::string err          = "boom";
    const size_t      n_writes     = (XGrammarBackend::kMaxInvalidCacheBytes / payload_size) * 4 + 16;

    for (size_t i = 0; i < n_writes; ++i) {
        std::string body(payload_size, 'a');
        // Make every key distinct so no in-place update path is taken.
        std::string suffix = std::to_string(i);
        std::copy(suffix.begin(), suffix.end(), body.begin());
        backend.setCacheInvalid({"json", body}, err);
    }

    // Exercise the most-recently-written key still being a hit.
    std::string last_body(payload_size, 'a');
    std::string last_suffix = std::to_string(n_writes - 1);
    std::copy(last_suffix.begin(), last_suffix.end(), last_body.begin());
    EXPECT_FALSE(backend.getCachedInvalid({"json", last_body}).empty()) << "MRU entry must survive eviction";

    // Oldest key must have been evicted under the byte budget.
    std::string first_body(payload_size, 'a');
    first_body[0] = '0';
    EXPECT_TRUE(backend.getCachedInvalid({"json", first_body}).empty()) << "oldest entry must be evicted";
}

// Oversize keys must be rejected at setCacheInvalid entry — mirrors getOrCompile
// so callers (e.g. LogitsProcessorFactory) can't punch through the size guard
// and inflate per-entry bytes past 2*kMaxKeyStringBytes before LRU eviction.
TEST(XGrammarBackendTest, InvalidCacheRejectsOversizeKey) {
    XGrammarBackend   backend(makeTokenizerInfoJson(), defaultOptions());
    const std::string huge_key(XGrammarBackend::kMaxKeyStringBytes + 1, 'k');
    backend.setCacheInvalid({"json", huge_key}, "boom");
    EXPECT_TRUE(backend.getCachedInvalid({"json", huge_key}).empty())
        << "oversize key must not enter invalid_cache_";
}

// Oversize error_messages get truncated before they enter invalid_cache_, so
// they cannot inflate per-entry bytes past kMaxErrorMessageBytes.
TEST(XGrammarBackendTest, InvalidCacheErrorMessageTruncated) {
    XGrammarBackend   backend(makeTokenizerInfoJson(), defaultOptions());
    const std::string huge_err(8 * 1024, 'E');
    backend.setCacheInvalid({"json", "k_trunc"}, huge_err);
    const std::string stored = backend.getCachedInvalid({"json", "k_trunc"});
    EXPECT_LE(stored.size(), XGrammarBackend::kMaxErrorMessageBytes + 32);
    EXPECT_NE(stored.find("...[truncated]"), std::string::npos);
}

TEST(XGrammarBackendTest, CreateMatcherProducesUsableObject) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    auto            compiled = backend.compileNow({"json", "$$ANY$$"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled, /*require_reasoning=*/false, /*think_end_token_ids=*/std::nullopt);
    ASSERT_TRUE(matcher);
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
    EXPECT_FALSE(matcher->isTerminated());
}

// ---- sanitizeStructuralTag (private; reached via -fno-access-control) ----

TEST(XGrammarBackendTest, SanitizeNewFormatRecursesIntoSequenceElements) {
    const std::string in =
        R"({"format":{"type":"sequence","elements":[{"type":"json_schema"},{"type":"qwen_xml_parameter"}]}})";
    const std::string out = XGrammarBackend::sanitizeStructuralTag(in);
    // Both inner nodes should now carry an empty json_schema.
    size_t first = out.find(R"("json_schema":{})");
    ASSERT_NE(first, std::string::npos);
    size_t second = out.find(R"("json_schema":{})", first + 1);
    EXPECT_NE(second, std::string::npos);
}

TEST(XGrammarBackendTest, SanitizeLeavesPresentJsonSchemaUntouched) {
    const std::string in  = R"({"format":{"type":"json_schema","json_schema":{"type":"integer"}}})";
    const std::string out = XGrammarBackend::sanitizeStructuralTag(in);
    EXPECT_NE(out.find(R"("type":"integer")"), std::string::npos);
    // No spurious empty-dict insertion.
    EXPECT_EQ(out.find(R"("json_schema":{})"), std::string::npos);
}

// ---- RtpGrammarMatcher rollback ----------------------------------------

TEST(RtpGrammarMatcherTest, RollbackRestoresAcceptedCount) {
    XGrammarBackend backend(makeTokenizerInfoJson(), defaultOptions());
    auto            compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled, /*require_reasoning=*/false, /*think_end_token_ids=*/std::nullopt);
    constexpr int kA = 'a';
    EXPECT_TRUE(matcher->acceptToken(kA));
    EXPECT_EQ(matcher->numAcceptedTokens(), 1);
    matcher->rollback(1);
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
}

}  // namespace
}  // namespace rtp_llm
