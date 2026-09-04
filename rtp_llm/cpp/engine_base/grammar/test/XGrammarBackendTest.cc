// XGrammarBackend + RtpGrammarMatcher unit tests (native-C++ path, no Python).

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfo.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

#include "absl/status/status.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

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
    cfg.compile_timeout_ms           = 30000;
    cfg.terminate_without_stop_token = terminate_without_stop_token;
    return cfg;
}

std::shared_ptr<XGrammarBackend> makeBackend(bool terminate_without_stop_token = false) {
    return XGrammarBackend::create(makeTokenizerInfo().SerializeJSON(), grammarConfig(terminate_without_stop_token));
}

std::shared_ptr<XGrammarBackend> makeBackend(const GrammarConfig& config) {
    return XGrammarBackend::create(makeTokenizerInfo().SerializeJSON(), config);
}

GrammarKeyCpp jsonKey(const std::string& schema) {
    return GrammarKeyCpp{"json", schema};
}

const char* kSimpleSchema = R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})";
const char* kSchemaA = R"({"type":"object","properties":{"aa":{"type":"integer"}},"required":["aa"]})";
const char* kSchemaB = R"({"type":"object","properties":{"bb":{"type":"integer"}},"required":["bb"]})";
const char* kSchemaC = R"({"type":"object","properties":{"cc":{"type":"integer"}},"required":["cc"]})";

class BlockingCompile {
public:
    XGrammarBackend::CompileFn fn() {
        return [this](const GrammarKeyCpp& key) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ++entered_;
                entered_cv_.notify_all();
                release_cv_.wait(lock, [this] { return released_; });
            }
            GrammarCompileResult result;
            result.status = absl::InvalidArgumentError("blocking fake compile for " + key.key_type);
            return result;
        };
    }

    bool waitForEntered(int count, std::chrono::milliseconds budget = std::chrono::seconds(30)) {
        std::unique_lock<std::mutex> lock(mutex_);
        return entered_cv_.wait_for(lock, budget, [&] { return entered_ >= count; });
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        release_cv_.notify_all();
    }

    int entered() {
        std::lock_guard<std::mutex> lock(mutex_);
        return entered_;
    }

private:
    std::mutex              mutex_;
    std::condition_variable entered_cv_;
    std::condition_variable release_cv_;
    int                     entered_  = 0;
    bool                    released_ = false;
};

class ScopedRelease {
public:
    ScopedRelease(BlockingCompile& compile, std::shared_ptr<XGrammarBackend>& backend):
        compile_(compile), backend_(backend) {}
    ~ScopedRelease() {
        compile_.release();
        backend_.reset();
    }

private:
    BlockingCompile&                 compile_;
    std::shared_ptr<XGrammarBackend>& backend_;
};

class ScopedJoin {
public:
    ScopedJoin(BlockingCompile& compile, std::vector<std::thread>& threads): compile_(compile), threads_(threads) {}
    ~ScopedJoin() {
        compile_.release();
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

private:
    BlockingCompile&          compile_;
    std::vector<std::thread>& threads_;
};

bool waitFor(const std::function<bool()>& predicate,
             std::chrono::milliseconds budget = std::chrono::seconds(30)) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return predicate();
}

std::vector<int64_t> measureEntryBytes(const std::vector<std::string>& schemas) {
    auto config                 = grammarConfig();
    config.compiler_cache_bytes = -1;
    auto                 backend = makeBackend(config);
    std::vector<int64_t> bytes;
    int64_t              seen = 0;
    for (const auto& schema : schemas) {
        if (!backend->compileNow(jsonKey(schema)).compiled) {
            return {};
        }
        const int64_t total = backend->stats().verdict_cache_bytes;
        bytes.push_back(total - seen);
        seen = total;
    }
    return bytes;
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

TEST(XGrammarBackendTest, ValidAndInvalidVerdictsAreCached) {
    auto backend = makeBackend();

    ASSERT_TRUE(backend->compileNow(jsonKey(kSimpleSchema)).compiled);
    ASSERT_TRUE(backend->compileNow(jsonKey(kSimpleSchema)).compiled);
    EXPECT_EQ(backend->stats().compile_total, 1);
    EXPECT_EQ(backend->stats().cache_size, 1);

    const auto invalid_key = jsonKey("{ not json at all");
    auto       invalid     = backend->compileNow(invalid_key);
    ASSERT_EQ(invalid.status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(backend->compileNow(invalid_key).status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(backend->stats().compile_total, 2);
    EXPECT_EQ(backend->stats().invalid_cache_size, 1);
}

TEST(XGrammarBackendTest, NonPositiveBoundedCompileSettingsFailFast) {
    auto config = grammarConfig();
    config.compile_timeout_ms = 0;
    EXPECT_THROW(makeBackend(config), std::invalid_argument);

    config                      = grammarConfig();
    config.compile_concurrency  = 0;
    EXPECT_THROW(makeBackend(config), std::invalid_argument);

    config                    = grammarConfig();
    config.compile_queue_size = 0;
    EXPECT_THROW(makeBackend(config), std::invalid_argument);
}

TEST(XGrammarBackendTest, TimeoutIsRetryableAndBackgroundCompileWarmsCache) {
    auto config                = grammarConfig();
    config.compile_timeout_ms  = 50;
    config.compile_concurrency = 1;
    auto backend               = makeBackend(config);

    BlockingCompile compile;
    backend->setCompileFnForTest(compile.fn());
    ScopedRelease cleanup(compile, backend);

    const auto key      = jsonKey(kSimpleSchema);
    auto       timed_out = backend->compileNow(key);
    EXPECT_EQ(timed_out.status.code(), absl::StatusCode::kResourceExhausted);
    EXPECT_EQ(backend->stats().compile_timeout, 1);
    EXPECT_EQ(backend->stats().cache_size, 0);
    EXPECT_EQ(backend->stats().invalid_cache_size, 0);

    compile.release();
    ASSERT_TRUE(waitFor([&] { return backend->stats().invalid_cache_size == 1; }));
    auto retry = backend->compileNow(key);
    EXPECT_EQ(retry.status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(compile.entered(), 1);
    EXPECT_EQ(backend->stats().compile_total, 1);
    EXPECT_TRUE(waitFor([&] { return backend->stats().inflight == 0; }));
}

TEST(XGrammarBackendTest, ConcurrentCallersShareOneCompile) {
    auto config                = grammarConfig();
    config.compile_concurrency = 2;
    auto backend               = makeBackend(config);

    BlockingCompile compile;
    backend->setCompileFnForTest(compile.fn());

    constexpr int            kCallers = 4;
    std::atomic<int>         invalid_results{0};
    std::vector<std::thread> threads;
    threads.reserve(kCallers);
    ScopedJoin cleanup(compile, threads);
    for (int i = 0; i < kCallers; ++i) {
        threads.emplace_back([&] {
            if (backend->compileNow(jsonKey(kSimpleSchema)).status.code() == absl::StatusCode::kInvalidArgument) {
                invalid_results.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    ASSERT_TRUE(compile.waitForEntered(1));
    ASSERT_TRUE(waitFor([&] { return backend->stats().compile_dedup == kCallers - 1; }));
    compile.release();
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(invalid_results.load(std::memory_order_relaxed), kCallers);
    EXPECT_EQ(compile.entered(), 1);
    EXPECT_EQ(backend->stats().compile_total, 1);
    EXPECT_EQ(backend->stats().inflight, 0);
}

TEST(XGrammarBackendTest, InflightGaugeIsOneFromInsertionThroughCompileReport) {
    auto backend = makeBackend();

    BlockingCompile compile;
    backend->setCompileFnForTest(compile.fn());

    std::mutex           metrics_mutex;
    std::vector<int64_t> lookup_inflight;
    std::vector<int64_t> compile_inflight;
    backend->setMetricsReportFnForTest([&](const RtpLLMGrammarMetricsCollector& collector) {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        if (collector.compile_qps) {
            compile_inflight.push_back(collector.compile_inflight);
        } else {
            lookup_inflight.push_back(collector.compile_inflight);
        }
    });

    GrammarCompileResult    result;
    std::vector<std::thread> threads;
    ScopedJoin               cleanup(compile, threads);
    threads.emplace_back([&] { result = backend->compileNow(jsonKey(kSimpleSchema)); });

    ASSERT_TRUE(compile.waitForEntered(1));
    EXPECT_EQ(backend->stats().inflight, 1);
    compile.release();
    threads.front().join();

    EXPECT_EQ(result.status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(backend->stats().inflight, 0);
    std::lock_guard<std::mutex> lock(metrics_mutex);
    ASSERT_EQ(lookup_inflight.size(), 1u);
    EXPECT_EQ(lookup_inflight.front(), 1);
    ASSERT_EQ(compile_inflight.size(), 1u);
    EXPECT_EQ(compile_inflight.front(), 1);
}

TEST(XGrammarBackendTest, FullCompileQueueRejectsWithoutCaching) {
    auto config                = grammarConfig();
    config.compile_timeout_ms  = 50;
    config.compile_concurrency = 1;
    config.compile_queue_size  = 1;
    auto backend               = makeBackend(config);

    BlockingCompile compile;
    backend->setCompileFnForTest(compile.fn());
    ScopedRelease cleanup(compile, backend);

    ASSERT_EQ(backend->compileNow(jsonKey("occupy-worker")).status.code(),
              absl::StatusCode::kResourceExhausted);
    ASSERT_TRUE(compile.waitForEntered(1));

    bool saw_rejection = false;
    for (int i = 0; i < 8 && !saw_rejection; ++i) {
        auto result = backend->compileNow(jsonKey("queued-" + std::to_string(i)));
        ASSERT_EQ(result.status.code(), absl::StatusCode::kResourceExhausted);
        saw_rejection = result.status.message().find("too many outstanding") != std::string::npos;
    }

    EXPECT_TRUE(saw_rejection);
    EXPECT_GT(backend->stats().compile_rejected, 0);
    EXPECT_EQ(backend->stats().cache_size, 0);
    EXPECT_EQ(backend->stats().invalid_cache_size, 0);
}

TEST(XGrammarBackendTest, AllocationFailureIsTransientAndNotCached) {
    auto backend = makeBackend();
    std::atomic<int> attempts{0};
    backend->setCompileFnForTest([&](const GrammarKeyCpp&) -> GrammarCompileResult {
        attempts.fetch_add(1, std::memory_order_relaxed);
        throw std::bad_alloc();
    });

    const auto key = jsonKey(kSimpleSchema);
    EXPECT_EQ(backend->compileNow(key).status.code(), absl::StatusCode::kResourceExhausted);
    EXPECT_EQ(backend->compileNow(key).status.code(), absl::StatusCode::kResourceExhausted);
    EXPECT_EQ(attempts.load(std::memory_order_relaxed), 2);
    EXPECT_EQ(backend->stats().cache_size, 0);
    EXPECT_EQ(backend->stats().invalid_cache_size, 0);
}

TEST(XGrammarBackendTest, OrdinaryRuntimeFailureIsTransientAndNotCached) {
    auto backend = makeBackend();
    std::atomic<int> attempts{0};
    backend->setCompileFnForTest([&](const GrammarKeyCpp&) -> GrammarCompileResult {
        attempts.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("thread creation failed");
    });

    const auto key = jsonKey(kSimpleSchema);
    EXPECT_EQ(backend->compileNow(key).status.code(), absl::StatusCode::kUnknown);
    EXPECT_EQ(backend->compileNow(key).status.code(), absl::StatusCode::kUnknown);
    EXPECT_EQ(attempts.load(std::memory_order_relaxed), 2);
    EXPECT_EQ(backend->stats().invalid_cache_size, 0);
}

TEST(XGrammarBackendTest, ExplicitSyntaxRuntimeFailuresAreDeterministicAndCached) {
    const std::vector<std::string> messages = {
        "Regex parsing error at position 4",
        "EBNF lexer error at line 2",
        "grammar parser error at byte 4",
        "unexpected token: syntax error",
    };

    for (const auto& message : messages) {
        SCOPED_TRACE(message);
        auto             backend = makeBackend();
        std::atomic<int> attempts{0};
        backend->setCompileFnForTest([&](const GrammarKeyCpp&) -> GrammarCompileResult {
            attempts.fetch_add(1, std::memory_order_relaxed);
            throw std::runtime_error(message);
        });

        const GrammarKeyCpp key{"regex", message};
        EXPECT_EQ(backend->compileNow(key).status.code(), absl::StatusCode::kInvalidArgument);
        EXPECT_EQ(backend->compileNow(key).status.code(), absl::StatusCode::kInvalidArgument);
        EXPECT_EQ(attempts.load(std::memory_order_relaxed), 1);
        EXPECT_EQ(backend->stats().invalid_cache_size, 1);
    }
}

TEST(XGrammarBackendTest, ClearDropsVerdictsAndByteAccounting) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend->compileNow(jsonKey(kSimpleSchema)).compiled);
    ASSERT_EQ(backend->compileNow(jsonKey("{ not json")).status.code(), absl::StatusCode::kInvalidArgument);
    ASSERT_GT(backend->stats().verdict_cache_bytes, 0);

    backend->clear();
    EXPECT_EQ(backend->stats().cache_size, 0);
    EXPECT_EQ(backend->stats().invalid_cache_size, 0);
    EXPECT_EQ(backend->stats().verdict_cache_bytes, 0);

    ASSERT_TRUE(backend->compileNow(jsonKey(kSimpleSchema)).compiled);
    EXPECT_GT(backend->stats().verdict_cache_bytes, 0);
}

TEST(XGrammarBackendTest, ByteBudgetEvictsLeastRecentlyUsedVerdict) {
    const auto entry_bytes = measureEntryBytes({kSchemaA, kSchemaB, kSchemaC});
    ASSERT_EQ(entry_bytes.size(), 3u);
    ASSERT_GT(entry_bytes[0], 0);
    ASSERT_EQ(entry_bytes[0], entry_bytes[1]);
    ASSERT_EQ(entry_bytes[1], entry_bytes[2]);

    auto config = grammarConfig();
    // The configured value is one total budget. Half is reserved for xgrammar and
    // half for verdicts, so size the total for exactly two verdict entries.
    config.compiler_cache_bytes = 2 * (entry_bytes[0] + entry_bytes[1]);
    auto backend                = makeBackend(config);

    const auto initial_stats = backend->stats();
    EXPECT_EQ(initial_stats.total_cache_budget_bytes, config.compiler_cache_bytes);
    EXPECT_EQ(initial_stats.compiler_cache_budget_bytes + initial_stats.verdict_cache_budget_bytes,
              config.compiler_cache_bytes);

    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaA)).compiled);
    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaB)).compiled);
    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaA)).compiled);
    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaC)).compiled);
    EXPECT_EQ(backend->stats().cache_size, 2);
    EXPECT_EQ(backend->stats().cache_evicted, 1);
    EXPECT_LE(backend->stats().verdict_cache_bytes, backend->stats().verdict_cache_budget_bytes);

    const auto compile_total = backend->stats().compile_total;
    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaB)).compiled);
    EXPECT_EQ(backend->stats().compile_total, compile_total + 1);
}

TEST(XGrammarBackendTest, OversizedVerdictIsNotCachedOrAllowedToExceedSharedBudget) {
    auto config                 = grammarConfig();
    config.compiler_cache_bytes = 1;
    auto backend                = makeBackend(config);
    std::atomic<int> attempts{0};
    backend->setCompileFnForTest([&](const GrammarKeyCpp&) {
        attempts.fetch_add(1, std::memory_order_relaxed);
        GrammarCompileResult result;
        result.status = absl::InvalidArgumentError("invalid grammar");
        return result;
    });

    const auto key = jsonKey(kSimpleSchema);
    EXPECT_EQ(backend->compileNow(key).status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(backend->compileNow(key).status.code(), absl::StatusCode::kInvalidArgument);

    const auto stats = backend->stats();
    EXPECT_EQ(stats.compiler_cache_budget_bytes, 1);
    EXPECT_EQ(stats.verdict_cache_budget_bytes, 0);
    EXPECT_EQ(stats.verdict_cache_bytes, 0);
    EXPECT_EQ(stats.invalid_cache_size, 0);
    EXPECT_EQ(stats.cache_oversized, 2);
    EXPECT_EQ(attempts.load(std::memory_order_relaxed), 2);
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
