#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include <xgrammar/tokenizer_info.h>

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

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

std::unique_ptr<XGrammarBackendCpp> makeBackend(const XGrammarBackendOptions& options) {
    return std::make_unique<XGrammarBackendCpp>(makeTokenizerInfoJson(), options);
}

XGrammarBackendOptions baseOptions() {
    XGrammarBackendOptions options;
    options.max_compiler_threads = 1;
    // Tests using real compiles only need the pooled path to work; a wide budget keeps them from racing a
    // clock on loaded CI hardware. The timeout tests override this explicitly.
    options.compile_timeout_ms = 30000;
    return options;
}

GrammarKeyCpp jsonKey(const std::string& schema) {
    return GrammarKeyCpp{"json", schema};
}

const char* kSimpleSchema = R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})";

// Three distinct grammars shaped alike, so their accounted sizes match and a ceiling can be derived that
// holds exactly two of them. The byte-budget tests below rely on that equality and assert it.
const char* kSchemaA = R"({"type":"object","properties":{"aa":{"type":"integer"}},"required":["aa"]})";
const char* kSchemaB = R"({"type":"object","properties":{"bb":{"type":"integer"}},"required":["bb"]})";
const char* kSchemaC = R"({"type":"object","properties":{"cc":{"type":"integer"}},"required":["cc"]})";

// A compile that blocks until the test releases it. Real xgrammar compiles are far too fast against this
// test's 128-token vocabulary to observe the timeout, dedup and queue-full paths without racing a wall
// clock, and making them genuinely slow would cost minutes of CI time and gigabytes of RAM. Driving the
// injected compile explicitly makes those paths deterministic instead.
class BlockingCompile {
public:
    XGrammarBackendCpp::CompileFn fn() {
        return [this](const GrammarKeyCpp& key) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ++entered_;
                entered_cv_.notify_all();
                release_cv_.wait(lock, [this] { return released_; });
            }
            if (throw_bad_alloc_) {
                throw std::bad_alloc();
            }
            CompileResult out;
            out.is_invalid    = true;
            out.error_message = "blocking fake compile for " + key.key_type;
            return out;
        };
    }

    void throwBadAlloc() {
        throw_bad_alloc_ = true;
    }

    bool waitForEntered(int count, std::chrono::milliseconds budget = 30s) {
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
    int                     entered_         = 0;
    bool                    released_        = false;
    bool                    throw_bad_alloc_ = false;
};

// Releases the fake compile before the backend is destroyed. The destructor joins the pool, so a still
// blocked compile would hang teardown rather than fail the test.
class ScopedRelease {
public:
    ScopedRelease(BlockingCompile& compile, std::unique_ptr<XGrammarBackendCpp>& backend):
        compile_(compile), backend_(backend) {}
    ~ScopedRelease() {
        compile_.release();
        backend_.reset();
    }

private:
    BlockingCompile&                     compile_;
    std::unique_ptr<XGrammarBackendCpp>& backend_;
};

// Releases the fake compile and joins the caller threads however the test leaves the scope. A failed
// ASSERT returns with the threads still joinable, and destroying a joinable std::thread calls
// std::terminate, which would take the whole test binary down instead of failing one case.
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

bool waitFor(const std::function<bool()>& predicate, std::chrono::milliseconds budget = 30s) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(5ms);
    }
    return predicate();
}

// The accounted cost of one cached grammar comes from xgrammar's own size estimate, which is an internal
// detail that may change with the dependency. The byte-budget tests therefore measure it against an
// unlimited cache and derive their ceiling from the measurement instead of hard-coding a number.
std::vector<int64_t> measureEntryBytes(const std::vector<std::string>& schemas) {
    auto options                 = baseOptions();
    options.compiler_cache_bytes = 0;
    auto                 backend = makeBackend(options);
    std::vector<int64_t> bytes;
    int64_t              seen = 0;
    for (const auto& schema : schemas) {
        if (!backend->compileNow(jsonKey(schema)).compiled) {
            return {};
        }
        const int64_t total = backend->stats().cache_bytes;
        bytes.push_back(total - seen);
        seen = total;
    }
    return bytes;
}

TEST(XGrammarBackendCppTest, CompilesAndCachesValidGrammar) {
    auto backend = makeBackend(baseOptions());
    auto key     = jsonKey(kSimpleSchema);

    ASSERT_FALSE(backend->getCached(key));

    auto result = backend->compileNow(key);
    ASSERT_TRUE(result.compiled);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_FALSE(result.is_overloaded);

    // Published by the backend itself, so the caller never has to write the cache back.
    EXPECT_TRUE(backend->getCached(key));
    EXPECT_EQ(backend->stats().cache_size, 1);
    EXPECT_EQ(backend->stats().compile_total, 1);

    // A second request is served from cache rather than recompiled.
    ASSERT_TRUE(backend->compileNow(key).compiled);
    EXPECT_EQ(backend->stats().compile_total, 1);
}

TEST(XGrammarBackendCppTest, InvalidGrammarIsCachedAndNotRecompiled) {
    auto backend = makeBackend(baseOptions());
    auto key     = jsonKey("{ not json at all");

    auto result = backend->compileNow(key);
    EXPECT_FALSE(result.compiled);
    EXPECT_TRUE(result.is_invalid);
    EXPECT_FALSE(result.is_overloaded);
    EXPECT_FALSE(result.error_message.empty());

    EXPECT_FALSE(backend->getCachedInvalid(key).empty());
    EXPECT_EQ(backend->stats().invalid_cache_size, 1);
    EXPECT_EQ(backend->stats().compile_invalid, 1);

    EXPECT_TRUE(backend->compileNow(key).is_invalid);
    EXPECT_EQ(backend->stats().compile_total, 1);
}

TEST(XGrammarBackendCppTest, UnknownGrammarTypeIsInvalid) {
    auto backend = makeBackend(baseOptions());

    auto result = backend->compileNow(GrammarKeyCpp{"no_such_type", kSimpleSchema});
    EXPECT_TRUE(result.is_invalid);
    EXPECT_FALSE(result.is_overloaded);
    EXPECT_NE(result.error_message.find("unknown grammar type"), std::string::npos);
}

TEST(XGrammarBackendCppTest, DisabledTimeoutBypassesTheBudget) {
    auto options               = baseOptions();
    options.compile_timeout_ms = 0;
    auto backend               = makeBackend(options);

    const auto      caller = std::this_thread::get_id();
    std::thread::id ran_on;
    backend->setCompileFnForTest([&](const GrammarKeyCpp&) {
        ran_on = std::this_thread::get_id();
        // Longer than any budget a pooled path would have applied.
        std::this_thread::sleep_for(50ms);
        CompileResult out;
        out.is_invalid    = true;
        out.error_message = "slow but permitted";
        return out;
    });

    auto result = backend->compileNow(jsonKey(kSimpleSchema));
    EXPECT_TRUE(result.is_invalid);
    EXPECT_FALSE(result.is_overloaded);
    // No pool exists, so the compile must have run inline on the caller thread.
    EXPECT_EQ(ran_on, caller);
    EXPECT_EQ(backend->stats().compile_timeout, 0);
}

TEST(XGrammarBackendCppTest, TimeoutReportsOverloadWithoutPoisoningCache) {
    auto options                = baseOptions();
    options.compile_timeout_ms  = 50;
    options.compile_concurrency = 1;
    auto backend                = makeBackend(options);

    BlockingCompile compile;
    backend->setCompileFnForTest(compile.fn());
    ScopedRelease cleanup(compile, backend);

    auto key    = jsonKey(kSimpleSchema);
    auto result = backend->compileNow(key);

    EXPECT_TRUE(result.is_overloaded);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_FALSE(result.compiled);
    EXPECT_NE(result.error_message.find("50ms budget"), std::string::npos);
    EXPECT_EQ(backend->stats().compile_timeout, 1);

    // An overload is not a verdict on the grammar, so neither cache may record it.
    EXPECT_TRUE(backend->getCachedInvalid(key).empty());
    EXPECT_FALSE(backend->getCached(key));
    EXPECT_EQ(backend->stats().cache_size, 0);
    EXPECT_EQ(backend->stats().invalid_cache_size, 0);
}

TEST(XGrammarBackendCppTest, TimedOutCompileStillPublishesSoRetryIsCheap) {
    auto options                = baseOptions();
    options.compile_timeout_ms  = 50;
    options.compile_concurrency = 1;
    auto backend                = makeBackend(options);

    BlockingCompile compile;
    backend->setCompileFnForTest(compile.fn());
    // Without this an assertion failure below would leave a worker blocked inside the fake, and the
    // backend destructor joins the pool: the binary would hang instead of reporting the failure.
    ScopedRelease cleanup(compile, backend);

    auto key = jsonKey(kSimpleSchema);
    ASSERT_TRUE(backend->compileNow(key).is_overloaded);

    // The abandoned compile keeps running, so its cost is paid once rather than once per retry.
    compile.release();
    ASSERT_TRUE(waitFor([&] { return backend->stats().invalid_cache_size == 1; }));
    EXPECT_EQ(compile.entered(), 1);

    auto retry = backend->compileNow(key);
    EXPECT_TRUE(retry.is_invalid);
    EXPECT_FALSE(retry.is_overloaded);
    EXPECT_EQ(backend->stats().compile_total, 1);
    EXPECT_TRUE(waitFor([&] { return backend->stats().inflight == 0; }));
}

TEST(XGrammarBackendCppTest, ConcurrentCallersShareOneCompile) {
    auto options                = baseOptions();
    options.compile_timeout_ms  = 30000;
    options.compile_concurrency = 2;
    auto backend                = makeBackend(options);

    BlockingCompile compile;
    backend->setCompileFnForTest(compile.fn());

    constexpr int            kCallers = 4;
    auto                     key      = jsonKey(kSimpleSchema);
    std::atomic<int>         invalid_results{0};
    std::vector<std::thread> threads;
    threads.reserve(kCallers);
    ScopedJoin cleanup(compile, threads);
    for (int i = 0; i < kCallers; ++i) {
        threads.emplace_back([&] {
            if (backend->compileNow(key).is_invalid) {
                invalid_results.fetch_add(1);
            }
        });
    }

    // Hold the leader inside the compile until every follower has joined it, so the dedup count is exact
    // rather than a race between the followers and the leader publishing to the cache.
    ASSERT_TRUE(compile.waitForEntered(1));
    ASSERT_TRUE(waitFor([&] { return backend->stats().compile_dedup == kCallers - 1; }));
    compile.release();
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(invalid_results.load(), kCallers);
    EXPECT_EQ(compile.entered(), 1);
    EXPECT_EQ(backend->stats().compile_total, 1);
    EXPECT_EQ(backend->stats().compile_dedup, kCallers - 1);
    EXPECT_EQ(backend->stats().inflight, 0);
}

TEST(XGrammarBackendCppTest, OutstandingCompilesAreBounded) {
    auto options                = baseOptions();
    options.compile_timeout_ms  = 50;
    options.compile_concurrency = 1;
    options.compile_queue_size  = 1;
    auto backend                = makeBackend(options);

    BlockingCompile compile;
    backend->setCompileFnForTest(compile.fn());
    ScopedRelease cleanup(compile, backend);

    // Occupy the running slot first, so the queue accounting is not disturbed by a worker picking items up.
    ASSERT_TRUE(backend->compileNow(jsonKey("occupy-the-worker")).is_overloaded);
    ASSERT_TRUE(compile.waitForEntered(1));

    // Distinct keys, otherwise dedup would collapse them onto the running compile.
    bool saw_rejection = false;
    for (int i = 0; i < 8 && !saw_rejection; ++i) {
        auto result = backend->compileNow(jsonKey("fill-" + std::to_string(i)));
        ASSERT_TRUE(result.is_overloaded);
        if (result.error_message.find("too many outstanding") != std::string::npos) {
            saw_rejection = true;
        }
    }

    EXPECT_TRUE(saw_rejection) << "queue cap never rejected a compile";
    EXPECT_GT(backend->stats().compile_rejected, 0);
    // A rejection must not be mistaken for a verdict on the grammar.
    EXPECT_EQ(backend->stats().cache_size, 0);
    EXPECT_EQ(backend->stats().invalid_cache_size, 0);
}

TEST(XGrammarBackendCppTest, AllocationFailureIsTransientNotInvalid) {
    auto options               = baseOptions();
    options.compile_timeout_ms = 0;
    auto backend               = makeBackend(options);

    backend->setCompileFnForTest([](const GrammarKeyCpp&) -> CompileResult { throw std::bad_alloc(); });

    auto key    = jsonKey(kSimpleSchema);
    auto result = backend->compileNow(key);

    // Caching an OOM as "invalid grammar" would permanently reject a valid schema after one transient
    // failure, which is exactly the pathological-compile case this guard exists for.
    EXPECT_TRUE(result.is_overloaded);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_TRUE(backend->getCachedInvalid(key).empty());
    EXPECT_EQ(backend->stats().invalid_cache_size, 0);
    EXPECT_EQ(backend->stats().cache_size, 0);
}

TEST(XGrammarBackendCppTest, CompilesEveryGrammarTypeThroughThePool) {
    auto options               = baseOptions();
    options.compile_timeout_ms = 30000;
    auto backend               = makeBackend(options);

    EXPECT_TRUE(backend->compileNow(jsonKey(kSimpleSchema)).compiled);
    EXPECT_TRUE(backend->compileNow(GrammarKeyCpp{"regex", "[a-z]+"}).compiled);
    EXPECT_TRUE(backend->compileNow(GrammarKeyCpp{"ebnf", "root ::= \"a\""}).compiled);
    EXPECT_EQ(backend->stats().cache_size, 3);
    EXPECT_EQ(backend->stats().inflight, 0);
}

TEST(XGrammarBackendCppTest, ClearDropsCachedVerdicts) {
    auto backend = makeBackend(baseOptions());

    ASSERT_TRUE(backend->compileNow(jsonKey(kSimpleSchema)).compiled);
    ASSERT_TRUE(backend->compileNow(jsonKey("{ not json at all")).is_invalid);
    ASSERT_EQ(backend->stats().cache_size, 1);
    ASSERT_EQ(backend->stats().invalid_cache_size, 1);

    backend->clear();
    EXPECT_EQ(backend->stats().cache_size, 0);
    EXPECT_EQ(backend->stats().invalid_cache_size, 0);
    // The byte accounting has to be reset with the entries, otherwise the cache would evict against a
    // phantom occupancy for the rest of the process.
    EXPECT_EQ(backend->stats().cache_bytes, 0);

    ASSERT_TRUE(backend->compileNow(jsonKey(kSimpleSchema)).compiled);
    EXPECT_EQ(backend->stats().cache_size, 1);
    EXPECT_GT(backend->stats().cache_bytes, 0);
}

TEST(XGrammarBackendCppTest, EvictsLeastRecentlyUsedGrammarOverBudget) {
    auto entry_bytes = measureEntryBytes({kSchemaA, kSchemaB, kSchemaC});
    ASSERT_EQ(entry_bytes.size(), 3u);
    ASSERT_GT(entry_bytes[0], 0);
    ASSERT_EQ(entry_bytes[0], entry_bytes[1]);
    ASSERT_EQ(entry_bytes[1], entry_bytes[2]) << "schemas must cost the same for one eviction to make room";

    auto options                 = baseOptions();
    options.compiler_cache_bytes = entry_bytes[0] + entry_bytes[1];
    auto backend                 = makeBackend(options);

    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaA)).compiled);
    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaB)).compiled);
    ASSERT_EQ(backend->stats().cache_size, 2);
    ASSERT_EQ(backend->stats().cache_evicted, 0);

    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaC)).compiled);

    EXPECT_LE(backend->stats().cache_bytes, options.compiler_cache_bytes);
    EXPECT_EQ(backend->stats().cache_size, 2);
    EXPECT_EQ(backend->stats().cache_evicted, 1);
    // The first grammar was the least recently used, so it is the one that pays for the third.
    EXPECT_FALSE(backend->getCached(jsonKey(kSchemaA)));
    EXPECT_TRUE(backend->getCached(jsonKey(kSchemaB)));
    EXPECT_TRUE(backend->getCached(jsonKey(kSchemaC)));
}

TEST(XGrammarBackendCppTest, EvictionKeepsAHeldGrammarUsable) {
    auto entry_bytes = measureEntryBytes({kSchemaA, kSchemaB});
    ASSERT_EQ(entry_bytes.size(), 2u);
    ASSERT_EQ(entry_bytes[0], entry_bytes[1]);

    auto options = baseOptions();
    // Room for one grammar, so the second compile must evict the first.
    options.compiler_cache_bytes = entry_bytes[0];
    auto backend                 = makeBackend(options);

    auto held = backend->compileNow(jsonKey(kSchemaA)).compiled;
    ASSERT_TRUE(held);
    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaB)).compiled);
    ASSERT_FALSE(backend->getCached(jsonKey(kSchemaA)));

    // The caller owns its grammar, so a request still streaming when the cache drops it keeps working.
    // Without this the ceiling would trade an OOM for a use-after-free.
    EXPECT_EQ(held.use_count(), 1);
    EXPECT_GT(held->MemorySizeBytes(), 0u);
}

TEST(XGrammarBackendCppTest, CacheHitProtectsGrammarFromEviction) {
    auto entry_bytes = measureEntryBytes({kSchemaA, kSchemaB, kSchemaC});
    ASSERT_EQ(entry_bytes.size(), 3u);
    ASSERT_EQ(entry_bytes[0], entry_bytes[1]);
    ASSERT_EQ(entry_bytes[1], entry_bytes[2]) << "schemas must cost the same for one eviction to make room";

    auto options                 = baseOptions();
    options.compiler_cache_bytes = entry_bytes[0] + entry_bytes[1];
    auto backend                 = makeBackend(options);

    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaA)).compiled);
    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaB)).compiled);

    // Reads have to count as use, otherwise a grammar every request needs would be evicted by whichever
    // one-off schema happened to compile most recently.
    ASSERT_TRUE(backend->getCached(jsonKey(kSchemaA)));
    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaC)).compiled);

    EXPECT_EQ(backend->stats().cache_evicted, 1);
    EXPECT_TRUE(backend->getCached(jsonKey(kSchemaA)));
    EXPECT_FALSE(backend->getCached(jsonKey(kSchemaB)));
}

TEST(XGrammarBackendCppTest, GrammarLargerThanCeilingIsKeptSoRetriesDoNotRecompile) {
    auto entry_bytes = measureEntryBytes({kSchemaA});
    ASSERT_EQ(entry_bytes.size(), 1u);

    auto options = baseOptions();
    // One byte short of what this grammar needs.
    options.compiler_cache_bytes = entry_bytes[0] - 1;
    ASSERT_GT(options.compiler_cache_bytes, 0);
    auto backend = makeBackend(options);

    auto key    = jsonKey(kSchemaA);
    auto result = backend->compileNow(key);

    // Dropping it instead would recompile the grammar on every single request, so the ceiling gives way by
    // this one entry. Nothing else is resident to evict, and the next store of another grammar drops it.
    EXPECT_TRUE(result.compiled);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_FALSE(result.is_overloaded);
    EXPECT_EQ(backend->stats().cache_size, 1);
    EXPECT_EQ(backend->stats().cache_bytes, entry_bytes[0]);
    // Counted apart from evictions: this is the cache exceeding its bound, not trimming itself to it.
    EXPECT_EQ(backend->stats().cache_oversized, 1);
    EXPECT_EQ(backend->stats().cache_evicted, 0);

    EXPECT_TRUE(backend->compileNow(key).compiled);
    EXPECT_EQ(backend->stats().compile_total, 1);

    // The overshoot is transient: storing any other verdict evicts the oversized one.
    ASSERT_TRUE(backend->compileNow(jsonKey(kSchemaB)).compiled);
    EXPECT_EQ(backend->stats().cache_size, 1);
    EXPECT_EQ(backend->stats().cache_evicted, 1);
    EXPECT_FALSE(backend->getCached(key));
}

TEST(XGrammarBackendCppTest, OversizedGrammarThatAlsoTimesOutStillConverges) {
    // The two guards meet here: a grammar slow enough to blow the wait budget and large enough to exceed
    // the cache ceiling. If the oversized verdict were dropped, the retry would recompile it, time out
    // again and drop it again - a livelock that burns a compile slot per attempt and never serves anyone.
    auto options = baseOptions();
    // Below the flat per-entry overhead, so any verdict at all is the oversized case.
    options.compiler_cache_bytes = 1;
    options.compile_timeout_ms   = 50;
    options.compile_concurrency  = 1;
    auto backend                 = makeBackend(options);

    BlockingCompile compile;
    backend->setCompileFnForTest(compile.fn());
    ScopedRelease cleanup(compile, backend);

    auto key = jsonKey(kSimpleSchema);
    ASSERT_TRUE(backend->compileNow(key).is_overloaded);
    ASSERT_EQ(backend->stats().compile_timeout, 1);

    compile.release();
    // Waits on the oversized counter rather than the cache size: the store publishes the entry under the
    // cache lock and only bumps this counter after releasing it, so the size alone would race.
    ASSERT_TRUE(waitFor([&] { return backend->stats().cache_oversized == 1; }));
    EXPECT_EQ(backend->stats().invalid_cache_size, 1);

    // The retry is served from cache rather than starting the same doomed compile over again.
    auto retry = backend->compileNow(key);
    EXPECT_TRUE(retry.is_invalid);
    EXPECT_FALSE(retry.is_overloaded);
    EXPECT_EQ(backend->stats().compile_total, 1);
    EXPECT_EQ(compile.entered(), 1);
}

TEST(XGrammarBackendCppTest, InvalidVerdictsShareTheByteBudget) {
    // Invalid verdicts are keyed by the whole schema text and hold their error message, so leaving them
    // outside the ceiling would reopen the unbounded growth this guard closes.
    const std::string message(1024, 'x');

    auto options               = baseOptions();
    options.compile_timeout_ms = 0;
    // Room for one verdict of this size, not two.
    options.compiler_cache_bytes = static_cast<int64_t>(message.size()) + 512;
    auto backend                 = makeBackend(options);
    backend->setCompileFnForTest([&](const GrammarKeyCpp&) {
        CompileResult out;
        out.is_invalid    = true;
        out.error_message = message;
        return out;
    });

    auto first  = jsonKey("first-broken-schema");
    auto second = jsonKey("second-broken-schema");
    ASSERT_TRUE(backend->compileNow(first).is_invalid);
    ASSERT_EQ(backend->stats().invalid_cache_size, 1);

    ASSERT_TRUE(backend->compileNow(second).is_invalid);

    EXPECT_EQ(backend->stats().invalid_cache_size, 1);
    EXPECT_LE(backend->stats().cache_bytes, options.compiler_cache_bytes);
    EXPECT_EQ(backend->stats().cache_evicted, 1);
    EXPECT_TRUE(backend->getCachedInvalid(first).empty());
    EXPECT_FALSE(backend->getCachedInvalid(second).empty());
}

TEST(XGrammarBackendCppTest, NonPositiveCeilingKeepsEveryVerdict) {
    auto options                 = baseOptions();
    options.compiler_cache_bytes = 0;
    auto backend                 = makeBackend(options);

    for (const auto* schema : {kSchemaA, kSchemaB, kSchemaC}) {
        ASSERT_TRUE(backend->compileNow(jsonKey(schema)).compiled);
    }

    EXPECT_EQ(backend->stats().cache_size, 3);
    EXPECT_GT(backend->stats().cache_bytes, 0);
    EXPECT_EQ(backend->stats().cache_evicted, 0);
}

}  // namespace
}  // namespace rtp_llm
