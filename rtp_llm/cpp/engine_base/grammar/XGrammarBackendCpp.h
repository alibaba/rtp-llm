#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/grammar.h>
#include <xgrammar/tokenizer_info.h>

#include "autil/ThreadPool.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class RtpGrammarMatcher;
class RtpLLMGrammarMetricsCollector;

struct GrammarKeyCpp {
    std::string key_type;
    std::string key_string;

    bool empty() const noexcept {
        return key_type.empty() || key_string.empty();
    }

    std::string id() const {
        std::string out;
        out.reserve(key_type.size() + key_string.size() + 1);
        out.append(key_type);
        out.push_back('\x1f');
        out.append(key_string);
        return out;
    }
};

struct CompileResult {
    std::shared_ptr<xgrammar::CompiledGrammar> compiled;
    // The grammar itself is broken and can never compile, so the verdict is permanently cacheable.
    bool is_invalid = false;
    // The compile did not finish within its budget, or was refused because too many compiles were
    // already outstanding. The grammar may well be valid, so the verdict must not be cached.
    bool        is_overloaded = false;
    std::string error_message;
};

struct XGrammarBackendOptions {
    bool any_whitespace        = true;
    bool strict_mode           = true;
    int  max_compiler_threads  = 8;
    bool enable_compiler_cache = true;
    // Ceiling on cached compiled grammars, measured with xgrammar's own MemorySizeBytes() estimate. It is
    // handed to xgrammar's compiler cache and applied again to the verdict cache here, whose strong
    // references would otherwise keep alive exactly what xgrammar evicts. The two hold the same
    // CompiledGrammar objects, so the resident total is not twice the ceiling: only xgrammar's rule-level
    // cache, one third of its budget, is distinct, putting the per-process bound near 4/3 of this. Every
    // rank owns a backend, so a DP8 node holds eight times that. The ceiling is soft by one entry: a
    // verdict too large to fit under it on its own is kept anyway, because dropping it would recompile the
    // grammar on every request. <=0 removes the ceiling and restores unbounded growth.
    int64_t compiler_cache_bytes = 2LL << 30;
    // Wall-clock budget a caller waits for one compile. <=0 keeps the legacy unbounded synchronous
    // compile on the caller thread. The default is short on purpose: the caller thread is an enqueue
    // thread, so waiting there costs latency for a request whose retry will hit the cache anyway.
    int compile_timeout_ms = 50;
    // Compiles allowed to run at once. Each compile internally fans out over max_compiler_threads, so
    // this multiplies CPU usage. Before this guard every caller compiled inline with no bound at all,
    // so this is a cap on what used to be unbounded, not a restriction of previous behaviour.
    int compile_concurrency = 16;
    // Queued compiles. The bound is soft: autil frees a slot when a worker picks an item up rather than
    // when it finishes, so up to compile_queue_size + compile_concurrency compiles can be outstanding.
    int                             compile_queue_size = 64;
    std::optional<std::vector<int>> override_stop_tokens;
};

// Cumulative counters plus resident sizes, used for metrics reporting and assertions in tests.
struct GrammarBackendStats {
    int64_t compile_total      = 0;
    int64_t compile_invalid    = 0;
    int64_t compile_timeout    = 0;
    int64_t compile_rejected   = 0;
    int64_t compile_dedup      = 0;
    int64_t cache_hit          = 0;
    int64_t cache_miss         = 0;
    int64_t invalid_hit        = 0;
    int64_t cache_size         = 0;
    int64_t invalid_cache_size = 0;
    // Accounted size of the verdict cache, how many verdicts have been dropped to keep it there, and how
    // many single verdicts were large enough that keeping them pushed the cache past its ceiling.
    int64_t cache_bytes     = 0;
    int64_t cache_evicted   = 0;
    int64_t cache_oversized = 0;
    int64_t inflight        = 0;
};

class XGrammarBackendCpp {
public:
    XGrammarBackendCpp(const std::string&            tokenizer_info_json,
                       const XGrammarBackendOptions& options,
                       kmonitor::MetricsReporterPtr  metrics_reporter = nullptr);
    ~XGrammarBackendCpp();

    XGrammarBackendCpp(const XGrammarBackendCpp&)            = delete;
    XGrammarBackendCpp& operator=(const XGrammarBackendCpp&) = delete;

    std::shared_ptr<xgrammar::CompiledGrammar> getCached(const GrammarKeyCpp& key) const;
    std::string                                getCachedInvalid(const GrammarKeyCpp& key) const;

    // Compiles `key`, publishing the verdict into the cache before returning. Concurrent callers of
    // the same key share a single compile. When a compile exceeds `compile_timeout_ms` the caller
    // gets an overloaded result while the compile keeps running in the background, so its cost is
    // paid once instead of once per retry.
    CompileResult compileNow(const GrammarKeyCpp& key);

    std::shared_ptr<RtpGrammarMatcher> createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                                     bool                                       require_reasoning,
                                                     std::optional<std::vector<int>>            think_end_token_ids,
                                                     bool terminate_without_stop_token = false);

    GrammarBackendStats stats() const;

    void clear();

    // Replaces the xgrammar call inside the compile path so tests can drive the timeout, dedup and
    // queue-cap behaviour with a fake that blocks or throws on demand, instead of racing a real compile
    // whose cost depends on xgrammar internals. Must be installed before any compile is issued.
    using CompileFn = std::function<CompileResult(const GrammarKeyCpp&)>;
    void setCompileFnForTest(CompileFn fn);

private:
    static std::string sanitizeStructuralTag(const std::string& tag_json);

    // Removes the in-flight entry for one key exactly once, on whichever path leaves the compile task.
    class InflightGuard {
    public:
        InflightGuard(XGrammarBackendCpp& owner, const std::string& id): owner_(owner), id_(id) {}
        ~InflightGuard() {
            release();
        }

        InflightGuard(const InflightGuard&)            = delete;
        InflightGuard& operator=(const InflightGuard&) = delete;

        void release() noexcept;

    private:
        XGrammarBackendCpp& owner_;
        const std::string&  id_;
        bool                released_ = false;
    };

    std::optional<CompileResult> lookupVerdict(const std::string& id) const;
    CompileResult                invokeCompiler(const GrammarKeyCpp& key);
    CompileResult                compileSync(const GrammarKeyCpp& key);
    CompileResult                runCompileTask(const GrammarKeyCpp& key, const std::string& id);
    void                         storeResult(const GrammarKeyCpp& key, const CompileResult& result);
    void                         reportLookup(bool hit, bool invalid_hit) const;
    void                         reportCompile(const CompileResult& result, int64_t latency_us) const;
    void                         reportOverload() const;
    void                         fillResidentGauges(RtpLLMGrammarMetricsCollector& collector) const;

    // Flat cost of the map node, bucket slot and LRU node that every cached verdict needs on top of its
    // payload. Approximate on purpose: it only has to keep the accounting from under-counting entries by
    // an order of magnitude.
    static constexpr int64_t kEntryOverheadBytes = 128;

    // One cached verdict: either a compiled grammar or the reason the grammar can never compile.
    struct CacheEntry {
        // Null marks an invalid verdict, whose reason is error_message.
        std::shared_ptr<xgrammar::CompiledGrammar> compiled;
        std::string                                error_message;
        int64_t                                    bytes = 0;
        std::list<std::string>::iterator           lru_it;
    };
    using CacheMap = std::unordered_map<std::string, CacheEntry>;

    // The three helpers below all require cache_mutex_ to be held.
    void    touchLocked(CacheEntry& entry) const;
    void    eraseLocked(CacheMap::iterator it);
    int64_t evictLocked();

private:
    const XGrammarBackendOptions options_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    xgrammar::TokenizerInfo      tokenizer_info_;
    xgrammar::GrammarCompiler    compiler_;
    // Empty in production; see setCompileFnForTest.
    CompileFn compile_fn_;

    // A lookup refreshes LRU order, so the whole cache is mutable behind the const read API.
    mutable std::mutex cache_mutex_;
    mutable CacheMap   cache_;
    // Most recently used at the front. Holds a second copy of each id, which is charged to the entry's
    // byte cost so a cache of huge schemas cannot silently overshoot the ceiling.
    mutable std::list<std::string> lru_;
    mutable int64_t                cache_bytes_   = 0;
    mutable int64_t                invalid_count_ = 0;

    // Null when compile_timeout_ms <= 0, which selects the synchronous compile path.
    autil::ThreadPoolBasePtr                                           compile_pool_;
    mutable std::mutex                                                 inflight_mutex_;
    std::unordered_map<std::string, std::shared_future<CompileResult>> inflight_;

    mutable std::atomic<int64_t> compile_total_{0};
    mutable std::atomic<int64_t> compile_invalid_{0};
    mutable std::atomic<int64_t> compile_timeout_{0};
    mutable std::atomic<int64_t> compile_rejected_{0};
    mutable std::atomic<int64_t> compile_dedup_{0};
    mutable std::atomic<int64_t> cache_hit_{0};
    mutable std::atomic<int64_t> cache_miss_{0};
    mutable std::atomic<int64_t> invalid_hit_{0};
    mutable std::atomic<int64_t> cache_evicted_{0};
    mutable std::atomic<int64_t> cache_oversized_{0};
};

using XGrammarBackendCppPtr = std::shared_ptr<XGrammarBackendCpp>;

}  // namespace rtp_llm
