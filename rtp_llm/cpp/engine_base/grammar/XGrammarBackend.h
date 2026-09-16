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
#include <utility>

#include <xgrammar/compiler.h>
#include <xgrammar/grammar.h>
#include <xgrammar/tokenizer_info.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "autil/ThreadPool.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class RtpGrammarMatcher;
class RtpLLMGrammarMetricsCollector;

// Grammar request key shared by scheduler and backend.
struct GrammarKeyCpp {
    std::string key_type;    // "json" / "regex" / "ebnf" / "structural_tag"
    std::string key_string;  // schema / pattern / EBNF / structural tag JSON

    bool empty() const noexcept {
        return key_type.empty();
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

struct GrammarCompileResult {
    std::shared_ptr<xgrammar::CompiledGrammar> compiled;
    absl::Status                               status = absl::OkStatus();
};

struct GrammarBackendStats {
    int64_t compile_total               = 0;
    int64_t compile_invalid             = 0;
    int64_t compile_timeout             = 0;
    int64_t compile_rejected            = 0;
    int64_t compile_dedup               = 0;
    int64_t cache_hit                   = 0;
    int64_t cache_miss                  = 0;
    int64_t invalid_hit                 = 0;
    int64_t cache_size                  = 0;
    int64_t invalid_cache_size          = 0;
    int64_t verdict_cache_bytes         = 0;
    int64_t total_cache_budget_bytes    = 0;
    int64_t compiler_cache_budget_bytes = 0;
    int64_t verdict_cache_budget_bytes  = 0;
    int64_t cache_evicted               = 0;
    int64_t cache_oversized             = 0;
    int64_t inflight                    = 0;
};

// Owns the xgrammar compiler and a bounded, singleflight verdict cache; thread-safe, no GIL.
class XGrammarBackend {
public:
    ~XGrammarBackend();

    XGrammarBackend(const XGrammarBackend&)            = delete;
    XGrammarBackend& operator=(const XGrammarBackend&) = delete;
    XGrammarBackend(XGrammarBackend&&)                 = delete;
    XGrammarBackend& operator=(XGrammarBackend&&)      = delete;

    // Empty tokenizer info intentionally disables structured output. Non-empty
    // tokenizer info is a startup compatibility contract: deserialization or
    // backend construction failures must propagate and abort engine startup.
    static std::shared_ptr<XGrammarBackend> create(const std::string&           tokenizer_info_json,
                                                   const GrammarConfig&         cfg,
                                                   kmonitor::MetricsReporterPtr metrics_reporter = nullptr);

    // Creates a fresh per-stream matcher from a grammar key. Matcher state itself
    // is never shared across streams.
    absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>> createMatcherFromKey(const GrammarKeyCpp& key);

    GrammarBackendStats stats() const;
    void                clear();

    // Test seams for deterministic timeout, queue, singleflight, failure classification, and metrics tests.
    using CompileFn       = std::function<GrammarCompileResult(const GrammarKeyCpp&)>;
    using MetricsReportFn = std::function<void(const RtpLLMGrammarMetricsCollector&)>;
    void                 setCompileFnForTest(CompileFn fn);
    void                 setMetricsReportFnForTest(MetricsReportFn fn);
    GrammarCompileResult compileNow(const GrammarKeyCpp& key);

private:
    struct Options {
        bool    any_whitespace                    = true;
        bool    strict_mode                       = true;
        bool    terminate_without_stop_token      = false;
        int     max_compiler_threads              = 8;
        int64_t total_cache_budget_bytes          = 2LL << 30;
        int64_t compiler_cache_budget_bytes       = 1LL << 30;
        int64_t verdict_cache_budget_bytes        = 1LL << 30;
        int     compile_timeout_ms                = 2000;
        int     compile_concurrency               = 1;
        int     compile_queue_size                = 2;
    };

    struct CacheEntry {
        std::shared_ptr<xgrammar::CompiledGrammar> compiled;
        std::string                                error_message;
        int64_t                                    bytes = 0;
        std::list<std::string>::iterator           lru_it;
    };
    using CacheMap = std::unordered_map<std::string, CacheEntry>;

    class InflightGuard {
    public:
        InflightGuard(XGrammarBackend& owner, std::string id): owner_(owner), id_(std::move(id)) {}
        ~InflightGuard();

        InflightGuard(const InflightGuard&)            = delete;
        InflightGuard& operator=(const InflightGuard&) = delete;

        void release() noexcept;

    private:
        XGrammarBackend& owner_;
        std::string      id_;
        bool             released_ = false;
    };

    XGrammarBackend(const xgrammar::TokenizerInfo& tokenizer_info,
                    const Options&                 options,
                    kmonitor::MetricsReporterPtr   metrics_reporter);

    static Options optionsFromConfig(const GrammarConfig& cfg);

    absl::StatusOr<std::shared_ptr<xgrammar::CompiledGrammar>> compile(const GrammarKeyCpp& key);
    GrammarCompileResult                                       invokeCompiler(const GrammarKeyCpp& key);
    GrammarCompileResult                                       compileSync(const GrammarKeyCpp& key);
    GrammarCompileResult runCompileTask(const GrammarKeyCpp& key, const std::string& id);

    absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>>
    createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled);

    std::optional<GrammarCompileResult> lookupVerdict(const std::string& id) const;
    void                                storeResult(const GrammarKeyCpp& key, const GrammarCompileResult& result);
    void                                touchLocked(CacheEntry& entry) const;
    void                                eraseLocked(CacheMap::iterator it);
    int64_t                             evictLocked();

    void reportLookup(bool hit, bool invalid_hit, std::optional<int64_t> inflight = std::nullopt) const;
    void reportCompile(const GrammarCompileResult& result, int64_t latency_us) const;
    void reportOverload() const;
    void fillResidentGauges(RtpLLMGrammarMetricsCollector& collector,
                            std::optional<int64_t>          inflight = std::nullopt) const;

    static constexpr int64_t kEntryOverheadBytes = 128;

private:
    const Options                options_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    xgrammar::TokenizerInfo      tokenizer_info_;
    xgrammar::GrammarCompiler    compiler_;
    CompileFn                    compile_fn_;
    MetricsReportFn              metrics_report_fn_for_test_;

    mutable std::mutex cache_mutex_;
    mutable CacheMap   cache_;
    mutable std::list<std::string> lru_;
    mutable int64_t                cache_bytes_   = 0;
    mutable int64_t                invalid_count_ = 0;

    autil::ThreadPoolBasePtr                                                  compile_pool_;
    mutable std::mutex                                                        inflight_mutex_;
    std::unordered_map<std::string, std::shared_future<GrammarCompileResult>> inflight_;

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

}  // namespace rtp_llm
