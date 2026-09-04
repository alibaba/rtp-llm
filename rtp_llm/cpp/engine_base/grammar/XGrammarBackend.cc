#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <exception>
#include <iterator>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include <xgrammar/exception.h>

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

int64_t elapsedMsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
}

std::string serializationErrorToString(const xgrammar::SerializationError& error) {
    return std::visit([](const auto& e) { return e.GetType() + ": " + std::string(e.what()); }, error);
}

bool isInvalid(const GrammarCompileResult& result) {
    return result.status.code() == absl::StatusCode::kInvalidArgument;
}

bool isTransient(const GrammarCompileResult& result) {
    return !result.compiled && !isInvalid(result);
}

bool isExplicitGrammarParseError(const std::runtime_error& error) {
    std::string message = error.what();
    std::transform(message.begin(), message.end(), message.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    constexpr const char* markers[] = {
        "invalid json",
        "json parse",
        "json parsing error",
        "failed to parse json",
        "invalid regex",
        "regex parse",
        "regex parsing error",
        "failed to parse regex",
        "invalid ebnf",
        "ebnf parse",
        "ebnf parsing error",
        "ebnf lexer error",
        "failed to parse ebnf",
        "invalid grammar",
        "grammar parse",
        "grammar parsing error",
        "grammar lexer error",
        "invalid structural tag",
        "structural tag parse",
        "structural tag parsing error",
        "syntax error",
        "parser error",
        "lexer error",
    };
    return std::any_of(std::begin(markers), std::end(markers), [&message](const char* marker) {
        return message.find(marker) != std::string::npos;
    });
}

}  // namespace

std::shared_ptr<XGrammarBackend> XGrammarBackend::create(const std::string&           tokenizer_info_json,
                                                         const GrammarConfig&         cfg,
                                                         kmonitor::MetricsReporterPtr metrics_reporter) {
    try {
        if (tokenizer_info_json.empty()) {
            RTP_LLM_LOG_INFO("XGrammarBackend::create: structured output disabled (TokenizerInfo empty)");
            return nullptr;
        }
        Options opts   = optionsFromConfig(cfg);
        auto    result = xgrammar::TokenizerInfo::DeserializeJSON(tokenizer_info_json);
        if (std::holds_alternative<xgrammar::SerializationError>(result)) {
            throw std::runtime_error("tokenizer info deserialize failed: "
                                     + serializationErrorToString(std::get<xgrammar::SerializationError>(result)));
        }
        const auto& serialized_tokenizer_info = std::get<xgrammar::TokenizerInfo>(result);
        // xgrammar derives its token-id lookup in the constructor but does not serialize it.
        // Rebuild from the decoded vocabulary without decoding byte-level tokens twice.
        const xgrammar::TokenizerInfo tokenizer_info(serialized_tokenizer_info.GetDecodedVocab(),
                                                     xgrammar::VocabType::RAW,
                                                     serialized_tokenizer_info.GetVocabSize(),
                                                     serialized_tokenizer_info.GetStopTokenIds(),
                                                     serialized_tokenizer_info.GetAddPrefixSpace());
        if (tokenizer_info.GetVocabSize() <= 0) {
            throw std::runtime_error("tokenizer vocab is empty");
        }
        auto backend = std::shared_ptr<XGrammarBackend>(
            new XGrammarBackend(tokenizer_info, opts, std::move(metrics_reporter)));
        RTP_LLM_LOG_INFO("XGrammarBackend::create: ready with serialized TokenizerInfo "
                         "(json_bytes=%zu, threads=%d)",
                         tokenizer_info_json.size(),
                         opts.max_compiler_threads);
        return backend;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("XGrammarBackend::create: initialization failed (%s); aborting startup", e.what());
        throw;
    } catch (...) {
        RTP_LLM_LOG_ERROR("XGrammarBackend::create: initialization failed (unknown); aborting startup");
        throw;
    }
}

XGrammarBackend::Options XGrammarBackend::optionsFromConfig(const GrammarConfig& cfg) {
    if (cfg.num_workers <= 0) {
        throw std::invalid_argument("grammar num_workers must be greater than 0 after config derivation");
    }
    if (cfg.compile_timeout_ms <= 0) {
        throw std::invalid_argument("grammar compile_timeout_ms must be greater than 0");
    }
    if (cfg.compile_concurrency <= 0) {
        throw std::invalid_argument("grammar compile_concurrency must be greater than 0");
    }
    if (cfg.compile_queue_size <= 0) {
        throw std::invalid_argument("grammar compile_queue_size must be greater than 0");
    }

    Options opts;
    opts.any_whitespace               = !cfg.constrained_json_disable_any_whitespace;
    opts.strict_mode                  = true;
    opts.terminate_without_stop_token = cfg.terminate_without_stop_token;
    opts.max_compiler_threads         = cfg.num_workers;
    opts.compile_timeout_ms           = cfg.compile_timeout_ms;
    opts.compile_concurrency          = cfg.compile_concurrency;
    opts.compile_queue_size           = cfg.compile_queue_size;
    if (cfg.compiler_cache_bytes <= 0) {
        opts.total_cache_budget_bytes    = -1;
        opts.compiler_cache_budget_bytes = -1;
        opts.verdict_cache_budget_bytes  = -1;
    } else {
        opts.total_cache_budget_bytes    = cfg.compiler_cache_bytes;
        opts.compiler_cache_budget_bytes =
            cfg.compiler_cache_bytes / 2 + cfg.compiler_cache_bytes % 2;
        opts.verdict_cache_budget_bytes = cfg.compiler_cache_bytes / 2;
    }
    return opts;
}

XGrammarBackend::XGrammarBackend(const xgrammar::TokenizerInfo& tokenizer_info,
                                 const XGrammarBackend::Options& options,
                                 kmonitor::MetricsReporterPtr metrics_reporter):
    options_(options),
    metrics_reporter_(std::move(metrics_reporter)),
    tokenizer_info_(tokenizer_info),
    compiler_(tokenizer_info_,
              options.max_compiler_threads,
              /*enable_cache=*/true,
              options.compiler_cache_budget_bytes) {
    const size_t concurrency = static_cast<size_t>(options_.compile_concurrency);
    const size_t queue_size  = static_cast<size_t>(options_.compile_queue_size);
    auto         pool        = std::make_shared<autil::LockFreeThreadPool>(
        concurrency, queue_size, nullptr, "XGrammarCompile", /*stopIfHasException=*/false);
    if (!pool->start()) {
        throw std::runtime_error("XGrammarBackend: failed to start grammar compile thread pool");
    }
    compile_pool_ = std::move(pool);
    RTP_LLM_LOG_INFO("XGrammarBackend init: vocab_size=%d, any_whitespace=%d, strict_mode=%d, "
                     "terminate_without_stop_token=%d, compiler_threads=%d, compile_timeout_ms=%d, "
                     "compile_concurrency=%d, compile_queue_size=%d, total_cache_budget_bytes=%lld, "
                     "compiler_cache_budget_bytes=%lld, verdict_cache_budget_bytes=%lld",
                     tokenizer_info_.GetVocabSize(),
                     static_cast<int>(options_.any_whitespace),
                     static_cast<int>(options_.strict_mode),
                     static_cast<int>(options_.terminate_without_stop_token),
                     options_.max_compiler_threads,
                     options_.compile_timeout_ms,
                     options_.compile_concurrency,
                     options_.compile_queue_size,
                     static_cast<long long>(options_.total_cache_budget_bytes),
                     static_cast<long long>(options_.compiler_cache_budget_bytes),
                     static_cast<long long>(options_.verdict_cache_budget_bytes));
}

XGrammarBackend::~XGrammarBackend() {
    if (compile_pool_) {
        // LockFreeThreadPool::stop() joins running workers, keeping this backend alive until every
        // background compile has stopped accessing its compiler and caches.
        RTP_LLM_LOG_INFO("XGrammarBackend shutdown: joining grammar compile pool");
        compile_pool_->stop();
        compile_pool_.reset();
        RTP_LLM_LOG_INFO("XGrammarBackend shutdown: grammar compile pool joined");
    }
}

void XGrammarBackend::touchLocked(CacheEntry& entry) const {
    lru_.splice(lru_.begin(), lru_, entry.lru_it);
}

void XGrammarBackend::eraseLocked(CacheMap::iterator it) {
    cache_bytes_ -= it->second.bytes;
    if (!it->second.compiled) {
        --invalid_count_;
    }
    lru_.erase(it->second.lru_it);
    cache_.erase(it);
}

int64_t XGrammarBackend::evictLocked() {
    const int64_t limit = options_.verdict_cache_budget_bytes;
    if (limit < 0) {
        return 0;
    }
    int64_t dropped = 0;
    while (cache_bytes_ > limit && !cache_.empty()) {
        auto it = cache_.find(lru_.back());
        if (it == cache_.end()) {
            lru_.pop_back();
            continue;
        }
        eraseLocked(it);
        ++dropped;
    }
    return dropped;
}

void XGrammarBackend::storeResult(const GrammarKeyCpp& key, const GrammarCompileResult& result) {
    // Only successful compiles and deterministic user-input failures are verdicts. Resource failures,
    // aborted work, and unknown exceptions are transient and must never poison the cache.
    if (isTransient(result)) {
        return;
    }

    const auto id = key.id();
    int64_t    bytes = kEntryOverheadBytes + 2 * static_cast<int64_t>(id.size());
    std::string error_message;
    if (result.compiled) {
        bytes += static_cast<int64_t>(result.compiled->MemorySizeBytes());
    } else {
        error_message = std::string(result.status.message());
        bytes += static_cast<int64_t>(error_message.size());
    }

    const int64_t limit = options_.verdict_cache_budget_bytes;
    if (limit >= 0 && bytes > limit) {
        cache_oversized_.fetch_add(1, std::memory_order_relaxed);
        RTP_LLM_LOG_WARNING("xgrammar verdict exceeds its shared-budget partition and will not be cached: "
                            "type=%s, bytes=%lld, verdict_budget=%lld, total_budget=%lld",
                            key.key_type.c_str(),
                            static_cast<long long>(bytes),
                            static_cast<long long>(limit),
                            static_cast<long long>(options_.total_cache_budget_bytes));
        return;
    }

    CacheEntry pending;
    pending.compiled      = result.compiled;
    pending.error_message = std::move(error_message);
    pending.bytes         = bytes;

    int64_t dropped  = 0;
    int64_t resident = 0;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (auto it = cache_.find(id); it != cache_.end()) {
            eraseLocked(it);
        }
        lru_.push_front(id);
        pending.lru_it = lru_.begin();
        cache_.emplace(id, std::move(pending));
        cache_bytes_ += bytes;
        if (!result.compiled) {
            ++invalid_count_;
        }
        dropped  = evictLocked();
        resident = cache_bytes_;
    }

    if (dropped > 0) {
        cache_evicted_.fetch_add(dropped, std::memory_order_relaxed);
        RTP_LLM_LOG_INFO("xgrammar verdict cache evicted %lld entries to stay under %lld bytes, now %lld bytes",
                         static_cast<long long>(dropped),
                         static_cast<long long>(limit),
                         static_cast<long long>(resident));
    }
}

XGrammarBackend::InflightGuard::~InflightGuard() {
    release();
}

void XGrammarBackend::InflightGuard::release() noexcept {
    if (released_) {
        return;
    }
    released_ = true;
    if (id_.empty()) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(owner_.inflight_mutex_);
        owner_.inflight_.erase(id_);
    } catch (...) {
    }
}

std::optional<GrammarCompileResult> XGrammarBackend::lookupVerdict(const std::string& id) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto                        it = cache_.find(id);
    if (it == cache_.end()) {
        return std::nullopt;
    }
    touchLocked(it->second);
    GrammarCompileResult cached;
    if (it->second.compiled) {
        cached.compiled = it->second.compiled;
    } else {
        cached.status = absl::InvalidArgumentError(it->second.error_message);
    }
    return cached;
}

GrammarCompileResult XGrammarBackend::invokeCompiler(const GrammarKeyCpp& key) {
    auto wrap = [](xgrammar::CompiledGrammar&& compiled) {
        GrammarCompileResult out;
        out.compiled = std::make_shared<xgrammar::CompiledGrammar>(std::move(compiled));
        return out;
    };

    const auto& grammar = key.key_string;
    if (key.key_type == "json") {
        return wrap(compiler_.CompileJSONSchema(
            grammar, options_.any_whitespace, std::nullopt, std::nullopt, options_.strict_mode));
    }
    if (key.key_type == "regex") {
        return wrap(compiler_.CompileRegex(grammar));
    }
    if (key.key_type == "ebnf") {
        return wrap(compiler_.CompileGrammar(grammar));
    }
    if (key.key_type == "structural_tag") {
        return wrap(compiler_.CompileStructuralTag(grammar));
    }
    GrammarCompileResult unknown;
    unknown.status = absl::InvalidArgumentError("unknown grammar type: " + key.key_type);
    return unknown;
}

GrammarCompileResult XGrammarBackend::compileSync(const GrammarKeyCpp& key) {
    compile_total_.fetch_add(1, std::memory_order_relaxed);
    const auto           begin = std::chrono::steady_clock::now();
    GrammarCompileResult result;
    try {
        result = compile_fn_ ? compile_fn_(key) : invokeCompiler(key);
        if (!result.compiled && result.status.ok()) {
            result.status = absl::UnknownError("grammar compiler returned no result");
        }
    } catch (const std::bad_alloc& e) {
        result.status = absl::ResourceExhaustedError(std::string("grammar compile ran out of memory: ") + e.what());
    } catch (const xgrammar::InvalidJSONError& e) {
        result.status = absl::InvalidArgumentError(e.what());
    } catch (const xgrammar::InvalidJSONSchemaError& e) {
        result.status = absl::InvalidArgumentError(e.what());
    } catch (const xgrammar::InvalidStructuralTagError& e) {
        result.status = absl::InvalidArgumentError(e.what());
    } catch (const std::invalid_argument& e) {
        result.status = absl::InvalidArgumentError(e.what());
    } catch (const std::runtime_error& e) {
        result.status = isExplicitGrammarParseError(e) ?
                            absl::InvalidArgumentError(e.what()) :
                            absl::UnknownError(std::string("grammar compiler runtime failure: ") + e.what());
    } catch (const std::exception& e) {
        result.status = absl::UnknownError(std::string("unexpected grammar compile error: ") + e.what());
    } catch (...) {
        result.status = absl::UnknownError("unknown exception during grammar compile");
    }

    if (isInvalid(result)) {
        compile_invalid_.fetch_add(1, std::memory_order_relaxed);
    }

    const auto elapsed_ms = elapsedMsSince(begin);
    if (result.compiled) {
        RTP_LLM_LOG_DEBUG("xgrammar compile ok: type=%s, len=%zu, elapsed_ms=%lld, bytes=%zu",
                          key.key_type.c_str(),
                          key.key_string.size(),
                          static_cast<long long>(elapsed_ms),
                          result.compiled->MemorySizeBytes());
    } else {
        RTP_LLM_LOG_WARNING("xgrammar compile failed: type=%s, len=%zu, elapsed_ms=%lld, invalid=%d, err=%s",
                            key.key_type.c_str(),
                            key.key_string.size(),
                            static_cast<long long>(elapsed_ms),
                            static_cast<int>(isInvalid(result)),
                            std::string(result.status.message()).c_str());
    }
    return result;
}

GrammarCompileResult XGrammarBackend::runCompileTask(const GrammarKeyCpp& key, const std::string& id) {
    InflightGuard guard(*this, id);

    const auto begin  = std::chrono::steady_clock::now();
    auto       result = compileSync(key);
    // Publish and report while the singleflight entry still represents this running task. Removing it
    // first would make the compile event under-report the inflight gauge as zero.
    storeResult(key, result);
    const auto latency_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
    reportCompile(result, latency_us);
    guard.release();
    return result;
}

GrammarCompileResult XGrammarBackend::compileNow(const GrammarKeyCpp& key) {
    const auto id = key.id();

    if (auto cached = lookupVerdict(id)) {
        reportLookup(cached->compiled != nullptr, isInvalid(*cached));
        return *cached;
    }

    if (!compile_pool_) {
        reportLookup(/*hit=*/false, /*invalid_hit=*/false);
        return runCompileTask(key, /*id=*/std::string());
    }

    std::shared_future<GrammarCompileResult> future;
    std::optional<GrammarCompileResult>      raced_cached;
    int64_t                                  lookup_inflight = 0;
    bool                                     rejected        = false;
    {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        // Lock order is inflight_mutex_ then cache_mutex_; no cache path acquires them in reverse.
        if (auto cached = lookupVerdict(id)) {
            raced_cached = std::move(*cached);
        } else if (auto it = inflight_.find(id); it != inflight_.end()) {
            future = it->second;
            compile_dedup_.fetch_add(1, std::memory_order_relaxed);
        } else {
            auto task = std::make_shared<std::packaged_task<GrammarCompileResult()>>(
                [this, key, id] { return runCompileTask(key, id); });
            future = task->get_future().share();
            inflight_.emplace(id, future);
            autil::ThreadPoolBase::ERROR_TYPE error = autil::ThreadPoolBase::ERROR_NONE;
            try {
                // Never execute inline when full: doing so would bypass both the concurrency bound and
                // the caller wait budget.
                error = compile_pool_->pushTask([task] { (*task)(); }, /*isBlocked=*/false, /*executeWhenFail=*/false);
            } catch (...) {
                error = autil::ThreadPoolBase::ERROR_POOL_ITEM_IS_NULL;
            }
            if (error != autil::ThreadPoolBase::ERROR_NONE) {
                inflight_.erase(id);
                compile_rejected_.fetch_add(1, std::memory_order_relaxed);
                rejected = true;
            }
        }
        lookup_inflight = static_cast<int64_t>(inflight_.size());
    }

    if (raced_cached) {
        reportLookup(raced_cached->compiled != nullptr, isInvalid(*raced_cached));
        return *raced_cached;
    }

    // Report the miss with the count captured after singleflight insertion. A fast worker may finish
    // before this thread emits metrics, but it cannot change the gauge associated with this lookup.
    reportLookup(/*hit=*/false, /*invalid_hit=*/false, lookup_inflight);

    if (rejected) {
        GrammarCompileResult out;
        out.status = absl::ResourceExhaustedError("too many outstanding grammar compiles (compile_queue_size="
                                                  + std::to_string(options_.compile_queue_size) + ")");
        RTP_LLM_LOG_WARNING("xgrammar compile rejected: type=%s, len=%zu", key.key_type.c_str(), key.key_string.size());
        reportOverload();
        return out;
    }

    if (future.wait_for(std::chrono::milliseconds(options_.compile_timeout_ms)) != std::future_status::ready) {
        compile_timeout_.fetch_add(1, std::memory_order_relaxed);
        GrammarCompileResult timed_out;
        timed_out.status = absl::ResourceExhaustedError(
            "grammar compile exceeded " + std::to_string(options_.compile_timeout_ms) + "ms budget");
        RTP_LLM_LOG_WARNING("xgrammar compile timeout: type=%s, len=%zu, timeout_ms=%d",
                            key.key_type.c_str(),
                            key.key_string.size(),
                            options_.compile_timeout_ms);
        reportOverload();
        return timed_out;
    }

    try {
        return future.get();
    } catch (const std::exception& e) {
        GrammarCompileResult broken;
        broken.status = absl::UnknownError(std::string("grammar compile aborted: ") + e.what());
        RTP_LLM_LOG_WARNING("xgrammar compile aborted: type=%s, len=%zu, err=%s",
                            key.key_type.c_str(),
                            key.key_string.size(),
                            std::string(broken.status.message()).c_str());
        reportOverload();
        return broken;
    } catch (...) {
        GrammarCompileResult broken;
        broken.status = absl::UnknownError("grammar compile aborted");
        RTP_LLM_LOG_WARNING(
            "xgrammar compile aborted: type=%s, len=%zu", key.key_type.c_str(), key.key_string.size());
        reportOverload();
        return broken;
    }
}

absl::StatusOr<std::shared_ptr<xgrammar::CompiledGrammar>> XGrammarBackend::compile(const GrammarKeyCpp& key) {
    auto result = compileNow(key);
    if (result.compiled) {
        return result.compiled;
    }
    return result.status.ok() ? absl::UnknownError("grammar compiler returned no result") : result.status;
}

absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>> XGrammarBackend::createMatcherFromKey(const GrammarKeyCpp& key) {
    auto compiled_or = compile(key);
    if (!compiled_or.ok()) {
        const std::string error = compiled_or.status().message().empty() ?
                                      "unknown compile error" :
                                      std::string(compiled_or.status().message());
        return absl::Status(compiled_or.status().code(), "Failed to compile " + key.key_type + " grammar: " + error);
    }
    return createMatcher(std::move(compiled_or.value()));
}

absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>>
XGrammarBackend::createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled) {
    if (!compiled) {
        return absl::InvalidArgumentError("createMatcher requires a non-null CompiledGrammar");
    }
    try {
        return std::make_shared<RtpGrammarMatcher>(std::move(compiled), options_.terminate_without_stop_token);
    } catch (const std::exception& e) {
        return absl::InvalidArgumentError(std::string("grammar matcher install failed: ") + e.what());
    } catch (...) {
        return absl::UnknownError("grammar matcher install failed: unknown");
    }
}

GrammarBackendStats XGrammarBackend::stats() const {
    GrammarBackendStats out;
    out.compile_total    = compile_total_.load(std::memory_order_relaxed);
    out.compile_invalid  = compile_invalid_.load(std::memory_order_relaxed);
    out.compile_timeout  = compile_timeout_.load(std::memory_order_relaxed);
    out.compile_rejected = compile_rejected_.load(std::memory_order_relaxed);
    out.compile_dedup    = compile_dedup_.load(std::memory_order_relaxed);
    out.cache_hit        = cache_hit_.load(std::memory_order_relaxed);
    out.cache_miss       = cache_miss_.load(std::memory_order_relaxed);
    out.invalid_hit      = invalid_hit_.load(std::memory_order_relaxed);
    out.cache_evicted               = cache_evicted_.load(std::memory_order_relaxed);
    out.cache_oversized             = cache_oversized_.load(std::memory_order_relaxed);
    out.total_cache_budget_bytes    = options_.total_cache_budget_bytes;
    out.compiler_cache_budget_bytes = options_.compiler_cache_budget_bytes;
    out.verdict_cache_budget_bytes  = options_.verdict_cache_budget_bytes;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        out.cache_size          = static_cast<int64_t>(cache_.size()) - invalid_count_;
        out.invalid_cache_size  = invalid_count_;
        out.verdict_cache_bytes = cache_bytes_;
    }
    {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        out.inflight = static_cast<int64_t>(inflight_.size());
    }
    return out;
}

void XGrammarBackend::reportLookup(bool hit, bool invalid_hit, std::optional<int64_t> inflight) const {
    if (invalid_hit) {
        invalid_hit_.fetch_add(1, std::memory_order_relaxed);
    } else if (hit) {
        cache_hit_.fetch_add(1, std::memory_order_relaxed);
    } else {
        cache_miss_.fetch_add(1, std::memory_order_relaxed);
    }
    if (!metrics_reporter_ && !metrics_report_fn_for_test_) {
        return;
    }
    RtpLLMGrammarMetricsCollector collector;
    collector.cache_hit_qps = hit || invalid_hit;
    fillResidentGauges(collector, inflight);
    if (metrics_report_fn_for_test_) {
        metrics_report_fn_for_test_(collector);
    }
    if (metrics_reporter_) {
        metrics_reporter_->report<RtpLLMGrammarMetrics, RtpLLMGrammarMetricsCollector>(nullptr, &collector);
    }
}

void XGrammarBackend::reportCompile(const GrammarCompileResult& result, int64_t latency_us) const {
    if (!metrics_reporter_ && !metrics_report_fn_for_test_) {
        return;
    }
    RtpLLMGrammarMetricsCollector collector;
    collector.compile_qps         = true;
    collector.compile_invalid_qps = isInvalid(result);
    collector.compile_latency_us  = latency_us;
    fillResidentGauges(collector);
    if (metrics_report_fn_for_test_) {
        metrics_report_fn_for_test_(collector);
    }
    if (metrics_reporter_) {
        metrics_reporter_->report<RtpLLMGrammarMetrics, RtpLLMGrammarMetricsCollector>(nullptr, &collector);
    }
}

void XGrammarBackend::reportOverload() const {
    if (!metrics_reporter_ && !metrics_report_fn_for_test_) {
        return;
    }
    RtpLLMGrammarMetricsCollector collector;
    collector.overload_qps = true;
    fillResidentGauges(collector);
    if (metrics_report_fn_for_test_) {
        metrics_report_fn_for_test_(collector);
    }
    if (metrics_reporter_) {
        metrics_reporter_->report<RtpLLMGrammarMetrics, RtpLLMGrammarMetricsCollector>(nullptr, &collector);
    }
}

void XGrammarBackend::fillResidentGauges(RtpLLMGrammarMetricsCollector& collector,
                                         std::optional<int64_t>          inflight) const {
    collector.total_cache_budget_bytes    = options_.total_cache_budget_bytes;
    collector.compiler_cache_budget_bytes = options_.compiler_cache_budget_bytes;
    collector.verdict_cache_budget_bytes  = options_.verdict_cache_budget_bytes;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        collector.verdict_cache_bytes = cache_bytes_;
    }
    if (inflight) {
        collector.compile_inflight = *inflight;
    } else {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        collector.compile_inflight = static_cast<int64_t>(inflight_.size());
    }
}

void XGrammarBackend::setCompileFnForTest(CompileFn fn) {
    compile_fn_ = std::move(fn);
}

void XGrammarBackend::setMetricsReportFnForTest(MetricsReportFn fn) {
    metrics_report_fn_for_test_ = std::move(fn);
}

void XGrammarBackend::clear() {
    // Only call while no compile is in flight. A running task may republish its result after this reset,
    // and GrammarCompiler::ClearCache() is not synchronized with compiler calls.
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    lru_.clear();
    cache_bytes_   = 0;
    invalid_count_ = 0;
    compiler_.ClearCache();
}

}  // namespace rtp_llm
