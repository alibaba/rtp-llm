#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <variant>

#include <xgrammar/exception.h>

#include "autil/LockFreeThreadPool.h"
#include "autil/legacy/any.h"
#include "autil/legacy/json.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

using JsonArray = autil::legacy::json::JsonArray;
using JsonMap   = autil::legacy::json::JsonMap;

void sanitizeStructuralFormat(autil::legacy::Any& any) {
    auto* map = autil::legacy::AnyCast<JsonMap>(&any);
    if (!map) {
        return;
    }

    std::string fmt_type;
    if (auto type_it = map->find("type"); type_it != map->end()) {
        if (auto* s = autil::legacy::AnyCast<std::string>(&type_it->second)) {
            fmt_type = *s;
        }
    }

    if ((fmt_type == "json_schema" || fmt_type == "qwen_xml_parameter") && map->find("json_schema") == map->end()) {
        (*map)["json_schema"] = JsonMap{};
    }

    if (fmt_type == "tag") {
        if (auto it = map->find("content"); it != map->end()) {
            sanitizeStructuralFormat(it->second);
        }
    } else if (fmt_type == "sequence" || fmt_type == "or") {
        if (auto it = map->find("elements"); it != map->end()) {
            if (auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second)) {
                for (auto& el : *arr) {
                    sanitizeStructuralFormat(el);
                }
            }
        }
    } else if (fmt_type == "triggered_tags" || fmt_type == "tags_with_separator") {
        if (auto it = map->find("tags"); it != map->end()) {
            if (auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second)) {
                for (auto& tag : *arr) {
                    sanitizeStructuralFormat(tag);
                }
            }
        }
    }
}

void sanitizeLegacyStructures(JsonMap& root) {
    auto it = root.find("structures");
    if (it == root.end()) {
        return;
    }
    auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second);
    if (!arr) {
        return;
    }
    for (auto& item : *arr) {
        if (auto* map = autil::legacy::AnyCast<JsonMap>(&item)) {
            if (map->find("schema") == map->end()) {
                (*map)["schema"] = JsonMap{};
            }
        }
    }
}

// JSON Schema keywords whose value is an instance rather than a schema. xgrammar serializes such a value into
// the grammar verbatim, so rewriting it would change the literal the model is required to emit.
bool isInstanceValuedKeyword(const std::string& key) {
    return key == "const" || key == "enum" || key == "default" || key == "examples";
}

// Drops `minLength` / `maxLength` from every schema node reachable from `any`, reporting whether anything was
// removed.
//
// xgrammar lowers a string length bound into a counted repetition, so the remaining-length counter becomes
// part of the grammar state: every token generated inside the string lands in a state the adaptive token mask
// cache has never seen, and each decode step pays a full vocabulary scan. Away from the bounds the mask it
// computes is the one the unbounded field yields, so the scan buys nothing there. The lowering also drops the
// escape branch of the string rule in the pinned version, which makes legal escaped content unemittable.
// Until both are fixed upstream the bound is worse than no bound, so it is removed before compiling.
//
// A member only counts as the keyword when its value is a number. Under `properties` the same name denotes a
// field and carries a schema instead, and the keyword is inert on non-string types, so that test alone decides
// and no type check is needed -- which also covers the string nodes whose type xgrammar infers rather than
// reads, such as the one under `propertyNames`.
//
// `in_json_schema` tracks whether xgrammar interprets this node as JSON Schema. The structural-tag DSL around
// it may carry unrelated payloads, so keywords are only honoured once a `json_schema` value or a legacy
// StructuralTagItem `schema` value has been entered.
bool stripStringLengthBounds(autil::legacy::Any& any, bool in_json_schema) {
    if (auto* arr = autil::legacy::AnyCast<JsonArray>(&any)) {
        bool stripped = false;
        for (auto& el : *arr) {
            stripped |= stripStringLengthBounds(el, in_json_schema);
        }
        return stripped;
    }
    auto* map = autil::legacy::AnyCast<JsonMap>(&any);
    if (!map) {
        return false;
    }

    bool stripped = false;
    if (in_json_schema) {
        for (const char* key : {"minLength", "maxLength"}) {
            auto it = map->find(key);
            if (it != map->end() && autil::legacy::json::IsJsonNumber(it->second)) {
                map->erase(it);
                stripped = true;
            }
        }
    }

    const bool legacy_schema_item = map->count("schema") && map->count("begin") && map->count("end");
    for (auto& [key, value] : *map) {
        if (isInstanceValuedKeyword(key)) {
            continue;
        }
        const bool child_in_json_schema =
            in_json_schema || key == "json_schema" || (legacy_schema_item && key == "schema");
        stripped |= stripStringLengthBounds(value, child_in_json_schema);
    }
    return stripped;
}

}  // namespace

std::string XGrammarBackendCpp::sanitizeStructuralTag(const std::string& tag_json) {
    autil::legacy::Any any;
    try {
        autil::legacy::json::ParseJson(tag_json, any);
    } catch (...) {
        return tag_json;
    }
    auto* root = autil::legacy::AnyCast<JsonMap>(&any);
    if (!root) {
        return tag_json;
    }
    if (root->find("structures") != root->end()) {
        sanitizeLegacyStructures(*root);
    } else if (auto fmt = root->find("format"); fmt != root->end()) {
        sanitizeStructuralFormat(fmt->second);
    }
    const bool stripped = stripStringLengthBounds(any, /*in_json_schema=*/false);
    std::string sanitized;
    try {
        sanitized = autil::legacy::json::ToString(any, true);
    } catch (...) {
        return tag_json;
    }
    if (stripped) {
        RTP_LLM_LOG_WARNING("XGrammarBackendCpp: removed string minLength/maxLength from a structural tag before "
                            "compiling; the bound would invalidate the token mask cache on every decode step");
    }
    return sanitized;
}

std::string XGrammarBackendCpp::sanitizeJsonSchema(const std::string& schema_json) {
    autil::legacy::Any any;
    try {
        autil::legacy::json::ParseJson(schema_json, any);
    } catch (...) {
        return schema_json;
    }
    // Re-serializing sorts object keys, which would reorder `properties` and with it the order the schema
    // forces fields to be generated in. An untouched schema therefore has to reach xgrammar verbatim; a
    // stripped one pays that reordering, which is why nothing else is rewritten here.
    if (!stripStringLengthBounds(any, /*in_json_schema=*/true)) {
        return schema_json;
    }
    std::string sanitized;
    try {
        sanitized = autil::legacy::json::ToString(any, true);
    } catch (...) {
        return schema_json;
    }
    RTP_LLM_LOG_WARNING("XGrammarBackendCpp: removed string minLength/maxLength from a json schema before "
                        "compiling; the bound would invalidate the token mask cache on every decode step");
    return sanitized;
}

XGrammarBackendCpp::XGrammarBackendCpp(const std::string&            tokenizer_info_json,
                                       const XGrammarBackendOptions& options,
                                       kmonitor::MetricsReporterPtr  metrics_reporter):
    options_(options),
    metrics_reporter_(std::move(metrics_reporter)),
    tokenizer_info_([&] {
        auto result = xgrammar::TokenizerInfo::DeserializeJSON(tokenizer_info_json);
        if (std::holds_alternative<xgrammar::TokenizerInfo>(result)) {
            return std::get<xgrammar::TokenizerInfo>(std::move(result));
        }
        auto error = std::get<xgrammar::SerializationError>(result);
        throw std::runtime_error(std::string("XGrammarBackendCpp: failed to deserialize TokenizerInfo: ")
                                 + std::visit([](const auto& e) { return std::string(e.what()); }, error));
    }()),
    compiler_(tokenizer_info_,
              std::max(1, options.max_compiler_threads),
              options.enable_compiler_cache,
              // xgrammar reads 0 as a zero-sized cache and aborts below -1, so fold our <=0 convention
              // onto its unlimited sentinel.
              options.compiler_cache_bytes > 0 ? options.compiler_cache_bytes : -1) {
    if (options_.compile_timeout_ms > 0) {
        const size_t concurrency = static_cast<size_t>(std::max(1, options_.compile_concurrency));
        const size_t queue_size  = static_cast<size_t>(std::max(1, options_.compile_queue_size));
        auto         pool        = std::make_shared<autil::LockFreeThreadPool>(
            concurrency, queue_size, nullptr, "XGrammarCompile", /*stopIfHasException=*/false);
        if (!pool->start()) {
            throw std::runtime_error("XGrammarBackendCpp: failed to start grammar compile thread pool");
        }
        compile_pool_ = std::move(pool);
    }
    RTP_LLM_LOG_INFO("XGrammarBackendCpp init: vocab_size=%d, any_whitespace=%d, strict_mode=%d, compiler_threads=%d, "
                     "compile_timeout_ms=%d, compile_concurrency=%d, compile_queue_size=%d, cache_bytes_limit=%lld",
                     tokenizer_info_.GetVocabSize(),
                     static_cast<int>(options_.any_whitespace),
                     static_cast<int>(options_.strict_mode),
                     std::max(1, options_.max_compiler_threads),
                     options_.compile_timeout_ms,
                     options_.compile_concurrency,
                     options_.compile_queue_size,
                     static_cast<long long>(options_.compiler_cache_bytes));
}

XGrammarBackendCpp::~XGrammarBackendCpp() {
    if (compile_pool_) {
        // autil's LockFreeThreadPool::stop() ignores the STOP_TYPE and always joins its workers, so this
        // blocks until any running compile finishes. That is what keeps `this` alive for the whole task
        // body, but it also means destroying the backend while a pathological compile is running takes as
        // long as that compile. Callers must not hold a lock that requests need across this destructor.
        RTP_LLM_LOG_INFO("XGrammarBackendCpp shutdown: joining grammar compile pool");
        compile_pool_->stop();
        compile_pool_.reset();
        RTP_LLM_LOG_INFO("XGrammarBackendCpp shutdown: grammar compile pool joined");
    }
}

std::shared_ptr<xgrammar::CompiledGrammar> XGrammarBackendCpp::getCached(const GrammarKeyCpp& key) const {
    std::shared_ptr<xgrammar::CompiledGrammar> compiled;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto                        it = cache_.find(key.id());
        if (it != cache_.end() && it->second.compiled) {
            compiled = it->second.compiled;
            touchLocked(it->second);
        }
    }
    reportLookup(compiled != nullptr, /*invalid_hit=*/false);
    return compiled;
}

std::string XGrammarBackendCpp::getCachedInvalid(const GrammarKeyCpp& key) const {
    std::string error_message;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto                        it = cache_.find(key.id());
        if (it != cache_.end() && !it->second.compiled) {
            error_message = it->second.error_message;
            touchLocked(it->second);
        }
    }
    if (!error_message.empty()) {
        reportLookup(/*hit=*/false, /*invalid_hit=*/true);
    }
    return error_message;
}

void XGrammarBackendCpp::touchLocked(CacheEntry& entry) const {
    // splice moves the node without invalidating the iterator we keep in the entry.
    lru_.splice(lru_.begin(), lru_, entry.lru_it);
}

void XGrammarBackendCpp::eraseLocked(CacheMap::iterator it) {
    cache_bytes_ -= it->second.bytes;
    if (!it->second.compiled) {
        --invalid_count_;
    }
    lru_.erase(it->second.lru_it);
    cache_.erase(it);
}

int64_t XGrammarBackendCpp::evictLocked() {
    const int64_t limit = options_.compiler_cache_bytes;
    if (limit <= 0) {
        return 0;
    }
    int64_t dropped = 0;
    // Stops with one entry left rather than emptying the cache. A verdict larger than the whole ceiling
    // would otherwise never stay resident, so every retry would recompile it - and for a grammar slow
    // enough to blow the wait budget that never converges: the caller keeps getting a retryable overload
    // and the retry keeps paying for a compile whose result is thrown away. Overshooting the ceiling by
    // the newest entry is the lesser evil, especially as its compiled tables are shared with xgrammar's
    // own cache, which is bounded by the same number.
    while (cache_bytes_ > limit && cache_.size() > 1) {
        auto it = cache_.find(lru_.back());
        if (it == cache_.end()) {
            // A store whose map insert threw leaves its LRU node behind; drop it and carry on.
            lru_.pop_back();
            continue;
        }
        eraseLocked(it);
        ++dropped;
    }
    return dropped;
}

void XGrammarBackendCpp::storeResult(const GrammarKeyCpp& key, const CompileResult& result) {
    // An overloaded verdict says nothing about the grammar, so caching it would poison later attempts.
    if (result.is_overloaded) {
        return;
    }
    if (!result.compiled && !result.is_invalid) {
        return;
    }

    const auto id = key.id();
    // MemorySizeBytes() walks the compiled tables, so keep it out of the critical section. The id is
    // charged twice because the map key and the LRU node each hold a copy of it, plus a flat per-entry
    // overhead for the two nodes themselves: without it a cache of millions of short invalid verdicts
    // would account for a fraction of what it actually costs.
    int64_t bytes =
        kEntryOverheadBytes + 2 * static_cast<int64_t>(id.size()) + static_cast<int64_t>(result.error_message.size());
    if (result.compiled) {
        bytes += static_cast<int64_t>(result.compiled->MemorySizeBytes());
    }

    const int64_t limit = options_.compiler_cache_bytes;
    // Copying the error message can be expensive for a large schema, so build the entry before locking.
    CacheEntry pending;
    pending.compiled      = result.compiled;
    pending.error_message = result.error_message;
    pending.bytes         = bytes;

    int64_t dropped  = 0;
    int64_t resident = 0;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (auto it = cache_.find(id); it != cache_.end()) {
            eraseLocked(it);
        }
        // The LRU node goes in first: if publishing the map entry then throws, the worst outcome is a
        // stale id in lru_, which evictLocked drops. Publishing the entry first would instead expose a
        // singular lru_it to the next lookup.
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

    if (limit > 0 && resident > limit) {
        // This one verdict outgrows the whole ceiling, so it is kept at the cost of exceeding it: the
        // alternative is recompiling it on every request. It is the only entry left and the next store of
        // anything else evicts it, so the overshoot lasts exactly as long as the grammar stays in use.
        cache_oversized_.fetch_add(1, std::memory_order_relaxed);
        RTP_LLM_LOG_WARNING("xgrammar verdict alone exceeds the cache ceiling, keeping it: type=%s, bytes=%lld,"
                            " limit=%lld",
                            key.key_type.c_str(),
                            static_cast<long long>(bytes),
                            static_cast<long long>(limit));
    }
    if (dropped > 0) {
        cache_evicted_.fetch_add(dropped, std::memory_order_relaxed);
        RTP_LLM_LOG_INFO("xgrammar cache evicted %lld verdicts to stay under %lld bytes, now %lld bytes",
                         static_cast<long long>(dropped),
                         static_cast<long long>(limit),
                         static_cast<long long>(resident));
    }
}

void XGrammarBackendCpp::InflightGuard::release() noexcept {
    if (released_) {
        return;
    }
    released_ = true;
    if (id_.empty()) {
        // Synchronous path: the compile never registered an in-flight entry.
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(owner_.inflight_mutex_);
        owner_.inflight_.erase(id_);
    } catch (...) {}
}

std::optional<CompileResult> XGrammarBackendCpp::lookupVerdict(const std::string& id) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto                        it = cache_.find(id);
    if (it == cache_.end()) {
        return std::nullopt;
    }
    touchLocked(it->second);
    CompileResult cached;
    if (it->second.compiled) {
        cached.compiled = it->second.compiled;
    } else {
        cached.is_invalid    = true;
        cached.error_message = it->second.error_message;
    }
    return cached;
}

CompileResult XGrammarBackendCpp::invokeCompiler(const GrammarKeyCpp& key) {
    // Exceptions propagate to compileSync, which owns the invalid-vs-transient classification.
    auto wrap = [](xgrammar::CompiledGrammar&& compiled) {
        CompileResult out;
        out.compiled = std::make_shared<xgrammar::CompiledGrammar>(std::move(compiled));
        return out;
    };

    const auto& grammar = key.key_string;
    if (key.key_type == "json") {
        return wrap(grammar == "$$ANY$$" ? compiler_.CompileBuiltinJSONGrammar() :
                                           compiler_.CompileJSONSchema(sanitizeJsonSchema(grammar),
                                                                       options_.any_whitespace,
                                                                       std::nullopt,
                                                                       std::nullopt,
                                                                       options_.strict_mode));
    }
    if (key.key_type == "regex") {
        return wrap(compiler_.CompileRegex(grammar));
    }
    if (key.key_type == "ebnf") {
        return wrap(compiler_.CompileGrammar(grammar));
    }
    if (key.key_type == "structural_tag") {
        return wrap(compiler_.CompileStructuralTag(sanitizeStructuralTag(grammar)));
    }
    CompileResult unknown;
    unknown.is_invalid    = true;
    unknown.error_message = "unknown grammar type: " + key.key_type;
    return unknown;
}

CompileResult XGrammarBackendCpp::compileSync(const GrammarKeyCpp& key) {
    compile_total_.fetch_add(1, std::memory_order_relaxed);
    const auto    begin = std::chrono::steady_clock::now();
    CompileResult result;
    try {
        // compile_fn_ is installed only by tests, before any compile can be in flight, so an
        // unsynchronized read is safe here.
        result = compile_fn_ ? compile_fn_(key) : invokeCompiler(key);
    } catch (const std::bad_alloc&) {
        // Transient, not a verdict on the grammar: caching it would turn one OOM into a permanent
        // rejection of a valid schema, which is exactly the pathological-compile case this guards.
        result.is_overloaded = true;
        result.error_message = "grammar compile ran out of memory";
    } catch (const std::exception& e) {
        result.is_invalid    = true;
        result.error_message = e.what();
    } catch (...) {
        result.is_overloaded = true;
        result.error_message = "unknown exception during grammar compile";
    }

    if (result.is_invalid) {
        compile_invalid_.fetch_add(1, std::memory_order_relaxed);
    }

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
    if (result.compiled) {
        RTP_LLM_LOG_DEBUG("xgrammar compile ok: type=%s, len=%zu, elapsed_ms=%lld",
                          key.key_type.c_str(),
                          key.key_string.size(),
                          static_cast<long long>(elapsed_ms));
    } else {
        RTP_LLM_LOG_WARNING("xgrammar compile failed: type=%s, len=%zu, elapsed_ms=%lld, err=%s",
                            key.key_type.c_str(),
                            key.key_string.size(),
                            static_cast<long long>(elapsed_ms),
                            result.error_message.c_str());
    }
    return result;
}

CompileResult XGrammarBackendCpp::runCompileTask(const GrammarKeyCpp& key, const std::string& id) {
    // The in-flight entry must go away on every exit path. If an exception (e.g. bad_alloc while
    // publishing the verdict) skipped the removal, the entry would linger forever holding a
    // ready-with-exception future and permanently short-circuit every later request for this key.
    InflightGuard guard(*this, id);

    const auto begin  = std::chrono::steady_clock::now();
    auto       result = compileSync(key);
    // Publish before dropping the in-flight entry, so a caller arriving in between sees the cache.
    storeResult(key, result);
    guard.release();

    const auto latency_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
    reportCompile(result, latency_us);
    return result;
}

CompileResult XGrammarBackendCpp::compileNow(const GrammarKeyCpp& key) {
    const auto id = key.id();

    if (auto cached = lookupVerdict(id)) {
        return *cached;
    }

    if (!compile_pool_) {
        return runCompileTask(key, /*id=*/std::string());
    }

    std::shared_future<CompileResult> future;
    bool                              rejected = false;
    {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        // Re-probe under inflight_mutex_: without it two callers that both miss the cache can each
        // become leader for the same key, because the cache probe above is a separate critical section.
        // Lock order is inflight_mutex_ -> cache_mutex_ and never the reverse.
        if (auto cached = lookupVerdict(id)) {
            return *cached;
        }
        if (auto it = inflight_.find(id); it != inflight_.end()) {
            future = it->second;
            compile_dedup_.fetch_add(1, std::memory_order_relaxed);
        } else {
            auto task = std::make_shared<std::packaged_task<CompileResult()>>(
                [this, key, id] { return runCompileTask(key, id); });
            future = task->get_future().share();
            inflight_.emplace(id, future);
            // pushTask instead of async(): async() falls back to running the lambda inline on this
            // thread when the queue is full, which would defeat both the cap and the timeout.
            autil::ThreadPoolBase::ERROR_TYPE ec = autil::ThreadPoolBase::ERROR_NONE;
            try {
                ec = compile_pool_->pushTask([task] { (*task)(); }, /*isBlocked=*/false, /*executeWhenFail=*/false);
            } catch (...) {
                // Allocating the work item can throw. Leaving the entry behind would strand a future
                // nothing will ever complete, so every later caller for this key would burn the full
                // timeout forever.
                ec = autil::ThreadPoolBase::ERROR_POOL_ITEM_IS_NULL;
            }
            if (ec != autil::ThreadPoolBase::ERROR_NONE) {
                inflight_.erase(id);
                compile_rejected_.fetch_add(1, std::memory_order_relaxed);
                rejected = true;
            }
        }
    }

    if (rejected) {
        CompileResult out;
        out.is_overloaded = true;
        out.error_message = "too many outstanding grammar compiles (compile_queue_size="
                            + std::to_string(options_.compile_queue_size) + ")";
        RTP_LLM_LOG_WARNING("xgrammar compile rejected: type=%s, len=%zu", key.key_type.c_str(), key.key_string.size());
        reportOverload();
        return out;
    }

    if (future.wait_for(std::chrono::milliseconds(options_.compile_timeout_ms)) != std::future_status::ready) {
        compile_timeout_.fetch_add(1, std::memory_order_relaxed);
        CompileResult timed_out;
        timed_out.is_overloaded = true;
        timed_out.error_message =
            "grammar compile exceeded " + std::to_string(options_.compile_timeout_ms) + "ms budget";
        RTP_LLM_LOG_WARNING("xgrammar compile timeout: type=%s, len=%zu, timeout_ms=%d",
                            key.key_type.c_str(),
                            key.key_string.size(),
                            options_.compile_timeout_ms);
        reportOverload();
        return timed_out;
    }

    CompileResult broken;
    broken.is_overloaded = true;
    try {
        return future.get();
    } catch (const std::exception& e) {
        // The pool dropped the task (broken_promise) or the task threw. Either way the grammar is
        // unjudged, so this must read as transient rather than invalid.
        broken.error_message = std::string("grammar compile aborted: ") + e.what();
    } catch (...) {
        broken.error_message = "grammar compile aborted";
    }
    RTP_LLM_LOG_WARNING("xgrammar compile aborted: type=%s, len=%zu, err=%s",
                        key.key_type.c_str(),
                        key.key_string.size(),
                        broken.error_message.c_str());
    reportOverload();
    return broken;
}

GrammarBackendStats XGrammarBackendCpp::stats() const {
    GrammarBackendStats out;
    out.compile_total    = compile_total_.load(std::memory_order_relaxed);
    out.compile_invalid  = compile_invalid_.load(std::memory_order_relaxed);
    out.compile_timeout  = compile_timeout_.load(std::memory_order_relaxed);
    out.compile_rejected = compile_rejected_.load(std::memory_order_relaxed);
    out.compile_dedup    = compile_dedup_.load(std::memory_order_relaxed);
    out.cache_hit        = cache_hit_.load(std::memory_order_relaxed);
    out.cache_miss       = cache_miss_.load(std::memory_order_relaxed);
    out.invalid_hit      = invalid_hit_.load(std::memory_order_relaxed);
    out.cache_evicted    = cache_evicted_.load(std::memory_order_relaxed);
    out.cache_oversized  = cache_oversized_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        out.cache_size         = static_cast<int64_t>(cache_.size()) - invalid_count_;
        out.invalid_cache_size = invalid_count_;
        out.cache_bytes        = cache_bytes_;
    }
    {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        out.inflight = static_cast<int64_t>(inflight_.size());
    }
    return out;
}

void XGrammarBackendCpp::reportLookup(bool hit, bool invalid_hit) const {
    if (invalid_hit) {
        invalid_hit_.fetch_add(1, std::memory_order_relaxed);
    } else if (hit) {
        cache_hit_.fetch_add(1, std::memory_order_relaxed);
    } else {
        cache_miss_.fetch_add(1, std::memory_order_relaxed);
    }
    if (!metrics_reporter_) {
        return;
    }
    RtpLLMGrammarMetricsCollector collector;
    // Any cached verdict counts as a hit, invalid ones included: what the rate answers is how much of the
    // load the cache absorbs, and a cached rejection absorbs just as much as a cached grammar. Misses are
    // compile_qps, so exporting them again would only add a second name for the same events.
    collector.cache_hit_qps = hit || invalid_hit;
    // Lookups are the only path that still runs once every request is served from cache, so the
    // resident gauges have to ride along here or they go silent in exactly the healthy steady state.
    fillResidentGauges(collector);
    metrics_reporter_->report<RtpLLMGrammarMetrics, RtpLLMGrammarMetricsCollector>(nullptr, &collector);
}

void XGrammarBackendCpp::reportCompile(const CompileResult& result, int64_t latency_us) const {
    if (!metrics_reporter_) {
        return;
    }
    RtpLLMGrammarMetricsCollector collector;
    collector.compile_qps         = true;
    collector.compile_invalid_qps = result.is_invalid;
    collector.compile_latency_us  = latency_us;
    fillResidentGauges(collector);
    metrics_reporter_->report<RtpLLMGrammarMetrics, RtpLLMGrammarMetricsCollector>(nullptr, &collector);
}

void XGrammarBackendCpp::reportOverload() const {
    if (!metrics_reporter_) {
        return;
    }
    RtpLLMGrammarMetricsCollector collector;
    // Queue rejection, wait-budget timeout and aborted task share one rate: all three hand the caller the
    // same retryable verdict, and which one it was only matters while reading the WARNING logs.
    collector.overload_qps = true;
    fillResidentGauges(collector);
    metrics_reporter_->report<RtpLLMGrammarMetrics, RtpLLMGrammarMetricsCollector>(nullptr, &collector);
}

void XGrammarBackendCpp::fillResidentGauges(RtpLLMGrammarMetricsCollector& collector) const {
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        // Bytes rather than entry count: bytes are what the ceiling bounds and what the host runs out of.
        collector.cache_bytes = cache_bytes_;
    }
    {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        collector.compile_inflight = static_cast<int64_t>(inflight_.size());
    }
}

std::shared_ptr<RtpGrammarMatcher>
XGrammarBackendCpp::createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                  bool                                       require_reasoning,
                                  std::optional<std::vector<int>>            think_end_token_ids,
                                  bool                                       terminate_without_stop_token) {
    return std::make_shared<RtpGrammarMatcher>(std::move(compiled),
                                               require_reasoning,
                                               std::move(think_end_token_ids),
                                               options_.override_stop_tokens,
                                               terminate_without_stop_token,
                                               /*max_rollback_tokens=*/200);
}

void XGrammarBackendCpp::setCompileFnForTest(CompileFn fn) {
    compile_fn_ = std::move(fn);
}

void XGrammarBackendCpp::clear() {
    // Only safe while no compile is in flight: inflight_ is left alone on purpose, so a task that is
    // still running will republish its verdict into the cache we just emptied, and compiler_.ClearCache()
    // is not synchronized against the compiles running on the pool threads.
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    lru_.clear();
    cache_bytes_   = 0;
    invalid_count_ = 0;
    compiler_.ClearCache();
}

}  // namespace rtp_llm
