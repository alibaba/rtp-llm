#include "rtp_llm/cpp/engine_base/schedulers/GrammarManager.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <future>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

void GrammarManager::decrementInFlightLocked(const std::string& kid) {
    auto it = in_flight_.find(kid);
    if (it == in_flight_.end()) {
        return;
    }
    if (--it->second.ref_count == 0) {
        in_flight_.erase(it);
        // No subscribers left → prune any not-yet-popped CompileTask for kid.
        auto new_end = std::remove_if(
            compile_tasks_.begin(), compile_tasks_.end(),
            [&](const CompileTask& t) { return t.key.id() == kid; });
        compile_tasks_.erase(new_end, compile_tasks_.end());
    }
}

bool GrammarManager::removeFromQueueLocked(const GenerateStreamPtr& stream) {
    for (auto it = grammar_queue_.begin(); it != grammar_queue_.end(); ++it) {
        if (it->stream == stream) {
            decrementInFlightLocked(it->key.id());
            grammar_queue_.erase(it);
            return true;
        }
    }
    return false;
}

GrammarManager::GrammarManager(std::shared_ptr<XGrammarBackendCpp> backend, GrammarConfig grammar_config):
    backend_(std::move(backend)) {
    if (grammar_config.compile_timeout_ms > 0) {
        grammar_compile_timeout_ms_ = grammar_config.compile_timeout_ms;
    }
    if (!hasBackend()) {
        RTP_LLM_LOG_INFO("GrammarManager init: backend=disabled, compile_timeout_ms=%lld",
                         static_cast<long long>(grammar_compile_timeout_ms_));
        return;
    }

    int num_workers = std::max(1, grammar_config.num_workers);
    RTP_LLM_LOG_INFO("GrammarManager init: backend=cpp, compile_timeout_ms=%lld, num_workers=%d",
                     static_cast<long long>(grammar_compile_timeout_ms_),
                     num_workers);

    workers_.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

GrammarManager::~GrammarManager() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
        compile_tasks_.clear();
    }
    worker_cv_.notify_all();

    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(grammar_compile_timeout_ms_);
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        worker_cv_.wait_until(lock, deadline, [this] { return alive_workers_.load() == 0; });
    }

    int stuck = alive_workers_.load();
    if (stuck > 0) {
        RTP_LLM_LOG_ERROR("GrammarManager shutdown: %d worker(s) stuck in compileNow after %lld ms, aborting process",
                          stuck, static_cast<long long>(grammar_compile_timeout_ms_));
        std::_Exit(1);
    }
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

size_t GrammarManager::size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return grammar_queue_.size();
}

bool GrammarManager::hasWaitingGrammars() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return !grammar_queue_.empty();
}

bool GrammarManager::hasActionableGrammar() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (grammar_queue_.empty()) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    for (const auto& entry : grammar_queue_) {
        if (!entry.stream || !entry.stream->isActive()) {
            return true;
        }
        if (entry.future.valid() && entry.future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            return true;
        }
        if (now >= entry.deadline) {
            return true;
        }
    }
    return false;
}

bool GrammarManager::isGrammarRequested(const GenerateStreamPtr& stream) const {
    auto& config = stream->generateConfig();
    return config->json_schema.has_value() || config->regex.has_value() || config->ebnf.has_value()
           || config->structural_tag.has_value();
}

GrammarKeyCpp GrammarManager::extractGrammarKey(const GenerateStreamPtr& stream) const {
    auto& config = stream->generateConfig();
    if (config->json_schema.has_value()) {
        return {"json", config->json_schema.value()};
    } else if (config->regex.has_value()) {
        return {"regex", config->regex.value()};
    } else if (config->ebnf.has_value()) {
        return {"ebnf", config->ebnf.value()};
    } else if (config->structural_tag.has_value()) {
        return {"structural_tag", config->structural_tag.value()};
    }
    return {};
}

void GrammarManager::replayPrefillTokensToGrammar(const GenerateStreamPtr& stream, RtpGrammarMatcher& matcher) {
    size_t output_len = stream->outputTokenLen();
    if (output_len == 0) {
        return;
    }
    auto all_tokens = stream->completeTokenIdsVec(0);
    int  input_len  = stream->inputLength();
    RTP_LLM_LOG_DEBUG("stream [%ld] grammar replay prefill tokens: output_len=%zu, input_len=%d",
                      stream->streamId(), output_len, input_len);
    std::vector<int32_t> token_ids;
    token_ids.reserve(output_len);
    for (size_t i = 0; i < output_len; ++i) {
        token_ids.push_back(static_cast<int32_t>(all_tokens[input_len + i]));
    }
    if (!matcher.acceptTokens(token_ids)) {
        RTP_LLM_LOG_WARNING("stream [%ld] grammar replay rejected token (parser refused)", stream->streamId());
        stream->reportError(ErrorCode::INVALID_PARAMS, "grammar replay prefill tokens error: parser refused");
    }
}

void GrammarManager::installMatcherOnStream(const GenerateStreamPtr&                   stream,
                                            std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                            const GrammarKeyCpp&                       key,
                                            bool                                       require_reasoning,
                                            bool                                       cache_hit,
                                            int64_t                                    compile_time_us) {
    auto matcher                            = backend_->createMatcher(std::move(compiled), require_reasoning);
    matcher->mutableStats().cache_hit       = cache_hit;
    matcher->mutableStats().compile_time_us = compile_time_us;
    matcher->mutableStats().dispatch_type   = key.key_type;
    matcher->initReasoning(require_reasoning);
    replayPrefillTokensToGrammar(stream, *matcher);
    stream->setGrammarMatcher(std::move(matcher));
}

bool GrammarManager::processReqWithGrammar(const GenerateStreamPtr& stream) {
    assert(stream);
    RTP_LLM_LOG_DEBUG("stream [%ld] processReqWithGrammar ENTER", stream->streamId());

    if (!hasBackend()) {
        if (isGrammarRequested(stream)) {
            stream->reportError(ErrorCode::INVALID_PARAMS,
                                "Grammar-based generation requested but grammar backend is disabled");
        }
        return false;
    }

    if (!isGrammarRequested(stream)) {
        stream->clearGrammarMatcher();
        return false;
    }

    GrammarKeyCpp key               = extractGrammarKey(stream);
    const bool    require_reasoning = stream->generateConfig()->in_think_mode;

    RTP_LLM_LOG_DEBUG("stream [%ld] grammar preprocess begin: key=%s, require_reasoning=%d",
                      stream->streamId(), key.brief().c_str(), static_cast<int>(require_reasoning));

    if (auto compiled = backend_->getCached(key); compiled) {
        installMatcherOnStream(stream, std::move(compiled), key, require_reasoning,
                               /*cache_hit=*/true, /*compile_time_us=*/0);
        RTP_LLM_LOG_DEBUG("stream [%ld] grammar cache hit accepted: key=%s",
                          stream->streamId(), key.brief().c_str());
        return false;
    }

    if (auto err = backend_->getCachedInvalid(key); !err.empty()) {
        stream->reportError(ErrorCode::INVALID_PARAMS,
                            "Failed to compile " + key.key_type + " grammar: " + err);
        return false;
    }

    // Slow path: queue an async compile, deduplicating identical in-flight keys.
    GrammarEntry entry;
    entry.stream            = stream;
    entry.key               = key;
    entry.require_reasoning = require_reasoning;
    entry.deadline          = std::chrono::steady_clock::now() + std::chrono::milliseconds(grammar_compile_timeout_ms_);

    const std::string kid                 = key.id();
    bool              submitted_new_task  = false;
    size_t            queue_size_after    = 0;
    size_t            compile_tasks_after = 0;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        auto [it, inserted] = in_flight_.try_emplace(kid);
        auto& slot          = it->second;

        if (inserted || !slot.future.valid()) {
            std::promise<GrammarReadyPayload> promise;
            slot.future = promise.get_future().share();

            CompileTask task;
            task.key               = key;
            task.require_reasoning = require_reasoning;
            task.promise           = std::move(promise);
            compile_tasks_.emplace_back(std::move(task));
            submitted_new_task = true;
        }
        slot.ref_count++;
        entry.future = slot.future;
        grammar_queue_.emplace_back(std::move(entry));
        queue_size_after    = grammar_queue_.size();
        compile_tasks_after = compile_tasks_.size();
    }
    if (submitted_new_task) {
        worker_cv_.notify_one();
    }

    RTP_LLM_LOG_DEBUG("stream [%ld] grammar async compile %s: key=%s, queue_size=%zu, pending_tasks=%zu",
                      stream->streamId(),
                      submitted_new_task ? "queued" : "subscribed to in-flight",
                      key.brief().c_str(), queue_size_after, compile_tasks_after);
    return true;
}

void GrammarManager::pollAndDrainLocked(std::list<GrammarEntry>& ready, std::list<GrammarEntry>& failed) {
    const auto now = std::chrono::steady_clock::now();
    for (auto it = grammar_queue_.begin(); it != grammar_queue_.end();) {
        auto& entry = *it;
        if (!entry.stream || !entry.stream->isActive()) {
            decrementInFlightLocked(entry.key.id());
            ready.splice(ready.end(), grammar_queue_, it++);
            continue;
        }
        if (entry.future.valid() && entry.future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            decrementInFlightLocked(entry.key.id());
            ready.splice(ready.end(), grammar_queue_, it++);
            continue;
        }
        if (now >= entry.deadline) {
            auto waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now - (entry.deadline - std::chrono::milliseconds(grammar_compile_timeout_ms_)))
                                 .count();
            RTP_LLM_LOG_WARNING("stream [%ld] grammar wait timeout: waited_ms=%lld, limit_ms=%lld",
                                entry.stream->streamId(),
                                static_cast<long long>(waited_ms),
                                static_cast<long long>(grammar_compile_timeout_ms_));
            decrementInFlightLocked(entry.key.id());
            failed.splice(failed.end(), grammar_queue_, it++);
            continue;
        }
        ++it;
    }
}

void GrammarManager::installReadyMatchers(std::list<GrammarEntry>&      ready,
                                          std::list<GenerateStreamPtr>& return_reqs) {
    // Multiple entries may share a key (in-flight dedup); cache write once
    // per kid, but every stream still gets its own fresh matcher.
    std::unordered_set<std::string> cache_written;
    for (auto& entry : ready) {
        if (!entry.stream) {
            continue;
        }
        return_reqs.emplace_back(entry.stream);
        if (!entry.stream->isActive()) {
            continue;
        }

        GrammarReadyPayload payload;
        try {
            payload = entry.future.get();
        } catch (const std::exception& e) {
            entry.stream->reportError(ErrorCode::INVALID_PARAMS,
                                      std::string("grammar compile error: ") + e.what());
            continue;
        }

        const std::string kid = entry.key.id();
        if (payload.is_invalid || !payload.compiled) {
            // Cache only real schema rejections; transient system errors retry.
            if (payload.is_invalid && cache_written.insert(kid).second) {
                backend_->setCacheInvalid(entry.key, payload.error_msg);
            }
            std::string err = payload.error_msg.empty() ? "unknown compile error" : payload.error_msg;
            entry.stream->reportError(ErrorCode::INVALID_PARAMS,
                                      "Failed to compile " + entry.key.key_type + " grammar: " + err);
            continue;
        }

        if (cache_written.insert(kid).second) {
            backend_->setCache(entry.key, payload.compiled);
        }
        installMatcherOnStream(entry.stream, payload.compiled, entry.key,
                               entry.require_reasoning,
                               /*cache_hit=*/false, payload.compile_time_us);
        RTP_LLM_LOG_DEBUG("stream [%ld] grammar ready -> waiting candidate: key=%s",
                          entry.stream->streamId(), entry.key.brief().c_str());
    }
}

void GrammarManager::reportFailedTimeouts(std::list<GrammarEntry>&      failed,
                                          std::list<GenerateStreamPtr>& return_reqs) {
    // Timeouts are NOT cached as invalid. A timeout reflects environment
    // (CPU contention, cold lazy init, memory pressure) more often than a
    // genuinely unparseable schema, and permanently rejecting a healthy
    // schema after one slow attempt is a worse failure mode than letting
    // the next request retry. Real schema rejections still poison the cache
    // — installReadyMatchers does that on payload.is_invalid only.
    for (auto& entry : failed) {
        if (!entry.stream) {
            continue;
        }
        return_reqs.emplace_back(entry.stream);
        entry.stream->reportError(ErrorCode::GENERATE_TIMEOUT, "Grammar preprocessing timed out");
    }
}

std::list<GenerateStreamPtr> GrammarManager::getReadyGrammarRequests() {
    std::list<GrammarEntry> ready;
    std::list<GrammarEntry> failed;
    size_t                  queue_size_before = 0;
    size_t                  queue_size_after  = 0;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_size_before = grammar_queue_.size();
        if (grammar_queue_.empty()) {
            return {};
        }
        pollAndDrainLocked(ready, failed);
        queue_size_after = grammar_queue_.size();
    }

    if (ready.empty() && failed.empty()) {
        return {};
    }

    RTP_LLM_LOG_DEBUG("grammar poll: ready=%zu, failed=%zu, queue=%zu->%zu",
                      ready.size(), failed.size(), queue_size_before, queue_size_after);

    std::list<GenerateStreamPtr> return_reqs;
    installReadyMatchers(ready, return_reqs);
    reportFailedTimeouts(failed, return_reqs);

    RTP_LLM_LOG_DEBUG("grammar poll done: returning %zu streams", return_reqs.size());
    return return_reqs;
}

void GrammarManager::abortRequests(const GenerateStreamPtr& stream) {
    assert(stream);
    bool removed;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        removed = removeFromQueueLocked(stream);
    }

    if (!removed) {
        RTP_LLM_LOG_DEBUG("abortRequests: stream [%ld] not in grammar queue (no-op)", stream->streamId());
        return;
    }
    RTP_LLM_LOG_DEBUG("abortRequests: stream [%ld] removed from grammar queue", stream->streamId());
    stream->reportError(ErrorCode::CANCELLED, "Aborted");
}

void GrammarManager::abortAll() {
    // Drain under lock, reportError outside — stream callbacks may take other
    // locks (scheduler.lock_), and holding queue_mutex_ across that risks
    // inversion vs processReqWithGrammar's in_flight_ insert.
    std::list<GrammarEntry> drained;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        drained.swap(grammar_queue_);
        compile_tasks_.clear();
        in_flight_.clear();
    }

    RTP_LLM_LOG_INFO("GrammarManager abortAll: drained=%zu", drained.size());
    for (auto& entry : drained) {
        if (!entry.stream) {
            continue;
        }
        entry.stream->reportError(ErrorCode::CANCELLED, "scheduler stopped");
        entry.stream->clearGrammarMatcher();
    }
}

void GrammarManager::cleanupStream(const GenerateStreamPtr& stream) {
    assert(stream);
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (removeFromQueueLocked(stream)) {
            RTP_LLM_LOG_DEBUG("cleanupStream: stream [%ld] removed from grammar queue", stream->streamId());
        }
    }
    stream->clearGrammarMatcher();
}

void GrammarManager::workerLoop() {
    alive_workers_.fetch_add(1, std::memory_order_relaxed);
    auto on_exit = [this] {
        alive_workers_.fetch_sub(1, std::memory_order_relaxed);
        worker_cv_.notify_all();
    };

    for (;;) {
        std::optional<CompileTask> popped;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            worker_cv_.wait(lock, [this] { return stop_ || !compile_tasks_.empty(); });
            if (stop_ && compile_tasks_.empty()) {
                on_exit();
                return;
            }
            popped.emplace(std::move(compile_tasks_.front()));
            compile_tasks_.pop_front();
        }

        if (stop_.load(std::memory_order_relaxed)) {
            try {
                popped->promise.set_value({nullptr, false, "shutdown", 0});
            } catch (const std::future_error&) {}
            continue;
        }

        RTP_LLM_LOG_DEBUG("grammar worker picked up task: key=%s, require_reasoning=%d",
                          popped->key.brief().c_str(), static_cast<int>(popped->require_reasoning));

        const auto t_start = std::chrono::steady_clock::now();

        GrammarReadyPayload payload;
        try {
            CompileResult result = backend_->compileNow(popped->key);
            payload.compiled     = std::move(result.compiled);
            payload.is_invalid   = result.is_invalid;
            payload.error_msg    = std::move(result.error_message);
        } catch (const std::exception& e) {
            payload.compiled   = nullptr;
            payload.is_invalid = false;
            payload.error_msg  = e.what();
        }

        const auto t_end       = std::chrono::steady_clock::now();
        const auto elapsed_us  = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
        payload.compile_time_us = elapsed_us;
        RTP_LLM_LOG_DEBUG("grammar worker compileNow done: key=%s, ok=%d, invalid=%d, elapsed_ms=%lld, err=%s",
                          popped->key.brief().c_str(),
                          static_cast<int>(payload.compiled != nullptr),
                          static_cast<int>(payload.is_invalid),
                          static_cast<long long>(elapsed_us / 1000),
                          payload.error_msg.empty() ? "" : payload.error_msg.c_str());

        // Cache eagerly so the result survives even when all subscribers
        // have timed out. Without this, a schema that compiles in 61s
        // (just past the 60s default timeout) would never be cached and
        // every future request would re-compile and re-timeout.
        if (payload.compiled) {
            backend_->setCache(popped->key, payload.compiled);
        } else if (payload.is_invalid) {
            backend_->setCacheInvalid(popped->key, payload.error_msg);
        }

        try {
            popped->promise.set_value(std::move(payload));
        } catch (const std::future_error& fe) {
            RTP_LLM_LOG_WARNING("grammar worker set_value future_error: key=%s, what=%s",
                                popped->key.brief().c_str(), fe.what());
        }
    }
}

}  // namespace rtp_llm
