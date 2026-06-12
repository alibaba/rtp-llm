#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarCompiler.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarSchemaValidator.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::mutex                       GrammarCompiler::singleton_mutex_;
std::unique_ptr<GrammarCompiler> GrammarCompiler::singleton_;
bool                             GrammarCompiler::initialized_ = false;
std::optional<size_t>            GrammarCompiler::config_fingerprint_;

namespace {
bool isDisabledName(const std::string& name) noexcept {
    return name.empty() || name == "none" || name == "None" || name == "NONE";
}

size_t mixFingerprint(size_t seed, size_t value) {
    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

size_t grammarConfigFingerprint(const GrammarConfig& cfg) {
    // Every field consumed by GrammarCompiler / XGrammarBackendOptions must be mixed in:
    // initialize() compares fingerprints to decide whether a re-init is a no-op vs a
    // misuse error, so leaving a field out causes silent acceptance of a stale config.
    size_t h = std::hash<std::string>{}(cfg.tokenizer_info_json);
    h        = mixFingerprint(h, std::hash<std::string>{}(cfg.grammar_backend));
    h        = mixFingerprint(h, static_cast<size_t>(cfg.constrained_json_disable_any_whitespace));
    h        = mixFingerprint(h, static_cast<size_t>(cfg.num_workers));
    h        = mixFingerprint(h, static_cast<size_t>(cfg.compile_timeout_ms));
    h        = mixFingerprint(h, static_cast<size_t>(cfg.mask_wait_timeout_ms));
    for (int32_t token : cfg.override_stop_tokens) {
        h = mixFingerprint(h, static_cast<size_t>(token));
    }
    return h;
}

XGrammarBackendOptions backendOptionsFromConfig(const GrammarConfig& cfg) {
    XGrammarBackendOptions opts;
    opts.any_whitespace        = !cfg.constrained_json_disable_any_whitespace;
    opts.strict_mode           = true;
    opts.max_compiler_threads  = std::max(1, cfg.num_workers);
    opts.enable_compiler_cache = true;
    opts.compiler_cache_bytes  = -1;
    if (!cfg.override_stop_tokens.empty()) {
        opts.override_stop_tokens = cfg.override_stop_tokens;
    }
    return opts;
}
}  // namespace

std::shared_ptr<XGrammarBackend> GrammarCompiler::buildBackend(const GrammarConfig& cfg) noexcept {
    try {
        if (isDisabledName(cfg.grammar_backend) || cfg.tokenizer_info_json.empty()) {
            RTP_LLM_LOG_INFO("GrammarCompiler: structured output disabled (backend=%s, tokenizer_info_json_empty=%d)",
                             cfg.grammar_backend.c_str(),
                             static_cast<int>(cfg.tokenizer_info_json.empty()));
            return nullptr;
        }
        if (cfg.grammar_backend != "xgrammar") {
            RTP_LLM_LOG_ERROR("GrammarCompiler: unknown grammar_backend='%s' -> disabled", cfg.grammar_backend.c_str());
            return nullptr;
        }

        XGrammarBackendOptions opts = backendOptionsFromConfig(cfg);
        auto backend = std::make_shared<XGrammarBackend>(cfg.tokenizer_info_json, opts);
        if (!backend) {
            RTP_LLM_LOG_ERROR("GrammarCompiler: XGrammarBackend construction returned null -> disabled");
            return nullptr;
        }
        if (!backend->isEnabled()) {
            RTP_LLM_LOG_ERROR("GrammarCompiler: XGrammarBackend reports !isEnabled() (backend unhealthy) -> disabled");
            return nullptr;
        }
        RTP_LLM_LOG_INFO("GrammarCompiler: xgrammar backend ready (override_stop_tokens=%zu, num_workers=%d)",
                         cfg.override_stop_tokens.size(),
                         opts.max_compiler_threads);
        return backend;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("GrammarCompiler: backend build threw (%s); disabling grammar", e.what());
        return nullptr;
    }
}

void GrammarCompiler::initialize(const GrammarConfig& cfg) {
    std::lock_guard<std::mutex> lock(singleton_mutex_);
    const size_t                fingerprint = grammarConfigFingerprint(cfg);
    if (initialized_) {
        if (config_fingerprint_.has_value() && *config_fingerprint_ == fingerprint) {
            RTP_LLM_LOG_INFO("GrammarCompiler already initialized with matching config; ignoring re-init");
            return;
        }
        // Reject silently keeping the first install: in a multi-model process
        // (e.g. embed + chat hosted side-by-side) "old config wins" silently
        // routes a new model's grammar requests through the wrong tokenizer,
        // and the only symptom is schema-illegal tokens at sample time. Surface
        // the misuse loudly so callers either resetForTest() or split into
        // separate processes.
        RTP_LLM_LOG_ERROR(
            "GrammarCompiler: initialize() called again with a DIFFERENT grammar config "
            "(fingerprint 0x%zx vs first 0x%zx). Multiple engines with different grammar "
            "configs in one process is unsupported.",
            fingerprint,
            config_fingerprint_.value_or(0));
        throw std::runtime_error(
            "GrammarCompiler: re-initialize attempted with a different config; "
            "multiple grammar configs in one process are unsupported");
    }
    auto backend = buildBackend(cfg);
    singleton_.reset(new GrammarCompiler(std::move(backend), cfg));
    config_fingerprint_ = fingerprint;
    initialized_        = true;
}

GrammarCompiler& GrammarCompiler::instance() {
    std::lock_guard<std::mutex> lock(singleton_mutex_);
    if (!singleton_) {
        // Never initialized (or reset by a test): behave as a disabled compiler
        // without claiming the process singleton. A later initialize() must still
        // be able to install the configured backend. Because of that, callers must
        // NOT cache the returned reference across a possible initialize() — always
        // call instance() fresh — or they may keep pointing at this disabled stub.
        static GrammarCompiler disabled(nullptr, GrammarConfig());
        // 只在第一次返回 stub 时打 warning：稳态下被反复调用会刷屏。生产环境
        // 走到这里说明 initialize() 没跑（pybind 注入顺序错），是配置 bug 而不是
        // 业务异常 —— 必须可见。
        static std::once_flag warn_once;
        std::call_once(warn_once, [] {
            RTP_LLM_LOG_WARNING(
                "GrammarCompiler::instance() returned disabled stub: initialize() was never called. "
                "Grammar requests will be rejected at admission until the engine wires GrammarCompiler::initialize.");
        });
        return disabled;
    }
    return *singleton_;
}

void GrammarCompiler::resetForTest() noexcept {
    std::unique_ptr<GrammarCompiler> old;
    {
        std::lock_guard<std::mutex> lock(singleton_mutex_);
        old = std::move(singleton_);
        initialized_        = false;
        config_fingerprint_.reset();
    }
    // Destructor runs outside the lock — joins workers (or detaches with state
    // kept alive via shared_ptr). Callers must ensure no concurrent instance()
    // users exist.
    old.reset();
}

GrammarCompiler::GrammarCompiler(std::shared_ptr<XGrammarBackend> backend, const GrammarConfig& cfg):
    state_(std::make_shared<WorkerState>()) {
    state_->backend = std::move(backend);
    if (cfg.compile_timeout_ms > 0) {
        grammar_compile_timeout_ms_ = cfg.compile_timeout_ms;
    }
    if (cfg.mask_wait_timeout_ms > 0) {
        mask_wait_timeout_ms_ = cfg.mask_wait_timeout_ms;
    }
    if (!enabled()) {
        RTP_LLM_LOG_INFO("GrammarCompiler init: backend=disabled, compile_timeout_ms=%lld",
                         static_cast<long long>(grammar_compile_timeout_ms_));
        return;
    }

    int num_workers = std::max(1, cfg.num_workers);
    RTP_LLM_LOG_INFO("GrammarCompiler init: backend=cpp, compile_timeout_ms=%lld, num_workers=%d",
                     static_cast<long long>(grammar_compile_timeout_ms_),
                     num_workers);

    workers_.reserve(num_workers);
    try {
        for (int i = 0; i < num_workers; ++i) {
            // Capture state_ by value: each worker owns a strong reference to
            // the shared state, so detach-on-shutdown leaves the worker with
            // a still-live mutex / queue / backend until it exits.
            workers_.emplace_back([state = state_] { workerLoop(state); });
        }
    } catch (const std::exception& e) {
        // std::thread construction can throw under resource exhaustion. Join
        // already-spawned workers and drop the backend so we degrade to disabled
        // mode instead of leaving joinable threads dangling.
        RTP_LLM_LOG_ERROR("GrammarCompiler init: failed to spawn worker thread (%s); "
                          "tearing down %zu spawned worker(s) and disabling grammar.",
                          e.what(), workers_.size());
        {
            std::lock_guard<std::mutex> lock(state_->queue_mutex);
            state_->stop = true;
        }
        state_->worker_cv.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
        workers_.clear();
        state_->stop    = false;
        state_->backend = nullptr;
    }
}

GrammarCompiler::~GrammarCompiler() {
    {
        std::lock_guard<std::mutex> lock(state_->queue_mutex);
        state_->stop = true;
        for (auto& task : state_->compile_tasks) {
            try {
                task.promise.set_value({nullptr, false, false, "grammar compiler shutting down", 0});
            } catch (const std::future_error&) {}
        }
        state_->compile_tasks.clear();
    }
    state_->worker_cv.notify_all();

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(grammar_compile_timeout_ms_);
    {
        std::unique_lock<std::mutex> lock(state_->queue_mutex);
        state_->worker_cv.wait_until(lock, deadline,
                                      [s = state_.get()] { return s->alive_workers.load() == 0; });
    }

    int stuck = state_->alive_workers.load();
    if (stuck > 0) {
        // Detach instead of abort: graceful shutdown getting escalated to a
        // hard crash because a single xgrammar compile took longer than the
        // timeout is a worse experience than letting the worker leak. Detach
        // is safe here because the worker captured a shared_ptr<WorkerState>
        // — the state survives this destructor and gets reaped when the last
        // detached worker exits (or the OS reaps it at process _exit, which
        // is the steady-state path: GrammarCompiler is a singleton destroyed
        // only at process teardown).
        RTP_LLM_LOG_ERROR("GrammarCompiler shutdown: %d worker(s) stuck in compileNow after %lld ms; "
                          "detaching (state kept alive via shared_ptr until workers exit)",
                          stuck, static_cast<long long>(grammar_compile_timeout_ms_));
        for (auto& t : workers_) {
            if (t.joinable()) {
                t.detach();
            }
        }
        return;
    }
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

size_t GrammarCompiler::pendingTasks() const {
    std::lock_guard<std::mutex> lock(state_->queue_mutex);
    return state_->compile_tasks.size();
}

std::shared_future<GrammarReadyPayload> GrammarCompiler::makeReadyFuture(GrammarReadyPayload payload) const {
    std::promise<GrammarReadyPayload> promise;
    auto                              future = promise.get_future().share();
    promise.set_value(std::move(payload));
    return future;
}

std::unique_ptr<RtpGrammarMatcher> GrammarCompiler::createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                                                  bool terminate_without_stop_token) {
    return state_->backend->createMatcher(std::move(compiled), terminate_without_stop_token);
}

std::shared_future<GrammarReadyPayload> GrammarCompiler::submit(const GrammarKeyCpp& key) {
    // Fast paths: serve cache hits / cached-invalids without touching the queue.
    if (auto compiled = state_->backend->getCached(key); compiled) {
        GrammarReadyPayload payload;
        payload.compiled        = std::move(compiled);
        payload.cache_hit       = true;
        payload.compile_time_us = 0;
        return makeReadyFuture(std::move(payload));
    }
    if (auto err = state_->backend->getCachedInvalid(key); !err.empty()) {
        GrammarReadyPayload payload;
        payload.is_invalid = true;
        payload.error_msg  = std::move(err);
        return makeReadyFuture(std::move(payload));
    }

    const std::string kid = key.id();

    // Check singleflight before expensive validation: if another thread
    // already submitted the same key, subscribe to its future directly.
    {
        std::lock_guard<std::mutex> lock(state_->queue_mutex);
        auto it = state_->in_flight.find(kid);
        if (it != state_->in_flight.end() && it->second.valid()) {
            RTP_LLM_LOG_INFO("grammar submit subscribed to in-flight: key=%s", key.brief().c_str());
            return it->second;
        }
    }

    // Validate outside the lock — only the first submitter pays this cost.
    // Concurrent duplicates that arrived before the in_flight entry was
    // installed will also validate, but the second lock acquisition below
    // will find the entry and subscribe instead of double-queuing.
    auto validate_result = validateGrammarKey(key);
    if (validate_result.status != GrammarValidateStatus::Ok) {
        state_->backend->setCacheInvalid(key, validate_result.detail);
        GrammarReadyPayload payload;
        payload.is_invalid = true;
        payload.error_msg  = std::move(validate_result.detail);
        return makeReadyFuture(std::move(payload));
    }

    bool submitted_new_task = false;
    std::shared_future<GrammarReadyPayload> future;
    {
        std::lock_guard<std::mutex> lock(state_->queue_mutex);
        auto it = state_->in_flight.find(kid);
        if (it != state_->in_flight.end() && it->second.valid()) {
            future = it->second;
        } else {
            std::promise<GrammarReadyPayload> promise;
            future                  = promise.get_future().share();
            state_->in_flight[kid]  = future;
            CompileTask task;
            task.key     = key;
            task.promise = std::move(promise);
            state_->compile_tasks.emplace_back(std::move(task));
            submitted_new_task = true;
        }
    }
    if (submitted_new_task) {
        state_->worker_cv.notify_one();
    }
    RTP_LLM_LOG_INFO("grammar submit %s: key=%s",
                      submitted_new_task ? "queued" : "subscribed to in-flight", key.brief().c_str());
    return future;
}

void GrammarCompiler::workerLoop(std::shared_ptr<WorkerState> state) {
    state->alive_workers.fetch_add(1, std::memory_order_relaxed);
    auto on_exit = [&state] {
        state->alive_workers.fetch_sub(1, std::memory_order_relaxed);
        state->worker_cv.notify_all();
    };

    for (;;) {
        std::optional<CompileTask> popped;
        {
            std::unique_lock<std::mutex> lock(state->queue_mutex);
            state->worker_cv.wait(lock, [&state] { return state->stop || !state->compile_tasks.empty(); });
            if (state->stop && state->compile_tasks.empty()) {
                on_exit();
                return;
            }
            popped.emplace(std::move(state->compile_tasks.front()));
            state->compile_tasks.pop_front();
        }

        const std::string kid = popped->key.id();
        auto              erase_in_flight = [&state, &kid] {
            std::lock_guard<std::mutex> lock(state->queue_mutex);
            auto it = state->in_flight.find(kid);
            if (it != state->in_flight.end()) {
                state->in_flight.erase(it);
            }
        };

        if (state->stop.load(std::memory_order_relaxed)) {
            try {
                popped->promise.set_value({nullptr, false, false, "shutdown", 0});
            } catch (const std::future_error&) {}
            erase_in_flight();
            continue;
        }

        // Catch-all around the whole task body: compileNow has its own
        // exception->payload handling, but setCache / logging / set_value can
        // still throw (e.g. bad_alloc). An exception escaping this thread would
        // std::terminate the process and leave every subscriber's future
        // permanently unfulfilled. Convert any escape into a system-error
        // payload so subscribers unblock and the worker keeps serving.
        try {
            RTP_LLM_LOG_INFO("grammar worker picked up task: key=%s", popped->key.brief().c_str());

            const auto t_start = std::chrono::steady_clock::now();

            GrammarReadyPayload payload;
            try {
                CompileResult result = state->backend->compileNow(popped->key);
                payload.compiled     = std::move(result.compiled);
                payload.is_invalid   = result.is_invalid;
                payload.error_msg    = std::move(result.error_message);
            } catch (const std::exception& e) {
                payload.compiled   = nullptr;
                payload.is_invalid = false;
                payload.error_msg  = e.what();
            }

            const auto t_end      = std::chrono::steady_clock::now();
            const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
            payload.compile_time_us = elapsed_us;
            RTP_LLM_LOG_INFO("grammar worker compileNow done: key=%s, ok=%d, invalid=%d, elapsed_ms=%lld, err=%s",
                             popped->key.brief().c_str(),
                             static_cast<int>(payload.compiled != nullptr),
                             static_cast<int>(payload.is_invalid),
                             static_cast<long long>(elapsed_us / 1000),
                             payload.error_msg.empty() ? "" : payload.error_msg.c_str());

            // Cache eagerly so the result survives even when all subscribers
            // have timed out. Without this, a schema that compiles in 61s (just
            // past the 60s default timeout) would never be cached and every
            // future request would re-compile and re-timeout.
            if (payload.compiled) {
                state->backend->setCache(popped->key, payload.compiled);
            } else if (payload.is_invalid) {
                state->backend->setCacheInvalid(popped->key, payload.error_msg);
            }

            popped->promise.set_value(std::move(payload));
        } catch (const std::future_error& fe) {
            RTP_LLM_LOG_WARNING("grammar worker set_value future_error: key=%s, what=%s",
                                popped->key.brief().c_str(), fe.what());
        } catch (const std::bad_alloc& e) {
            // 资源型瞬时失败：不缓存，下一次相同 key 重新走 compile，避免把
            // 内存压力固化成永久 invalid 状态。
            RTP_LLM_LOG_ERROR("grammar worker task body bad_alloc (transient); not caching, key=%s",
                              popped->key.brief().c_str());
            try {
                popped->promise.set_value({nullptr, false, false, std::string("grammar worker bad_alloc: ") + e.what(), 0});
            } catch (const std::future_error&) {}
        } catch (const std::exception& e) {
            // 任务体内 setCache / 序列化等抛出的非资源型异常：把 key 标 invalid
            // 缓存，避免后续相同 schema 反复重试触发 retry storm（subscribers
            // 仍能从 promise 立即拿到错误）。
            const std::string err = std::string("grammar worker exception: ") + e.what();
            RTP_LLM_LOG_ERROR("grammar worker task body threw (%s); marking key invalid to suppress retry storm: key=%s",
                              e.what(), popped->key.brief().c_str());
            try {
                if (state->backend) {
                    state->backend->setCacheInvalid(popped->key, err);
                }
            } catch (...) {}
            try {
                popped->promise.set_value({nullptr, true, false, err, 0});
            } catch (const std::future_error&) {}
        } catch (...) {
            RTP_LLM_LOG_ERROR("grammar worker task body threw unknown exception; key=%s", popped->key.brief().c_str());
            try {
                popped->promise.set_value({nullptr, false, false, "grammar worker unknown exception", 0});
            } catch (const std::future_error&) {}
        }

        // Remove the singleflight slot AFTER the result has been cached + the
        // promise fulfilled: any concurrent submit that raced in still found the
        // (now-ready) future; later submits hit the backend cache instead. A
        // system-error result is intentionally not cached, so a subsequent
        // submit re-queues a fresh compile (retry semantics preserved).
        erase_in_flight();
    }
}

}  // namespace rtp_llm
