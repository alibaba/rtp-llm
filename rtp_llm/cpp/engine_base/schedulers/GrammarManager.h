#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <xgrammar/grammar.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// Worker → scheduler payload. (compiled != null) succeeded;
// (is_invalid) → cacheable schema rejection; otherwise system-level error
// (not cached, retry next time).
struct GrammarReadyPayload {
    std::shared_ptr<xgrammar::CompiledGrammar> compiled;
    bool                                       is_invalid      = false;
    std::string                                error_msg;
    int64_t                                    compile_time_us = 0;
};

// Schedules grammar compiles on a C++ worker pool, dedups in-flight identical
// keys via shared_future, replays prefill tokens into freshly-compiled
// matchers, and surfaces queue/compile timeouts. No GIL on any path.
class GrammarManager {
public:
    // backend=nullptr → "disabled" mode (cc_test ctors). Public methods become
    // no-ops; no workers are spawned.
    explicit GrammarManager(std::shared_ptr<XGrammarBackendCpp> backend       = nullptr,
                            GrammarConfig                       grammar_config = GrammarConfig());
    ~GrammarManager();  // blocking join — see ctor docstring in the .cc

    GrammarManager(const GrammarManager&)            = delete;
    GrammarManager& operator=(const GrammarManager&) = delete;

    size_t size() const;
    void   clear();
    bool   hasWaitingGrammars() const;

    // True iff the scheduler should call getReadyGrammarRequests now (avoids
    // spinning when every entry is still waiting on its compile future).
    bool hasActionableGrammar() const;

    // Returns true iff stream was queued for async compile (caller must NOT
    // enqueue into waiting_streams_ yet).
    bool processReqWithGrammar(const GenerateStreamPtr& stream);

    // Reaps finished compiles, installs matchers, returns streams ready to
    // proceed to waiting_streams_.
    std::list<GenerateStreamPtr> getReadyGrammarRequests();

    void abortRequests(const GenerateStreamPtr& stream);
    void abortAll();
    void cleanupStream(const GenerateStreamPtr& stream);

private:
    struct GrammarEntry {
        GenerateStreamPtr                       stream;
        GrammarKeyCpp                           key;
        bool                                    require_reasoning = false;
        std::shared_future<GrammarReadyPayload> future;
        std::chrono::steady_clock::time_point   deadline;
    };

    struct CompileTask {
        GrammarKeyCpp                     key;
        bool                              require_reasoning;
        std::promise<GrammarReadyPayload> promise;
    };

    // ref_count = how many GrammarEntry currently subscribe to `future`.
    // Drops to 0 → slot erased and any unstarted CompileTask for this kid
    // pruned (no readers remain).
    struct InFlightSlot {
        std::shared_future<GrammarReadyPayload> future;
        size_t                                  ref_count = 0;
    };

    void workerLoop();

    // Caller MUST hold queue_mutex_.
    void decrementInFlightLocked(const std::string& kid);
    bool removeFromQueueLocked(const GenerateStreamPtr& stream);

    bool          isGrammarRequested(const GenerateStreamPtr& stream) const;
    GrammarKeyCpp extractGrammarKey(const GenerateStreamPtr& stream) const;

    // Late-attached matchers must catch up on tokens already emitted in prefill.
    void replayPrefillTokensToGrammar(const GenerateStreamPtr& stream, RtpGrammarMatcher& matcher);

    // Build matcher around `compiled`, fill stats, init reasoning, replay
    // prefill tokens, attach to stream. Used by both the cache-hit and the
    // fresh-compile paths.
    void installMatcherOnStream(const GenerateStreamPtr&                   stream,
                                std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                const GrammarKeyCpp&                       key,
                                bool                                       require_reasoning,
                                bool                                       cache_hit,
                                int64_t                                    compile_time_us);

    // getReadyGrammarRequests phases. Each takes/returns plain lists; caller
    // composes the final return list. Held lock contract documented per fn.
    void pollAndDrainLocked(std::list<GrammarEntry>& ready, std::list<GrammarEntry>& failed);
    void installReadyMatchers(std::list<GrammarEntry>&        ready,
                              std::list<GenerateStreamPtr>&   return_reqs);
    void reportFailedTimeouts(std::list<GrammarEntry>&        failed,
                              std::list<GenerateStreamPtr>&   return_reqs);

    bool hasBackend() const noexcept { return backend_ != nullptr; }

    std::shared_ptr<XGrammarBackendCpp> backend_;

    mutable std::mutex      queue_mutex_;
    std::condition_variable worker_cv_;
    std::list<GrammarEntry> grammar_queue_;
    std::deque<CompileTask> compile_tasks_;

    // keyed by GrammarKeyCpp::id(). Invariant: each grammar_queue_ entry
    // contributes 1 to its slot's ref_count.
    std::unordered_map<std::string, InFlightSlot> in_flight_;

    std::vector<std::thread> workers_;
    std::atomic<bool>        stop_{false};
    std::atomic<int>         alive_workers_{0};

    int64_t grammar_compile_timeout_ms_ = 60000;
};

}  // namespace rtp_llm
