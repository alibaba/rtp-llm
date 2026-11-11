#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rtp_llm {

// Actual graph backend name for the build target: "cuda" on NVIDIA, "hip"
// on MI308X. Reported in capture/decode events.
inline constexpr const char* awiGraphBackendName() {
#if USING_ROCM
    return "hip";
#else
    return "cuda";
#endif
}

// One observation record per the AWI adaptation evaluator contract.
// Events are captured from real engine execution paths (step loop, graph
// runner, cache manager, scheduler admission); nothing here may be
// synthesized from requested configuration alone.
struct AwiEvent {
    std::string type;             // capture|prefill|decode|resident|reuse|capacity_failure
    int64_t     rank{-1};         // recording rank
    std::string operation_id;     // armed benchmark operation, empty if none
    int64_t     request_id{-1};   // engine-internal request id
    std::string awi_request_id;   // X-AWI-Request-ID passthrough

    // prefill: whole window including all chunks, common host clock
    int64_t input_tokens{-1};
    int64_t start_ns{-1};
    int64_t end_ns{-1};

    // decode: one aggregated event per scheduler step per request
    int64_t     step{-1};          // 1..N-1; first output token belongs to prefill
    std::string backend;           // "cuda" | "hip", from actual runner
    int         graph_replay{-1};  // 1 replayed, 0 eager fallback, -1 unknown
    double      elapsed_ms{-1};    // synchronized step incl host/communication overhead

    // resident snapshot
    int64_t kv_tokens{-1};
    int64_t reserved_tokens{-1};

    // reuse
    int64_t hit_tokens{-1};

    // capacity_failure
    int64_t     concurrency{-1};
    int64_t     context_tokens{-1};
    std::string failure_kind;  // device_oom|kv_capacity_exceeded

    // capture / misc structured detail
    std::string detail_json;
};

// Armed operation-scoped admission/residency probe (POST /awi/barrier).
struct AwiOperation {
    std::string                     operation_id;
    std::unordered_set<std::string> awi_request_ids;
    std::unordered_set<int64_t>     request_ids;  // resolved internal ids
    int64_t                         context_tokens{0};
    bool                            released{false};
    // One-shot guard so exactly one resident snapshot is emitted per armed
    // operation (rank 0 only; the engine step loop test-and-sets this).
    bool resident_emitted{false};
};

// Per-backend-process collector. Frontend aggregates all ranks through the
// observation RPC; each rank only records what it actually executed.
class AwiCollector {
public:
    static AwiCollector& instance();

    void setRank(int rank);
    int  rank() const;

    void                recordEvent(AwiEvent event);
    std::vector<AwiEvent> events(const std::string& operation_id) const;
    void                clearEvents();

    // Barrier operation state (written by the RPC handler, read by scheduler
    // and engine step loop). Empty operation_id disarms.
    void        armOperation(AwiOperation operation);
    bool        hasArmedOperation() const;
    AwiOperation armedOperation() const;
    void        bindInternalId(const std::string& awi_request_id, int64_t internal_id);
    void        markReleased();
    void        clearOperation();
    // Test-and-set: returns true exactly once per armed operation, for the
    // single resident snapshot the contract requires.
    bool        markResidentEmitted();

    // Startup facts injected once by the Python layer (build identity, weight
    // placement derived from real allocations, measured network fetch count).
    void        setBuildInfo(std::string internal_sha, std::string public_sha);
    void        setWeightsJson(std::string json);
    void        setStartupNetworkFetches(int64_t count);
    void        setConfigurationJson(std::string json);
    std::string buildInfoJson() const;
    std::string weightsJson() const;
    int64_t     startupNetworkFetches() const;
    std::string configurationJson() const;

    // Graph-runner replay epochs: incremented on every real graph replay,
    // separately for decode and prefill graphs. The engine step hook brackets
    // process() with epoch reads; a decode step whose decode epoch did not
    // advance is an eager fallback (graph_replay=0).
    void noteGraphReplay(bool decode) {
        (decode ? decode_replay_epoch_ : prefill_replay_epoch_)
            .fetch_add(1, std::memory_order_relaxed);
    }
    int64_t graphReplayEpoch(bool decode) const {
        return (decode ? decode_replay_epoch_ : prefill_replay_epoch_)
            .load(std::memory_order_relaxed);
    }

    // Executor collective-forward epoch: incremented on every real model
    // forward, which the tpSync contract makes identical on all TP ranks.
    // The engine step hook brackets process() with epoch reads to tell steps
    // that ran collective work from idle scheduler polls, so idle steps no
    // longer flood the observations payload with per-rank step events
    // (measured: 3.2M idle events => 432MB JSON on a 25-minute server).
    void noteCollectiveForward() {
        collective_forward_epoch_.fetch_add(1, std::memory_order_relaxed);
    }
    int64_t collectiveForwardEpoch() const {
        return collective_forward_epoch_.load(std::memory_order_relaxed);
    }

    // One-shot prefill/reuse event emission guard keyed by internal request id.
    bool markPrefillEmitted(int64_t request_id);

private:
    AwiCollector()                                       = default;
    AwiCollector(const AwiCollector&)                    = delete;
    AwiCollector& operator=(const AwiCollector&)         = delete;

    // Ring cap per rank. Idle step events are gated at the engine, so what
    // remains is real execution evidence: a full 1M-token validation pass
    // emits a few thousand events per rank. 50k leaves an order of magnitude
    // of headroom while bounding /awi/observations to a few MB per rank.
    static constexpr size_t kMaxEvents = 50000;

    mutable std::mutex       mu_;
    std::deque<AwiEvent>     events_;
    int                      rank_{-1};
    bool                     has_armed_{false};
    AwiOperation             armed_;
    std::string              internal_sha_;
    std::string              public_sha_;
    std::string              weights_json_;
    std::string              configuration_json_;
    int64_t                  startup_network_fetches_{-1};  // -1 = not measured yet
    std::unordered_set<int64_t> prefill_emitted_;
    std::atomic<int64_t>        decode_replay_epoch_{0};
    std::atomic<int64_t>        prefill_replay_epoch_{0};
    std::atomic<int64_t>        collective_forward_epoch_{0};
};

}  // namespace rtp_llm
