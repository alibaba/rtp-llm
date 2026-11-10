#pragma once

// AWI evaluator instrumentation registry (per backend-rank process singleton).
//
// Stores genuine runtime observations required by the evaluator contract:
//   * operation-scoped admission barriers (armed via updateSchedulerInfo,
//     decided inside the FIFO scheduler against real KV block accounting),
//   * prefill windows, per-step decode timings (with graph_replay flag),
//     resident scheduler snapshots, memory-KV reuse hit tokens, capacity
//     failures and graph capture events,
//   * build/configuration metadata and per-rank weight placement reported
//     by the Python model layer.
//
// All records come from live engine/scheduler/cache execution paths; nothing
// here synthesizes events from requested settings. The frontend aggregates
// per-rank queryJson() outputs across the world group.

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rtp_llm {

struct AwiBarrierOp {
    std::string              operation_id;
    std::vector<std::string> request_ids;
    int64_t                  context_tokens {0};
    int64_t                  arm_time_ns {0};
    bool                     released {false};   // capacity fit observed; admission unlocked
    bool                     failed {false};     // genuine capacity failure reported
    bool                     resident_recorded {false};
};

struct AwiPrefillEvent {
    std::string request_id;
    int64_t     input_tokens {0};
    int64_t     start_ns {0};
    int64_t     end_ns {0};
};

struct AwiDecodeEvent {
    std::string request_id;
    int64_t     step {0};
    std::string backend;
    bool        graph_replay {false};
    double      elapsed_ms {0.0};
};

struct AwiSeqSnapshot {
    std::string request_id;
    int64_t     input_tokens {0};
    int64_t     kv_tokens {0};
    int64_t     reserved_tokens {0};
};

struct AwiReuseEvent {
    std::string request_id;
    int64_t     hit_tokens {0};
};

struct AwiCapacityFailureEvent {
    std::string operation_id;
    int64_t     concurrency {0};
    int64_t     context_tokens {0};
    std::string failure_kind;  // device_oom | kv_capacity_exceeded
};

class AwiObservationRegistry {
public:
    static AwiObservationRegistry& instance();

    // ---- barrier arming (scheduler-info channel) ----
    void armBarrier(const std::string& operation_id, std::vector<std::string> request_ids, int64_t context_tokens);
    bool isArmed(const std::string& request_id, std::string* operation_id_out) const;
    bool belongsToOp(const std::string& operation_id, const std::string& request_id) const;
    size_t opPeerCount(const std::string& operation_id) const;
    bool opResolved(const std::string& operation_id) const;
    void releaseBarrier(const std::string& operation_id);
    void failBarrier(const std::string& operation_id, const std::string& failure_kind, int64_t concurrency);
    bool markResidentRecorded(const std::string& operation_id);

    // ---- event recording (engine/scheduler/cache paths) ----
    void recordPrefill(const AwiPrefillEvent& e);
    void recordDecode(const AwiDecodeEvent& e);
    void recordResident(const std::vector<AwiSeqSnapshot>& sequences);
    void recordReuse(const AwiReuseEvent& e);
    void recordCapture(const std::string& backend);
    void recordStartupNetworkFetch();

    // ---- engine step-tap helpers (only armed/tracked requests are stored) ----
    bool isTracked(const std::string& request_id) const;
    // Chunked-prefill window tracking: first step sets start_ns, later steps
    // extend end_ns; finalize emits the aggregated event exactly once.
    void prefillStep(const std::string& request_id, int64_t input_tokens, int64_t step_start_ns, int64_t step_end_ns);
    void finalizePrefill(const std::string& request_id);
    // Strictly increasing per-request decode step counter (contract: 1..N-1).
    int64_t nextDecodeStep(const std::string& request_id);
    // Graph replay evidence: incremented from the real CudaGraphRunner replay
    // path; the engine samples around executor_->process() for a per-step flag.
    void     noteGraphReplay();
    uint64_t graphReplayCount() const;
    // Reuse hits are recorded once per request (first nonzero reuseLength).
    void recordReuseOnce(const std::string& request_id, int64_t hit_tokens);
    // Barrier bookkeeping for the resident snapshot (single emit per op).
    bool opAwaitingResident(std::string* operation_id_out) const;
    bool opRequestIds(const std::string& operation_id, std::vector<std::string>* ids_out) const;
    // Device-OOM capacity failure; attributed to the newest armed operation.
    void recordDeviceOom(int64_t concurrency_hint);

    // ---- metadata (python layer, once per rank) ----
    void setBuildInfo(const std::string& internal_sha, const std::string& public_sha);
    void setConfiguration(int64_t context_tokens, bool memory_kv_reuse, const std::string& graph_backend);
    void setWeightsPlacementJson(const std::string& weights_json_array);
    void setRank(int64_t world_rank);

    // ---- query (per-rank JSON; frontend aggregates) ----
    std::string queryJson(const std::string& operation_id) const;

    static int64_t nowNs();

private:
    AwiObservationRegistry() = default;

    static std::string jsonEscape(const std::string& in);

    struct PendingPrefill {
        int64_t input_tokens {0};
        int64_t start_ns {0};
        int64_t end_ns {0};
        bool    finalized {false};
    };

    mutable std::mutex                                            mu_;
    std::unordered_map<std::string, AwiBarrierOp>                 ops_;
    std::unordered_map<std::string, std::string>                  request_to_op_;
    std::unordered_map<std::string, PendingPrefill>               prefill_pending_;
    std::unordered_map<std::string, int64_t>                      decode_steps_;
    std::unordered_set<std::string>                               reuse_recorded_;
    std::atomic<uint64_t>                                         graph_replay_count_ {0};
    std::deque<AwiPrefillEvent>                                   prefill_events_;
    std::deque<AwiDecodeEvent>                                    decode_events_;
    std::deque<std::vector<AwiSeqSnapshot>>                       resident_events_;
    std::deque<AwiReuseEvent>                                     reuse_events_;
    std::deque<AwiCapacityFailureEvent>                           capacity_events_;
    std::deque<std::string>                                       capture_backends_;
    int64_t                                                       startup_network_fetches_ {0};
    std::string                                                   internal_sha_;
    std::string                                                   public_sha_;
    int64_t                                                       cfg_context_tokens_ {0};
    bool                                                          cfg_memory_kv_reuse_ {false};
    std::string                                                   cfg_graph_backend_;
    std::string                                                   weights_json_;
    int64_t                                                       rank_world_rank_ {-1};
};

}  // namespace rtp_llm
