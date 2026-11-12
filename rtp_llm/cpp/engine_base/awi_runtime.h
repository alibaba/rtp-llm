#pragma once

// Runtime instrumentation for the evaluator /awi protocol.
//
// This is deliberately a process-local, dependency-free observation store.  It
// is fed by real engine/scheduler/allocator code paths (scheduler snapshots,
// CudaGraphRunner capture/replay, KV allocation results).  It never invents an
// event from a requested configuration.

#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace rtp_llm {
namespace awi {

struct SequenceSnapshot {
    std::string request_id;
    int64_t     input_tokens    = 0;
    int64_t     kv_tokens       = 0;
    int64_t     reserved_tokens = 0;
};

struct Event {
    std::string              type;
    std::string              operation_id;
    std::string              request_id;
    int64_t                  step          = -1;
    int64_t                  input_tokens  = 0;
    int64_t                  start_ns      = 0;
    int64_t                  end_ns        = 0;
    double                   elapsed_ms    = 0.0;
    bool                     graph_replay  = false;
    std::string              backend;
    std::string              source;
    int64_t                  hit_tokens    = 0;
    int64_t                  concurrency   = 0;
    int64_t                  context_tokens = 0;
    std::string              failure_kind;
    int32_t                  rank          = 0;
    int32_t                  world_size    = 1;
    std::vector<SequenceSnapshot> sequences;
};

// A snapshot handed from the scheduler to the runtime.
struct ResidentSnapshot {
    std::vector<SequenceSnapshot> sequences;
    int64_t                       free_blocks    = 0;
    int64_t                       available_blocks = 0;
    int64_t                       total_blocks   = 0;
    std::string                   source;
};

class AwiRuntime {
public:
    static AwiRuntime& instance();

    void setParallel(int32_t rank, int32_t world_size);
    int32_t rank() const {
        return rank_;
    }
    int32_t worldSize() const {
        return world_size_;
    }

    // Actual accelerator backend of this build/runtime.
    static const std::string& graphBackend();
    static bool               graphBackendIsHip();

    // Build provenance.  Resolution order: env -> compile-time macro -> empty.
    static std::string internalCommit();
    static std::string publicCommit();
    static void        resetBuildCacheForTest();

    // Request binding.  stream_id is the engine-local integer id; request_id is
    // the evaluator supplied X-AWI-Request-ID string.
    void bindRequest(int64_t stream_id, const std::string& request_id);
    void unbindRequest(int64_t stream_id);
    std::string requestIdForStream(int64_t stream_id) const;

    // Graph capture/replay observed from CudaGraphRunner.
    void recordCapture(const std::string& backend, const std::string& source);
    void recordGraphForward(const std::string& backend, bool replayed, const std::string& source);

    // Scheduler/engine step observations.
    void recordPrefillChunk(const std::string& request_id,
                            int64_t            input_tokens,
                            int64_t            start_ns,
                            int64_t            end_ns,
                            bool               final_chunk,
                            int32_t            rank,
                            int32_t            world_size,
                            const std::string& source);
    void recordDecode(const std::string& request_id,
                      int64_t            step,
                      int64_t            elapsed_ns,
                      bool               graph_replay,
                      const std::string& backend,
                      int32_t            rank,
                      int32_t            world_size,
                      const std::string& source);
    void recordResident(const ResidentSnapshot& snapshot, const std::string& operation_id, const std::string& source);
    void recordReuse(const std::string& request_id,
                     int64_t            hit_tokens,
                     int64_t            input_tokens,
                     const std::string& source);
    void recordCapacityFailure(const std::string& operation_id,
                               int64_t            concurrency,
                               int64_t            context_tokens,
                               const std::string& failure_kind,
                               const std::string& detail,
                               const std::string& source);

    // Barrier probe.  request_ids are evaluator ids (header values).
    void armBarrier(const std::string& operation_id,
                    const std::vector<std::string>& request_ids,
                    int64_t                         context_tokens);
    void clearBarrier();
    bool barrierActive() const;
    bool isBarrierRequest(int64_t stream_id) const;
    void setRequestTarget(int64_t stream_id, int64_t target_tokens);
    // Called by the scheduler while a decode is resident.  Returns true only
    // when every barrier request has its full reservation materialized.
    bool observeResidentAndCheckReady(int64_t stream_id,
                                      int64_t input_tokens,
                                      int64_t kv_tokens,
                                      int64_t reserved_tokens);
    bool barrierReady() const;
    int64_t barrierContextTokens() const;
    std::string barrierOperationForObservation() const;
    size_t barrierRequestCount() const;
    // Returns true exactly once per armed barrier once the resident horizon is
    // complete.  The engine then takes one real scheduler snapshot.
    bool consumeResidentSnapshotRequest();

    // True once any AWI request has been bound.  Used to keep instrumentation
    // out of normal smoke traffic.
    bool recordingEnabled() const;

    // Serialize all observations.  operation_id filters events to that
    // operation plus global capture events; an empty id returns everything.
    // weights_json is a raw JSON array supplied by the weights census.
    std::string observationsJson(const std::string& operation_id, const std::string& weights_json) const;
    std::string eventsJson(const std::string& operation_id) const;

    static int64_t nowNs();
    static int64_t nowUs();

private:
    AwiRuntime() = default;

    struct RequestState {
        bool    bound          = false;
        int64_t input_tokens   = 0;
        int64_t kv_tokens      = 0;
        int64_t reserved_tokens = 0;
        int64_t target_tokens  = 0;
        bool    seen           = false;
    };

    std::string operationForRequestLocked(const std::string& request_id) const;
    void        appendEventLocked(const Event& event);
    void        writeJournalLocked(const Event& event);
    bool        barrierReadyLocked() const;

    mutable std::mutex                                    mutex_;
    std::map<int64_t, std::string>                        stream_to_request_;
    std::map<std::string, int64_t>                        request_to_stream_;
    std::map<std::string, RequestState>                   request_state_;
    std::map<std::string, std::string>                    request_operation_;
    std::vector<Event>                                    events_;
    std::string                                           barrier_operation_;
    std::set<std::string>                                 barrier_requests_;
    int64_t                                               barrier_context_tokens_ = 0;
    bool                                                  recording_              = false;
    bool                                                  resident_snapshot_recorded_ = false;
    int32_t                                               rank_                   = 0;
    int32_t                                               world_size_             = 1;
    // Per-forward graph status is written by PyWrappedModel on the engine
    // thread.  It is read immediately after executor_->process().
    bool                                                  last_graph_replay_      = false;
    std::string                                           last_graph_backend_;
    bool                                                  graph_seen_this_step_   = false;
    bool                                                  telemetry_file_ready_   = false;
    std::string                                           telemetry_path_;
};

// Helper used by PyWrappedModel::forward to bracket a single forward.
void beginForwardGraphObservation();
void observeForwardGraphReplay(const std::string& backend, bool replayed, const std::string& source);
// Returns true and fills backend/replayed if a graph forward happened since the
// last beginForwardGraphObservation().
bool takeForwardGraphObservation(std::string& backend, bool& replayed);

}  // namespace awi
}  // namespace rtp_llm