#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeWorkloadGenerator.h"

namespace rtp_llm::benchmark {

// Device blocks allocated at admission time and held by the benchmark request
// with REQUEST refs.
//
// load_target_blocks:  targets for non-joined lower-tier load descriptors.
// suffix_blocks:       device blocks for the unmatched tail of the request path.
// joined_holder_blocks: joined descriptor target_blocks carrying one deduplicated
//                       REQUEST ref owned by this request.
struct PreparedRequestResources {
    std::vector<std::vector<BlockIdxType>> load_target_blocks;    // per group set
    std::vector<std::vector<BlockIdxType>> suffix_blocks;         // per group set
    std::vector<std::vector<BlockIdxType>> joined_holder_blocks;  // per group set
    bool                                   load_targets_allocated{false};
    bool                                   suffix_allocated{false};
    bool                                   joined_holder_allocated{false};

    bool holdsBlocks() const {
        return load_targets_allocated || suffix_allocated || joined_holder_allocated;
    }
};

// Abstract cache surface driven exclusively by the single foreground scheduler
// thread. The real implementation adapts BlockTreeCache; fakes inject failures
// and observe call patterns in unit tests.
class OnlineCacheApi {
public:
    virtual ~OnlineCacheApi() = default;

    struct MatchOutcome {
        size_t matched_device_blocks{0};
        size_t host_matched_blocks{0};
        // Depth of the longest path found in the tree (device + lower-tier
        // positions). The load covers everything beyond matched_device_blocks,
        // so this is the actual reuse depth used for suffix preparation.
        size_t                            actual_matched_depth{0};
        std::vector<MultiNodeResource>    matched_device_resources;  // REQUEST refs held
        std::shared_ptr<LoadAsyncContext> load_ticket;               // nullable
    };

    virtual MatchOutcome match(const PathKeys& path) = 0;

    // Allocate device targets for every lower-tier load desc. On failure
    // releases everything already allocated and returns false with nothing
    // held. Only called when the outcome carries a load ticket.
    virtual bool allocateLoadTargets(const MatchOutcome& outcome, PreparedRequestResources& out) = 0;

    // Allocate device blocks for the unmatched suffix (one block per path
    // position per group set). On failure releases load targets and suffix
    // blocks and returns false.
    virtual bool allocateSuffixBlocks(size_t suffix_block_count, PreparedRequestResources& out) = 0;

    // Take REQUEST refs on the joined descriptor's real target_blocks so the
    // joined request holds the descriptor blocks independently of the loader's
    // transfer holder. Called only when the outcome carries a joined-load
    // ticket. Returns false if referencing fails.
    virtual bool holdJoinedBlocks(const MatchOutcome& outcome, PreparedRequestResources& out) = 0;

    // Set targets and commit the load ticket. On failure releases every held
    // block and returns false.
    virtual bool commitLoad(const std::shared_ptr<LoadAsyncContext>& ticket, PreparedRequestResources& out) = 0;

    // Full-path insert using the prepared blocks, then release every request
    // ref (matched prefix, load targets, suffix, joined holder) through
    // onBlocksReleased. `actual_matched_depth` determines where suffix blocks
    // are placed; no KV allocation happens here.
    virtual void publishInsert(const PathKeys&                 path,
                               size_t                          actual_matched_depth,
                               PreparedRequestResources&       out,
                               std::vector<MultiNodeResource>& matched_resources) = 0;

    // Release match-held REQUEST refs without any prepared blocks (used when
    // admission fails before allocation).
    virtual void releaseMatched(std::vector<MultiNodeResource>& resources) = 0;

    // Release every held block (load targets, suffix, joined holder, matched
    // prefix). Used by rollback and cleanup; must be idempotent.
    virtual void rollback(PreparedRequestResources& out, std::vector<MultiNodeResource>& matched_resources) = 0;
};

// Benchmark-local bounded polling helper used by the real adapter.
bool boundedPendingTaskDrain(const std::function<size_t()>& pending_count, std::chrono::milliseconds budget);

enum class OnlineRequestState : uint8_t {
    WAITING,
    LOADING_CACHE,
    READY,
    FINISHED,
};

// One logical request context. Only the scheduler thread reads or modifies it.
struct OnlineRequestContext {
    size_t   request_id{0};
    PathKeys path;
    size_t   planned_reuse_blocks{0};
    size_t   target_tokens{0};

    OnlineRequestState                    state{OnlineRequestState::WAITING};
    std::chrono::steady_clock::time_point admitted_at{};
    std::chrono::steady_clock::time_point match_at{};
    size_t                                matched_depth{0};
    size_t                                matched_device_blocks{0};
    size_t                                host_matched_blocks{0};
    std::vector<MultiNodeResource>        matched_device_resources;
    std::shared_ptr<LoadAsyncContext>     load_ticket;
    PreparedRequestResources              prepared;
    bool                                  tokens_counted{false};

    // Dependency metadata from the trace descriptor.
    size_t  family_id{0};
    size_t  epoch_id{0};
    size_t  generation{0};
    int64_t predecessor_id{-1};
    bool    is_continuation{false};
};

struct OnlineSchedulerMetrics {
    // Per-call latency samples (ns)
    std::vector<int64_t> match_ns;
    std::vector<int64_t> insert_ns;
    std::vector<int64_t> load_commit_ns;
    std::vector<int64_t> match_to_ready_ns;

    // Lifecycle counters
    size_t forward_batches{0};
    size_t forward_requests{0};
    size_t completed_transactions{0};
    size_t dropped_waiting_at_deadline{0};
    size_t trace_exhaustions{0};
    size_t lifecycle_timeouts{0};

    // Load outcomes
    size_t loads_committed{0};
    size_t loads_succeeded{0};
    size_t loads_failed{0};
    size_t loads_cancelled{0};
    size_t cancel_request_failed{0};
    size_t load_target_allocation_failed{0};
    size_t suffix_allocation_failed{0};
    size_t load_commit_failed{0};
    size_t admission_allocation_retries{0};
    size_t joined_holder_failed{0};
    size_t joined_holder_blocks_total{0};

    // Reuse accounting
    size_t               planned_reuse_blocks_total{0};
    size_t               actual_matched_depth_total{0};
    size_t               device_matched_blocks_total{0};
    size_t               host_matched_blocks_total{0};
    size_t               insert_path_keys_total{0};
    size_t               insert_new_nodes_total{0};
    size_t               unexpected_extra_match_count{0};
    std::vector<int64_t> planned_reuse_blocks_samples;
    std::vector<int64_t> actual_matched_depth_samples;
    std::vector<int64_t> reuse_delta_blocks_samples;  // actual - planned

    // Lifecycle peaks
    size_t              active_requests_peak{0};
    size_t              waiting_requests_peak{0};
    size_t              loading_requests_peak{0};
    size_t              load_tickets_pending_peak{0};
    size_t              held_request_blocks_peak{0};
    size_t              ready_batch_max{0};
    std::vector<size_t> ready_batch_sizes;

    int64_t scheduler_no_ready_wait_ns{0};

    // Dependency tracking
    size_t dependency_skip_count{0};          // DEPENDENCY_NOT_READY skips
    size_t dependency_waiting_peak{0};        // peak number of pending dependency waits
    size_t dependency_failed_descendants{0};  // descendants blocked by parent failure

    // Completed request-shape coverage for the measured phase.
    size_t                              completed_base_transactions{0};
    size_t                              completed_continuation_transactions{0};
    size_t                              max_completed_generation{0};
    std::map<size_t, size_t>            completed_requests_by_family;
    std::vector<int64_t>                completed_generation_samples;
    std::set<size_t>                    completed_continuation_families;
    std::set<std::pair<size_t, size_t>> completed_family_epochs;
};

// Single-threaded online request state machine:
// WAITING -> LOADING_CACHE|READY -> FINISHED, with bounded
// cleanup for allocation failure, load failure/cancel, lifecycle timeout and
// the measured deadline. Exactly one thread (the caller) drives every cache
// foreground API call; the cache task pool performs load/evict/store in the
// background. Ready requests are forwarded in batches with one fixed sleep.
//
// Continuation dependencies: admission skips (DEPENDENCY_NOT_READY) requests
// whose parent has not yet been published, without blocking eligible families.
// Token-budget exhaustion stops the admission pass (TOKEN_BUDGET_BLOCKED).
// Request outcomes persist across warmup/measured phase boundaries.
class OnlineTreeScheduler {
public:
    OnlineTreeScheduler(OnlineCacheApi& cache, const OnlineTreeWorkloadConfig& config);

    // Runs one phase (warmup or measured) for `duration`. Admission and
    // backlog refill stop at the deadline; already-admitted requests drain to
    // completion. Consumes trace entries starting at `next_trace_index`.
    // `measured_ns` spans from entry to the completion of the last admitted
    // request. Returns false if any request exceeded the lifecycle timeout or
    // the trace was exhausted before the deadline.
    bool runPhase(const std::vector<OnlineRequestDescriptor>& trace,
                  size_t&                                     next_trace_index,
                  std::chrono::milliseconds                   duration,
                  int64_t&                                    measured_ns);

    // Active (non-FINISHED) contexts after runPhase; must be zero after a
    // completed phase.
    size_t activeContexts() const {
        return contexts_.size();
    }
    size_t pendingLoadTickets() const;

    const OnlineSchedulerMetrics& metrics() const {
        return metrics_;
    }
    OnlineSchedulerMetrics takeMetrics() {
        OnlineSchedulerMetrics snapshot = metrics_;
        metrics_                        = OnlineSchedulerMetrics{};
        return snapshot;
    }

private:
    enum class RequestOutcome : uint8_t { PENDING, PUBLISHED, FAILED, ABANDONED };

    // Admission return: OK admitted or terminally handled, BLOCKED by the
    // token budget, SKIP while a dependency is unresolved.
    enum class AdmitResult { OK, BLOCKED, SKIP };

    OnlineRequestContext makeContext(const OnlineRequestDescriptor& descriptor);
    // Admission transaction: match -> allocate load targets -> hold joined blocks
    // -> allocate suffix -> commit load. Returns BLOCKED when the head request
    // cannot fit the token budget or SKIP when a continuation's parent has not
    // been published. Terminal and handled failures return OK so admission can
    // continue with other requests.
    AdmitResult admit(OnlineRequestContext& ctx);
    // Idempotent cleanup: release prepared blocks and matched refs, decrement
    // the token budget and mark the context FINISHED.
    void   cleanupRequest(OnlineRequestContext& ctx);
    void   recordOutcome(const OnlineRequestContext& ctx, RequestOutcome outcome);
    RequestOutcome predecessorOutcome(const OnlineRequestContext& ctx) const;
    void   removeFinished();

    OnlineCacheApi&                  cache_;
    OnlineTreeWorkloadConfig         config_;
    std::deque<OnlineRequestContext> contexts_;
    OnlineSchedulerMetrics           metrics_;
    size_t                           active_tokens_{0};
    // Request outcomes persist across phases; not reset by takeMetrics().
    std::vector<RequestOutcome>     request_outcomes_;
};

}  // namespace rtp_llm::benchmark
