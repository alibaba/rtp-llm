#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <grpcpp/alarm.h>
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"

namespace rtp_llm {

class PriorityCancelExecutor;

struct DeferredPrefillContext {
    struct StartOperationResult {
        bool started{false};
        bool priority_finalizer_claimed{false};
    };

    AtomicGuardPtr                   request_guard;
    std::shared_ptr<GenerateInputPB> input;
    // Members are destroyed in reverse order: context must go before input,
    // because RPCContext keeps a raw pointer into input.
    std::unique_ptr<PrefillGenerateContext> context;
    std::shared_ptr<grpc::Alarm>            ttl_alarm;
    // Retries of finalizePriorityPreemption's "wait for scheduler FINISHED"
    // turn. Touched only by the single finalizer claimant that owns this
    // deferred context (priority_finalize_claimed_), so a plain counter is
    // sufficient; it lives here (not in a global) so the budget is per request.
    int64_t priority_finalize_scheduler_wait_iters = 0;

    void cancel(const grpc::Status& status);
    // Return true exactly once when the caller becomes the asynchronous
    // finalization owner.
    bool                 finishOperation();
    StartOperationResult tryStartOperation();
    bool                 requestPriorityFinalization();

private:
    std::mutex operation_mu_;
    bool       operation_active_{true};
    bool       priority_finalize_requested_{false};
    bool       priority_finalize_claimed_{false};
};

// Tracks cancel-visible active contexts and Fetch-visible prepared contexts.
class DeferredPrefillContextMap: public std::enable_shared_from_this<DeferredPrefillContextMap> {
public:
    grpc::Status registerActive(int64_t request_id, const std::shared_ptr<DeferredPrefillContext>& context);
    grpc::Status store(int64_t request_id, const std::shared_ptr<DeferredPrefillContext>& context);
    grpc::Status
    armTtl(int64_t request_id, const std::shared_ptr<DeferredPrefillContext>& context, std::chrono::milliseconds ttl);
    grpc::Status                            take(int64_t request_id, std::shared_ptr<DeferredPrefillContext>& context);
    std::shared_ptr<DeferredPrefillContext> remove(int64_t request_id, const DeferredPrefillContext* expected);
    PriorityCancelResult                    cancelByPriorityPreemption(int64_t                                  request_id,
                                                                       std::shared_ptr<DeferredPrefillContext>& context,
                                                                       bool*                                    newly_installed = nullptr);
    PriorityCancelResult                    cancelByPriorityPreemption(int64_t request_id) {
        std::shared_ptr<DeferredPrefillContext> ignored;
        return cancelByPriorityPreemption(request_id, ignored);
    }
    void publishPriorityPreemptionCanceled(int64_t request_id, const DeferredPrefillContext* expected);
    // W-2 ledger-sweep linkage: unconditional (any-owner) mirror of
    // publishPriorityPreemptionCanceled's active-entry removal, invoked by the
    // runtime-meta stale-overlay sweep hook after it published the aged typed
    // CANCELED record. Keeps the tombstone downgrade so duplicate Cancels stay
    // idempotent. Must be called WITHOUT the runtime-meta lock held.
    void   dropActiveEntryAfterLedgerSweep(int64_t request_id);
    void   finish(int64_t request_id, const DeferredPrefillContext* expected);
    void   stopAccepting();
    void   cancelAll(const grpc::Status& status);
    size_t size() const;

private:
    enum class PriorityPreemptionTombstoneKind : uint8_t {
        ABSENT_FENCE,
        ACTIVE_CANCEL,
    };

    struct PriorityPreemptionTombstone {
        int64_t                         expires_at_ms;
        PriorityPreemptionTombstoneKind kind;
    };

    void expire(int64_t request_id, const DeferredPrefillContext* expected);
    void sweepPriorityPreemptionTombstones(int64_t now_ms);
    void rememberRecentlySeenRequest(int64_t request_id, int64_t now_ms);
    void sweepRecentlySeenRequests(int64_t now_ms);
    // mu_ must be held. A single helper keeps missing-active and
    // active-cancel tombstones on the same lifetime/expiry path.
    void installPriorityPreemptionTombstone(int64_t request_id, int64_t now_ms, PriorityPreemptionTombstoneKind kind);

    mutable std::mutex                                                   mu_;
    std::unordered_map<int64_t, std::shared_ptr<DeferredPrefillContext>> contexts_;
    std::unordered_map<int64_t, std::weak_ptr<DeferredPrefillContext>>   active_contexts_;
    // Request-id reuse is explicitly out of scope. A latched tombstone keeps
    // duplicate Cancel and a future FetchResponse idempotent after the active
    // context has moved to asynchronous cleanup.
    std::unordered_map<int64_t, PriorityPreemptionTombstone> priority_preemption_tombstones_;
    std::deque<std::pair<int64_t, int64_t>>                  priority_preemption_tombstone_expiries_;
    // Distinguishes a truly never-registered request from one whose active
    // context has already completed. Without this bounded history, a late
    // Cancel could install an ABSENT_FENCE for completed work and falsely
    // report TOMBSTONED to the master.
    std::unordered_map<int64_t, int64_t>    recently_seen_requests_;
    std::deque<std::pair<int64_t, int64_t>> recently_seen_request_expiries_;
    bool                                    stopping_{false};
};

// Batch-enqueue prefill server for PD separation.
//
// EnqueueGroup prepares and admits each request, then stores its context.
// FetchResponse atomically takes that context and continues the existing
// synchronous PrefillRpcServer::finishStream path.
//
// EnqueueGroup is written to read like the single-request GenerateStreamCall: a linear top level that
// delegates to named phase methods (admitGroup -> acceptGroup -> buildSlotContexts
// -> prepareGroup -> enqueueGroupStreams -> publishSlot).
class PrefillBatchRpcServer: public PrefillRpcServer {
public:
    PrefillBatchRpcServer();
    ~PrefillBatchRpcServer() override;

    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;

    grpc::Status
    EnqueueBatch(grpc::ServerContext* context, const EnqueueBatchRequestPB* request, EnqueueBatchResponsePB* response);

    virtual grpc::Status
    EnqueueGroup(grpc::ServerContext* context, const EnqueueGroupRequestPB* request, EnqueueBatchResponsePB* response);

    grpc::Status FetchResponse(grpc::ServerContext*                   context,
                               const FetchRequestPB*                  request,
                               grpc::ServerWriter<GenerateOutputsPB>* writer);

    void beginShutdown();

private:
    PriorityCancelResult onCancelRequest(int64_t request_id) override;
    void finalizePriorityPreemption(int64_t request_id, std::shared_ptr<DeferredPrefillContext> deferred);
    void finishSlotOperation(int64_t request_id, const std::shared_ptr<DeferredPrefillContext>& deferred);

    // One accepted request inside a group; carried across the EnqueueGroup phase methods.
    struct BatchSlot {
        std::shared_ptr<GenerateInputPB>        input;
        std::shared_ptr<DeferredPrefillContext> deferred;
        grpc::Status                            registration_status = grpc::Status::OK;
        int64_t                                 fetch_attach_timeout_ms{0};
        // True when admitGroup installed RTP_LLM_BATCH_STREAM_DEFAULT_TIMEOUT_MS
        // on this slot's input copy. R1: an injected default must bound only
        // the prefill-side chain (stream checkTimeout / nextOutput wait); it
        // must NOT flow into the downstream GenerateInputPB nor into the
        // context's request_timeout_ms (which would tighten the P->D streaming
        // RPC deadline below max_rpc_timeout_ms and truncate decode).
        bool default_timeout_injected{false};
    };

    struct PrepareResult {
        bool         prepared     = false;
        grpc::Status stage_status = grpc::Status::OK;
    };

    struct ReadySlot {
        BatchSlot*                              slot = nullptr;
        std::shared_ptr<DeferredPrefillContext> deferred;
    };

    // ---- EnqueueGroup phases (mirror GenerateStreamCall's linear structure) ----
    // Validate and copy inputs. Fills `slots`; returns the status to
    // propagate (OK with empty slots means "nothing to run", the caller returns immediately).
    grpc::Status
    admitGroup(const EnqueueGroupRequestPB* request, EnqueueBatchResponsePB* response, std::vector<BatchSlot>& slots);
    // Prepare and enqueue the group synchronously; ACK only streams admitted by the scheduler.
    grpc::Status acceptGroup(std::vector<BatchSlot> slots, EnqueueBatchResponsePB* response);
    void         buildSlotContexts(std::vector<BatchSlot>& slots);
    // prepareAllocateResource-with-retry per slot on prepare_resource_worker_pool_.
    std::vector<PrepareResult> prepareGroup(std::vector<BatchSlot>& slots);
    // engine_->enqueueMultiple for the stored slots. Scheduler rejections are removed from the context map and written
    // to response; admitted slots remain for publication.
    grpc::Status enqueueGroupStreams(std::vector<ReadySlot>& ready_slots, EnqueueBatchResponsePB* response);
    std::shared_ptr<DeferredPrefillContext> storeSlot(BatchSlot& slot, EnqueueBatchResponsePB* response);
    void                                    publishSlot(ReadySlot& ready_slot, EnqueueBatchResponsePB* response);
    void rejectSlot(ReadySlot& ready_slot, const grpc::Status& status, EnqueueBatchResponsePB* response);

    // ---- Batch infrastructure ----
    void initThreadPools();
    void schedulePriorityFinalization(int64_t request_id, std::shared_ptr<DeferredPrefillContext> deferred);

private:
    std::shared_ptr<DeferredPrefillContextMap> deferred_contexts_ = std::make_shared<DeferredPrefillContextMap>();
    std::atomic<bool>                          stopping_{false};
    autil::ThreadPoolBasePtr                   prepare_resource_worker_pool_;
    std::unique_ptr<PriorityCancelExecutor>    priority_cancel_executor_;
};

}  // namespace rtp_llm
