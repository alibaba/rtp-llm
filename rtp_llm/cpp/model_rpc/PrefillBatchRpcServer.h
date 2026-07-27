#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <grpcpp/alarm.h>
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"

namespace rtp_llm {

struct DeferredPrefillContext {
    AtomicGuardPtr                   request_guard;
    std::shared_ptr<GenerateInputPB> input;
    // Members are destroyed in reverse order: context must go before input,
    // because RPCContext keeps a raw pointer into input.
    std::unique_ptr<PrefillGenerateContext> context;
    std::shared_ptr<grpc::Alarm>            ttl_alarm;

    void cancel(const grpc::Status& status);
};

// Owns contexts that have been prepared but not yet claimed by FetchResponse.
class DeferredPrefillContextMap: public std::enable_shared_from_this<DeferredPrefillContextMap> {
public:
    grpc::Status store(int64_t request_id, const std::shared_ptr<DeferredPrefillContext>& context);
    grpc::Status
    armTtl(int64_t request_id, const std::shared_ptr<DeferredPrefillContext>& context, std::chrono::milliseconds ttl);
    grpc::Status                            take(int64_t request_id, std::shared_ptr<DeferredPrefillContext>& context);
    std::shared_ptr<DeferredPrefillContext> remove(int64_t request_id, const DeferredPrefillContext* expected);
    void                                    stopAccepting();
    void                                    cancelAll(const grpc::Status& status);
    size_t                                  size() const;

private:
    void expire(int64_t request_id, const DeferredPrefillContext* expected);

    mutable std::mutex                                                   mu_;
    std::unordered_map<int64_t, std::shared_ptr<DeferredPrefillContext>> contexts_;
    bool                                                                 stopping_{false};
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
    PrefillBatchRpcServer() = default;
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
    // One accepted request inside a group; carried across the EnqueueGroup phase methods.
    struct BatchSlot {
        std::shared_ptr<GenerateInputPB>        input;
        std::unique_ptr<PrefillGenerateContext> prefill_context;
        AtomicGuardPtr                          request_guard;
        int64_t                                 fetch_attach_timeout_ms{0};
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

private:
    std::shared_ptr<DeferredPrefillContextMap> deferred_contexts_ = std::make_shared<DeferredPrefillContextMap>();
    std::atomic<bool>                          stopping_{false};
    autil::ThreadPoolBasePtr                   prepare_resource_worker_pool_;
};

}  // namespace rtp_llm
