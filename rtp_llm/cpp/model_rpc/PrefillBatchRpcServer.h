#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillMetrics.h"
#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"

namespace rtp_llm {

// Batch-enqueue prefill server for PD separation.
//
// This class isolates the entire batch path (EnqueueBatch / EnqueueGroup / FetchResponse and
// the thread pool, response registry and pool metrics behind them) from the single-request prefill
// server. It inherits PrefillRpcServer and reuses its shared building blocks — prepareAllocateResource
// and finishStream — for each request in a group; the base class is never mutated by the batch path,
// so the single-request behavior stays exactly as it was.
//
// EnqueueGroup is written to read like the single-request GenerateStreamCall: a linear top level that
// delegates to named phase methods (admitGroup -> acceptGroup -> buildSlotContexts -> prepareGroup
// -> enqueueGroupStreams -> launchSlotRunner -> runSlotStream).
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

private:
    // One accepted request inside a group; carried across the EnqueueGroup phase methods.
    struct BatchSlot {
        std::shared_ptr<GenerateInputPB>        input;
        std::unique_ptr<PrefillGenerateContext> prefill_context;
        AtomicGuardPtr                          request_guard;
    };

    struct PrepareResult {
        bool         prepared     = false;
        grpc::Status stage_status = grpc::Status::OK;
    };

    struct ReadySlot {
        BatchSlot*                           slot = nullptr;
        std::shared_ptr<ResponseBufferEntry> entry;
    };

    struct SlotRunnerState {
        std::unique_ptr<PrefillGenerateContext> prefill_context;
        std::shared_ptr<GenerateInputPB>        input;
        std::shared_ptr<ResponseBufferWriter>   writer;
        std::shared_ptr<ResponseBufferEntry>    entry;
        AtomicGuardPtr                          request_guard;
    };

    // ---- EnqueueGroup phases (mirror GenerateStreamCall's linear structure) ----
    // Validate and copy inputs. Fills `slots`; returns the status to
    // propagate (OK with empty slots means "nothing to run", the caller returns immediately).
    grpc::Status
    admitGroup(const EnqueueGroupRequestPB* request, EnqueueBatchResponsePB* response, std::vector<BatchSlot>& slots);
    // Prepare and enqueue the group synchronously; ACK only streams admitted by the scheduler.
    grpc::Status acceptGroup(std::vector<BatchSlot> slots, EnqueueBatchResponsePB* response);
    void         buildSlotContexts(std::vector<BatchSlot>& slots);
    // prepareAllocateResource-with-retry per slot on slot_worker_pool_.
    std::vector<PrepareResult> prepareGroup(std::vector<BatchSlot>& slots);
    // engine_->enqueueMultiple for the prepared slots.
    grpc::Status enqueueGroupStreams(std::vector<ReadySlot>& ready_slots);
    // Submit the per-request finishStream runner after scheduler admission.
    grpc::Status launchSlotRunner(ReadySlot& ready_slot);
    // The finishStream driver for one accepted request (mirrors finishStream in the single-request path).
    void runSlotStream(SlotRunnerState state, int64_t request_id);
    void publishSlot(ReadySlot& ready_slot, EnqueueBatchResponsePB* response);
    void rejectSlot(ReadySlot& ready_slot, const grpc::Status& status, EnqueueBatchResponsePB* response);

    // ---- Batch infrastructure ----
    void startResponseRegistryGc();
    void stopResponseRegistryGc();
    bool tryStartAsyncResponseWorker();
    void finishAsyncResponseWorker();
    void stopAsyncResponseWorkers();
    void initThreadPools();
    void reportPoolMetrics();

private:
    ResponseBufferRegistry  response_registry_;
    std::atomic<bool>       response_gc_stop_{false};
    std::mutex              response_gc_mu_;
    std::condition_variable response_gc_cv_;
    std::thread             response_gc_thread_;
    bool                    response_worker_stop_{false};
    std::mutex              response_worker_mu_;
    std::condition_variable response_worker_cv_;
    size_t                  response_worker_count_{0};

    // Thread pool replacing std::async / std::thread::detach.
    autil::ThreadPoolBasePtr slot_worker_pool_;  // Prepare + async response runners.
    PoolMetrics              slot_pool_metrics_;
};

}  // namespace rtp_llm
