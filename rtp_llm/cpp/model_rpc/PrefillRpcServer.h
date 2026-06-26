#pragma once

#include <atomic>
#include <condition_variable>
#include "grpc++/grpc++.h"
#include <mutex>
#include <thread>
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"
#include "rtp_llm/cpp/cache/RecentCacheKeyWindow.h"
#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"

namespace rtp_llm {

// Pool-level health metrics, reported periodically
struct PoolMetrics {
    std::atomic<size_t> active    = 0;  // currently executing tasks
    std::atomic<size_t> queued    = 0;  // tasks waiting in queue
    std::atomic<size_t> completed = 0;  // total finished since creation
    std::atomic<size_t> rejected  = 0;  // pushTask refused (pool full)
    std::atomic<size_t> fallback  = 0;  // fallback to detached thread
};

class PrefillRpcServer: public RemoteRpcServer {
public:
    PrefillRpcServer() {}
    ~PrefillRpcServer() override;
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             mm_process_engine) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response);

    grpc::Status
    EnqueueBatch(grpc::ServerContext* context, const EnqueueBatchRequestPB* request, EnqueueBatchResponsePB* response);

    grpc::Status
    EnqueueGroup(grpc::ServerContext* context, const EnqueueGroupRequestPB* request, EnqueueBatchResponsePB* response);

    grpc::Status FetchResponse(grpc::ServerContext*                   context,
                               const FetchRequestPB*                  request,
                               grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status Cancel(grpc::ServerContext* context, const CancelRequestPB* request, EmptyPB* response);

private:
    grpc::Status syncPrefix(PrefillGenerateContext& prefill_context);
    grpc::Status finishStream(PrefillGenerateContext& prefill_context);
    ErrorInfo    waitStreamBeforeRun(std::shared_ptr<GenerateStream> stream);
    grpc::Status prepareAllocateResource(PrefillGenerateContext& prefill_context);
    void         getRpcConnection(PrefillGenerateContext& prefill_context);
    void         multimodalProcess(PrefillGenerateContext& prefill_context);
    void         remoteAllocateResource(PrefillGenerateContext& prefill_context);
    void         enqueueRequest(PrefillGenerateContext& prefill_context);
    void         remoteLoadCacheStart(PrefillGenerateContext& prefill_context);
    void         pollLocalOutput(PrefillGenerateContext& prefill_context);
    void         remoteLoadCacheEnd(PrefillGenerateContext& prefill_context);
    void         remoteGenerate(PrefillGenerateContext& prefill_context);
    void         pollRemoteOutput(PrefillGenerateContext& prefill_context);
    void         reportPrefillRecentCacheKeyMetricsOnce(PrefillGenerateContext& prefill_context);
    void         startResponseRegistryGc();
    void         stopResponseRegistryGc();
    bool         tryStartAsyncResponseWorker();
    void         finishAsyncResponseWorker();
    void         stopAsyncResponseWorkers();
    void         initThreadPools();
    void         reportPoolMetrics();
    std::string  batchTargetAddrForDpRank(int dp_rank) const;

private:
    std::string                           decode_cluster_name_;
    std::unique_ptr<RecentCacheKeyWindow> prefill_recent_cache_key_window_;
    ResponseBufferRegistry                response_registry_;
    std::atomic<bool>                     response_gc_stop_{false};
    std::mutex                            response_gc_mu_;
    std::condition_variable               response_gc_cv_;
    std::thread                           response_gc_thread_;
    std::atomic<bool>                     response_worker_stop_{false};
    std::mutex                            response_worker_mu_;
    std::condition_variable               response_worker_cv_;
    size_t                                response_worker_count_{0};

    // Thread pools replacing std::async / std::thread::detach
    autil::ThreadPoolBasePtr enqueue_worker_pool_;  // Dispatch only (L1 DP dispatch, fast ms-level)
    autil::ThreadPoolBasePtr worker_lambda_pool_;   // Heavy worker lambdas (L2/L3 coordination, I/O-bound ~12s)
    autil::ThreadPoolBasePtr slot_worker_pool_;     // L2 Prep + L3 Load + L4 Finish
    PoolMetrics              enqueue_pool_metrics_;
    PoolMetrics              worker_lambda_pool_metrics_;
    PoolMetrics              slot_pool_metrics_;
};

}  // namespace rtp_llm