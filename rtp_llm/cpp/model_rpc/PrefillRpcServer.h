#pragma once

#include <atomic>
#include <condition_variable>
#include "grpc++/grpc++.h"
#include <mutex>
#include <thread>
#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"
#include "rtp_llm/cpp/model_rpc/RecentCacheKeyWindow.h"
#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"
#include "rtp_llm/cpp/model_rpc/RequestSession.h"

namespace rtp_llm {

class PrefillRpcServer: public RemoteRpcServer {
public:
    PrefillRpcServer() {}
    ~PrefillRpcServer() override;
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response);

    grpc::Status
    BatchEnqueue(grpc::ServerContext* context, const BatchEnqueueRequestPB* request, BatchEnqueueResponsePB* response);

    grpc::Status FetchResponse(grpc::ServerContext*                   context,
                               const FetchRequestPB*                  request,
                               grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status AttachStream(grpc::ServerContext*                   context,
                              const AttachStreamRequestPB*           request,
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

private:
    std::string                           decode_cluster_name_;
    std::unique_ptr<RecentCacheKeyWindow> prefill_recent_cache_key_window_;
    SessionManager                        session_manager_;
    ResponseBufferRegistry                response_registry_;
    std::atomic<bool>                     response_gc_stop_{false};
    std::mutex                            response_gc_mu_;
    std::condition_variable               response_gc_cv_;
    std::thread                           response_gc_thread_;
    std::atomic<bool>                     response_worker_stop_{false};
    std::mutex                            response_worker_mu_;
    std::condition_variable               response_worker_cv_;
    size_t                                response_worker_count_{0};
};

}  // namespace rtp_llm
