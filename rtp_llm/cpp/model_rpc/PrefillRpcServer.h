#pragma once

#include <memory>
#include <string>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"
#include "rtp_llm/cpp/cache/RecentCacheKeyWindow.h"

namespace rtp_llm {

// Prefill-side gRPC server for PD (prefill/decode) separation — the single-request path.
//
//   GenerateStreamCall
//     → syncPrefix    (prepareAllocateResource with retry + enqueueRequest)
//     → finishStream  (remoteLoadCacheStart → pollLocalOutput → remoteLoadCacheEnd
//                       → remoteGenerate → pollRemoteOutput)
//
// The batch-enqueue path (EnqueueBatch / EnqueueGroup / FetchResponse and the thread pools,
// response registry and pool metrics behind it) lives entirely in the derived PrefillBatchRpcServer,
// which reuses finishStream / prepareAllocateResource from this class. This base is never mutated by
// the batch path, keeping the single-request behavior isolated.
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

protected:
    // Shared with the derived batch server (each batch slot reuses these).
    grpc::Status prepareAllocateResource(PrefillGenerateContext& prefill_context);
    grpc::Status finishStream(PrefillGenerateContext& prefill_context);

private:
    grpc::Status syncPrefix(PrefillGenerateContext& prefill_context);
    ErrorInfo    waitStreamBeforeRun(std::shared_ptr<GenerateStream> stream);
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

private:
    std::string                           decode_cluster_name_;
    std::unique_ptr<RecentCacheKeyWindow> prefill_recent_cache_key_window_;
};

}  // namespace rtp_llm
