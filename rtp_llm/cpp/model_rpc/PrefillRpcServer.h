#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"
#include "rtp_llm/cpp/model_rpc/RecentCacheKeyWindow.h"
#include "rtp_llm/cpp/model_rpc/ResponseBuffer.h"

namespace rtp_llm {

class PrefillRpcServer: public RemoteRpcServer {
public:
    PrefillRpcServer() {}
    ~PrefillRpcServer() {}
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response);

    // V1 FlexLB-controlled DP fan-out: sync prefix (alloc/enqueue/load-start) runs inline,
    // finish stream (poll/load-end/remote-gen/poll-remote) runs in a detached worker.
    grpc::Status syncPrefix(PrefillGenerateContext& prefill_context);
    grpc::Status finishStream(PrefillGenerateContext& prefill_context);

    // V1 external handlers — route into ResponseBuffer-based async queue.
    grpc::Status FetchResponse(grpc::ServerContext*                   context,
                               const FetchRequestPB*                  request,
                               grpc::ServerWriter<GenerateOutputsPB>* writer);
    grpc::Status Cancel(grpc::ServerContext* context, const CancelRequestPB* request, EmptyPB* response);

    // V1 single-slot admission entry: kicks off syncPrefix inline and detaches finishStream.
    // Invoked either by DP0's BatchEnqueue fan-out (peer stub call) or by Master bypass.
    grpc::Status Enqueue(grpc::ServerContext* context, const EnqueueRequestPB* request, EnqueueResponsePB* response);

    // V1 FlexLB → DP0 batch entry. DP0 fans each slot out via Enqueue (self inline,
    // peers through rpc_pool). Per-slot errors are isolated in response->acks().
    grpc::Status
    BatchEnqueue(grpc::ServerContext* context, const BatchEnqueueRequestPB* request, BatchEnqueueResponsePB* response);

    ResponseBufferRegistry& responseRegistry() {
        return response_registry_;
    }

private:
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
    void         reportPrefillRecentCacheKeyMetrics(PrefillGenerateContext& prefill_context);

private:
    std::string                           decode_cluster_name_;
    std::unique_ptr<RecentCacheKeyWindow> prefill_recent_cache_key_window_;
    ResponseBufferRegistry                response_registry_;
};

}  // namespace rtp_llm
