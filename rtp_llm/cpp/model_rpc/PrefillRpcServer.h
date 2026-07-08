#pragma once

#include <cstdint>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"
#include "rtp_llm/cpp/model_rpc/RecentCacheKeyWindow.h"

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
    void         reportPrefillRecentCacheKeyMetricsOnce(PrefillGenerateContext& prefill_context);
    static bool  shouldUsePdSeparation(const GenerateConfigPB& generate_config);

private:
    std::string                           decode_cluster_name_;
    std::unique_ptr<RecentCacheKeyWindow> prefill_recent_cache_key_window_;
};

}  // namespace rtp_llm
