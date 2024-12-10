#pragma once

#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/RRLoadBalancer.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/WRRLoadBalancer.h"
#include "maga_transformer/cpp/model_rpc/RemoteRpcServer.h"
#include "maga_transformer/cpp/model_rpc/PrefillGenerateContext.h"

using namespace fastertransformer;

namespace rtp_llm {

class PrefillRpcServer: public RemoteRpcServer {
public:
    PrefillRpcServer() {}
    ~PrefillRpcServer() {}
    grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;
    
    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status RemoteFinish(grpc::ServerContext* context,
                               const RemoteFinishRequestPB* request,
                               EmptyPB* response);
    bool ready();

private:
    void initLoadBalancer();
    LoadBalancerInitParams makeConfig();
    ErrorInfo waitStreamBeforeRun(std::shared_ptr<GenerateStream> stream);
    grpc::Status prepareAllocateResource(PrefillGenerateContext& prefill_context);
    void getRpcConnection(PrefillGenerateContext& prefill_context);
    void remoteAllocateResource(PrefillGenerateContext& prefill_context);
    void enqueueRequest(PrefillGenerateContext& prefill_context);
    void remoteLoadCacheStart(PrefillGenerateContext& prefill_context);
    void pollLocalOutput(PrefillGenerateContext& prefill_context);
    void remoteLoadCacheEnd(PrefillGenerateContext& prefill_context);
    void remoteGenerate(PrefillGenerateContext& prefill_context);
    void pollRemoteOutput(PrefillGenerateContext& prefill_context);

private:
    std::shared_ptr<BaseLoadBalancer> load_balancer_;
    std::string docode_cluster_name_;
};

}