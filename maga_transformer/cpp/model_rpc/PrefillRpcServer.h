#pragma once
#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/RRLoadBalancer.h"
#include "maga_transformer/cpp/model_rpc/RemoteRpcServer.h"

namespace rtp_llm {

struct RPCContext {
    RPCContext(grpc::ServerContext*                   context,
               const GenerateInputPB*                 request,
               grpc::ServerWriter<GenerateOutputsPB>* writer)
               : context(context), request(request), writer(writer) {}

    int64_t requestID() {
        return request->request_id();
    }

    grpc::ServerContext*                   context;
    const GenerateInputPB*                 request;
    grpc::ServerWriter<GenerateOutputsPB>* writer;
};

class PrefillRpcServer;

class PrefillGenerateContext: public GenerateContext {
public:
    PrefillGenerateContext(PrefillRpcServer* server, RPCContext& rpc_context)
                           : GenerateContext(rpc_context.requestID()), server(server), rpc_context(rpc_context) {}
    ~PrefillGenerateContext();
    void markRequestEnd();

public:
    PrefillRpcServer* server;
    RPCContext& rpc_context;

    std::string decode_addr;
    std::shared_ptr<RpcService::Stub> stub;
    std::shared_ptr<grpc::ClientContext> client_context;
    std::shared_ptr<grpc::ClientReaderWriter<GenerateRequestPB, GenerateOutputsPB>> client_stream;

    bool finished = false;
    ErrorCode error_code = ErrorCode::NONE;
    std::string error_msg;
    grpc::Status error_status = grpc::Status::OK;

    // tmp member
    GenerateOutputsPB* response = nullptr;
    int64_t remote_cost_time_us;
};

class PrefillRpcServer: public RemoteRpcServer {
public:
    PrefillRpcServer() {}
    ~PrefillRpcServer() {}
    grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;
    
    grpc::Status generate_stream(grpc::ServerContext*                   context,
                                 const GenerateInputPB*                 request,
                                 grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status remote_finish(grpc::ServerContext* context,
                               const RemoteFinishRequestPB* request,
                               EmptyPB* response);
    bool ready();

private:
    void initLoadBalancer();
    LoadBalancerInitParams makeConfig();
    absl::Status waitStreamBeforeRun(std::shared_ptr<GenerateStream> stream);
    grpc::Status prepareAllocateResource(PrefillGenerateContext& prefill_context);
    void getRpcConnection(PrefillGenerateContext& prefill_context);
    void remoteAllocateResource(PrefillGenerateContext& prefill_context);
    void enqueueRequest(PrefillGenerateContext& prefill_context);
    void remoteLoadCache(PrefillGenerateContext& prefill_context);
    void pollLocalOutput(PrefillGenerateContext& prefill_context);
    void remoteGenerate(PrefillGenerateContext& prefill_context);
    void pollRemoteOutput(PrefillGenerateContext& prefill_context);
    void reportTime(PrefillGenerateContext& prefill_context);

private:
    std::shared_ptr<RRLoadBalancer> load_balancer_;
    std::string docode_cluster_name_;
};

}