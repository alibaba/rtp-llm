#pragma once

#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/utils/ErrorCode.h"
#include "maga_transformer/cpp/model_rpc/RPCPool.h"
#include "maga_transformer/cpp/model_rpc/GenerateContext.h"
#include "maga_transformer/cpp/proto/model_rpc_service.grpc.pb.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"
#include "maga_transformer/cpp/model_rpc/RemoteServerResource.h"

namespace rtp_llm {

struct PrefillStatInfo {
    enum ExecuteStage {
        start                   = 0,
        getRpcConnection        = 1,
        multimodalProcess       = 2,
        remoteAllocateResource  = 3,
        enqueueRequest          = 4,
        remoteLoadCacheStart    = 5,
        pollLocalOutput         = 6,
        remoteLoadCacheEnd      = 7,
        RemoteGenerate          = 8,
        pollRemoteOutput        = 9,
        finish                  = 10
    };

    int64_t begin_time                      = 0;
    int64_t get_rpc_connection_rt_us        = 0;
    int64_t multimodal_process_rt_us        = 0;
    int64_t remote_allocate_resource_rt_us  = 0;
    int64_t enqueue_request_rt_us           = 0;
    int64_t remote_load_cache_start_rt_us   = 0;
    int64_t poll_local_output_rt_us         = 0;
    int64_t remote_load_cache_end_rt_us     = 0;
    int64_t remote_generate_rt_us           = 0;
    int64_t poll_remote_output_rt_us        = 0;
    ExecuteStage stage                      = start;

    ExecuteStage saveStage() const;
    void restoreStage(ExecuteStage stage);
    void nextStage();
};

struct RPCContext {
    int64_t requestID() {
        return request->request_id();
    }

    const GenerateInputPB*                 request;
    grpc::ServerWriter<GenerateOutputsPB>* writer;
};

class PrefillGenerateContext: public GenerateContext {
public:
    PrefillGenerateContext(RemoteServerResource* resource, RPCContext& rpc_context,
                           int64_t timeout_ms, grpc::ServerContext* server_context,
                           kmonitor::MetricsReporterPtr& metrics_reporter)
                           : GenerateContext(rpc_context.requestID(), timeout_ms, server_context, metrics_reporter),
                             resource(resource), rpc_context(rpc_context) {}
    ~PrefillGenerateContext();
    void reset() override;
    void nextStage();
    grpc::Status closeGrpcStream();
    void closeGrpcConnection();

private:
    void markRequestEnd();
    void printTime();
    void reportTime();
    void stopStream();

public:
    typedef grpc::ClientReaderWriter<GenerateRequestPB, GenerateOutputsPB> ClientStream;

    RemoteServerResource*                   resource;
    RPCContext                              rpc_context;
    std::shared_ptr<GenerateInput>          generate_input;

    std::string                             decode_addr;
    GrpcConnection                          grpc_connection;
    std::shared_ptr<RpcService::Stub>       stub;
    std::shared_ptr<grpc::ClientContext>    client_context;
    std::shared_ptr<ClientStream>           client_stream;
    bool                                    grpc_stream_closed = false;
    PrefillStatInfo                         stat_info;
    int64_t                                 loading_cache_requests  = 0;

    // for debug, will delete in future
    GenerateOutputsPB                       response;
    int64_t                                 remote_cost_time_us;
};

}  // namespace rtp_llm
