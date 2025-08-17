#pragma once

#include "rtp_llm/cpp/model_rpc/RemoteServerResource.h"
#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"


namespace rtp_llm {

struct PrefillRpcContext {
    PrefillRpcContext() {
        client_context   = std::make_shared<grpc::ClientContext>();
        completion_queue = std::make_shared<grpc::CompletionQueue>();
    }

    ~PrefillRpcContext() {
        completion_queue->Shutdown();
    }

    bool                                                                    finished = false;
    RemoteStoreRequestPB                                                    request;
    RemoteStoreResponsePB                                                   response;
    grpc::Status                                                            status;
    std::shared_ptr<RpcService::Stub>                                       stub;
    std::shared_ptr<grpc::ClientContext>                                    client_context;
    std::shared_ptr<grpc::CompletionQueue>                                  completion_queue;
    std::unique_ptr<grpc::ClientAsyncResponseReader<RemoteStoreResponsePB>> reader;
};

struct PrefillGenerateContextNew: public GenerateContext {
    PrefillGenerateContextNew(RemoteServerResource*             resource,
                              grpc::ServerContext*                server_context,
                              const RemoteGenerateRequestPBNew* request,
                              RemoteGenerateResponsePBNew*      response,
                              kmonitor::MetricsReporterPtr&     metrics_reporter,
                              std::shared_ptr<RpcServerRuntimeMeta> meta):
        GenerateContext(request->input().request_id(),
                        request->input().generate_config().timeout_ms(),
                        server_context,
                        metrics_reporter,
                        meta),
        resource(resource),
        request(request),
        response(response) {
        request_begin_time_us = currentTimeUs();
        for (int i = 0; i < request->addrs_size(); ++i) {
            decode_workers.push_back(request->addrs(i));
        }
    }

    ~PrefillGenerateContextNew() { 
        stopStream(); 
        notifyRequestEndForAllRank();
        reportTime();
    }

private:
    void reportTime();
    void notifyRequestEndForAllRank();
    void notifyRequestEnd(int index);

public:
    ErrorInfo init(const std::shared_ptr<EngineBase>& engine);
    void stopStream();

public:
    RemoteServerResource*             resource;
    const RemoteGenerateRequestPBNew* request;
    RemoteGenerateResponsePBNew*      response;
    std::shared_ptr<GenerateInput>    generate_input;

    std::vector<std::string> decode_workers;

    std::vector<std::shared_ptr<PrefillRpcContext>> rpc_contexts;

    int64_t request_begin_time_us             = 0;
    int64_t notify_store_cache_done_time_us   = 0;
    int64_t generate_first_token_done_time_us = 0;
    int64_t wait_store_cache_done_time_us     = 0;
    int64_t min_response_done_time_us         = 1lu << 60;
    int64_t max_response_done_time_us         = 0;
};

}  // namespace rtp_llm
