#pragma once

#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"

namespace rtp_llm {

struct DecodeRpcContextNew {
    DecodeRpcContextNew() {
        client_context   = std::make_shared<grpc::ClientContext>();
        completion_queue = std::make_shared<grpc::CompletionQueue>();
    }

    ~DecodeRpcContextNew() {
        completion_queue->Shutdown();
    }

    bool                                                                          finished = false;
    RemoteGenerateRequestPBNew                                                    request;
    RemoteGenerateResponsePBNew                                                   response;
    grpc::Status                                                                  status;
    std::shared_ptr<RpcService::Stub>                                             stub;
    std::shared_ptr<grpc::ClientContext>                                          client_context;
    std::shared_ptr<grpc::CompletionQueue>                                        completion_queue;
    std::unique_ptr<grpc::ClientAsyncResponseReader<RemoteGenerateResponsePBNew>> reader;
};

struct DecodeGenerateContextNew: public GenerateContext {

    DecodeGenerateContextNew(grpc::ServerContext*                   server_context,
                             const GenerateInputPB*                 request,
                             grpc::ServerWriter<GenerateOutputsPB>* response_writer,
                             kmonitor::MetricsReporterPtr&          metrics_reporter,
                             std::shared_ptr<RpcServerRuntimeMeta>  meta):
        GenerateContext(
            request->request_id(), request->generate_config().timeout_ms(), server_context, metrics_reporter, meta),
        request(request),
        response_writer(response_writer) {
        request_begin_time_us = currentTimeUs();
    }

    ~DecodeGenerateContextNew();

    ErrorInfo init(const std::shared_ptr<EngineBase>& engine);

private:
    void reportTime();

public:
    const GenerateInputPB*                 request;
    grpc::ServerWriter<GenerateOutputsPB>* response_writer;
    std::shared_ptr<GenerateInput>         generate_input;

    RemoteGenerateRequestPBNew  remote_generate_request;
    RemoteGenerateResponsePBNew remote_generate_response;

    int64_t request_begin_time_us                 = 0;
    int64_t prepare_generate_context_done_time_us = 0;
    int64_t load_cache_from_prefill_done_time_us  = 0;
    int64_t local_generate_done_time_us           = 0;
};

}  // namespace rtp_llm