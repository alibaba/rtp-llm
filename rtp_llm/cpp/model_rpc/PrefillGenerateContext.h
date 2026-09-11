#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/model_rpc/RemoteServerResource.h"

namespace rtp_llm {

enum class PrefillTerminalCause : uint8_t {
    ACTIVE              = 0,
    PRIORITY_PREEMPTION = 1,
    OTHER               = 2,
};

enum class PriorityPreemptionRequestResult : uint8_t {
    INSTALLED         = 0,
    ALREADY_INSTALLED = 1,
    REJECTED          = 2,
};

struct PrefillStatInfo {
    enum ExecuteStage {
        start                  = 0,
        getRpcConnection       = 1,
        multimodalProcess      = 2,
        remoteAllocateResource = 3,
        enqueueRequest         = 4,
        remoteLoadCacheStart   = 5,
        pollLocalOutput        = 6,
        remoteLoadCacheEnd     = 7,
        RemoteGenerate         = 8,
        pollRemoteOutput       = 9,
        finish                 = 10
    };

    int64_t      begin_time                            = 0;
    int64_t      get_rpc_connection_rt_us              = 0;
    int64_t      multimodal_process_rt_us              = 0;
    int64_t      remote_allocate_resource_rt_us        = 0;
    int64_t      enqueue_request_rt_us                 = 0;
    int64_t      remote_load_cache_start_rt_us         = 0;
    int64_t      remote_load_cache_wait_stream_rt_us   = 0;
    int64_t      remote_load_cache_write_request_rt_us = 0;
    int64_t      poll_local_output_rt_us               = 0;
    int64_t      remote_load_cache_end_rt_us           = 0;
    int64_t      remote_generate_rt_us                 = 0;
    int64_t      poll_remote_output_rt_us              = 0;
    ExecuteStage stage                                 = start;

    ExecuteStage saveStage() const;
    void         restoreStage(ExecuteStage stage);
    void         nextStage();
    void         finishStage();
};

struct RPCContext {
    int64_t requestID() {
        return request->request_id();
    }

    const GenerateInputPB*                              request;
    grpc::internal::WriterInterface<GenerateOutputsPB>* writer;
};

class PrefillGenerateContext: public GenerateContext {
public:
    PrefillGenerateContext(RemoteServerResource*                 resource,
                           RPCContext&                           rpc_context,
                           int64_t                               timeout_ms,
                           grpc::ServerContext*                  server_context,
                           kmonitor::MetricsReporterPtr&         metrics_reporter,
                           std::shared_ptr<RpcServerRuntimeMeta> meta,
                           int64_t                               prefill_stop_stream_wait_timeout_ms = 2000):
        GenerateContext(rpc_context.requestID(), timeout_ms, server_context, metrics_reporter, meta),
        task_identity_{rpc_context.requestID(),
                       rpc_context.request && rpc_context.request->has_group_id() ?
                           rpc_context.request->group_id().value() :
                           -1},
        resource(resource),
        rpc_context(rpc_context),
        cancel_state(std::make_shared<std::atomic<bool>>(false)),
        prefill_stop_stream_wait_timeout_ms_(prefill_stop_stream_wait_timeout_ms) {
        prefill_worker_cache_store_addrs = resource->workers;
    }
    ~PrefillGenerateContext();
    void         setStream(const std::shared_ptr<GenerateStream>& stream) override;
    void         reset() override;
    bool         isRequestCancelled() const override;
    PriorityPreemptionRequestResult requestPriorityPreempt();
    bool         isPriorityPreempted() const;
    bool         tryMarkOtherTerminal();
    PrefillTerminalCause terminalCause() const;
    void         tryCancelDownstream();
    bool         finalizePriorityPreemption();
    void         setLocalStreamSchedulerOwned(bool owned);
    // Linearizes ordinary runtime-meta removal with installation of the
    // priority-preemption first cause and its CANCELING overlay.
    void         dequeueStreamFromRuntimeMeta();
    void         nextStage();
    grpc::Status closeGrpcStream();
    void         closeGrpcConnection();

private:
    int64_t requestIdForCacheStore() const;
    void    markLocalRequestEnd(int64_t request_id);
    void    markRemoteWorkersRequestEnd(int64_t request_id);
    void    markRequestEnd();
    void    reportTime();
    void    stopStream();

    // The batch envelope exists before QueryConverter/local enqueue. Use the
    // same immutable identity for early Cancel and late stream registration.
    const TaskIdentity task_identity_;

public:
    typedef grpc::ClientReaderWriterInterface<GenerateRequestPB, GenerateOutputsPB> ClientStream;

    RemoteServerResource*                resource;
    RPCContext                           rpc_context;
    std::shared_ptr<GenerateInput>       generate_input;
    std::string                          decode_addr;
    std::vector<std::string>             prefill_worker_cache_store_addrs;
    GrpcConnection                       grpc_connection;
    std::shared_ptr<RpcService::Stub>    stub;
    std::shared_ptr<grpc::ClientContext> client_context;
    std::shared_ptr<ClientStream>        client_stream;
    std::shared_ptr<std::atomic<bool>>   cancel_state;
    bool                                 grpc_stream_closed             = false;
    grpc::Status                         last_grpc_stream_closed_status = grpc::Status::OK;
    PrefillStatInfo                      stat_info;
    const std::atomic<size_t>*           loading_cache_requests               = nullptr;
    int64_t                              prefill_stop_stream_wait_timeout_ms_ = 2000;

private:
    std::atomic<PrefillTerminalCause> terminal_cause_{PrefillTerminalCause::ACTIVE};
    std::mutex                        terminal_transition_mu_;
    std::mutex        priority_finalize_mu_;
    bool              priority_finalized_{false};
    bool              local_stream_scheduler_owned_{false};
};

}  // namespace rtp_llm
