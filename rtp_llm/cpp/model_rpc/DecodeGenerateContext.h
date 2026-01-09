#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/proto/model_rpc_service.pb.h"

namespace rtp_llm {

typedef grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB> ServerStream;

struct DecodeStatInfo {
    enum ExecuteStage {
        start                  = 0,
        prepareGenerateContext = 1,
        allocateResource       = 2,
        loadCacheFromPrefill   = 3,
        localGenerate          = 4,
        finish                 = 5
    };

    int64_t      begin_time                     = 0;
    int64_t      prepare_generate_context_rt_us = 0;
    int64_t      allocate_resource_rt_us        = 0;
    int64_t      load_cache_from_prefill_rt_us  = 0;
    int64_t      local_generate_rt_us           = 0;
    int64_t      load_cache_min_rt_us           = 0;
    int64_t      load_cache_max_rt_us           = 0;
    int64_t      load_cache_polling_cost_us     = 0;
    ExecuteStage stage                          = start;

    ExecuteStage saveStage() const;
    void         restoreStage(ExecuteStage stage);
    void         nextStage();
};

struct DecodeRpcContext {
    ServerStream* grpc_stream;
};

class DecodeGenerateContext: public GenerateContext {
public:
    DecodeGenerateContext(DecodeRpcContext&                     rpc_context,
                          int64_t                               timeout_ms,
                          grpc::ServerContext*                  server_context,
                          kmonitor::MetricsReporterPtr&         metrics_reporter,
                          std::shared_ptr<RpcServerRuntimeMeta> meta):
        GenerateContext(0, timeout_ms, server_context, metrics_reporter, meta), rpc_context(rpc_context) {}
    ~DecodeGenerateContext();
    void reportTime();
    struct TimeInfo {
        void    updateRequestBegineTime();
        void    updateLoadBeginTime();
        void    updateLoadEndTime();
        void    updateGenerateBeginTime();
        void    updateGenerateEndTime();
        int64_t loadCacheTimeMs() const;

        int64_t request_begin_time_us;
        int64_t load_begin_time_us;
        int64_t load_end_time_us;
        int64_t generate_begin_time_us;
        int64_t generate_end_time_us;
    };

public:
    DecodeRpcContext&        rpc_context;
    std::vector<std::string> peer_addrs;  // prefill worker addrs
    GenerateRequestPB        allocate_request;
    DecodeStatInfo           stat_info;
    int64_t                  loading_cache_requests = 0;

    // for debug, will delete in future
    TimeInfo time_info;
};

}  // namespace rtp_llm
