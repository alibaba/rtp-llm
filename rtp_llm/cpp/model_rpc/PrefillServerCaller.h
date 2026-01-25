#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class PrefillServerCallerContext {
public:
    PrefillServerCallerContext(const std::string& prefill_addr,
                               int64_t            decode_polling_call_prefill_ms,
                               const std::string& unique_key);
    ~PrefillServerCallerContext();

    // 非阻塞检查 Prefill 是否完成
    void checkDone();

    // 检查是否完成
    bool done() const {
        return finished;
    }

    // 检查是否成功
    bool success() const {
        return finished && status.ok() && response_received_;
    }

public:
    std::string                                                 prefill_addr;
    std::string                                                 unique_key;
    std::shared_ptr<grpc::ClientContext>                        client_context;
    std::shared_ptr<grpc::CompletionQueue>                      completion_queue;
    GenerateInputPB                                             request;
    GenerateOutputsPB                                           response;
    grpc::Status                                                status;
    std::shared_ptr<RpcService::Stub>                           stub;
    std::unique_ptr<grpc::ClientAsyncReader<GenerateOutputsPB>> reader;
    bool                                                        finished = false;

private:
    int64_t request_begin_time_us_;
    int64_t decode_polling_call_prefill_ms_;
    bool    response_received_ = false;  // 是否已收到响应
    bool    finish_started_    = false;  // 是否已启动 Finish 操作
};

class PrefillServerCaller {
public:
    PrefillServerCaller(const std::string& process_id, int64_t decode_polling_call_prefill_ms);
    ~PrefillServerCaller() = default;

    // 调用 Prefill 服务器
    std::shared_ptr<PrefillServerCallerContext> callPrefill(const GenerateInputPB* request,
                                                            const std::string&     ip,
                                                            uint32_t               port,
                                                            const std::string&     unique_key,
                                                            int64_t                deadline_us);

    grpc::Status callPrefill(grpc::ServerContext*                   server_context,
                             const GenerateInputPB*                 request,
                             grpc::ServerWriter<GenerateOutputsPB>* response_writer);

private:
    std::shared_ptr<RPCPool> rpc_pool_;
    std::string              process_id_;
    int64_t                  decode_polling_call_prefill_ms_;
};

}  // namespace rtp_llm
