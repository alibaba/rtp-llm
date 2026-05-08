#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/PrefillServerCallerContext.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {

class PrefillServerCaller {
public:
    explicit PrefillServerCaller(const std::string& process_id);

    std::shared_ptr<PrefillServerCallerContext> callPrefill(const GenerateInputPB* request,
                                                            const std::string&     ip,
                                                            uint32_t               port,
                                                            const std::string&     unique_key,
                                                            int64_t                deadline_us);

    grpc::Status callPrefill(grpc::ServerContext*                   server_context,
                             const GenerateInputPB*                 request,
                             grpc::ServerWriter<GenerateOutputsPB>* response_writer);

    int getPrefillTpSize(const std::string& ip, uint32_t port, int32_t request_timeout_ms);

private:
    std::shared_ptr<RPCPool> rpc_pool_;
    std::string              process_id_;

    mutable std::shared_mutex            prefill_tp_cache_mutex_;
    std::unordered_map<std::string, int> prefill_tp_cache_;
};

}  // namespace rtp_llm
