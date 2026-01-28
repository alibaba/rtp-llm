#pragma once

#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <grpc++/grpc++.h>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnectorServerCaller {
public:
    P2PConnectorServerCaller(const std::vector<std::string>& worker_addrs);
    ~P2PConnectorServerCaller() = default;

public:
    struct Result {
        Result(): success_(false), timeout_ms(0), request_id(0), start_time_us(currentTimeUs()) {}
        ~Result() {
            if (completion_queue) {
                completion_queue->Shutdown();
            }
        }

        bool success() const {
            return success_;
        }
        bool done() const {
            return done_;
        }
        void    checkDone();
        void    cancel();
        int64_t totalCostTimeUs() const {
            return total_cost_time_us;
        }

        bool                                                                              success_ = false;
        bool                                                                              done_    = false;
        std::shared_ptr<RpcService::Stub>                                                 stub;
        std::shared_ptr<grpc::ClientContext>                                              client_context;
        P2PConnectorStartLoadRequestPB                                                    request;
        P2PConnectorStartLoadResponsePB                                                   response;
        std::shared_ptr<grpc::CompletionQueue>                                            completion_queue;
        std::unique_ptr<grpc::ClientAsyncResponseReader<P2PConnectorStartLoadResponsePB>> reader;
        grpc::Status                                                                      status;
        std::string                                                                       server_addr;
        int                                                                               timeout_ms;
        int64_t                                                                           request_id;
        int64_t                                                                           start_time_us;
        int64_t                                                                           total_cost_time_us;
        IGenerateStreamPtr                                                                generate_stream;
        ErrorCode   error_code = ErrorCode::NONE_ERROR;
        std::string error_message;
    };

    std::shared_ptr<Result> load(int64_t                   request_id,
                                 const std::string&        prefill_ip,
                                 uint32_t                  prefill_port,
                                 const std::string&        unique_key,
                                 int64_t                   deadline_ms,
                                 const IGenerateStreamPtr& generate_stream);

private:
    std::vector<std::string>    worker_addrs_;
    std::shared_ptr<RPCPool>    rpc_pool_;
    std::vector<TPWorkerInfoPB> tp_worker_infos_;
};

}  // namespace rtp_llm
