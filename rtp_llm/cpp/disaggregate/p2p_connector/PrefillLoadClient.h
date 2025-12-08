#pragma once

#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/CommonDefs.h"
#include <grpc++/grpc++.h>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class PrefillLoadClient {
public:
    PrefillLoadClient(const GptInitParameter& gpt_init_parameter);
    ~PrefillLoadClient() = default;

public:
    struct Result {
        Result(): success_(false), timeout_ms_(0), request_id_(0) {}
        ~Result() {
            if (completion_queue_) {
                completion_queue_->Shutdown();
            }
        }

        bool success() const {
            return success_;
        }
        bool waitDone();

        // TODO: test this
        void cancel();

        bool                                                                              success_;
        std::shared_ptr<RpcService::Stub>                                                 stub;
        std::shared_ptr<grpc::ClientContext>                                              client_context;
        P2PConnectorStartLoadRequestPB                                                    request;
        P2PConnectorStartLoadResponsePB                                                   response;
        std::shared_ptr<grpc::CompletionQueue>                                            completion_queue_;
        std::unique_ptr<grpc::ClientAsyncResponseReader<P2PConnectorStartLoadResponsePB>> reader_;
        grpc::Status                                                                      status;
        std::string                                                                       server_addr;
        int                                                                               timeout_ms_;
        int64_t                                                                           request_id_;
    };

    std::shared_ptr<Result> load(int64_t            request_id,
                                 const std::string& prefill_ip,
                                 uint32_t           prefill_port,
                                 const std::string& unique_key,
                                 int64_t            deadline_ms);

private:
    const GptInitParameter&     gpt_init_parameter_;
    std::shared_ptr<RPCPool>    rpc_pool_;
    std::vector<TPWorkerInfoPB> tp_worker_infos_;
};

}  // namespace rtp_llm
