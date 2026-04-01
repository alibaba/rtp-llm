#pragma once

#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/IGenerateStream.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <grpc++/grpc++.h>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class PrefillLoadCaller {
public:
    /// @param worker_addrs Decode worker 地址列表，每项格式为 ip:cache_store_port:grpc_port（冒号分隔三段）
    PrefillLoadCaller(const std::vector<std::string>& worker_addrs);
    ~PrefillLoadCaller() = default;

public:
    struct Result {
        Result(): success_(false), timeout_ms(0), request_id(0), start_time_us(currentTimeUs()) {}
        ~Result() {
            shutdownAndDrainCompletionQueue();
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

    private:
        bool pollCompletionQueue();
        void updateStreamFromResponse();
        /// After TryCancel or when tearing down, must Shutdown CQ and drain Next() until false to avoid gRPC leaks.
        void shutdownAndDrainCompletionQueue();

    public:
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

        bool completion_queue_shutdown_drained_{false};
    };

    /// @brief 向 Prefill server 发起异步 StartLoad RPC，通知其开始向 Decode 发送 KV cache
    std::shared_ptr<Result> load(int64_t                   request_id,
                                 const std::string&        prefill_ip,
                                 uint32_t                  prefill_port,
                                 const std::string&        unique_key,
                                 int64_t                   deadline_ms,
                                 const IGenerateStreamPtr& generate_stream);

private:
    bool buildAndStartAsyncRpc(const std::shared_ptr<Result>& result,
                               const std::string&             unique_key,
                               int64_t                        deadline_ms,
                               int64_t                        request_id);

    std::vector<std::string>    worker_addrs_;
    std::shared_ptr<RPCPool>    rpc_pool_;
    std::vector<TPWorkerInfoPB> tp_worker_infos_;
};

}  // namespace rtp_llm
