#pragma once

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <grpc++/grpc++.h>
#include <atomic>
#include <functional>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <optional>

namespace rtp_llm {

// Side-channel payload for P2P bypass (carries first token, reuse, SP info, position_ids)
struct P2PSideChannelPayload {
    bool                 has_first_token  = false;
    int64_t              first_token_id   = 0;
    int32_t              total_reuse_len  = 0;
    int32_t              local_reuse_len  = 0;
    int32_t              remote_reuse_len = 0;
    int32_t              memory_reuse_len = 0;
    int32_t              disk_reuse_len   = 0;
    std::vector<int>     propose_tokens;
    TensorPB             propose_probs;
    TensorPB             propose_hidden;
    std::vector<int32_t> position_ids;
    bool                 has_data = false;
};

class DecodeLoadHelper {
public:
    /// @param worker_addrs Decode worker 地址列表，每项格式为 host:cache_store_port:grpc_port
    /// 或 [IPv6]:cache_store_port:grpc_port
    DecodeLoadHelper(const std::vector<std::string>& worker_addrs);
    ~DecodeLoadHelper() = default;

public:
    struct Result {
        Result(): success_(false), timeout_ms(0), request_id(0), start_time_us(currentTimeUs()) {}
        ~Result() = default;

        bool success() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return success_;
        }
        bool done() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return done_;
        }
        FirstError::Snapshot firstError() const {
            return first_error_.snapshot();
        }
        void    complete(bool ok);
        void    setDoneCallback(std::function<void()> callback);
        void    cancel();
        int64_t totalCostTimeUs() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return total_cost_time_us;
        }

    private:
        FirstError first_error_;
        void updateStreamFromResponse();

    public:
        bool                                                                              success_ = false;
        bool                                                                              done_    = false;
        std::shared_ptr<RpcService::Stub>                                                 stub;
        std::shared_ptr<grpc::ClientContext>                                              client_context;
        P2PConnectorStartLoadRequestPB                                                    request;
        P2PConnectorStartLoadResponsePB                                                   response;

        std::unique_ptr<grpc::ClientAsyncResponseReader<P2PConnectorStartLoadResponsePB>> reader;
        grpc::Status                                                                      status;
        std::string                                                                       server_addr;
        std::string                                                                       unique_key;
        int                                                                               timeout_ms{0};
        int64_t                                                                           request_id{0};
        int64_t                                                                           start_time_us{0};
        int64_t                                                                           total_cost_time_us{0};
        ErrorCode   error_code = ErrorCode::NONE_ERROR;
        std::string error_message;

        // P2P bypass: parsed side-channel payload.
        P2PSideChannelPayload side_channel_payload;

        // The shared CQ owns this result until physical Finish, even after cancel().

    private:
        std::atomic<bool>  cancel_requested_{false};
        mutable std::mutex state_mutex_;
        std::function<void()> done_callback_;
    };

    /// @brief 向 Prefill server 发起异步 StartLoad RPC，通知其开始向 Decode 发送 KV cache
    std::shared_ptr<Result> load(int64_t            request_id,
                                 const std::string& prefill_ip,
                                 uint32_t           prefill_port,
                                 const std::string& unique_key,
                                 int64_t            request_deadline_ms,
                                 int64_t            transfer_deadline_ms,
                                 bool               no_transfer = false,
                                 uint64_t           plan_digest = 0,
                                 const std::vector<int>& active_route_ids = {});

private:
    bool buildAndStartAsyncRpc(const std::shared_ptr<Result>& result,
                               const std::string&             unique_key,
                               int64_t                        request_deadline_ms,
                               int64_t                        transfer_deadline_ms,
                               int64_t                        request_id,
                               bool                           no_transfer,
                               uint64_t                       plan_digest,
                               const std::vector<int>&        active_route_ids);

    std::vector<std::string>    worker_addrs_;
    std::shared_ptr<RPCPool>    rpc_pool_;
    std::vector<TPWorkerInfoPB> tp_worker_infos_;
};

}  // namespace rtp_llm
