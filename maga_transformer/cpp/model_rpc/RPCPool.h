#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "grpc++/grpc++.h"
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {

class RPCPool {
public:
    struct Connection {
        std::shared_ptr<grpc::Channel> channel;
        std::shared_ptr<RpcService::Stub> stub;
    };

public:
    absl::StatusOr<Connection> getConnection(std::string peer) {
        std::lock_guard<std::mutex> guard(mutex_);
        auto iter = connection_pool_.find(peer);
        if (iter == connection_pool_.end()
            || iter->second.channel->GetState(false) == GRPC_CHANNEL_SHUTDOWN
            || iter->second.channel->GetState(false) == GRPC_CHANNEL_TRANSIENT_FAILURE) {
            grpc::ChannelArguments args;
            args.SetInt(GRPC_ARG_MAX_CONNECTION_IDLE_MS, 60000);
            args.SetInt(GRPC_ARG_MAX_CONCURRENT_STREAMS, 200);
            auto grpc_channel = grpc::CreateCustomChannel(peer, grpc::InsecureChannelCredentials(), args);
            if (!grpc_channel) { 
                std::string error_msg = "create grpc channel for " + peer + " failed";
                return absl::InternalError(error_msg);
            }
            // TODO(xinfei.sxf) grpc_stub是unique ptr，本意是阻止共享。
            // 多个请求，使用同一个grpc stub/channel会串行发送吗，特别是stream请求并发度。
            auto grpc_stub    = RpcService::NewStub(grpc_channel);
            if (!grpc_stub) {
                std::string error_msg = "create grpc stub for " + peer + " failed";
                return absl::InternalError(error_msg);
            }
            Connection connection = {grpc_channel, std::move(grpc_stub)};
            connection_pool_[peer] = connection;
            return connection;
        } else {
            return iter->second;
        }
    }

    // TODO(xinfei.sxf) add watch for grpc channel state changed to closed

private:
    std::mutex mutex_;
    std::unordered_map<std::string, Connection> connection_pool_;
};

}  // namespace rtp_llm
