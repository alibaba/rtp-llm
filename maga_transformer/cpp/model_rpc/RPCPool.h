#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "grpc++/grpc++.h"
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/proto/model_rpc_service.grpc.pb.h"
#include "maga_transformer/cpp/utils/Logger.h"

namespace rtp_llm {

template <typename T>
struct Connection {
    std::shared_ptr<grpc::Channel> channel;
    std::shared_ptr<typename T::Stub> stub;
};

template <typename T>
class Pool {
public:
    absl::StatusOr<Connection<T>> getConnection(std::string peer) {
        std::lock_guard<std::mutex> guard(mutex_);
        auto iter = connection_pool_.find(peer);
        if (iter == connection_pool_.end()
            || iter->second.channel->GetState(true) == GRPC_CHANNEL_SHUTDOWN
            || iter->second.channel->GetState(true) == GRPC_CHANNEL_TRANSIENT_FAILURE) {
            grpc::ChannelArguments args;
            args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, 1024 * 1024 * 1024);
            args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, 1024 * 1024 * 1024);
            args.SetInt(GRPC_ARG_MAX_CONNECTION_IDLE_MS, 60000);
            args.SetInt(GRPC_ARG_MAX_CONCURRENT_STREAMS, 5000);
            args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
            auto grpc_channel = grpc::CreateCustomChannel(peer, grpc::InsecureChannelCredentials(), args);
            if (!grpc_channel) {
                std::string error_msg = "create grpc channel for " + peer + " failed";
                return absl::InternalError(error_msg);
            }
            auto channel_status = grpc_channel->GetState(true);
            if (channel_status != GRPC_CHANNEL_READY) {
                std::chrono::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(10);
                bool isChannelReady = grpc_channel->WaitForConnected(deadline);
                if (!(isChannelReady && grpc_channel->GetState(false) == GRPC_CHANNEL_READY)) {
                    FT_LOG_WARNING("wait channel ready failed channel, isChannelReady %d, current status %ld", isChannelReady, grpc_channel->GetState(false));
                    std::string error_msg = "create grpc channel connection for " + peer + " failed, not ready";
                    return absl::InternalError(error_msg);
                }
            }

            auto grpc_stub    = T::NewStub(grpc_channel);
            if (!grpc_stub) {
                std::string error_msg = "create grpc stub for " + peer + " failed";
                return absl::InternalError(error_msg);
            }
            Connection<T> connection = {grpc_channel, std::move(grpc_stub)};
            connection_pool_[peer] = connection;
            return connection;
        } else {
            return iter->second;
        }
    }

    void removeConnection(std::string peer) {
        std::lock_guard<std::mutex> guard(mutex_);
        connection_pool_.erase(peer);
    }

    // TODO(xinfei.sxf) add watch for grpc channel state changed to closed

private:
    std::mutex mutex_;
    std::unordered_map<std::string, Connection<T>> connection_pool_;
};

using RPCPool = Pool<RpcService>;
using MultimodalRpcPool = Pool<MultimodalRpcService>;
using GrpcConnection = Connection<RpcService>;

}  // namespace rtp_llm
