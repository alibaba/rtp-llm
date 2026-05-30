#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <climits>
#include <cstdlib>
#include "grpc++/grpc++.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

inline int readGrpcEnvIntAtLoad(const char* env_name, int default_value) {
    const char* env_value = std::getenv(env_name);
    if (env_value == nullptr || env_value[0] == '\0') {
        return default_value;
    }

    char* end    = nullptr;
    long  parsed = std::strtol(env_value, &end, 10);
    if (end == env_value || parsed <= 0 || parsed > INT_MAX) {
        RTP_LLM_LOG_WARNING("invalid %s=%s, fallback to %d", env_name, env_value, default_value);
        return default_value;
    }
    return static_cast<int>(parsed);
}

inline const int kGrpcKeepAliveTimeMs    = readGrpcEnvIntAtLoad("RTP_LLM_GRPC_KEEPALIVE_TIME_MS", 10000);
inline const int kGrpcKeepAliveTimeoutMs = readGrpcEnvIntAtLoad("RTP_LLM_GRPC_KEEPALIVE_TIMEOUT_MS", 5000);

inline int getGrpcEnvInt(const char* env_name, int default_value) {
    const std::string name(env_name);
    if (name == "RTP_LLM_GRPC_KEEPALIVE_TIME_MS") {
        return kGrpcKeepAliveTimeMs;
    }
    if (name == "RTP_LLM_GRPC_KEEPALIVE_TIMEOUT_MS") {
        return kGrpcKeepAliveTimeoutMs;
    }
    return default_value;
}

template<typename T>
struct Connection {
    std::shared_ptr<grpc::Channel>    channel;
    std::shared_ptr<typename T::Stub> stub;
};

template<typename T>
class Pool {
public:
    absl::StatusOr<Connection<T>> getConnection(std::string peer) {
        std::lock_guard<std::mutex> guard(mutex_);
        auto                        iter = connection_pool_.find(peer);
        // Check if we need to create a new connection
        bool need_new_connection = iter == connection_pool_.end();
        if (!need_new_connection) {
            // Check if existing connection is in a bad state
            auto channel_state = iter->second.channel->GetState(true);
            need_new_connection =
                (channel_state == GRPC_CHANNEL_SHUTDOWN || channel_state == GRPC_CHANNEL_TRANSIENT_FAILURE);
            // Remove bad connection from pool if needed
            if (need_new_connection) {
                connection_pool_.erase(iter);
            }
        }

        if (need_new_connection) {
            grpc::ChannelArguments args;
            args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);
            args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
            args.SetInt(GRPC_ARG_MAX_CONCURRENT_STREAMS, 100000);
            args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, getGrpcEnvInt("RTP_LLM_GRPC_KEEPALIVE_TIME_MS", 10000));
            // 需配合 GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS 使用，例如 500
            args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, getGrpcEnvInt("RTP_LLM_GRPC_KEEPALIVE_TIMEOUT_MS", 5000));
            args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
            auto grpc_channel = grpc::CreateCustomChannel(peer, grpc::InsecureChannelCredentials(), args);
            if (!grpc_channel) {
                std::string error_msg = "create grpc channel for " + peer + " failed";
                return absl::InternalError(error_msg);
            }

            auto grpc_stub = T::NewStub(grpc_channel);
            if (!grpc_stub) {
                std::string error_msg = "create grpc stub for " + peer + " failed";
                return absl::InternalError(error_msg);
            }
            Connection<T> connection = {grpc_channel, std::move(grpc_stub)};
            connection_pool_[peer]   = connection;
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
    std::mutex                                     mutex_;
    std::unordered_map<std::string, Connection<T>> connection_pool_;
};

using RPCPool           = Pool<RpcService>;
using MultimodalRpcPool = Pool<MultimodalRpcService>;
using GrpcConnection    = Connection<RpcService>;

}  // namespace rtp_llm
