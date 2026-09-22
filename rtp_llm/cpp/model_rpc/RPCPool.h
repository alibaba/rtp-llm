#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "grpc++/grpc++.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

template<typename T>
struct Connection {
    std::shared_ptr<grpc::Channel>    channel;
    std::shared_ptr<typename T::Stub> stub;
};

template<typename T>
class Pool {
public:
    absl::StatusOr<Connection<T>> getConnection(std::string peer) {
        // Declare before the lock so every return unlocks before logging. These
        // snapshots deliberately do not retain the removed Channel or Stub.
        struct DiagnosticSnapshot {
            const std::string& peer;
            bool looked_up = false;
            bool removed = false;
            bool created = false;
            grpc_connectivity_state observed_state = GRPC_CHANNEL_IDLE;
            void* previous_channel_ptr = nullptr;
            void* created_channel_ptr = nullptr;

            ~DiagnosticSnapshot() noexcept {
                try {
                    if (looked_up) {
                        RTP_LLM_LOG_INFO("[KV_RPC] event=POOL_LOOKUP peer=%s channel_ptr=%p observed_state=%d",
                                         peer.c_str(), previous_channel_ptr, static_cast<int>(observed_state));
                    }
                    if (removed) {
                        RTP_LLM_LOG_INFO("[KV_RPC] event=POOL_REMOVE peer=%s channel_ptr=%p reason=observed_bad_state",
                                         peer.c_str(), previous_channel_ptr);
                    }
                    if (created) {
                        RTP_LLM_LOG_INFO("[KV_RPC] event=POOL_CREATE peer=%s channel_ptr=%p",
                                         peer.c_str(), created_channel_ptr);
                    }
                } catch (...) {
                    // Logging must not change connection lookup or exception unwinding.
                }
            }
        } diagnostic{peer};
        std::lock_guard<std::mutex> guard(mutex_);
        auto                        iter = connection_pool_.find(peer);
        // Check if we need to create a new connection
        bool need_new_connection = iter == connection_pool_.end();
        if (!need_new_connection) {
            // Check if existing connection is in a bad state
            auto channel_state = iter->second.channel->GetState(true);
            diagnostic.looked_up = true;
            diagnostic.previous_channel_ptr = iter->second.channel.get();
            diagnostic.observed_state = channel_state;
            need_new_connection =
                (channel_state == GRPC_CHANNEL_SHUTDOWN || channel_state == GRPC_CHANNEL_TRANSIENT_FAILURE);
            // Remove bad connection from pool if needed
            if (need_new_connection) {
                diagnostic.removed = true;
                connection_pool_.erase(iter);
            }
        }

        if (need_new_connection) {
            grpc::ChannelArguments args;
            args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);
            args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
            args.SetInt(GRPC_ARG_MAX_CONCURRENT_STREAMS, 100000);
            args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
            // 需配合 GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS 使用，例如 500
            args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);
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
            diagnostic.created = true;
            diagnostic.created_channel_ptr = grpc_channel.get();
            return connection;
        } else {
            return iter->second;
        }
    }

    void removeConnection(std::string peer) {
        bool removed = false;
        void* channel_ptr = nullptr;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            auto iter = connection_pool_.find(peer);
            if (iter != connection_pool_.end()) {
                channel_ptr = iter->second.channel.get();
                removed = true;
            }
            connection_pool_.erase(peer);
        }
        if (removed) {
            try {
                RTP_LLM_LOG_INFO("[KV_RPC] event=POOL_REMOVE peer=%s channel_ptr=%p reason=explicit_remove",
                                 peer.c_str(), channel_ptr);
            } catch (...) {
                // The requested removal already completed; logging is best effort.
            }
        }
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
