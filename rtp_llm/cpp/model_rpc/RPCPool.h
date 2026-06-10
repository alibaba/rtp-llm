#pragma once

#include <chrono>
#include <memory>
#include <mutex>
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
        // [PD-DIAG] Stage timestamps. Production has shown getConnection blocking
        // 18-22s on a single peer; we want to know exactly which step is the
        // bottleneck (lock acquire vs GetState vs CreateCustomChannel vs NewStub).
        const auto t_enter = std::chrono::steady_clock::now();
        auto       t_locked = t_enter;

        Connection<T>               cached_connection;
        std::shared_ptr<std::mutex> peer_mutex;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            t_locked                = std::chrono::steady_clock::now();
            auto iter               = connection_pool_.find(peer);
            if (iter != connection_pool_.end()) {
                cached_connection = iter->second;
            }
            peer_mutex = getPeerMutexLocked(peer);
        }
        std::chrono::steady_clock::time_point t_after_get_state = t_locked;
        grpc_connectivity_state                channel_state    = GRPC_CHANNEL_IDLE;
        bool need_new_connection = !cached_connection.channel || !cached_connection.stub;
        if (!need_new_connection) {
            // Check if existing connection is in a bad state. GetState(true) is
            // documented as non-blocking but in practice can sit for seconds when
            // the channel is in TRANSIENT_FAILURE under reconnect backoff.
            channel_state     = cached_connection.channel->GetState(true);
            t_after_get_state = std::chrono::steady_clock::now();
            need_new_connection =
                (channel_state == GRPC_CHANNEL_SHUTDOWN || channel_state == GRPC_CHANNEL_TRANSIENT_FAILURE);
        }
        if (!need_new_connection) {
            logSlowGetConnection(peer,
                                 channel_state,
                                 false,
                                 t_enter,
                                 t_locked,
                                 t_after_get_state,
                                 t_after_get_state,
                                 t_after_get_state);
            return cached_connection;
        }

        std::lock_guard<std::mutex> peer_guard(*peer_mutex);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            auto                        iter = connection_pool_.find(peer);
            if (iter != connection_pool_.end()) {
                cached_connection = iter->second;
            } else {
                cached_connection = Connection<T>();
            }
        }

        channel_state       = GRPC_CHANNEL_IDLE;
        t_after_get_state   = t_locked;
        need_new_connection = !cached_connection.channel || !cached_connection.stub;
        if (!need_new_connection) {
            channel_state     = cached_connection.channel->GetState(true);
            t_after_get_state = std::chrono::steady_clock::now();
            need_new_connection =
                (channel_state == GRPC_CHANNEL_SHUTDOWN || channel_state == GRPC_CHANNEL_TRANSIENT_FAILURE);
        }
        if (!need_new_connection) {
            logSlowGetConnection(peer,
                                 channel_state,
                                 false,
                                 t_enter,
                                 t_locked,
                                 t_after_get_state,
                                 t_after_get_state,
                                 t_after_get_state);
            return cached_connection;
        }

        std::chrono::steady_clock::time_point t_after_create = t_after_get_state;
        std::chrono::steady_clock::time_point t_after_stub   = t_after_get_state;
        grpc::ChannelArguments args;
        args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);
        args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
        args.SetInt(GRPC_ARG_MAX_CONCURRENT_STREAMS, 100000);
        args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
        // 需配合 GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS 使用，例如 500
        args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);
        args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
        auto grpc_channel = grpc::CreateCustomChannel(peer, grpc::InsecureChannelCredentials(), args);
        t_after_create    = std::chrono::steady_clock::now();
        if (!grpc_channel) {
            std::string error_msg = "create grpc channel for " + peer + " failed";
            return absl::InternalError(error_msg);
        }

        auto grpc_stub = T::NewStub(grpc_channel);
        t_after_stub   = std::chrono::steady_clock::now();
        if (!grpc_stub) {
            std::string error_msg = "create grpc stub for " + peer + " failed";
            return absl::InternalError(error_msg);
        }

        Connection<T> connection = {grpc_channel, std::move(grpc_stub)};
        {
            std::lock_guard<std::mutex> guard(mutex_);
            connection_pool_[peer] = connection;
        }
        logSlowGetConnection(peer,
                             channel_state,
                             true,
                             t_enter,
                             t_locked,
                             t_after_get_state,
                             t_after_create,
                             t_after_stub);
        return connection;
    }

    void removeConnection(std::string peer) {
        std::lock_guard<std::mutex> guard(mutex_);
        connection_pool_.erase(peer);
    }

    // TODO(xinfei.sxf) add watch for grpc channel state changed to closed

private:
    std::shared_ptr<std::mutex> getPeerMutexLocked(const std::string& peer) {
        auto [it, inserted] = peer_mutexes_.try_emplace(peer, nullptr);
        if (!it->second) {
            it->second = std::make_shared<std::mutex>();
        }
        return it->second;
    }

    static const char* channelStateName(grpc_connectivity_state s) {
        switch (s) {
            case GRPC_CHANNEL_IDLE: return "IDLE";
            case GRPC_CHANNEL_CONNECTING: return "CONNECTING";
            case GRPC_CHANNEL_READY: return "READY";
            case GRPC_CHANNEL_TRANSIENT_FAILURE: return "TRANSIENT_FAILURE";
            case GRPC_CHANNEL_SHUTDOWN: return "SHUTDOWN";
            default: return "UNKNOWN";
        }
    }

    /// [PD-DIAG] Emit a single WARN with each stage's cost when total getConnection
    /// time exceeds 100ms. Lets us pin down whether the slow getConnection call is
    /// stuck on the lock, on GetState(true), on CreateCustomChannel, or on NewStub.
    static void logSlowGetConnection(const std::string&                    peer,
                                     grpc_connectivity_state               channel_state,
                                     bool                                  need_new_connection,
                                     std::chrono::steady_clock::time_point t_enter,
                                     std::chrono::steady_clock::time_point t_locked,
                                     std::chrono::steady_clock::time_point t_after_get_state,
                                     std::chrono::steady_clock::time_point t_after_create,
                                     std::chrono::steady_clock::time_point t_after_stub) {
        const auto total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_after_stub - t_enter).count();
        if (total_us < 100 * 1000) {
            return;
        }
        const auto lock_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_locked - t_enter).count();
        const auto get_state_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_after_get_state - t_locked).count();
        const auto create_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_after_create - t_after_get_state).count();
        const auto stub_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_after_stub - t_after_create).count();
        RTP_LLM_LOG_WARNING(
            "[PD-DIAG] RpcPool::getConnection slow, peer=%s, channel_state=%s, need_new=%d, "
            "total_us=%lld, lock_acq_us=%lld, get_state_us=%lld, create_channel_us=%lld, new_stub_us=%lld",
            peer.c_str(),
            channelStateName(channel_state),
            need_new_connection ? 1 : 0,
            (long long)total_us,
            (long long)lock_us,
            (long long)get_state_us,
            (long long)create_us,
            (long long)stub_us);
    }

    std::mutex                                     mutex_;
    std::unordered_map<std::string, Connection<T>> connection_pool_;
    std::unordered_map<std::string, std::shared_ptr<std::mutex>> peer_mutexes_;
};

using RPCPool           = Pool<RpcService>;
using MultimodalRpcPool = Pool<MultimodalRpcService>;
using GrpcConnection    = Connection<RpcService>;

}  // namespace rtp_llm
