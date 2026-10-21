#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <list>
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
        // This guard is declared before peer_guard, so the per-peer lock is
        // unlocked before its last registry reference can be removed.
        struct PeerMutexRelease {
            Pool&                             pool;
            const std::string&                peer;
            std::shared_ptr<std::mutex>&       mutex;
            ~PeerMutexRelease() {
                pool.releasePeerMutex(peer, mutex);
            }
        } peer_mutex_release{*this, peer, peer_mutex};
        {
            std::lock_guard<std::mutex> guard(mutex_);
            t_locked                = std::chrono::steady_clock::now();
            auto iter               = connection_pool_.find(peer);
            if (iter != connection_pool_.end()) {
                cached_connection = iter->second.connection;
                connection_lru_.splice(connection_lru_.begin(), connection_lru_, iter->second.position);
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
        Connection<T> previous_connection = std::move(cached_connection);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            auto                        iter = connection_pool_.find(peer);
            if (iter != connection_pool_.end()) {
                cached_connection = iter->second.connection;
                connection_lru_.splice(connection_lru_.begin(), connection_lru_, iter->second.position);
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
        Connection<T> evicted_connection;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            evicted_connection = cacheConnectionLocked(peer, connection);
        }
        // Destroy an evicted channel outside mutex_: gRPC teardown must not
        // serialize unrelated peers behind the pool-wide lock.
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

    absl::StatusOr<Connection<T>> getReadyConnection(std::string peer, std::chrono::milliseconds timeout) {
        return getReadyConnection(std::move(peer), std::chrono::system_clock::now() + timeout);
    }

    absl::StatusOr<Connection<T>>
    getReadyConnection(std::string                                peer,
                       std::chrono::system_clock::time_point       deadline,
                       const std::function<bool()>&                is_cancelled = nullptr) {
        auto connection_status = getConnection(peer);
        if (!connection_status.ok()) {
            return connection_status.status();
        }

        auto           connection = connection_status.value();
        const auto     begin_time = std::chrono::steady_clock::now();
        const auto     state      = connection.channel->GetState(false);
        constexpr auto kCancellationPollInterval = std::chrono::milliseconds(10);
        while (true) {
            if (is_cancelled && is_cancelled()) {
                return absl::CancelledError("grpc channel readiness wait for " + peer + " was cancelled");
            }
            const auto now = std::chrono::system_clock::now();
            if (now >= deadline) {
                break;
            }
            if (connection.channel->WaitForConnected(std::min(deadline, now + kCancellationPollInterval))) {
                return connection;
            }
        }

        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin_time)
                .count();
        const auto final_state = connection.channel->GetState(false);
        RTP_LLM_LOG_WARNING(
            "grpc channel not ready, peer [%s], channel [%p], state [%d], final state [%d], elapsed [%ld]ms",
            peer.c_str(),
            static_cast<void*>(connection.channel.get()),
            static_cast<int>(state),
            static_cast<int>(final_state),
            elapsed_ms);
        removeConnection(peer, connection.channel);
        return absl::UnavailableError("grpc channel for " + peer + " is not ready");
    }

    void removeConnection(const std::string& peer, const std::shared_ptr<grpc::Channel>& expected_channel) {
        Connection<T> removed_connection;
        std::lock_guard<std::mutex> guard(mutex_);
        auto                        iter = connection_pool_.find(peer);
        if (iter != connection_pool_.end() && iter->second.connection.channel == expected_channel) {
            removed_connection = std::move(iter->second.connection);
            connection_lru_.erase(iter->second.position);
            connection_pool_.erase(iter);
        }
    }

    // TODO(xinfei.sxf) add watch for grpc channel state changed to closed

private:
    // Cache ownership is bounded even when routing visits continuously changing
    // peers. Eviction drops only the cache reference; active RPCs keep their
    // shared channel/stub and are never cancelled by cache maintenance.
    static constexpr size_t kMaxCachedConnections = 1024;
    struct CachedConnection {
        Connection<T> connection;
        std::list<std::string>::iterator position;
    };

    Connection<T> cacheConnectionLocked(const std::string& peer, const Connection<T>& connection) {
        Connection<T> evicted;
        auto it = connection_pool_.find(peer);
        if (it != connection_pool_.end()) {
            evicted = it->second.connection;
            it->second.connection = connection;
            connection_lru_.splice(connection_lru_.begin(), connection_lru_, it->second.position);
            return evicted;
        }
        connection_lru_.push_front(peer);
        try {
            connection_pool_.emplace(peer, CachedConnection{connection, connection_lru_.begin()});
        } catch (...) {
            connection_lru_.pop_front();
            throw;
        }
        if (connection_pool_.size() > kMaxCachedConnections) {
            auto oldest = connection_pool_.find(connection_lru_.back());
            evicted = std::move(oldest->second.connection);
            connection_pool_.erase(oldest);
            connection_lru_.pop_back();
        }
        return evicted;
    }

    std::shared_ptr<std::mutex> getPeerMutexLocked(const std::string& peer) {
        auto it = peer_mutexes_.find(peer);
        if (it != peer_mutexes_.end()) {
            return it->second;
        }
        auto peer_mutex = std::make_shared<std::mutex>();
        peer_mutexes_.emplace(peer, peer_mutex);
        return peer_mutex;
    }

    void releasePeerMutex(const std::string& peer, std::shared_ptr<std::mutex>& peer_mutex) {
        if (!peer_mutex) {
            return;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        auto it = peer_mutexes_.find(peer);
        const bool same_mutex = it != peer_mutexes_.end() && it->second == peer_mutex;
        // Drop this caller's reference under mutex_ as well: simultaneous exits
        // must not both observe the other's reference and leave an orphan entry.
        peer_mutex.reset();
        // Acquisition and removal share mutex_, so no waiter can receive a new
        // mutex while another caller still holds the old one.
        if (same_mutex && it->second.use_count() == 1) {
            peer_mutexes_.erase(it);
        }
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
    std::unordered_map<std::string, CachedConnection> connection_pool_;
    std::list<std::string> connection_lru_;
    std::unordered_map<std::string, std::shared_ptr<std::mutex>> peer_mutexes_;
};

using RPCPool           = Pool<RpcService>;
using MultimodalRpcPool = Pool<MultimodalRpcService>;
using GrpcConnection    = Connection<RpcService>;

}  // namespace rtp_llm
