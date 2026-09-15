#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "aios/network/anet/transport.h"
#include "aios/network/arpc/arpc/ANetRPCChannelManager.h"

namespace rtp_llm {

namespace transfer {

class TcpClient {
public:
    TcpClient() = default;
    ~TcpClient();

    /// @brief 初始化 transport 和 channel manager
    /// @param idle_ttl 为 0 时关闭 idle 淘汰与清扫；否则超过该时长未命中的缓存项可被驱逐
    /// @param sweep_every_n_calls 为 0 时关闭「每 N 次 getChannel 全表清扫」；仅保留 miss 路径清扫
    bool                                  init(int                       io_thread_count,
                                               std::chrono::milliseconds idle_ttl = std::chrono::milliseconds::zero(),
                                               std::uint64_t             sweep_every_n_calls = 0);
    std::shared_ptr<arpc::RPCChannelBase> getChannel(const std::string& ip, uint32_t port);

private:
    struct ChannelCacheEntry {
        std::shared_ptr<arpc::RPCChannelBase> channel;
        std::chrono::steady_clock::time_point last_used;
    };

    void                                  stop();
    std::shared_ptr<arpc::RPCChannelBase> openChannel(const std::string& spec);
    /// @pre \p channel_map_mutex_ is held by the caller; this method does not lock.
    void evictIdleExpired(std::chrono::steady_clock::time_point now);

    std::mutex                                         channel_map_mutex_;
    std::unordered_map<std::string, ChannelCacheEntry> channel_map_;

    std::chrono::steady_clock::duration channel_idle_ttl_{std::chrono::steady_clock::duration::zero()};
    std::uint64_t                       sweep_every_n_calls_{0};
    std::uint64_t                       get_channel_call_counter_{0};

    std::unique_ptr<anet::Transport>             rpc_channel_transport_;
    std::shared_ptr<arpc::ANetRPCChannelManager> rpc_channel_manager_;
};
}  // namespace transfer

}  // namespace rtp_llm
