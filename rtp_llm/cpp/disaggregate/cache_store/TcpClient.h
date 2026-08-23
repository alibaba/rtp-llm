#pragma once

#include "aios/network/anet/transport.h"
#include "aios/network/arpc/arpc/ANetRPCChannelManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpTransferConnection.h"

namespace rtp_llm {

class TcpClient {
public:
    TcpClient() = default;
    ~TcpClient();

public:
    bool                                  init(int io_thread_count);
    std::shared_ptr<arpc::RPCChannelBase> getChannel(const std::string& ip, uint32_t port);
    void                                  invalidateChannel(const std::string& ip, uint32_t port);
    std::shared_ptr<TransferConnection>   getTransferConnection(const std::string& ip, uint32_t port, int device_id);

private:
    void                                  stop();
    std::shared_ptr<arpc::RPCChannelBase> openChannel(const std::string& spec);

private:
    struct ChannelEntry {
        std::shared_ptr<arpc::RPCChannelBase> channel;
        int64_t                               last_use_ms{0};
    };

    // TODO: optimize to mutil channel for same host, one channel may affect performance
    std::mutex                                     channel_map_mutex_;
    std::unordered_map<std::string, ChannelEntry> channel_map_;
    // cached channels idle longer than this are dropped on next use, defending against
    // connections silently killed by middleboxes (LB/NAT idle timeout), 0 disables eviction
    int64_t channel_idle_timeout_ms_{60000};

    std::unique_ptr<anet::Transport>             rpc_channel_transport_;
    std::shared_ptr<arpc::ANetRPCChannelManager> rpc_channel_manager_;
};

}  // namespace rtp_llm
