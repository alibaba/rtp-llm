#pragma once

#include "aios/network/anet/transport.h"
#include "aios/network/arpc/arpc/ANetRPCChannelManager.h"

namespace rtp_llm {

namespace transfer {
class TcpClient {
public:
    TcpClient() = default;
    ~TcpClient();

public:
    bool                                  init(int io_thread_count);
    std::shared_ptr<arpc::RPCChannelBase> getChannel(const std::string& ip, uint32_t port);

private:
    void                                  stop();
    std::shared_ptr<arpc::RPCChannelBase> openChannel(const std::string& spec);

private:
    // TODO: optimize to mutil channel for same host, one channel may affect performance
    std::mutex                                                             channel_map_mutex_;
    std::unordered_map<std::string, std::shared_ptr<arpc::RPCChannelBase>> channel_map_;

    std::unique_ptr<anet::Transport>             rpc_channel_transport_;
    std::shared_ptr<arpc::ANetRPCChannelManager> rpc_channel_manager_;
};
}  // namespace transfer

}  // namespace rtp_llm
