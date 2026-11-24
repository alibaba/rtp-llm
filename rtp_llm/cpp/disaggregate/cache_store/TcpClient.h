#pragma once

#include "aios/network/anet/transport.h"
#include "aios/network/arpc/arpc/ANetRPCChannelManager.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace rtp_llm {

class TransferConnection;

class TcpClient {
public:
    TcpClient() = default;
    ~TcpClient();

public:
    bool                                  init(int io_thread_count);
    std::shared_ptr<arpc::RPCChannelBase> getChannel(const std::string& ip, uint32_t port);
    std::shared_ptr<TransferConnection>   getTransferConnection(const std::string& ip, uint32_t port, int device_id);

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

}  // namespace rtp_llm
