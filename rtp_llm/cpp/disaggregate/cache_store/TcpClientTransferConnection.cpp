#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpTransferConnection.h"

namespace rtp_llm {

std::shared_ptr<TransferConnection>
TcpClient::getTransferConnection(const std::string& ip, uint32_t port, int device_id) {
    auto channel = getChannel(ip, port);
    if (channel == nullptr) {
        return nullptr;
    }
    return std::make_shared<TcpTransferConnection>(channel, device_id);
}

}  // namespace rtp_llm
