#include "aios/network/rdma/accl/RdmaClient.h"

#include "aios/network/anet/addrspec.h"
#include "aios/network/rdma/accl/AcclClientWrapper.h"
#include "aios/network/rdma/accl/RdmaChannelClient.h"
#include "aios/network/rdma/accl/TimerManager.h"
#include "autil/StringUtil.h"

ARPC_BEGIN_NAMESPACE(arpc)
AUTIL_LOG_SETUP(arpc, RdmaClient);

RdmaClient::RdmaClient(const RdmaClientConfig& config): _config(config) {}

RdmaClient::~RdmaClient() {}

bool RdmaClient::Init() {
    _timerManager = std::make_shared<TimerManager>();
    _client       = std::make_shared<AcclClientWrapper>(_config);
    if (!_timerManager || !_client) {
        return false;
    }
    return _client->Init();
}

std::shared_ptr<RdmaChannelClient> RdmaClient::OpenChannel(const std::string& address) {
    anet::AddrSpec addrspec;
    if (addrspec.setAddrSpec(address.c_str()) != 0) {
        AUTIL_LOG(WARN, "rdma channel openChannel %s failed, parse address failed", address.c_str());
        return std::shared_ptr<RdmaChannelClient>();
    }

    std::shared_ptr<RdmaChannelClient> channel = std::make_shared<RdmaChannelClient>(
        addrspec.getProtocolFamily() == AF_INET6 ? ::accl::barex::TCP_V6 : ::accl::barex::TCP_V4,
        std::string(inet_ntoa(in_addr({addrspec.getNetworkAddr()}))),
        addrspec.getPort(),
        _timerManager,
        _client);
    channel->Init();
    return channel;
}

ARPC_END_NAMESPACE(arpc)