#include "aios/network/rdma/accl/RdmaServer.h"

#include "aios/network/arpc/arpc/metric/ServerMetricReporter.h"
#include "aios/network/rdma/accl/AcclServerWrapper.h"
#include "aios/network/rdma/accl/IRdmaServerAdapter.h"
#include "aios/network/rdma/accl/RdmaChannelServer.h"

ARPC_BEGIN_NAMESPACE(arpc);
AUTIL_LOG_SETUP(arpc, RdmaServer);

RdmaServer::RdmaServer(const RdmaServerConfig& config): _config(config) {}

RdmaServer::~RdmaServer() {
    _channels.clear();
}

bool RdmaServer::Init() {
    _server = std::make_shared<AcclServerWrapper>(_config);
    if (!_server) {
        return false;
    }
    return _server->Init();
}

bool RdmaServer::Listen(const std::string& address, const std::shared_ptr<IRdmaServerAdapter>& adapter) {
    auto channel = std::make_shared<RdmaChannelServer>(address, adapter, _server, _config.rpcMetricReporter);
    if (!channel) {
        AUTIL_LOG(WARN, "rdma server listen %s failed, new rdma channel server failed", address.c_str());
        return false;
    }

    if (!channel->Listen()) {
        AUTIL_LOG(WARN, "rdma server listen %s failed, do listen failed", address.c_str());
        return false;
    }

    _lock.lock();
    _channels[address] = channel;
    _lock.unlock();
    return true;
}

int RdmaServer::GetListenPort() const {
    if (_channels.empty()) {
        return -1;
    }
    return _channels.begin()->second->GetListenPort();
}

ARPC_END_NAMESPACE(arpc);