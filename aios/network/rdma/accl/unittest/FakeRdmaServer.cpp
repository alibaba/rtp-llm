#include "aios/network/rdma/accl/unittest/FakeRdmaServer.h"

ARPC_BEGIN_NAMESPACE(arpc)

FakeRdmaServer::FakeRdmaServer(const RdmaServerConfig& config) {
    _adapter = std::make_shared<FakeRdmaServerAdapter>();

    _server = std::make_shared<RdmaServer>(config);
    if (!_server->Init()) {
        _server.reset();
    }
}

bool FakeRdmaServer::Listen(int port) {
    if (!_server) {
        return false;
    }

    auto address = "tcp:localhost:" + std::to_string(port);
    return _server->Listen(address, _adapter);
}

ARPC_END_NAMESPACE(arpc)