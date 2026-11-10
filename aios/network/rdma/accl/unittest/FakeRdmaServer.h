#pragma once

#include "aios/network/rdma/accl/RdmaServer.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaServerAdapter.h"

ARPC_BEGIN_NAMESPACE(arpc)

class FakeRdmaServer {
public:
    FakeRdmaServer(const RdmaServerConfig& config);

public:
    bool Listen(int port);

protected:
    std::shared_ptr<FakeRdmaServerAdapter> _adapter;
    std::shared_ptr<RdmaServer>            _server;
};

ARPC_END_NAMESPACE(arpc)