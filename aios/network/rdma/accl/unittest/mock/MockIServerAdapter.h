#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/IRdmaServerAdapter.h"
#include "unittest/unittest.h"

ARPC_BEGIN_NAMESPACE(arpc);

class MockIRdmaServerAdapter: public IRdmaServerAdapter {
public:
    virtual ~MockIRdmaServerAdapter() {}

public:
    MOCK_METHOD3(HandlePacket,
                 int(const std::shared_ptr<RdmaMessage>&,
                     const std::shared_ptr<RdmaChannelServer>&,
                     const std::shared_ptr<ServerRPCStats>&));
};

ARPC_END_NAMESPACE(arpc);