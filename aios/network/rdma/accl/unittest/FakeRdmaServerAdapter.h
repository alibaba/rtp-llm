#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/IRdmaServerAdapter.h"

ARPC_BEGIN_NAMESPACE(arpc)

class FakeRdmaServerAdapter: public IRdmaServerAdapter {
public:
    virtual int HandlePacket(const std::shared_ptr<RdmaMessage>&       message,
                             const std::shared_ptr<RdmaChannelServer>& channel,
                             const std::shared_ptr<ServerRPCStats>&    stats) override;
};

ARPC_END_NAMESPACE(arpc)