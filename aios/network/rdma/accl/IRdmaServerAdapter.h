#pragma once

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaMessage;
class RdmaChannelServer;
class ServerRPCStats;

class IRdmaServerAdapter {
public:
    // Handle received request in msg
    // if handle success, should return 0 and user release msg
    // if handle failed, should return -1 and user should not release msg, (msg will needed for send error response)
    virtual int HandlePacket(const std::shared_ptr<RdmaMessage>&       msg,
                             const std::shared_ptr<RdmaChannelServer>& channel,
                             const std::shared_ptr<ServerRPCStats>&    stats) = 0;
    virtual ~IRdmaServerAdapter() {}
};

ARPC_END_NAMESPACE(arpc);