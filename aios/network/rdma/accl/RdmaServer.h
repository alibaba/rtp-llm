#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/arpc/arpc/metric/ServerMetricReporter.h"
#include "aios/network/rdma/accl/RdmaServerConfig.h"
#include "autil/Lock.h"
#include "autil/Log.h"

ARPC_BEGIN_NAMESPACE(arpc)

class RdmaChannelServer;
class IRdmaServerAdapter;
class AcclServerWrapper;

class RdmaServer {
public:
    RdmaServer(const RdmaServerConfig& config);
    ~RdmaServer();

public:
    bool Listen(const std::string& address, const std::shared_ptr<IRdmaServerAdapter>& adapter);
    bool Init();

    int GetListenPort() const;

    const RdmaServerConfig& config() const {
        return _config;
    }

private:
    RdmaServerConfig _config;

    autil::ThreadMutex                                        _lock;
    std::map<std::string, std::shared_ptr<RdmaChannelServer>> _channels;
    std::shared_ptr<AcclServerWrapper>                        _server;

private:
    AUTIL_LOG_DECLARE();
    friend class RdmaServerTest;
};

ARPC_END_NAMESPACE(arpc)