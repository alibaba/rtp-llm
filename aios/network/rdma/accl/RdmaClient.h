#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/RdmaClientConfig.h"
#include "autil/Lock.h"
#include "autil/Log.h"

ARPC_BEGIN_NAMESPACE(arpc)

class RdmaChannelClient;
class TimerManager;
class AcclClientWrapper;

class RdmaClient {
public:
    RdmaClient(const RdmaClientConfig& config);
    virtual ~RdmaClient();

public:
    bool Init();

    // Return RdmaChannelClient to Use
    // address should be like tcp:1.2.3.4:12345
    std::shared_ptr<RdmaChannelClient> OpenChannel(const std::string& address);
    const RdmaClientConfig&            config() const {
        return _config;
    }

private:
    RdmaClientConfig _config;

    std::shared_ptr<AcclClientWrapper> _client;
    std::shared_ptr<TimerManager>      _timerManager;

private:
    AUTIL_LOG_DECLARE();
    friend class RdmaClientTest;
};

ARPC_END_NAMESPACE(arpc)