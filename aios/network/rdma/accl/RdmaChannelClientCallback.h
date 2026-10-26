#pragma once

#include <memory>

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection_callback.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xmessage.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "autil/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelClient;
class RdmaMempool;

// Class used to process receive request
class RdmaChannelClientCallback: public ::accl::barex::XConnectionCallback {
public:
    RdmaChannelClientCallback(const std::weak_ptr<RdmaChannelClient>& channel,
                              const std::shared_ptr<RdmaMempool>&     rdmaMempool);
    ~RdmaChannelClientCallback();

public:
    virtual void OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>& connection,
                            const ::accl::barex::XMessage&                     msgs) override;
    virtual void OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) override;
    virtual void OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection) override;

private:
    std::weak_ptr<RdmaChannelClient> _channel;
    std::shared_ptr<RdmaMempool>     _mempool;

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc);
