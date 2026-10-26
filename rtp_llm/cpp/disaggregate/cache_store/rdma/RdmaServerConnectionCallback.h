#pragma once

#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection_callback.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xmessage.h"

namespace rtp_llm {

class RdmaServerHostConnectionsMap;

class RdmaServerConnectionCallback: public ::accl::barex::XConnectionCallback {
public:
    RdmaServerConnectionCallback(const std::shared_ptr<RdmaServerHostConnectionsMap>& host_connections_map);

public:
    void OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>& conn,
                    const ::accl::barex::XMessage&                     msgs) override;
    void OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& conn) override;
    void OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& conn) override;

private:
    std::weak_ptr<RdmaServerHostConnectionsMap> host_connections_map_;
};

}  // namespace rtp_llm