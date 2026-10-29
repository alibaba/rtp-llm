#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection_callback.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xmessage.h"

namespace rtp_llm {

class RdmaClientHostConnections;

class RdmaClientConnectionCallback: public ::accl::barex::XConnectionCallback {
public:
    RdmaClientConnectionCallback(const std::shared_ptr<RdmaClientHostConnections>& connection);

public:
    void OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>& conn,
                    const ::accl::barex::XMessage&                     msgs) override;
    void OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& conn) override;
    void OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& conn) override;

private:
    std::weak_ptr<RdmaClientHostConnections> connection_;
};

}  // namespace rtp_llm