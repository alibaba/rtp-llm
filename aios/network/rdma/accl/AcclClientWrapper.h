#pragma once

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xclient_transport.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/RdmaClientConfig.h"
#include "autil/Log.h"

namespace accl {
namespace barex {
class XClientTransport;
}
}  // namespace accl

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaMempool;
class AcclClientWrapper {
public:
    AcclClientWrapper(const RdmaClientConfig& config);

    virtual ~AcclClientWrapper();

public:
    bool Init();

    // 只支持一次连接, 不支持重连
    virtual bool Connect(::accl::barex::TcpProtoType                                protocol,
                         std::string                                                ip,
                         uint16_t                                                   port,
                         const std::shared_ptr<::accl::barex::XConnectionCallback>& callback);

    virtual void Disconnect(const std::shared_ptr<::accl::barex::XConnection>& connection);

    std::shared_ptr<RdmaMempool> GetMempool() const {
        return _mempool;
    }

protected:
    RdmaClientConfig                                 _config;
    std::shared_ptr<RdmaMempool>                     _mempool;
    std::shared_ptr<::accl::barex::XClientTransport> _client;

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc);