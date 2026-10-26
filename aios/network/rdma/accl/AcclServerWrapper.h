#pragma once

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/RdmaServerConfig.h"
#include "autil/Log.h"

namespace accl {
namespace barex {
class XServerTransport;
class XConnection;
class XConnectionCallback;
class XMessage;
}  // namespace barex
}  // namespace accl

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaMessage;
class RdmaMempool;

class AcclServerWrapper {
public:
    AcclServerWrapper(const RdmaServerConfig& config);
    virtual ~AcclServerWrapper();

public:
    virtual bool Listen(::accl::barex::TcpProtoType                                protocol,
                        uint32_t                                                   port,
                        const std::shared_ptr<::accl::barex::XConnectionCallback>& callback);

    bool Init();

    const std::shared_ptr<RdmaMempool>& GetMempool() const {
        return _mempool;
    }
    void SetMempool(const std::shared_ptr<RdmaMempool>& mempool) {
        _mempool = mempool;
    }

protected:
    AcclServerWrapper() {}

private:
    RdmaServerConfig _config;

    std::shared_ptr<RdmaMempool>                     _mempool;
    std::shared_ptr<::accl::barex::XServerTransport> _server;

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc);