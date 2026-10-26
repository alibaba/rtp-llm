#pragma once

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "autil/Log.h"

namespace accl {
namespace barex {
class XConnection;
}
}  // namespace accl

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaMessage;
class RdmaMempool;

class RdmaRequest {
public:
    RdmaRequest();
    virtual ~RdmaRequest();

public:
    virtual std::shared_ptr<RdmaMessage> Encode()                                        = 0;
    virtual bool                         Decode(const std::shared_ptr<RdmaMessage>& msg) = 0;

public:
    void SetTimeoutMs(double timeoutMs) {
        _timeoutMs = timeoutMs;
    }
    double GetTimeoutMs() const {
        return _timeoutMs;
    }

    void SetRequestId(uint32_t requestId) {
        _requestId = requestId;
    }
    uint32_t GetRequestId() const {
        return _requestId;
    }

    void SetConnection(const std::shared_ptr<::accl::barex::XConnection>& connection) {
        _connection = connection;
    }
    const std::shared_ptr<::accl::barex::XConnection>& GetConnection() const {
        return _connection;
    }

    void SetMempool(const std::shared_ptr<RdmaMempool>& mempool) {
        _mempool = mempool;
    }
    const std::shared_ptr<RdmaMempool>& GetMempool() const {
        return _mempool;
    }

    const std::string& GetPeerAddress() const {
        return _peerAddress;
    }
    void SetPeerAddress(const std::shared_ptr<::accl::barex::XConnection>& connection);

protected:
    double                                      _timeoutMs;
    uint32_t                                    _requestId;
    std::shared_ptr<::accl::barex::XConnection> _connection;
    std::shared_ptr<RdmaMempool>                _mempool;
    std::string                                 _peerAddress;

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc)