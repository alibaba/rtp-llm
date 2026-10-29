#include "aios/network/rdma/accl/AcclClientWrapper.h"

#include "aios/network/accl-barex/include/accl/barex/barex.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection_callback.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"

ARPC_BEGIN_NAMESPACE(arpc);
AUTIL_LOG_SETUP(arpc, AcclClientWrapper);

AcclClientWrapper::AcclClientWrapper(const RdmaClientConfig& config): _config(config) {
    RdmaMempoolFactory::Init();
}

AcclClientWrapper::~AcclClientWrapper() {
    RdmaMempoolFactory::Shutdown();
}

bool AcclClientWrapper::Init() {
    _mempool = RdmaMempoolFactory::GetMempool();
    if (!_mempool) {
        AUTIL_LOG(ERROR, "rdma channel init failed, new rdma mempool singleton failed");
        return false;
    }

    auto xmempool = _mempool->GetXMempool();
    if (!xmempool) {
        AUTIL_LOG(ERROR, "rdma channel init failed, get xmempool failed");
        return false;
    }

    ::accl::barex::XClientTransportConfig clientConfig(
        _config.rdmaIoThreadNum, _config.rdmaWorkerThreadNum, xmempool, _config.statsReporter);
    _client = ::accl::barex::XClientTransport::NewInstance(clientConfig);
    if (!_client) {
        AUTIL_LOG(ERROR, "rdma channel init failed, accl client init failed");
        return false;
    }
    return true;
}

bool AcclClientWrapper::Connect(::accl::barex::TcpProtoType                                protocol,
                                std::string                                                ip,
                                uint16_t                                                   port,
                                const std::shared_ptr<::accl::barex::XConnectionCallback>& callback) {
    if (_client) {
        _client->Connect(protocol, ip, port, callback);
        return true;
    }
    return false;
}

void AcclClientWrapper::Disconnect(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    if (_client && connection) {
        if (!_client->Disconnect(connection)) {
            AUTIL_LOG(ERROR, "rdma channel disconnect failed");
        }
    }
}

ARPC_END_NAMESPACE(arpc);