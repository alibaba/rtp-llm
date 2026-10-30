#include "aios/network/rdma/accl/AcclServerWrapper.h"

#include "aios/autil/autil/EnvUtil.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection_callback.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xmessage.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xserver_transport.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xstats_reporter.h"
#include "aios/network/accl-barex/include/accl/barex/xstats.h"
#include "aios/network/arpc/arpc/SyncClosure.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"

ARPC_BEGIN_NAMESPACE(arpc);
AUTIL_LOG_SETUP(arpc, AcclServerWrapper);

AcclServerWrapper::AcclServerWrapper(const RdmaServerConfig& config): _config(config) {
    RdmaMempoolFactory::Init();
}

AcclServerWrapper::~AcclServerWrapper() {
    RdmaMempoolFactory::Shutdown();
}

bool AcclServerWrapper::Init() {
    _mempool = RdmaMempoolFactory::GetMempool();
    if (!_mempool) {
        AUTIL_LOG(WARN, "rdma channel server init failed, new rdmamempool singleton failed");
        return false;
    }
    auto xmempool = _mempool->GetXMempool();
    if (!xmempool) {
        AUTIL_LOG(WARN, "rdma channel server init failed, get xmempool failed");
        return false;
    }

    ::accl::barex::XServerTransportConfig serverConfig(
        _config.rdmaIoThreadNum, _config.rdmaWorkerThreadNum, xmempool, _config.acclMetricReporter);
    _server = ::accl::barex::XServerTransport::NewInstance(serverConfig);
    if (!_server) {
        AUTIL_LOG(
            WARN,
            "rdma channel server init failed, new accl server failed, io thread count %lu, worker thread count %lu",
            _config.rdmaIoThreadNum,
            _config.rdmaWorkerThreadNum);
        return false;
    }

    return true;
}

bool AcclServerWrapper::Listen(::accl::barex::TcpProtoType                                protocol,
                               uint32_t                                                   port,
                               const std::shared_ptr<::accl::barex::XConnectionCallback>& callback) {
    if (_server && _server->Listen(protocol, port, callback) == ::accl::barex::BAREX_SUCCESS) {
        AUTIL_LOG(INFO, "accl server listen port %d success", port);
        return true;
    }
    AUTIL_LOG(INFO, "accl server listen port %d failed", port);
    return false;
}

ARPC_END_NAMESPACE(arpc);