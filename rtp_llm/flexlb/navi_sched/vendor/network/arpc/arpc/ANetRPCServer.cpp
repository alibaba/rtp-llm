#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCServer.h"

#include <assert.h>
#include <cstddef>
#include <google/protobuf/message.h>
#include <memory>
#include <new>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "rtp_llm/flexlb/navi_sched/vendor/autil/autil/Lock.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/delaydecodepacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/globalflags.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/PacketArg.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerAdapter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerClosure.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetApp.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetRPCMessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetRPCServerAdapter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/common/Exception.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

using namespace std;
using namespace anet;

ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(ANetRPCServer);

using namespace common;

ANetRPCServer::ANetRPCServer(Transport *transport, size_t threadNum, size_t queueSize)
    : RPCServer(threadNum, queueSize), _anetApp(transport) {
    ANetRPCMessageCodec *codec = new ANetRPCMessageCodec(_anetApp.GetPacketFactory());
    _messageCodec = codec;
    _serverAdapter = new ANetRPCServerAdapter(this, codec);
}

ANetRPCServer::~ANetRPCServer() {}

bool ANetRPCServer::Listen(const std::string &address, int timeout, int maxIdleTime, int backlog) {
    IOComponent *ioComponent =
        _anetApp.Listen(address, (ANetRPCServerAdapter *)_serverAdapter, timeout, maxIdleTime, backlog);

    if (NULL == ioComponent) {
        ARPC_LOG(ERROR, "listen on %s failed", address.c_str());
        return false;
    }

    autil::ScopedLock lock(_ioComponentMutex);
    _ioComponentList.push_back(ioComponent);
    return true;
}

bool ANetRPCServer::Listen(uint16_t port, int timeout, int maxIdleTime, int backlog) {
    IOComponent *ioComponent =
        _anetApp.Listen(port, (ANetRPCServerAdapter *)_serverAdapter, timeout, maxIdleTime, backlog);

    if (NULL == ioComponent) {
        ARPC_LOG(ERROR, "listen on %d failed", port);
        return false;
    }

    autil::ScopedLock lock(_ioComponentMutex);
    _ioComponentList.push_back(ioComponent);
    return true;
}

bool ANetRPCServer::Close() {
    Exception e;
    e.Init("", "", 0);
    ARPC_LOG(DEBUG, "[%p] RPCServer::Close: %s", this, e.what());
    if (_metricReporter) {
        _metricReporter->stop();
    }
    {
        autil::ScopedLock lock(_ioComponentMutex);
        list<IOComponent *>::iterator it = _ioComponentList.begin();

        for (; it != _ioComponentList.end(); ++it) {
            (*it)->close();
            (*it)->subRef();
        }

        _ioComponentList.clear();
    }
    if (_metricReporter) {
        SetMetricReporter({});
    }
    return true;
}

void ANetRPCServer::dump(std::ostringstream &out) {
    RPCServer::dump(out);
    // dump iocomponent
    {
        autil::ScopedLock lock(_ioComponentMutex);
        out << "IOComponent Count: " << _ioComponentList.size() << endl;

        for (list<IOComponent *>::iterator it = _ioComponentList.begin(); it != _ioComponentList.end(); ++it) {
            (*it)->dump(out);
        }
    }
}

int ANetRPCServer::GetListenPort() {
    if (_ioComponentList.empty()) {
        ARPC_LOG(ERROR, "io component list is empty");
        return -1;
    }
    anet::IOComponent *ioComponent = _ioComponentList.front();
    Socket *socket = ioComponent->getSocket();
    if (!socket) {
        ARPC_LOG(ERROR, "get socket failed");
        return -1;
    }
    return socket->getPort();
}

void ANetRPCServer::SetMetricReporter(const std::shared_ptr<ServerMetricReporter> &metricReporter) {
    _metricReporter = metricReporter;
    if (_serverAdapter != nullptr) {
        auto anetAdapter = dynamic_cast<ANetRPCServerAdapter *>(_serverAdapter);
        if (anetAdapter != nullptr) {
            anetAdapter->SetMetricReporter(metricReporter);
        }
    }
}

ARPC_END_NAMESPACE(arpc)
