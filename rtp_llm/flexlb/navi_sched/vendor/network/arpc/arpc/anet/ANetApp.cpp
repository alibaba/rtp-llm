#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetApp.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/defaultpacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transport.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

using namespace std;
using namespace anet;

ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(ANetApp);

ANetApp::ANetApp(Transport *transport) : _streamer(&_factory) {
    _ownTransport = false;

    if (transport == NULL) {
        _transport = new Transport();
        _ownTransport = true;
    } else {
        _transport = transport;
    }
}

ANetApp::~ANetApp() {
    if (_ownTransport) {
        StopPrivateTransport();
        delete _transport;
        _transport = NULL;
    }
}

IOComponent *
ANetApp::Listen(const std::string &address, IServerAdapter *serverAdapter, int timeout, int maxIdleTime, int backlog) {
    return _transport->listen(address.c_str(), &_streamer, serverAdapter, timeout, maxIdleTime, backlog);
}

IOComponent *
ANetApp::Listen(uint16_t port, IServerAdapter *serverAdapter, int timeout, int maxIdleTime, int backlog) {
    return _transport->listen(port, &_streamer, serverAdapter, timeout, maxIdleTime, backlog);
}

Connection *ANetApp::Connect(const std::string &address, bool autoReconn, anet::CONNPRIORITY prio) {
    return _transport->connect(address.c_str(), &_streamer, autoReconn, prio);
}

Connection *
ANetApp::Connect(const std::string &address, const std::string &bindAddr, bool autoReconn, anet::CONNPRIORITY prio) {
    return _transport->connectWithAddr(bindAddr.c_str(), address.c_str(), &_streamer, autoReconn, prio);
}

bool ANetApp::StartPrivateTransport() { return StartPrivateTransport("Arpc"); }

bool ANetApp::StartPrivateTransport(const std::string &name) {
    if (!_ownTransport) {
        return false;
    }

    _transport->start();
    _transport->setName(name);
    return true;
}

bool ANetApp::StopPrivateTransport() {
    if (!_ownTransport) {
        return false;
    }

    _transport->stop();
    _transport->wait();

    return true;
}

void ANetApp::dump(std::ostringstream &out) {
    if (_transport) {
        _transport->dump(out);
    }
}

ARPC_END_NAMESPACE(arpc);
