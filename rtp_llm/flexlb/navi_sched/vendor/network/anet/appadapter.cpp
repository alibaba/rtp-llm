#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/appadapter.h"

#include <iosfwd>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transport.h"

namespace anet {
class IConnection;
class IPacketStreamer;
} // namespace anet

using namespace std;

namespace anet {

AnetAcceptor::AnetAcceptor(
    const string &address, Transport *transport, IPacketStreamer *streamer, int postPacketTimeout, int maxIdleTime)
    : mAddress(address)
    , mTransport(transport)
    , mStreamer(streamer)
    , mPostPacketTimeout(postPacketTimeout)
    , mMaxIdleTime(maxIdleTime) {}

void AnetAcceptor::HandleAccept() {
    mTransport->listen(mAddress.c_str(), mStreamer, this, mPostPacketTimeout, mMaxIdleTime);
}

AnetConnector::AnetConnector(const string &address, Transport *transport, IPacketStreamer *streamer, bool autoReconn)
    : mAddress(address), mTransport(transport), mStreamer(streamer), mAutoReconn(autoReconn) {}

IConnection *AnetConnector::Connect() { return mTransport->connect(mAddress.c_str(), mStreamer, mAutoReconn); }

} // namespace anet
