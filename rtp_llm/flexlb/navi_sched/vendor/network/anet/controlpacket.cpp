
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/controlpacket.h"

namespace anet {
ControlPacket ControlPacket::BadPacket(CMD_BAD_PACKET, false);
ControlPacket ControlPacket::TimeoutPacket(CMD_TIMEOUT_PACKET, false);
ControlPacket ControlPacket::ConnectionClosedPacket(CMD_CONNECTION_CLOSED, false);
ControlPacket ControlPacket::ReceiveNewConnectionPacket(CMD_RECEIVE_NEW_CONNECTION, false);
ControlPacket ControlPacket::CloseConnectionPacket(CMD_CLOSE_CONNECTION, false);

ControlPacket::ControlPacket(int c, bool freeDelete) {
    _command = c;
    _freeDelete = freeDelete;
}

void ControlPacket::free() {
    if (_freeDelete) {
        delete this;
    }
}

} // namespace anet
