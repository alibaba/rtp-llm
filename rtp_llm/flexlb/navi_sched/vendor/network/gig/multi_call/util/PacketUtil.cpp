#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/PacketUtil.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/controlpacket.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, PacketUtil);

PacketUtil::PacketUtil() {
}

PacketUtil::~PacketUtil() {
}

bool PacketUtil::isControlPacket(anet::Packet *packet, int32_t &ec) {
    if (!packet) {
        return true;
    }
    if (!packet->isRegularPacket()) {
        auto controlPacket = dynamic_cast<anet::ControlPacket *>(packet);
        if (controlPacket) {
            AUTIL_LOG(WARN, "receive control packet: %s", controlPacket->what());
        }
        ec = controlPacket->getCommand();
        return true;
    }
    return false;
}

void PacketUtil::deletePacket(anet::Packet *packet) {
    if (packet) {
        packet->free();
    }
}

} // namespace multi_call
