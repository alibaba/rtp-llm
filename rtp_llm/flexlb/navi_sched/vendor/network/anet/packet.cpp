#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/channel.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipackethandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/timeutil.h"

BEGIN_ANET_NS();

Packet::Packet() {
    _next = NULL;
    _channel = NULL;
    _expireTime = 0;
    _timeoutMs = 0;
    _packetDequeueCB = NULL;
    memset(&_packetHeader, 0, sizeof(PacketHeader));
}

Packet::~Packet() {}

void Packet::setChannel(Channel *channel) {
    if (channel) {
        _channel = channel;
        setChannelId(channel->getId());
        _channel->setId(getChannelId());
    }
}

void Packet::setPcode(int32_t pcode) { _packetHeader._pcode = pcode; }

int32_t Packet::getPcode() { return _packetHeader._pcode; }

void Packet::setExpireTime(int milliseconds) {
    if (milliseconds < 0) {
        _expireTime = TimeUtil::PRE_MAX;
    } else if (milliseconds == 0) {
        _expireTime = 0;
    } else { /**@todo this design is not testing friendly*/
        _timeoutMs = milliseconds;
        _expireTime = TimeUtil::getTime() + static_cast<int64_t>(milliseconds) * static_cast<int64_t>(1000);
    }
}

void Packet::invokeDequeueCB(void) {
    if (_packetDequeueCB != NULL)
        (void)_packetDequeueCB->handlePacket(this, NULL);
}

END_ANET_NS();
