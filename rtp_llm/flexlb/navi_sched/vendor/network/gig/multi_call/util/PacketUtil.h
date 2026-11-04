#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

class PacketUtil
{
public:
    PacketUtil();
    ~PacketUtil();

private:
    PacketUtil(const PacketUtil &);
    PacketUtil &operator=(const PacketUtil &);

public:
    static void deletePacket(anet::Packet *packet);
    static bool isControlPacket(anet::Packet *packet, int32_t &ec);

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call

