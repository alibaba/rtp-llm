#pragma once
/**
 * File name: orderedpacketqueue.h
 * Author: huangrh
 * Create time: 2010-11-22 14:27:14
 * $Id$
 *
 * Description: ***add description here***
 *
 */

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packetqueue.h"

namespace anet {
class Packet;
} // namespace anet

BEGIN_ANET_NS();
class OrderedPacketQueue : public PacketQueue {
public:
    OrderedPacketQueue();
    ~OrderedPacketQueue();

public:
    virtual void push(Packet *packet);
    virtual void moveTo(PacketQueue *destQueue);
    virtual void moveBack(PacketQueue *srcQueue);
};

END_ANET_NS();
