#pragma once
#include <stddef.h>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/common.h"

namespace anet {
class Packet;
} // namespace anet

BEGIN_ANET_NS();
class PacketQueue {
public:
    PacketQueue();
    virtual ~PacketQueue();

public:
    Packet *pop();
    virtual void push(Packet *packet);
    size_t size();
    bool empty();

    /**
     * move all entries of current queue to the tail of destQueue
     */
    virtual void moveTo(PacketQueue *destQueue);

    /**
     * move all entries of  srcQueue back to the head of current queue
     */
    virtual void moveBack(PacketQueue *srcQueue);

    Packet *getTimeoutList(int64_t now);

protected:
    Packet *_head;
    Packet *_tail;
    size_t _size;
    friend class OrderedPacketQueue;
};

END_ANET_NS();

