#pragma once
/**
 * File name: defaultpacket.h
 * Author: zhangli
 * Create time: 2008-12-25 11:10:50
 * $Id$
 *
 * Description: ***add description here***
 *
 */

#include <stddef.h>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"

namespace anet {
class DataBuffer;

class DefaultPacket : public Packet {
public:
    DefaultPacket();
    ~DefaultPacket();

    bool setBody(const char *, size_t);
    bool appendBody(const char *, size_t);
    const char *getBody(size_t &) const;
    char *getBody(size_t &);
    const char *getBody() const;
    char *getBody();

    bool encode(DataBuffer *output) override;
    bool decode(DataBuffer *input, PacketHeader *header) override;

    int64_t getSpaceUsed() override;
    size_t getBodyLen() const;
    bool setCapacity(size_t size);
    size_t getCapacity();

protected:
    char *_body;
    size_t _bodyLength;
    size_t _capacity;
};

} /*end namespace anet*/
