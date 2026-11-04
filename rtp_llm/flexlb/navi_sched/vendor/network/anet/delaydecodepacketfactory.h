#pragma once
/**
 * File name: delaydecodepacketfactory.h
 * Author: lizhang
 * Create time: 2010-11-07 18:34:30
 * $Id$
 *
 * Description: ***add description here***
 *
 */


#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketfactory.h"

namespace anet {
class Packet;

class DelayDecodePacketFactory : public IPacketFactory {
public:
    DelayDecodePacketFactory();

    ~DelayDecodePacketFactory();

    virtual Packet *createPacket(int pcode);
};

} /*end namespace anet*/
