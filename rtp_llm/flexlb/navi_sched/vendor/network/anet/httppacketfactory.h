#pragma once
/**
 * File name: httppacketfactory.h
 * Author: zhangli
 * Create time: 2008-12-19 16:36:39
 * $Id$
 *
 * Description: ***add description here***
 *
 */

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketfactory.h"

namespace anet {
class Packet;

class HTTPPacketFactory : public IPacketFactory {
public:
    HTTPPacketFactory();
    Packet *createPacket(int pcode);
};

} /*end namespace anet*/
