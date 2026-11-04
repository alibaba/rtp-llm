/**
 * File name: httppacketfactory.cpp
 * Author: zhangli
 * Create time: 2008-12-19 16:36:39
 * $Id$
 *
 * Description: ***add description here***
 *
 */

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httppacketfactory.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httppacket.h"

namespace anet {
class Packet;

HTTPPacketFactory::HTTPPacketFactory() {}

Packet *HTTPPacketFactory::createPacket(int pcode) { return new HTTPPacket; }

} /*end namespace anet*/
