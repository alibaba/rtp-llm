/**
 * File name: defaultpacketfactory.cpp
 * Author: zhangli
 * Create time: 2008-12-25 11:38:19
 * $Id$
 *
 * Description: ***add description here***
 *
 */

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/defaultpacketfactory.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/defaultpacket.h"

namespace anet {
Packet *DefaultPacketFactory::createPacket(int pcode) { return new DefaultPacket; }

} /*end namespace anet*/
