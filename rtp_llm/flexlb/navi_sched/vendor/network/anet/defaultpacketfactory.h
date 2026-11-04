#pragma once
/**
 * File name: defaultpacketfactory.h
 * Author: zhangli
 * Create time: 2008-12-25 11:38:19
 * $Id$
 *
 * Description: ***add description here***
 *
 */

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketfactory.h"

namespace anet {
class Packet;
class DefaultPacketFactory : public IPacketFactory {
public:
    Packet *createPacket(int pcode);
};

} /*end namespace anet*/
