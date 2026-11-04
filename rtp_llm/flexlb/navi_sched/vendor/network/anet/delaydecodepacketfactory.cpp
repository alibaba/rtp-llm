/**
 * File name: delaydecodepacketfactory.cpp
 * Author: lizhang
 * Create time: 2010-11-07 18:38:34
 * $Id$
 *
 * Description: ***add description here***
 *
 */

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/delaydecodepacketfactory.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/delaydecodepacket.h"

namespace anet {
class Packet;

DelayDecodePacketFactory::DelayDecodePacketFactory() {}

DelayDecodePacketFactory::~DelayDecodePacketFactory() {}

Packet *DelayDecodePacketFactory::createPacket(int /*pcode*/) { return new DelayDecodePacket; }

} /*end namespace anet*/
