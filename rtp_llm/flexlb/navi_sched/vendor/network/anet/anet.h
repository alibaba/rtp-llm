#pragma once
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/advancepacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/advancepacketfactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/controlpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databufferserializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/defaultpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/defaultpacketfactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/defaultpacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/delaydecodepacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/delaydecodepacketfactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httppacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httppacketfactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httpstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketfactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipackethandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iserveradapter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/streamingcontext.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/timeutil.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transport.h"

// DO NOT export interfaces about logging implicitly
//#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"

/**legacy http related header files*/
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httppacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httprequestpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httpresponsepacket.h"

// for direct packet
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directpacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directstreamingcontext.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directtcpconnection.h"

#define BEGIN_ANET_NS() namespace anet {
#define END_ANET_NS() }
#define USE_ANET_NS() using namespace anet;

BEGIN_ANET_NS()

/* Redefine advance packet type to make it easier to use. */
typedef AdvancePacket<DelayDecodePacket> AdvanceDelayDecodePacket;
typedef AdvancePacket<DefaultPacket> AdvanceDefaultPacket;
typedef AdvancePacketFactory<DelayDecodePacket> AdvanceDelayDecodePacketFactory;
typedef AdvancePacketFactory<DefaultPacket> AdvanceDefaultPacketFactory;

END_ANET_NS()

#define ANET_VERSION 040000
