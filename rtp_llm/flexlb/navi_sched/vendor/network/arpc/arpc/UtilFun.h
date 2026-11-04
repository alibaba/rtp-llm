#pragma once
#include <memory>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"

ARPC_BEGIN_NAMESPACE(arpc);

ErrorMsg *
BuildErrorMsg(const std::string &errMsg, ErrorCode errCode, const std::shared_ptr<google::protobuf::Arena> &arena);

anet::DelayDecodePacket *BuildPacket(anet::PacketHeader *header, MessageSerializable *pSeri, bool ownSeri = true);

void DecodePacket(ANetRPCController *pController,
                  anet::DelayDecodePacket *pPacket,
                  RPCMessage *message,
                  const std::shared_ptr<google::protobuf::Arena> &arenaPtr,
                  bool bResponse = true);

ARPC_END_NAMESPACE(arpc);

