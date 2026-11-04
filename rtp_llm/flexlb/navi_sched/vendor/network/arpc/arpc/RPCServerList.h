#pragma once
#include <sstream>

#include "rtp_llm/flexlb/navi_sched/vendor/autil/autil/Lock.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"

ARPC_BEGIN_NAMESPACE(arpc);

class RPCServer;

const int MAX_RPCSERVERS = 1024;

typedef RPCServer *RPCServerListT[MAX_RPCSERVERS];

RPCServerListT &getRPCServerList();
autil::ThreadMutex &getRPCServerListMutex();

bool addRPCServerToList(RPCServer *);
bool delRPCServerFromList(RPCServer *);
int findRPCServerFromList(RPCServer *);
int getRPCServerCount(void);

int dumpRPCServerList(std::ostringstream &out);

ARPC_END_NAMESPACE(arpc);

