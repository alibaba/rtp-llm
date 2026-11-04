#pragma once

#include <sstream>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadmutex.h"

namespace anet {

class Transport;

/**
 * Global list is maintained for all transport objects in a process.
 * In Transport constructor and destructor, the object will be linked to or
 * removed from the global transport list.
 * The purpose of this list is to provide an entry point so that the connection
 * state can get accessed.
 */

/* We assume that in a process, there will be no more than 1024 transports
 * get used in the same time. This assumption may become invalid one day and we
 * just need to increase the value then. */
#define MAX_TRANSPORTS 1024

/* Redefine the transport pointer array. */
typedef Transport *TransportListT[MAX_TRANSPORTS];

/* Group of helper functions to manipulate the operation of transport list. */
TransportListT &GetTransportList();
ThreadMutex &GetTransportListMutex();

bool addTransportToList(Transport *t);
bool delTransportFromList(Transport *t);
int findTransportFromList(Transport *t);

int getTransportCount(void);

/* Dump the content of all the active transports in the system. */
int dumpTransportList(std::ostringstream &buf);

} // namespace anet
