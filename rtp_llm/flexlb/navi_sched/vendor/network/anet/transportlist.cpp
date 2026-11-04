#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transportlist.h"

#include <ostream>
#include <stddef.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/debug.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadmutex.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transport.h"

using namespace std;

namespace anet {

/* The variable to remember current transport count in list. */
static int TransportCount = 0;

#define INC_TRANSPORT() (TransportCount++)
#define DEC_TRANSPORT() (TransportCount--)
TransportListT &GetTransportList() {
    /* We rely on the fact that static data will be cleared to
     * NULL by the compiler. */
    static TransportListT TransportList;
    return TransportList;
}

ThreadMutex &GetTransportListMutex() {
    static ThreadMutex TransportListMutex;
    return TransportListMutex;
}

int getTransportCount() {
    ThreadMutex &TransportListMutex = GetTransportListMutex();
    TransportListMutex.lock();
    int rc = TransportCount;
    TransportListMutex.unlock();
    return rc;
}

bool addTransportToList(Transport *t) {
    bool added = false;
    int i = 0;
    TransportListT &TransportList = GetTransportList();
    ThreadMutex &TransportListMutex = GetTransportListMutex();

    /* We don't want to add duplicated items. */
    if (findTransportFromList(t) < 0) {
        TransportListMutex.lock();
        for (i = 0; i < MAX_TRANSPORTS; i++) {
            if (TransportList[i] == NULL) {
                TransportList[i] = t;
                INC_TRANSPORT();
                added = true;
                break;
            }
        }
        TransportListMutex.unlock();
        DBGASSERT(getTransportCount() <= MAX_TRANSPORTS);
        if (i == MAX_TRANSPORTS) {
            ANET_LOG(WARN, "Slot number limits to %d. No more slots for Transport: %p", MAX_TRANSPORTS, t);
        }
    }
    return added;
}

bool delTransportFromList(Transport *t) {
    bool deleted = false;
    int i = 0;

    TransportListT &TransportList = GetTransportList();
    ThreadMutex &TransportListMutex = GetTransportListMutex();

    TransportListMutex.lock();
    for (i = 0; i < MAX_TRANSPORTS; i++) {
        if (TransportList[i] == t) {
            TransportList[i] = NULL;
            DEC_TRANSPORT();
            deleted = true;
            break;
        }
    }
    TransportListMutex.unlock();
    DBGASSERT(getTransportCount() >= 0);

    return deleted;
}

int findTransportFromList(Transport *t) {
    int i = 0;
    int rc = -1;

    TransportListT &TransportList = GetTransportList();
    ThreadMutex &TransportListMutex = GetTransportListMutex();

    TransportListMutex.lock();
    for (i = 0; i < MAX_TRANSPORTS; i++) {
        if (TransportList[i] == t) {
            rc = i;
            break;
        }
    }
    TransportListMutex.unlock();

    return rc;
}

int dumpTransportList(ostringstream &buf) {
    int i = 0;
    TransportListT &TransportList = GetTransportList();
    ThreadMutex &TransportListMutex = GetTransportListMutex();

    int count = getTransportCount();
    buf << "===============================Dump of Transports===============================" << endl
        << "Total count of transports: " << count << endl;
    if (count == MAX_TRANSPORTS) {
        buf << "[Warning]: Transport count is up to limit(" << MAX_TRANSPORTS << ")." << endl
            << "           Dumped information maybe incomplete." << endl;
    }

    TransportListMutex.lock();
    for (i = 0; i < MAX_TRANSPORTS; i++) {
        if (TransportList[i] != NULL)
            TransportList[i]->dump(buf);
    }
    TransportListMutex.unlock();

    return 0;
}

} // namespace anet
