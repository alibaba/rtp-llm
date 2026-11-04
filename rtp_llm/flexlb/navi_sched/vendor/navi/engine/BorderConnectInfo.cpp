#include "navi/engine/BorderConnectInfo.h"
#include "navi/util/CommonUtil.h"

namespace navi {

std::ostream& operator<<(std::ostream& os,
                         const BorderConnectInfo &connectInfo)
{
    return os << "BorderConnectInfo ("
              << CommonUtil::getPortQueueType(connectInfo.queueType) << ","
              << CommonUtil::getStoreType(connectInfo.storeType) << ","
              << "from[" << connectInfo.fromPartInfo.ShortDebugString() << "],"
              << "to[" << connectInfo.toPartInfo.ShortDebugString() << "],"
              << "eof[" << connectInfo.eofPartInfo.ShortDebugString() << "])";
}

}
