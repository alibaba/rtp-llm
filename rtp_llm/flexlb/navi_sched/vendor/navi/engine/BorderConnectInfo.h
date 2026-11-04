#pragma once

#include "navi/common.h"
#include "navi/engine/PartInfo.h"

namespace navi {

struct BorderConnectInfo {
    BorderConnectInfo()
        : queueType(PQT_UNKNOWN)
        , storeType(PST_UNKNOWN)
    {
    }
    PortQueueType queueType;
    PortStoreType storeType;
    PartInfo fromPartInfo;
    PartInfo toPartInfo;
    PartInfo eofPartInfo;
};

extern std::ostream& operator<<(std::ostream& os,
                                const BorderConnectInfo &connectInfo);

}
