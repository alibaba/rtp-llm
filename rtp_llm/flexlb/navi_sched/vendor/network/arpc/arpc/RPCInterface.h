#pragma once

#include <stddef.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCChannelBase.h"

ARPC_BEGIN_NAMESPACE(arpc);

class RPCChannelManager {
public:
    virtual ~RPCChannelManager() {}

    // timout is miliseconds
    virtual RPCChannel *OpenChannel(const std::string &address,
                                    bool block = false,
                                    size_t queueSize = 50ul,
                                    int timeout = 5000,
                                    bool autoReconn = true,
                                    anet::CONNPRIORITY prio = anet::ANET_PRIORITY_NORMAL) = 0;
};

ARPC_END_NAMESPACE(arpc)

