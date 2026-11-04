#pragma once
#include <assert.h>
#include <stddef.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCInterface.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

/**
 * @ingroup ClientClasses
 * Base class for ARPCChannelManager.
 * Implemented useful functions.
 */
class RPCChannelManagerBase : public RPCChannelManager {
public:
    /**
     * Constructor
     */
    RPCChannelManagerBase();
    virtual ~RPCChannelManagerBase();

public:
    /**
     * Empty virtual base function, implementation is in ARPCChannelManager.
     */
    virtual RPCChannel *OpenChannel(const std::string &address,
                                    bool block = false,
                                    size_t queueSize = 50ul,
                                    int timeout = 5000,
                                    bool autoReconn = true,
                                    anet::CONNPRIORITY prio = anet::ANET_PRIORITY_NORMAL) override {
        assert(false);
        return NULL;
    }

    /**
     * No use for now.
     */
    virtual void Close();
};

TYPEDEF_PTR(RPCChannelManagerBase);
ARPC_END_NAMESPACE(arpc);

