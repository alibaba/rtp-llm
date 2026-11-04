#pragma once

#include <stddef.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCChannelManagerBase.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetApp.h"

ARPC_BEGIN_NAMESPACE(arpc)

class ANetRPCChannelManager : public RPCChannelManagerBase {
public:
    ANetRPCChannelManager(anet::Transport *transport = NULL);
    virtual ~ANetRPCChannelManager();

public:
    // queueSize : default queue size for rpcchannel
    // timout    : default timeout for rpcchannel(miliseconds)
    virtual RPCChannel *OpenChannel(const std::string &address,
                                    bool block = false,
                                    size_t queueSize = 50ul,
                                    int timeout = 5000,
                                    bool autoReconn = true,
                                    anet::CONNPRIORITY prio = anet::ANET_PRIORITY_NORMAL) override;
    /**
     * Start private transport in this channel manager.
     */
    bool StartPrivateTransport();

    bool StartPrivateTransport(const std::string &name);

    /**
     * Stop private transport in this channel manager.
     */
    bool StopPrivateTransport();

    anet::Transport *GetTransport();

    void SetMetricReporter(const std::shared_ptr<ClientMetricReporter> &metricReporter);

protected:
    /**
     * Tool function to create connection to remote server.
     * @return Connection object pointer
     */
    anet::Connection *Connect(const std::string &address,
                              bool block,
                              size_t queueSize,
                              int timeout,
                              bool autoReconn,
                              anet::CONNPRIORITY prio = anet::ANET_PRIORITY_NORMAL);

    /**
     * Check if this manager owns its private transport or not.
     */
    bool OwnTransport() { return _anetApp.OwnTransport(); }
    /**
     * Tool function to create connection with bind ip and port to remote server.
     * @return Connection object pointer
     */
    anet::Connection *Connect(const std::string &address,
                              const std::string &bindAddr,
                              size_t queueSize,
                              int timeout,
                              bool autoReconn,
                              anet::CONNPRIORITY prio = anet::ANET_PRIORITY_NORMAL);

protected:
    ANetApp _anetApp;
    std::shared_ptr<ClientMetricReporter> _metricReporter;
};

ARPC_END_NAMESPACE(arpc)

