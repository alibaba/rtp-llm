#pragma once

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/TPBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorServerCaller.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "autil/LoopThread.h"
#include <mutex>
#include <vector>

namespace rtp_llm {

// p2p connector always match all when pd sep
class P2PConnectorAsyncMatchContext: public KVCacheConnector::AsyncMatchContext {
public:
    P2PConnectorAsyncMatchContext(const KVCacheResourcePtr& resource): resource_(resource) {}
    virtual ~P2PConnectorAsyncMatchContext() {}

public:
    size_t                          matchedBlockCount() const override;
    KVCacheConnector::ConnectorType connectorType() const override;
    bool                            done() const override;
    bool                            success() const override;

private:
    KVCacheResourcePtr resource_;
};

class P2PConnectorAsyncReadContext: public AsyncContext {
public:
    P2PConnectorAsyncReadContext(const KVCacheResourcePtr&                                           resource,
                                 const std::shared_ptr<TPBroadcastClient::Result>&                   tp_sync_result,
                                 const std::shared_ptr<P2PConnectorServerCaller::Result>&            server_call_result,
                                 const std::shared_ptr<P2PConnectorClientSchedulerMetricsCollector>& collector):
        resource_(resource),
        tp_sync_result_(tp_sync_result),
        server_call_result_(server_call_result),
        collector_(collector) {}
    virtual ~P2PConnectorAsyncReadContext() {}

public:
    bool done() const override;
    bool success() const override;
    void checkDone();
    bool needCancel() const;
    void cancel(const std::shared_ptr<TPBroadcastClient>& tp_broadcast_client);

    std::string uniqueKey() const {
        return tp_sync_result_ ? tp_sync_result_->uniqueKey() : "";
    }

private:
    KVCacheResourcePtr                                           resource_;
    std::shared_ptr<TPBroadcastClient::Result>                   tp_sync_result_;
    std::shared_ptr<P2PConnectorServerCaller::Result>            server_call_result_;
    std::shared_ptr<P2PConnectorClientSchedulerMetricsCollector> collector_;
    std::shared_ptr<TPBroadcastClient::Result>                   cancel_result_;
};

class P2PConnectorAsyncWriteByLayerContext: public AsyncContext {
public:
    P2PConnectorAsyncWriteByLayerContext(const KVCacheResourcePtr& resource): resource_(resource) {}
    virtual ~P2PConnectorAsyncWriteByLayerContext() {}

public:
    bool done() const override;
    bool success() const override;

private:
    KVCacheResourcePtr resource_;
};

class P2PConnectorAsyncReadContextChecker {
public:
    P2PConnectorAsyncReadContextChecker() = default;
    ~P2PConnectorAsyncReadContextChecker();

public:
    bool   init(const kmonitor::MetricsReporterPtr&       metrics_reporter,
                const std::shared_ptr<TPBroadcastClient>& tp_broadcast_client);
    void   stop();
    void   addContext(const std::shared_ptr<P2PConnectorAsyncReadContext>& context);
    size_t inflightContextCount() const;

private:
    void checkOnce();

private:
    kmonitor::MetricsReporterPtr                               metrics_reporter_;
    std::shared_ptr<TPBroadcastClient>                         tp_broadcast_client_;
    mutable std::mutex                                         async_contexts_mutex_;
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> async_contexts_;
    autil::LoopThreadPtr                                       check_done_thread_;
};

}  // namespace rtp_llm