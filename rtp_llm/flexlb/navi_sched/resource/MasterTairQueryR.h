#pragma once

#include "aios/navi_sched/scheduler/MasterScheduleContextData.h"
#include "kmonitor/client/MetricsReporterR.h"
#include "aios/navi_sched/resource/TairKvcmGrpcClientR.h"
#include "aios/navi_sched/scheduler/TairKvcmGrpcProtocol.h"
#include "navi/engine/Resource.h"
#include <map>
#include <mutex>

namespace navi_sched {

class MasterTairQueryR : public navi::Resource {
private:
    enum class RpcStage { LEADER, LOOKUP, DONE };
    struct Service {
        std::mutex mutex;
        TairKvcmGrpcEndpoint leader;
        bool refreshRequired = false;
        int64_t discoveryStartedUs = 0;
        uint64_t revision = 0;
    };
    struct Query {
        RpcStage stage = RpcStage::DONE;
        std::shared_ptr<Service> service;
        TairKvcmGrpcEndpoint leader;
        uint64_t serviceRevision = 0;
        int64_t startedUs = 0;
        TairKvcmGrpcMessagePtr body;
        std::shared_ptr<void> token;
        bool sent = false;
        size_t retryCount = 0;
        size_t rpcCount = 0;
        size_t discoveryAttempts = 0;
    };
public:
    static const std::string RESOURCE_ID;
    struct Batch {
        bool complete() const { return remaining == 0; }
        bool wasSent(size_t index) const { return queries.at(index).sent; }
        size_t retryCount(size_t index) const { return queries.at(index).retryCount; }
        size_t rpcCount(size_t index) const { return queries.at(index).rpcCount; }
    private:
        friend class MasterTairQueryR;
        MasterScheduleContextDataPtr data;
        navi::AsyncPipePtr completionPipe;
        SchedulerConfigR::Config config;
        uint64_t round = 0;
        size_t remaining = 0;
        std::vector<Query> queries;
    };
    using BatchPtr = std::shared_ptr<Batch>;
    void def(navi::ResourceDefBuilder &builder) const override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    BatchPtr startBatch(MasterScheduleContextDataPtr data,
                        const std::vector<NaviSchedScheduleItem> &requests,
                        uint64_t scheduleId, const navi::AsyncPipePtr &pipe, uint64_t round);
    void completed(TairKvcmGrpcCompletionData &completion, const BatchPtr &batch);
    void expirePending(const BatchPtr &batch);
    void reportBatch(const Batch &batch) const;
private:
    void buildQueries(MasterScheduleContextData &data,
                      const std::vector<NaviSchedScheduleItem> &requests,
                      uint64_t scheduleId, const SchedulerConfigR::Config &config = {});
    std::shared_ptr<const NaviSchedTairKvcmConfig> canonicalPolicy(
        const std::shared_ptr<const NaviSchedTairKvcmConfig> &policy,
        const std::string &instanceId);
    void invalidateLeader(const Query &query);
    void startDiscovery(Batch &batch, size_t index);
    bool retry(Batch &batch, size_t index, const char *reason);
    void startLookup(Batch &batch, size_t index);
    void send(Batch &batch, size_t index, RpcStage stage,
              const TairKvcmGrpcEndpoint &endpoint, const std::string &uri,
              const TairKvcmGrpcMessagePtr &body);
    void finish(Batch &batch, size_t index, TairKvcmQueryStatus status);
    void reportRpcCompletion(const TairKvcmGrpcCompletionData &completion,
                              TairKvcmQueryStatus status, size_t responseBytes, bool late,
                              kv_cache_manager::proto::meta::ErrorCode errorCode) const;
    kmonitor::MetricsReporterR *_metricsReporterR = nullptr;
    mutable std::mutex _cacheMutex;
    TairKvcmGrpcClientR *_grpcClientR = nullptr;
    std::shared_ptr<Service> _service = std::make_shared<Service>();
    struct PolicyAlias {
        std::weak_ptr<const NaviSchedTairKvcmConfig> source;
        std::shared_ptr<const NaviSchedTairKvcmConfig> canonical;
    };
    std::map<std::pair<const NaviSchedTairKvcmConfig *, std::string>, PolicyAlias> _policyAliases;
};

} // namespace navi_sched
