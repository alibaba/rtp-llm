#pragma once

#include <multi_call/interface/SearchService.h>
#include "navi/engine/Resource.h"
#include "navi/resource/GigMetaR.h"
#include "navi/rpc_server/NaviRpcServerR.h"
#include "autil/LoopThread.h"

namespace navi {

class GigClientR : public navi::Resource, public multi_call::SnapshotChangeCallback
{
public:
    GigClientR();
    ~GigClientR();
public:
    GigClientR(const GigClientR &) = delete;
    GigClientR &operator=(const GigClientR &) = delete;
public:
    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    navi::SummaryErrorCode getSummary(const std::string &query,
            navi::SummaryBuilder &builder) const override;
public:
    static const std::string RESOURCE_ID;
public:
    // gig callback
    void Run() override;
private:
    void initFlowConfig();
    void notifyLoop();

public:
    const multi_call::SearchServicePtr &getSearchService() const {
        return _searchService;
    }

private:
    RESOURCE_DEPEND_DECLARE();
private:
    DECLARE_LOGGER();
    RESOURCE_DEPEND_ON_FALSE(NaviRpcServerR, _naviRpcServerR);
    multi_call::MultiCallConfig _initConfig;
    std::map<std::string, multi_call::FlowControlConfig> _flowConfigMap;
    std::string _nodePath;
    std::string _metricPath;
    multi_call::SearchServicePtr _searchService;
    multi_call::SourceIdTy _sourceId = multi_call::INVALID_SOURCE_ID;
    ProduceNotifierPtr _notifier;
    GigMetaRPtr _metaR;

    int64_t _notifyInterval = 0;
    bool _singlePartMeta = false;
    std::shared_ptr<autil::LoopThread> _notifyThread;
    autil::ThreadMutex _mtx;
    bool _needNotify = false;
};
}
