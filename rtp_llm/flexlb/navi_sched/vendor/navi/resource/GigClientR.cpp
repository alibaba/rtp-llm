#include "navi/resource/GigClientR.h"
#include "navi/builder/ResourceDefBuilder.h"
#include "navi/util/CommonUtil.h"
#include "navi/engine/NaviRegisterMacro.h"

using namespace multi_call;

namespace navi {

const std::string GigClientR::RESOURCE_ID = "navi.buildin.gig_client";

GigClientR::GigClientR() {
}

GigClientR::~GigClientR() {
    _notifyThread.reset();
    size_t count = 0;
    if (_searchService) {
        _searchService->stealSnapshotChangeCallback();
        CommonUtil::waitUseCount(_searchService, 1, 5 * 1000 * 1000);
    }
    NAVI_KERNEL_LOG(DEBUG,
                    "gig client resource destructed [%p], search "
                    "service [%p], use_count: [%lu], sleep count: %lu",
                    this, _searchService.get(), _searchService.use_count(),
                    count);
    if (_searchService.use_count() > 1) {
        NAVI_KERNEL_LOG(ERROR,
                        "gig destruct bt: %s",
                        navi::NAVI_TLS_LOGGER ? navi::NAVI_TLS_LOGGER->logger->getCurrentBtString().c_str() : "");
    }
}

void GigClientR::def(ResourceDefBuilder &builder) const {
    builder.name(RESOURCE_ID, RS_SNAPSHOT)
        .produce(GigMetaR::RESOURCE_ID);
}

bool GigClientR::config(navi::ResourceConfigContext &ctx) {
    ctx.Jsonize("init_config", _initConfig, _initConfig);
    ctx.Jsonize("flow_config", _flowConfigMap, _flowConfigMap);
    ctx.Jsonize("node_path", _nodePath, _nodePath);
    ctx.Jsonize("gig_client_notify_interval", _notifyInterval, _notifyInterval);
    ctx.Jsonize("single_part_meta", _singlePartMeta, _singlePartMeta);
    return true;
}

navi::ErrorCode GigClientR::init(navi::ResourceInitContext &ctx) {
    if (NAVI_TLS_LOGGER) {
        _logger = *NAVI_TLS_LOGGER;
    }
    _sourceId = CommonUtil::random64();
    _notifier = ctx.getProduceNotifier();
    _searchService = std::make_shared<SearchService>();
    NAVI_KERNEL_LOG(INFO, "search service created [%p]", _searchService.get());
    if (_naviRpcServerR) {
        auto gigRpcServer = _naviRpcServerR->getGigRpcServer();
        if (gigRpcServer) {
            _searchService->setThisSpec(gigRpcServer->getRpcSpec());
        }
    }
    if (!_searchService->init(_initConfig)) {
        NAVI_KERNEL_LOG(ERROR, "search service init failed [%p] config [%s]",
                        _searchService.get(), ToJsonString(_initConfig).c_str());
        return navi::EC_INIT_RESOURCE;
    }
    initFlowConfig();
    if (_notifier) {
        _searchService->setSnapshotChangeCallback(this);
    }
    if (_notifyInterval > 0) {
        NAVI_KERNEL_LOG(INFO, "notify interval [%ld]", _notifyInterval);
        _notifyThread = autil::LoopThread::createLoopThread([this]() { notifyLoop(); }, _notifyInterval);
        if (!_notifyThread) {
            NAVI_KERNEL_LOG(ERROR, "create notify loop failed");
            return navi::EC_INIT_RESOURCE;
        }
    }
    return navi::EC_NONE;
}

void GigClientR::initFlowConfig() {
    std::map<std::string, FlowControlConfigPtr> flowControlConfigMap;
    for (const auto &pair : _flowConfigMap) {
        const auto &strategy = pair.first;
        const auto &config = pair.second;

        FlowControlConfigPtr configPtr(new FlowControlConfig(config));
        flowControlConfigMap.emplace(strategy, configPtr);
    }
    _searchService->updateFlowConfig(flowControlConfigMap);
}

navi::SummaryErrorCode GigClientR::getSummary(const std::string &query,
        navi::SummaryBuilder &builder) const
{
    {
        auto table = builder.newTable("SubscribeResultTable");
        std::vector<multi_call::BizMetaInfo> bizMetaInfos;
        multi_call::TopoSignatureSetPtr gigMetaLocalSignatures;
        if (!_searchService->getBizMetaInfos(_sourceId, _singlePartMeta, bizMetaInfos, gigMetaLocalSignatures)) {
            table->setDesc("getBizMetaInfos failed");
        } else {
            auto bizColumn = table->addColumn("Biz");
            auto versionColumn = table->addColumn("Version");
            auto partCountColumn = table->addColumn("PartCount");
            auto subscribeCountColumn = table->addColumn("SubscribePartCount");
            auto stopCountColumn = table->addColumn("StopPartCount");
            auto healthCountColumn = table->addColumn("HealthPartCount");
            auto normalProviderCountColumn = table->addColumn("TotalNormalProviderCount");
            auto heartbeatHealthCountColumn = table->addColumn("HeartbeatHealthCount");
            auto minHeartbeatHealthCountColumn = table->addColumn("MinHeartbeatHealthCount");
            for (size_t i = 0; i < bizMetaInfos.size(); i++) {
                const auto &bizMeta = bizMetaInfos[i];
                const auto &bizName = bizMeta.bizName;
                for (const auto &pair : bizMeta.versions) {
                    auto version = pair.first;
                    const auto &versionInfo = pair.second;
                    bizColumn->addData(bizName);
                    versionColumn->addData(version);
                    partCountColumn->addData(versionInfo.partCount);
                    subscribeCountColumn->addData(versionInfo.subscribeCount);
                    stopCountColumn->addData(versionInfo.stopCount);
                    healthCountColumn->addData(versionInfo.healthCount);
                    normalProviderCountColumn->addData(versionInfo.normalProviderCount);
                    heartbeatHealthCountColumn->addData(versionInfo.heartbeatHealthCount);
                    minHeartbeatHealthCountColumn->addData(versionInfo.minHeartbeatHealthCount);
                }
            }
        }
    }
    {
        auto table = builder.newTable("InitConfigTable");
        table->setDesc(FastToJsonString(_initConfig));
    }
    {
        auto table = builder.newTable("完整流控参数");
        table->setDesc(_searchService->getAllFlowControlConfig());
    }
    {
        auto table = builder.newTable("配置流控参数");
        table->setDesc(FastToJsonString(_flowConfigMap));
    }
    return navi::SEC_OK;
}

void GigClientR::Run() {
    if (!_notifier) {
        return;
    }
    NaviLoggerScope scope(_logger);
    std::vector<BizMetaInfo> bizMetaInfos;
    multi_call::TopoSignatureSetPtr gigMetaLocalSignatures;
    if (!_searchService->getBizMetaInfos(_sourceId, _singlePartMeta, bizMetaInfos, gigMetaLocalSignatures)) {
        NAVI_LOG(ERROR, "getBizMetaInfos failed");
        return;
    }
    auto metaR = std::make_shared<GigMetaR>(bizMetaInfos, gigMetaLocalSignatures);
    if (_metaR && _metaR->isSame(metaR)) {
        return;
    }
    metaR->log();
    autil::ScopedLock lk(_mtx);
    _metaR = metaR;
    _needNotify = true;
    NAVI_LOG(INFO,
             "create GigMetaR [%p], localSubCount: %lu",
             metaR.get(),
             gigMetaLocalSignatures ? gigMetaLocalSignatures->size() : 0);
    if (!_notifyThread) {
        if (!_notifier->notify(metaR)) {
            NAVI_LOG(ERROR, "notify GigMetaR produce failed [%p]", metaR.get());
        }
    }
}

void GigClientR::notifyLoop() {
    NAVI_LOG(INFO, "interval notify, this [%p]", this);
    autil::ScopedLock lk(_mtx);
    if (!_needNotify) {
        NAVI_LOG(INFO, "no need interval notify");
        return;
    }
    auto newGigMetaR = std::make_shared<GigMetaR>(_metaR->getBizMetaInfos(), _metaR->getGigMetaLocalSignatures());
    NAVI_LOG(INFO, "notify with gig meta version [%ld]", newGigMetaR->version());
    _notifier->notify(newGigMetaR);
    _needNotify = false;
}

REGISTER_BUILDIN_RESOURCE(GigClientR);

}
