#pragma once

#include "navi/common.h"
#include "navi/config/NaviConfig.h"
#include "navi/engine/Biz.h"
#include "navi/engine/CreatorManager.h"
#include "autil/Lock.h"

namespace navi {

class ResourceMap;
struct NaviSnapshotSummary;
struct BizSummary;
class NaviSummaryDef;

class BizManager
{
public:
    BizManager(bool isRealNavi,
               const std::string &naviName,
               const std::shared_ptr<ModuleManager> &moduleManager);
    ~BizManager();
private:
    BizManager(const BizManager &);
    BizManager &operator=(const BizManager &);
public:
    bool preInit(const NaviLoggerPtr &logger,
                 BizManager *oldBizManager,
                 const NaviConfig &config,
                 const ResourceMap &rootResourceMapIn);
    GraphDef *cloneResourceGraph() const;
    bool postInit();
    bool start(multi_call::GigRpcServer *gigRpcServer, BizManager *oldBizManager);
    void initMetrics(kmonitor::MetricsReporter &reporter);
    GraphDef *createResourceGraph(const std::string &bizName,
                                  NaviPartId partCount,
                                  NaviPartId partId,
                                  const std::set<std::string> &resourceSet) const;
    bool isLocal(const LocationDef &location) const;
    BizPtr getBiz(NaviLogger *_logger, const std::string &bizName) const;
    bool getBizPartInfo(const std::string &bizName, NaviPartId &partCount, std::vector<NaviPartId> &partIds) const;
    const BizPtr &getBuildinBiz() const;
    const ModuleManagerPtr &getModuleManager() const;
    const CreatorManagerPtr &getBuildinCreatorManager() const;
    Resource *getSnapshotResource(const std::string &name) const;
    void fillSummary(NaviSnapshotSummary &summary) const;
    void fillProto(NaviSummaryDef &summaryDef) const;
    void updateProto(NaviSummaryDef &summaryDef) const;
    void cleanupModule();
    const std::vector<multi_call::SignatureTy> &getPublishSignatures() const {
        return _publishSignatures;
    }
    void stealPublishSignature(const std::vector<multi_call::SignatureTy> &stealList);
    std::vector<std::string> getBizList() const;
    void reclaimCache();
public:
    void setTestMode(TestMode testMode) { _testMode = testMode; }
    TestMode getTestMode() const { return _testMode; }

private:
    bool initModuleManager(BizManager *oldBizManager);
    bool createBuildinBiz(const NaviConfig &config);
    void initBuildinBizResourceList(
        const std::vector<NaviRegistryConfig> &configVec);
    bool initMemoryPoolR(const ResourceMap &rootResourceMap);
    bool initTypeMemoryPool();
    bool createBizs(const NaviConfig &config,
                    BizManager *oldBizManager);
    BizPtr createBiz(const std::string &bizName,
                     const std::string &configPath,
                     const NaviBizConfig &config,
                     const Biz *buildinBiz,
                     const Biz *oldBiz);
    bool initResourceGraph(const NaviConfig &config);
    void initCreatorId();
    BizPtr doGetBiz(const std::string &bizName) const;
private:
    static void addRequireAttr(GraphBuilder &builder);
    static void addNodeRequireAttr(N n, int64_t &count);
    static bool needUpdateRequire(N n);
private:
    DECLARE_LOGGER();
    bool _isRealNavi = false;
    TestMode _testMode = TM_NONE;
    std::string _naviName;
    std::vector<multi_call::SignatureTy> _publishSignatures;
    multi_call::GigRpcServer *_gigRpcServer = nullptr;
    std::shared_ptr<kmonitor::MetricsReporter> _metricsReporter;
    BizPtr _buildinBiz;
    NaviBizConfig _buildinConfig;
    ResourceMapPtr _buildinResourceMap;
    ResourceMap _memoryPoolRMap;
    MultiLayerResourceMap _snapshotMultiLayerResourceMap;
    BizMap _bizMap;
    BizMap _bizMapWithAlias;
    BizPtr _defaultBiz;
    ModuleManagerPtr _moduleManager;
    GraphDef *_graphDef;
};

NAVI_TYPEDEF_PTR(BizManager);

}
