#pragma once

#include "navi/config/NaviConfig.h"
#include "navi/engine/CreatorManager.h"
#include "navi/engine/Data.h"
#include "navi/engine/GraphInfo.h"
#include "navi/engine/LocalSubGraphCache.h"
#include "navi/engine/ResourceManager.h"
#include "navi/engine/ReplaceInfoMap.h"
#include "navi/builder/GraphBuilder.h"

namespace kmonitor {
class MetricsReporter;
}

namespace navi {

class Creator;
class Kernel;
class NodeDef;
class BizManager;
class GraphBuilder;
struct NaviSnapshotSummary;
class RegistryTableDef;
class BizSummaryDef;
struct BizSummary;
class ReplaceInfoMap;
struct LocalSubGraphInternal;

class Biz
{
public:
    Biz(BizManager *bizManager, const std::string &bizName);
    ~Biz();
private:
    Biz(const Biz &);
    Biz &operator=(const Biz &);
public:
    bool preInit(bool isRealNavi,
                 const std::string &configPath, const NaviBizConfig &config,
                 const ModuleManagerPtr &moduleManager, const Biz *buildinBiz,
                 const Biz *oldBiz);
    bool saveResource(NaviPartId partCount, NaviPartId partId, const ResourceMap &resourceMap);
    bool collectPublish(std::vector<multi_call::ServerBizTopoInfo> &infoVec) const;
    void startPublish(multi_call::GigRpcServer *gigRpcServer);
    bool publishResource(NaviPartId partCount, NaviPartId partId);
    void initMetrics(kmonitor::MetricsReporter &reporter);
    const NaviRegistryConfigMap &getBizStageResourceConfigMap() const;
    bool initResourceGraph(bool ignoreProduceError,
                           GraphId buildinGraphId,
                           GraphBuilder &builder,
                           GraphId &resultGraphId) const;
    N addResourceGraph(const std::string &resource,
                       const MultiLayerResourceMap &multiLayerResourceMap,
                       int32_t scope,
                       const std::string &requireNode,
                       int32_t requireNodeIndex,
                       const NaviUnorderedMap<std::string, std::string> &replaceMap,
                       GraphBuilder &builder) const;
    bool getResourceMap(NaviPartId partCount, NaviPartId partId, ResourceMapPtr &resourceMap) const;
    ResourceStage getResourceStage(const std::string &name) const;
    ErrorCode createResource(NaviPartId partCount,
                             NaviPartId partId,
                             const std::string &name,
                             ResourceStage minStage,
                             const MultiLayerResourceMap &multiLayerResourceMap,
                             const SubNamedDataMap *namedDataMap,
                             const ProduceInfo &produceInfo,
                             bool checkReuse,
                             Node *requireNode,
                             bool failAsError,
                             ResourcePtr &resource);
    Resource *buildResourceRecur(NaviPartId partCount,
                                 NaviPartId partId,
                                 const std::string &name,
                                 ResourceStage minStage,
                                 NaviConfigContext *inputCtx,
                                 const SubNamedDataMap *namedDataMap,
                                 const MultiLayerResourceMap &nodeResourceMap,
                                 const ReplaceInfoMap *replaceInfoMap,
                                 Node *requireNode,
                                 ResourceMap &inputResourceMap);
    const CreatorManagerPtr &getCreatorManager() const;
    const KernelCreator *getKernelCreator(const std::string &name) const;
    ResourceCreator *getResourceCreator(const std::string &name) const;
    const GraphInfo *getGraphInfo(const std::string &name) const;
    bool isSinglePart() const;
    bool hasPartId(NaviPartId partId) const;
    NaviPartId getPartCount() const;
    void getPartInfo(NaviPartId &partCount, std::vector<NaviPartId> &partIds) const;
    std::vector<NaviPartId> getPartIdsWithoutBizPart() const;
    const std::set<std::string> &getKernelSet() const;
    const std::set<std::string> &getKernelBlacklist() const;
    const autil::legacy::RapidValue *getKernelConfig(const std::string &name) const;
    const std::vector<std::pair<std::string, bool>> *getResourceDependResourceVec(const std::string &name) const;
    const NaviUnorderedSet<std::string> *getResourceReplacerSet(const std::string &name) const;
    const std::string &getConfigPath() const;
    const std::string &getName() const;
    void setDefault(bool value) {
        _isDefault = value;
    }
    bool isDefault() const {
        return _isDefault;
    }
    const BizPublishInfos &getPublishInfos() const {
        return _publishInfos;
    }
    const std::shared_ptr<multi_call::FlowControlConfig> &getFlowControlConfig() const {
        return _flowControlConfig;
    }
    const ReplaceInfoMap &getReplaceInfoMap() const {
        return _replaceInfoMap;
    }
    void reclaimCache() {
        _graphCache.reclaim();
    }
    std::shared_ptr<LocalSubGraphInternal> getCacheGraph(size_t key) const {
        return _graphCache.getGraph(key);
    }
    bool putCacheGraph(size_t key, const std::shared_ptr<LocalSubGraphInternal> &graph) {
        return _graphCache.putGraph(key, graph);
    }
    void fillSummary(NaviSnapshotSummary &summary) const;
    void fillProto(StringHashTable &stringHashTable, RegistryTableDef &registryTable, BizSummaryDef &bizDef) const;
    void updateProto(BizSummaryDef &bizDef) const;
public:
    TestMode getTestMode() const;

private:
    bool initCreatorManager(bool isRealNavi,
                            const NaviBizConfig &config,
                            const ModuleManagerPtr &moduleManager,
                            const Biz *buildinBiz);
    void initConfigPath(const std::string &configPath,
                        const NaviBizConfig &config);
    bool initConfigMap(const NaviBizConfig &config, const Biz *buildinBiz);
    bool initConfig(const NaviBizConfig &config, const Biz *buildinBiz);
    bool initKernelSet(bool isRealNavi, const NaviBizConfig &config);
    void addBuildinKernels();
    bool initKernelCreatorMap();
    bool initResourceStageMap(const NaviBizConfig &config, const Biz *buildinBiz);
    void mergeBuildinResource(const Biz *buildinBiz);
    void addToStageMap(ResourceStage stage, const std::string &resource,
                       bool require);
    bool createResourceManager(const NaviBizConfig &config,
                               const Biz *buildinBiz, const Biz *oldBiz);
    bool loadGraphs(const NaviBizConfig &config);
    bool initPublishInfos(const NaviBizConfig &config);
    bool initCache(const NaviBizConfig &config);
    ResourceManager *getResourceManager(NaviPartId partId) const;
    void fillKernelSummary(BizSummary &bizSummary) const;
    void fillResourceSummary(BizSummary &bizSummary) const;
private:
    BizManager *_bizManager;
    std::string _bizName;
    std::string _configPath;
    bool _isDefault = false;
    CreatorManagerPtr _creatorManager;
    std::vector<std::string> _modules;
    std::set<std::string> _configKernelList;
    std::set<std::string> _configKernelBlackList;
    std::set<std::string> _configResourcesList;
    std::set<std::string> _kernelSet;
    std::set<std::string> _kernelBlacklist;
    BizPublishInfos _publishInfos;
    std::shared_ptr<multi_call::FlowControlConfig> _flowControlConfig;
    ResourceStageMap _resourceStageMap;
    NaviUnorderedMap<std::string, const KernelCreator *> _kernelCreatorMap;
    NaviPartId _partCount;
    NaviUnorderedMap<NaviPartId, ResourceManager *> _resourceManagers;
    NaviRegistryConfigMap _kernelConfigMap;
    NaviRegistryConfigMap _resourceConfigMap;
    NaviRegistryConfigMap _bizStageResourceConfigMap;
    ReplaceInfoMap _replaceInfoMap;
    NaviUnorderedMap<std::string, GraphInfo *> _graphInfoMap;
    std::shared_ptr<kmonitor::MetricsReporter> _metricsReporter;
    LocalSubGraphCache _graphCache;
};

NAVI_TYPEDEF_PTR(Biz);
typedef NaviUnorderedMap<std::string, BizPtr> BizMap;
NAVI_TYPEDEF_PTR(BizMap);

}
