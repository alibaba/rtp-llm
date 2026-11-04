#pragma once

#include "autil/Lock.h"
#include "navi/config/NaviConfig.h"
#include "navi/engine/CreatorManager.h"
#include "navi/engine/CreatorRegistry.h"
#include "navi/engine/MultiLayerResourceMap.h"
#include "navi/engine/NamedData.h"
#include "navi/engine/Resource.h"

namespace multi_call {
class GigRpcServer;
}

namespace navi {

class ResourceCreator;
class Biz;
struct PartSummary;
class ProduceInfo;
class NaviRpcServerR;
class ReplaceInfoMap;
class BizPartSummaryDef;

typedef NaviUnorderedMap<ResourceStage, std::map<std::string, bool> >
    ResourceStageMap;

class ResourceManager
{
public:
    ResourceManager(const Biz *biz,
                    const std::string &bizName,
                    const std::string &configPath,
                    NaviPartId partCount,
                    NaviPartId partId,
                    const CreatorManager &creatorManager,
                    const NaviRegistryConfigMap &configMap);
    ~ResourceManager();
private:
    ResourceManager(const ResourceManager &);
    ResourceManager &operator=(const ResourceManager &);
public:
    bool preInit(ResourceManager *oldResourceManager);
    bool saveResource(const ResourceMap &resourceMap);
    bool collectPublish(std::vector<multi_call::ServerBizTopoInfo> &infoVec);
    void startPublish(multi_call::GigRpcServer *gigRpcServer);
    bool publishResource(ResourceManager *buildinResourceManager,
                         ResourceManager *bizResourceManager,
                         const ResourceStageMap &resourceStageMap);
    bool validateResource(const ResourceStageMap &resourceStageMap) const;
    ResourceMapPtr getMergedResourceMap() const;
    ErrorCode createResource(const std::string &name,
                             ResourceStage minStage,
                             const MultiLayerResourceMap &multiLayerResourceMap,
                             const SubNamedDataMap *namedDataMap,
                             const ProduceInfo &produceInfo,
                             bool checkReuse,
                             Node *requireNode,
                             bool failAsError,
                             ResourcePtr &resource);
    Resource *buildResourceRecur(const std::string &resourceName,
                                 ResourceStage minStage,
                                 NaviConfigContext *inputCtx,
                                 const SubNamedDataMap *namedDataMap,
                                 const MultiLayerResourceMap &nodeResourceMap,
                                 const ReplaceInfoMap *replaceInfoMap,
                                 Node *requireNode,
                                 bool failAsError,
                                 ResourceMap &inputResourceMap) const;
    NaviPartId getPartCount() const;
    bool hasConfig(const std::string &name) const;
    void fillSummary(PartSummary &partSummary) const;
    void fillProto(StringHashTable &stringHashTable, BizPartSummaryDef &summaryDef) const;
private:
    bool validateStage(const ResourceMapPtr &mergedResourceMap, const ResourceStageMap &resourceStageMap) const;
    Resource *getExistResource(const std::string &name, const MultiLayerResourceMap &multiLayerResourceMap);
    bool validateDepend(const Resource *resource,
                        const MultiLayerResourceMap &multiLayerResourceMap) const;
    NaviConfigContext *getConfigCtx(const std::string &name,
                                    NaviConfigContext *inputCtx,
                                    ResourceStage stage,
                                    NaviConfigContext *configCtx,
                                    Node *requireNode) const;
    ErrorCode collectDependMap(const ResourceCreator *creator,
                               ResourceStage minStage,
                               const SubNamedDataMap *namedDataMap,
                               const MultiLayerResourceMap &multiLayerResourceMap,
                               Node *requireNode,
                               bool failAsError,
                               ResourceMap &dependResourceMap);
    Resource *getResourceFromOld(const std::string &name) const;
    ResourcePtr newAndConfigResource(const std::string &name,
                                     const ResourceCreator *creator,
                                     NaviConfigContext *ctx,
                                     bool failAsError) const;
    ErrorCode initResource(const ResourcePtr &resourcePtr,
                           const SubNamedDataMap *namedDataMap,
                           const ProduceInfo *produceInfo,
                           const ResourceCreator *creator,
                           NaviConfigContext *configCtx,
                           Node *requireNode,
                           bool failAsError) const;
    bool collectStaticGraphInfo();
    bool updatePublishMeta();
    bool getPublishMetaMap(std::map<std::string, std::string> &metaMap) const;
    bool addStaticGraphMeta(std::map<std::string, std::string> &metaMap) const;
    void setMergedResourceMap(const ResourceMapPtr &resourceMap);
    void setResourceMap(const ResourceMapPtr &resourceMap);
    ResourceMapPtr getResourceMap() const;
private:
    static DataPtr getNamedData(const SubNamedDataMap *namedDataMap,
                                const std::string &name);
public:
    template <typename T>
    static inline bool
    bindResource(const std::string &name,
                 void *obj,
                 void *castObj,
                 T &baseResourceMap,
                 const DynamicResourceInfoMap &dynamicGroupMap,
                 const BindInfos &bindInfos,
                 const NaviVector<int32_t, 1> &selectorResult,
                 const char *bindType);
    template <typename T>
    static bool bindNamedData(const std::string &name,
                              T &obj,
                              void *castObj,
                              const SubNamedDataMap *namedDataMap,
                              const BindInfos &bindInfos,
                              const char *bindType);

private:
    const Biz *_biz;
    std::string _bizName;
    std::string _configPath;
    NaviPartId _partCount;
    NaviPartId _partId;
    bool _testMode = false;
    const CreatorManager &_creatorManager;
    const NaviRegistryConfigMap &_configMap;
    mutable autil::ThreadMutex _resourceMapLock;
    ResourceMapPtr _resourceMap;
    mutable autil::ReadWriteLock _mergeMapLock;
    ResourceMapPtr _mergedResourceMap;
    ResourceManager *_oldResourceManager;
    mutable autil::ThreadMutex _publishLock;
    multi_call::GigRpcServer *_gigRpcServer;
    std::shared_ptr<NaviUnorderedMap<std::string, StaticGraphInfoPtr>> _staticGraphInfoMap;
};

template <typename T>
inline bool ResourceManager::bindResource(
    const std::string &name,
    void *obj,
    void *castObj,
    T &baseResourceMap,
    const DynamicResourceInfoMap &dynamicGroupMap,
    const BindInfos &bindInfos,
    const NaviVector<int32_t, 1> &selectorResult,
    const char *bindType) {
    NAVI_KERNEL_LOG(SCHEDULE1,
                    "start bind resource for %s [%s] at [%p]",
                    bindType,
                    name.c_str(),
                    obj);
    for (const auto &info : bindInfos.staticBindVec) {
        const auto &dependName = info.name;
        auto dependResource = baseResourceMap.getResource(dependName);
        NAVI_KERNEL_LOG(SCHEDULE1,
                        "bind depend resource for %s [%s] at [%p], "
                        "dependName [%s] dependResource [%p] require [%d]",
                        bindType, name.c_str(), obj, dependName.c_str(),
                        dependResource, info.require);
        if (!dependResource && info.require) {
            NAVI_KERNEL_LOG(
                WARN,
                "init %s [%s] failed, lack required depend resource [%s]",
                bindType, name.c_str(), dependName.c_str());
            return false;
        }
        if (info.binder && !info.binder(obj, castObj, dependResource, nullptr, nullptr)) {
            NAVI_KERNEL_LOG(WARN,
                            "init %s [%s] failed, bind depend resource [%s] "
                            "failed, type cast failed, actual type [%s]",
                            bindType,
                            name.c_str(),
                            dependName.c_str(),
                            dependResource ? typeid(*dependResource).name() : "");
            return false;
        }
    }
    const auto &dynamicBindVec = bindInfos.dynamicBindVec;
    for (const auto &info : dynamicBindVec) {
        const auto &group = info.group;
        auto it = dynamicGroupMap.find(group);
        if (dynamicGroupMap.end() == it) {
            NAVI_KERNEL_LOG(WARN,
                            "dynamic group [%s] not exist in def of %s [%s]",
                            group.c_str(), bindType, name.c_str());
            return false;
        }
        auto dynamicGroupDef = it->second;
        auto count = dynamicGroupDef->depend_resources_size();
        for (int i = 0; i < count; i++) {
            const auto &dependDef = dynamicGroupDef->depend_resources(i);
            const auto &dependName = dependDef.name();
            auto require = dependDef.require();
            auto dependResource = baseResourceMap.getResource(dependName);
            NAVI_KERNEL_LOG(SCHEDULE1,
                            "bind depend resource for %s [%s] at [%p], "
                            "dependName [%s] dependResource [%p] require [%d]",
                            bindType, name.c_str(), obj, dependName.c_str(),
                            dependResource, require);
            if (!dependResource && require) {
                NAVI_KERNEL_LOG(
                    WARN,
                    "init %s [%s] failed, lack required depend resource [%s]",
                    bindType, name.c_str(), dependName.c_str());
                return false;
            }
            if (info.binder && !info.binder(obj, castObj, dependResource, nullptr, nullptr, nullptr)) {
                NAVI_KERNEL_LOG(
                    WARN,
                    "init %s [%s] failed, bind depend resource [%s] "
                    "failed, type cast failed",
                    bindType, name.c_str(), dependName.c_str());
                return false;
            }
        }
    }
    for (int32_t i = 0; i < bindInfos.selectBindVec.size(); i++) {
        const auto &info = bindInfos.selectBindVec[i];
        auto selectIndex = selectorResult[i];
        Resource *dependResource = nullptr;
        if (selectIndex < 0) {
            if (info.binder && !info.binder(obj, castObj, dependResource, nullptr, nullptr)) {
                NAVI_KERNEL_LOG(WARN,
                                "init %s [%s] failed, bind failed, select negative index [%d] on required depend "
                                "resource, selector [%d]",
                                bindType,
                                name.c_str(),
                                selectIndex,
                                i);
                return false;
            }
            continue;
        }
        const auto &dependName = info.candidates[selectIndex];
        dependResource = baseResourceMap.getResource(dependName);
        NAVI_KERNEL_LOG(SCHEDULE1,
                        "bind depend resource for %s [%s] at [%p], "
                        "dependName [%s] dependResource [%p] require [%d]",
                        bindType, name.c_str(), obj, dependName.c_str(),
                        dependResource, info.require);
        if (!dependResource && info.require) {
            NAVI_KERNEL_LOG(WARN,
                            "init %s [%s] failed, lack selected required depend resource [%s], selector [%d]",
                            bindType,
                            name.c_str(),
                            dependName.c_str(),
                            i);
            return false;
        }
        if (info.binder && !info.binder(obj, castObj, dependResource, nullptr, nullptr)) {
            NAVI_KERNEL_LOG(WARN,
                            "init %s [%s] failed, bind selected depend resource [%s] "
                            "failed, type cast failed, selector [%d]",
                            bindType, name.c_str(), dependName.c_str(), i);
            return false;
        }
    }
    return true;
}

template <typename T>
bool ResourceManager::bindNamedData(const std::string &name,
                                    T &obj,
                                    void *castObj,
                                    const SubNamedDataMap *namedDataMap,
                                    const BindInfos &bindInfos,
                                    const char *bindType) {
    for (const auto &info : bindInfos.namedDataBindVec) {
        const auto &dataName = info.name;
        auto data = getNamedData(namedDataMap, dataName);
        if (!data && info.require) {
            NAVI_KERNEL_LOG(WARN,
                            "init %s [%s] failed, lack required named data [%s]",
                            bindType, name.c_str(), dataName.c_str());
            return false;
        }
        if (info.binder && !info.binder(&obj, castObj, data.get(), nullptr)) {
            NAVI_KERNEL_LOG(WARN,
                            "init %s [%s] failed, bind named data [%s] failed, "
                            "type cast failed",
                            bindType, name.c_str(), dataName.c_str());
            return false;
        }
    }
    return true;
}

NAVI_TYPEDEF_PTR(ResourceManager);

}
