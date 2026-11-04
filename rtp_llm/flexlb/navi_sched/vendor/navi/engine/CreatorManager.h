#pragma once

#include "navi/config/NaviConfig.h"
#include "navi/engine/KernelCreator.h"
#include "navi/engine/ModuleManager.h"
#include "navi/engine/ResourceCreator.h"
#include "navi/engine/Type.h"
#include "navi/log/NaviLogger.h"
#include <regex>

namespace navi {

struct BizSummary;
class RegistryTableDef;
class BizSummaryDef;
class ReplaceInfoMap;

class CreatorManager
{
public:
    CreatorManager(const std::string &bizName,
                   const std::shared_ptr<CreatorManager> &parentManager);
    ~CreatorManager();
private:
    CreatorManager(const CreatorManager &);
    CreatorManager &operator=(const CreatorManager &);
public:
    bool init(bool isRealNavi,
              const std::string &configPath,
              const std::vector<std::string> &modules,
              const ModuleManagerPtr &moduleManager,
              const ReplaceInfoMap &replaceInfoMap,
              const NaviRegistryConfigMap &resourceConfigMap);
    bool postInit(RegistryIdGenerator &idGenerator);
    const KernelCreator *getKernelCreator(const std::string &kernelName) const;
    ResourceCreator *getResourceCreator(const std::string &resourceName) const;
    const std::vector<std::pair<std::string, bool>> *getResourceDependResourceVec(const std::string &name) const;
    const NaviUnorderedSet<std::string> *getResourceReplacerSet(const std::string &name) const;
    bool supportReplace(const std::string &from, const std::string &to);
    ResourceStage getResourceStage(const std::string &name) const;
    const Type *getType(const std::string &typeId) const;
    bool collectMatchedKernels(const std::string &kernelRegex, bool &emptyMatch,
                               std::set<std::string> &kernelSet) const;
    void fillModuleSummary(BizSummary &bizSummary) const;
    void fillProto(StringHashTable &stringHashTable, RegistryTableDef &registryTable, BizSummaryDef &bizDef) const;
    std::set<std::string> getKernelSet() const;
    void setTypeMemoryPoolR(const MemoryPoolRPtr &memoryPoolR);
private:
    bool initModules(const std::string &configPath,
                     const std::vector<std::string> &modules,
                     const ModuleManagerPtr &moduleManager);
    bool initBuildinModule();
    bool initRegistrys(bool isRealNavi,
                       const ReplaceInfoMap &replaceInfoMap,
                       const NaviRegistryConfigMap &resourceConfigMap);
    bool applyReplaceTo(const ReplaceInfoMap &replaceInfoMap);
    bool validateRegistry() const;
    bool validateKernelRegistry() const;
    bool validateResourceRegistry() const;
    bool validateTypeRegistry() const;
    bool initBuildMode(const NaviRegistryConfigMap &resourceConfigMap);
    bool initMergedMaps();
    bool initKernelMergedMap();
    bool initResourceMergedMap();
    bool initTypeMergedMap();
    void addToKernelMergedMap(const CreatorMap &creatorMap);
    void addToResourceMergedMap(const CreatorMap &creatorMap);
    bool initDependBy();
    bool initProduce();
    bool initReplace();
    bool addResourceDependBy(const std::string &name,
                             ResourceCreator *resourceCreator);
    bool addKernelDependBy(const std::string &name,
                           ResourceCreator *resourceCreator);
    Creator *getDependByResource(const std::string &name, bool &isClone);
    Creator *getDependByKernel(const std::string &name);
    bool initEnableBuilds();
    bool initKernelDependant();
    bool initResourceDependant();
    bool initCastOffset();
    bool initJsonizeInfo();
private:
    static const char *getRegexErrorStr(std::regex_constants::error_type code);
private:
    std::string _bizName;
    std::shared_ptr<CreatorManager> _parentManager;
    std::map<std::string, ModulePtr> _moduleMap;
    CreatorRegistry _resourceRegistry;
    CreatorRegistry _kernelRegistry;
    CreatorRegistry _typeRegistry;
    CreatorMap _cloneKernelCreatorMap;
    CreatorMap _cloneResourceCreatorMap;
    NaviUnorderedMap<std::string, KernelCreator *> _mergedKernelCreatorMap;
    NaviUnorderedMap<std::string, ResourceCreator *> _mergedResourceCreatorMap;
    NaviUnorderedMap<std::string, Type *> _mergedTypeMap;
};

NAVI_TYPEDEF_PTR(CreatorManager);

}
