#pragma once

#include "navi/builder/ResourceDefBuilder.h"
#include "navi/engine/CreatorRegistry.h"
#include "navi/proto/KernelDef.pb.h"
#include "navi/log/NaviLogger.h"

namespace navi {

class Resource;
class ResourceCreator;
class ResourceRegistryDef;
class CreatorManager;

typedef std::map<std::string, const DependByDef *> DependByMap;
typedef NaviUnorderedMap<std::string, DynamicResourceGroupDef *>
    DynamicResourceInfoMap;
typedef std::pair<ResourceCreator *, std::shared_ptr<Resource>> CreatorResourcePair;
typedef NaviUnorderedMap<std::string, CreatorResourcePair> CastInfoMap;

class ResourceCreator : public Creator
{
private:
    typedef std::shared_ptr<Resource> ResourcePtr;
    typedef std::function<ResourcePtr()> ResourceCreateFunc;
    typedef std::function<bool(Resource *other, int64_t &castOffset)> ResourceCastFunc;
public:
    ResourceCreator(const char *sourceFile,
                    int32_t line,
                    bool naviRegistry,
                    const ResourceDefBuilder &builder,
                    const ResourceCreateFunc &func,
                    const ResourceCastFunc &castFunc);
    virtual ~ResourceCreator();
private:
    ResourceCreator(const ResourceCreator &);
    ResourceCreator &operator=(const ResourceCreator &);
public:
    bool init(bool isBuildin, Module *module) override;
    const std::string &getName() const override;
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
    CreatorPtr clone() const;
    ResourceStage getStage() const;
    const std::shared_ptr<ResourceDef> &def() const;
    std::string getClassName() const override;
    ResourcePtr create() const;
    const std::vector<std::pair<std::string, bool>> &getDependResources() const {
        return _dependResourceVec;
    }
    const std::set<std::string> &getEnableBuildResourceSet() const {
        return _enableBuildResourceSet;
    }
    const std::set<std::string> &getEnableReplaceSet() const {
        return _enableReplaceSet;
    }
    const NaviUnorderedSet<std::string> &getReplaceBySet() const {
        return _replaceBySet;
    }
    bool supportReplace(const std::string &to) const {
        if (_replaceBySet.end() != _replaceBySet.find(to)) {
            return true;
        }
        return (_realName == to);
    }
    bool addReplacer(ResourceCreator *replacerCreator);
    bool addDependBy(const std::string &resource, const DependByDef *def);
    const DependByMap &getDependByResourceMap() const;
    const DependByMap &getDependByKernelMap() const;
    const BindInfos &getBindInfos() const;
    const DynamicResourceInfoMap &getDynamicGroupInfoMap() const;
    void addResourceDependant(const std::string &resource);
    void addKernelDependant(const std::string &kernel);
    const ProduceInfoDef *getProduceInfo() const;
    bool setProducer(const std::string &producer) {
        if (!_producer.empty()) {
            return false;
        }
        _producer = producer;
        return true;
    }
    const std::string &getProducer() const {
        return _producer;
    }
    void setBuildMode(ResourceBuildMode buildMode) {
        _def->set_build_mode(buildMode);
    }
    ResourceBuildMode getBuildMode() const {
        return _def->build_mode();
    }
    void setRealName(const std::string &realName) {
        _realName = realName;
    }
    bool isAsync() const {
        return _isAsync;
    }
    bool postInit(const NaviUnorderedMap<std::string, ResourceCreator *> &resourceCreatorMap);
    bool initCastOffset(CreatorManager *manager);
    int64_t &getCastOffset() {
        return _castOffset;
    }
    bool parallelBuild() const {
        return false;
    }
    void fillProto(StringHashTable &stringHashTable, ResourceRegistryDef &registryDef) const;
public:
    static ResourceBuildMode parseResourceBuildMode(const std::string &buildMode);
    static const char *getResourceBuildModeStr(ResourceBuildMode mode);
    static bool getCastInfo(const std::string &name,
                            CreatorManager *manager,
                            CastInfoMap &castInfoMap,
                            CreatorResourcePair &pair);
public:
    template <typename T1, typename T2>
    static bool
    initCastOffsetTyped(T1 *object,
                        int64_t &thisCastOffset,
                        const T2 &castFunc,
                        const DynamicResourceInfoMap &dynamicGroupMap,
                        const BindInfos &bindInfos,
                        CreatorManager *manager);

private:
    // for clone
    ResourceCreator(const char *sourceFile,
                    int32_t line,
                    bool naviRegistry,
                    ResourceDef *def,
                    const ResourceCreateFunc &func,
                    const ResourceCastFunc &castFunc,
                    const BindInfos &bindInfos);
    bool validateDef() const;
    bool initDependResources();
    bool initEnableBuilds(const NaviUnorderedMap<std::string, ResourceCreator *> &resourceCreatorMap);
    bool addResourceDependRecur(const std::string &name,
                                bool require,
                                const NaviUnorderedMap<std::string, ResourceCreator *> &resourceCreatorMap);
    bool initDynamicGroup();
    bool initDependBy();
    bool initReplace();
    bool addToDependResourceMap(const DependResourceDef &resourceDef);
private:
    std::shared_ptr<ResourceDef> _def;
    std::string _realName;
    bool _enableBuildInited = false;
    bool _isAsync = false;
    int64_t _castOffset = NAVI_INIT_CAST_OFFSET;
    ResourceCreateFunc _func;
    ResourceCastFunc _castFunc;
    std::map<std::string, bool> _dependResources;
    std::vector<std::pair<std::string, bool>> _dependResourceVec;
    DynamicResourceInfoMap _dynamicGroupMap;
    DependByMap _dependByResource;
    DependByMap _dependByKernel;
    BindInfos _bindInfos;
    std::set<std::string> _resourceDependant;
    std::set<std::string> _kernelDependant;
    std::string _producer;
    std::set<std::string> _enableBuildResourceSet;
    std::set<std::string> _enableReplaceSet;
    NaviUnorderedSet<std::string> _replaceBySet;
};

template <typename T1, typename T2>
bool ResourceCreator::initCastOffsetTyped(
    T1 *object,
    int64_t &thisCastOffset,
    const T2 &castFunc,
    const DynamicResourceInfoMap &dynamicGroupMap,
    const BindInfos &bindInfos,
    CreatorManager *manager)
{
    int64_t castOffset;
    if (!castFunc(object, castOffset)) {
        NAVI_KERNEL_LOG(ERROR, "cast to self failed");
        return false;
    }
    navi::updateCastOffset(castOffset, thisCastOffset);
    CastInfoMap castInfoMap;
    for (const auto &info : bindInfos.staticBindVec) {
        if (!info.binder) {
            continue;
        }
        CreatorResourcePair pair;
        if (!getCastInfo(info.name, manager, castInfoMap, pair)) {
            return false;
        }
        info.binder(object,
                    nullptr,
                    pair.second.get(),
                    &thisCastOffset,
                    &pair.first->getCastOffset());
    }
    for (const auto &info : bindInfos.dynamicBindVec) {
        if (!info.binder) {
            continue;
        }
        const auto &group = info.group;
        auto it = dynamicGroupMap.find(group);
        if (dynamicGroupMap.end() == it) {
            continue;
        }
        auto dynamicGroupDef = it->second;
        auto count = dynamicGroupDef->depend_resources_size();
        for (int i = 0; i < count; i++) {
            const auto &dependName = dynamicGroupDef->depend_resources(i).name();
            CreatorResourcePair pair;
            if (!getCastInfo(dependName, manager, castInfoMap, pair)) {
                return false;
            }
            info.binder(object,
                        nullptr,
                        pair.second.get(),
                        &thisCastOffset,
                        &pair.first->getCastOffset(),
                        nullptr);
        }
    }
    for (const auto &info : bindInfos.namedDataBindVec) {
        if (!info.binder) {
            continue;
        }
        info.binder(object, nullptr, nullptr, &thisCastOffset);
    }
    for (const auto &info : bindInfos.selectBindVec) {
        if (!info.binder) {
            continue;
        }
        for (const auto &dependName : info.candidates) {
            CreatorResourcePair pair;
            if (!getCastInfo(dependName, manager, castInfoMap, pair)) {
                return false;
            }
            info.binder(object,
                        nullptr,
                        pair.second.get(),
                        &thisCastOffset,
                        &pair.first->getCastOffset());
        }
    }
    return true;
}

}
