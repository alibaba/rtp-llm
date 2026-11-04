#pragma once

#include "navi/common.h"
#include "navi/engine/CreatorRegistry.h"

namespace navi {

class NaviLogger;
class Creator;

class CreatorRegistryImpl
{
public:
    CreatorRegistryImpl();
    ~CreatorRegistryImpl();
private:
    CreatorRegistryImpl(const CreatorRegistryImpl &);
    CreatorRegistryImpl &operator=(const CreatorRegistryImpl &);
public:
    bool addModuleInitFunc(const NaviModuleInitFunc &func);
    bool addCreatorFunc(const NaviCreatorFunc &func);
public:
    void addSubRegistry(const CreatorRegistryImpl *impl);
    void addJsonizeFunc(const NaviJsonizeInfoFunc &func);
    bool initModule(Module *module);
    bool initCreators(bool isRealNavi, bool isBuildin);
    void replaceCreatorMap(const CreatorMap &newCreatorMap);
    Creator *getCreator(const std::string &name) const;
    const CreatorMap &getCreatorMap() const;
    void clear();
    void setInited(bool inited=true);
    void fillJsonizeInfo(const NaviJsonizeInfoMap &jsonizeInfoMap);
    const NaviJsonizeInfoMap &getJsonizeInfoMap() const {
        return _jsonizeInfoMap;
    }
private:
    bool getCreators(bool isRealNavi, bool isBuildin, CreatorMap &creatorMap) const;
    void addJsonizeInfo(const NaviJsonizeInfoFunc &func);
private:
    bool _inited;
    Module *_module = nullptr;
    std::vector<NaviModuleInitFunc> _moduleInitFuncList;
    std::vector<NaviCreatorFunc> _funcList;
    std::set<NaviJsonizeInfoFunc> _jsonizeFuncSet;
    NaviJsonizeInfoMap _jsonizeInfoMap;
    CreatorMap _creatorMap;
    std::vector<const CreatorRegistryImpl *> _subRegistrys;
};

}
