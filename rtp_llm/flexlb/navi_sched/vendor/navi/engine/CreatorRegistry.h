#pragma once

#include "navi/common.h"
#include "navi/engine/CreatorStats.h"
#include "autil/legacy/jsonizable.h"

namespace navi {

class NaviLogger;
class CreatorRegistryImpl;
class Module;
struct RegistrySummary;
class StringHashTable;
class RegistryDef;
class NaviJsonizeInfo;

typedef NaviUnorderedMap<std::string, std::vector<NaviJsonizeInfo>> NaviJsonizeInfoMap;

class RegistryIdGenerator {
public:
    int64_t *getKernelCounter() {
        return &_kernelCounter;
    }
    int64_t *getResourceCounter() {
        return &_resourceCounter;
    }
    int64_t *getTypeCounter() {
        return &_typeCounter;
    }
private:
    int64_t _kernelCounter = 0;
    int64_t _resourceCounter = 0;
    int64_t _typeCounter = 0;
};

NAVI_TYPEDEF_PTR(RegistryIdGenerator);

class Creator : public autil::legacy::Jsonizable {
public:
    Creator(const char *sourceFile, int32_t line);
    virtual ~Creator();
    virtual const std::string &getName() const = 0;
    virtual bool init(bool isBuildin, Module *module) {
        setBuildin(isBuildin);
        setModule(module);
        return true;
    }
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
    const char *getSourceFile() const {
        return _sourceFile;
    }
    int32_t getLine() const {
        return _line;
    }
    void setId(int64_t *counter) {
        if (_id < 0) {
            _id = (*counter)++;
        }
    }
    int64_t getId() const {
        return _id;
    }
    void setBuildin(bool isBuildin) {
        _isBuildin = isBuildin;
    }
    bool isBuildin() const {
        return _isBuildin;
    }
    void setNaviRegistry(bool value) {
        _naviRegistry = value;
    }
    bool isNaviRegistry() const {
        return _naviRegistry;
    }
    void setModule(Module *module) {
        _module = module;
    }
    Module *getModule() const {
        return _module;
    }
    const std::string &getModulePath() const;
    void initJsonizeInfo(const NaviJsonizeInfoMap &jsonizeInfoMap);
    void fillProto(StringHashTable &stringHashTable, RegistryDef &def) const;
public:
    CreatorStats &getCreatorStats() const {
        return _stats;
    }
protected:
    virtual std::string getClassName() const {
        return std::string();
    }
protected:
    int64_t _id;
    bool _isBuildin;
    bool _naviRegistry;
    const char *_sourceFile;
    int32_t _line;
    Module *_module;
    std::vector<NaviJsonizeInfo> _jsonizeInfoVec;
    mutable CreatorStats _stats;
};

NAVI_TYPEDEF_PTR(Creator);

typedef NaviUnorderedMap<std::string, CreatorPtr> CreatorMap;

typedef void (*NaviJsonizeInfoFunc)(NaviJsonizeInfo &info);

class CreatorRegistry
{
public:
    CreatorRegistry();
    ~CreatorRegistry();
private:
    CreatorRegistry(const CreatorRegistry &);
    CreatorRegistry &operator=(const CreatorRegistry &);
public:
    bool addModuleInitFunc(const NaviModuleInitFunc &func);
    bool addCreatorFunc(const NaviCreatorFunc &func);
public:
    void addSubRegistry(const CreatorRegistry &registry);
    void addJsonizeFunc(NaviJsonizeInfoFunc func);
    bool initModule(Module *module);
    bool initCreators(bool isRealNavi, bool isBuildin);
    void replaceCreatorMap(const CreatorMap &newCreatorMap);
    Creator *getCreator(const std::string &name) const;
    const CreatorMap &getCreatorMap() const;
    void clear();
    void setInited(bool inited=true);
    const NaviJsonizeInfoMap &getJsonizeInfoMap() const;
    void fillJsonizeInfo(const NaviJsonizeInfoMap &jsonizeInfoMap);
public:
    static CreatorRegistry *current(RegistryType type);
    static void setCurrent(RegistryType type, CreatorRegistry *current);
    static CreatorRegistry *buildin(RegistryType type);
private:
    CreatorRegistryImpl *_impl;
private:
    static CreatorRegistry *CURRENT[RT_MAX];
    static CreatorRegistry *NAVI_BUILDIN[RT_MAX];
};

}
