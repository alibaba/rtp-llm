#pragma once

#include "navi/engine/CreatorRegistry.h"
#include <map>

namespace navi {

class Module {
public:
    Module(const std::string &path);
    ~Module();
public:
    bool init();
    CreatorRegistry &getResourceRegistry();
    CreatorRegistry &getKernelRegistry();
    CreatorRegistry &getTypeRegistry();
    void setHandle(void *handle);
    const std::string &getPath() const;
public:
    static const std::string &getModulePath(Module *module);
public:
    static const std::string BUILDIN_MODULE_ID;
private:
    std::string _path;
    void *_handle;
    CreatorRegistry _resourceRegistry;
    CreatorRegistry _kernelRegistry;
    CreatorRegistry _typeRegistry;
};

NAVI_TYPEDEF_PTR(Module);

class ModuleManager
{
public:
    ModuleManager();
    ~ModuleManager();
private:
    ModuleManager(const ModuleManager &);
    ModuleManager &operator=(const ModuleManager &);
public:
    ModulePtr loadModule(const std::string &configPath,
                         const std::string &modulePath);
    void cleanup();
    const std::map<std::string, ModulePtr> &getModuleMap() const {
        return _moduleMap;
    }
private:
    std::map<std::string, ModulePtr> _moduleMap;
};

NAVI_TYPEDEF_PTR(ModuleManager);

}
