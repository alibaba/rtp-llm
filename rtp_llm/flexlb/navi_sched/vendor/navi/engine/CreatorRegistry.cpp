#include "navi/engine/CreatorRegistry.h"
#include "navi/engine/NaviConfigContext.h"
#include "navi/engine/CreatorRegistryImpl.h"
#include "navi/engine/ModuleManager.h"
#include "navi/engine/StringHashTable.h"
#include "navi/proto/NaviSummary.pb.h"
#include "navi/log/NaviLogger.h"

namespace navi {

CreatorRegistry *CreatorRegistry::CURRENT[] = {};
CreatorRegistry *CreatorRegistry::NAVI_BUILDIN[] = {};

__attribute__((destructor)) void destructBuildinRegistry() {
    for (int i = 0; i < RT_MAX; i++) {
        delete CreatorRegistry::buildin(RegistryType(i));
    }
}

Creator::Creator(const char *sourceFile, int32_t line)
    : _id(-1)
    , _isBuildin(false)
    , _naviRegistry(false)
    , _sourceFile(sourceFile)
    , _line(line)
    , _module(nullptr)
{
}

Creator::~Creator() {}

const std::string &Creator::getModulePath() const {
    return Module::getModulePath(_module);
}

void Creator::initJsonizeInfo(const NaviJsonizeInfoMap &jsonizeInfoMap) {
    auto className = getClassName();
    auto it = jsonizeInfoMap.find(className);
    if (jsonizeInfoMap.end() != it) {
        _jsonizeInfoVec = it->second;
    }
}

void Creator::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("id", getId());
    json.Jsonize("name", getName());
    json.Jsonize("source", std::string(getSourceFile()) + ":" + std::to_string(getLine()));
    json.Jsonize("buildin", isBuildin());
    json.Jsonize("jsonize_infos", _jsonizeInfoVec);
}

void Creator::fillProto(StringHashTable &stringHashTable, RegistryDef &def) const {
    def.set_id(getId());
    def.set_is_build_in(isBuildin());
    def.set_navi_registry(isNaviRegistry());
    auto source = getSourceFile();
    def.set_source_file_hash(stringHashTable.getHash(source ? source : ""));
    def.set_line(getLine());
    def.set_module_hash(stringHashTable.getHash(getModulePath()));
    for (const auto &info : _jsonizeInfoVec) {
        auto jsonizeInfoDef = def.add_jsonize_info();
        info.fillProto(stringHashTable, *jsonizeInfoDef);
    }
}

CreatorRegistry::CreatorRegistry()
    : _impl(new CreatorRegistryImpl())
{
}

CreatorRegistry::~CreatorRegistry() {
    delete _impl;
}

CreatorRegistry *CreatorRegistry::current(RegistryType type) {
    if (!CURRENT[type]) {
        CURRENT[type] = buildin(type);
    }
    return CURRENT[type];
}

void CreatorRegistry::setCurrent(RegistryType type, CreatorRegistry *current) {
    CURRENT[type] = current;
}

CreatorRegistry *CreatorRegistry::buildin(RegistryType type) {
    if (!NAVI_BUILDIN[type]) {
        NAVI_BUILDIN[type] = new CreatorRegistry();
    }
    return NAVI_BUILDIN[type];
}

bool CreatorRegistry::addModuleInitFunc(const NaviModuleInitFunc &func) {
    return _impl->addModuleInitFunc(func);
}

bool CreatorRegistry::addCreatorFunc(const NaviCreatorFunc &func) {
    return _impl->addCreatorFunc(func);
}

void CreatorRegistry::addSubRegistry(const CreatorRegistry &registry) {
    _impl->addSubRegistry(registry._impl);
}

void CreatorRegistry::addJsonizeFunc(NaviJsonizeInfoFunc func) {
    _impl->addJsonizeFunc(func);
}

bool CreatorRegistry::initModule(Module *module) {
    return _impl->initModule(module);
}

bool CreatorRegistry::initCreators(bool isRealNavi, bool isBuildin) {
    return _impl->initCreators(isRealNavi, isBuildin);
}

void CreatorRegistry::replaceCreatorMap(const CreatorMap &newCreatorMap) {
    _impl->replaceCreatorMap(newCreatorMap);
}

Creator *CreatorRegistry::getCreator(const std::string &name) const {
    return _impl->getCreator(name);
}

const CreatorMap &CreatorRegistry::getCreatorMap() const {
    return _impl->getCreatorMap();
}

void CreatorRegistry::clear() {
    _impl->clear();
}

void CreatorRegistry::setInited(bool inited) {
    _impl->setInited(inited);
}

const NaviJsonizeInfoMap &CreatorRegistry::getJsonizeInfoMap() const {
    return _impl->getJsonizeInfoMap();
}

void CreatorRegistry::fillJsonizeInfo(const NaviJsonizeInfoMap &jsonizeInfoMap) {
    _impl->fillJsonizeInfo(jsonizeInfoMap);
}

}
