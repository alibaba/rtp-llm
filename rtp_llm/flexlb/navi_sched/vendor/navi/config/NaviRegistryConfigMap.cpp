#include "navi/config/NaviRegistryConfigMap.h"
#include "navi/config/NaviConfig.h"
#include "navi/log/NaviLogger.h"

using namespace autil::legacy;

namespace navi {

bool NaviRegistryConfig::loadConfig(const std::string &name,
                                    const std::string &configStr)
{
    assert(config == nullptr);
    _document.reset(new RapidDocument());
    auto ret = NaviConfig::parseToDocument(configStr, *_document);
    if (!ret) {
        NAVI_KERNEL_LOG(ERROR,
                        "parse resource config failed, resource [%s] config [%s]",
                        name.c_str(), configStr.c_str());
        return false;
    }
    this->name = name;
    this->config = _document.get();
    return true;
}

void NaviRegistryConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("name", name);
    json.Jsonize("config", config);
    json.Jsonize("build_mode", buildMode, buildMode);
    if (json.GetMode() == FROM_JSON) {
        if (!config || !config->IsObject()) {
            NAVI_KERNEL_LOG(ERROR, "field [config] is not json map, name [%s]",
                            name.c_str());
            AUTIL_LEGACY_THROW(autil::legacy::NotJsonizableException,
                               "invalid config");
        }
    }
}

bool NaviRegistryConfigMap::init(const std::vector<NaviRegistryConfig> &configVec) {
    for (const auto &config : configVec) {
        const auto &name = config.name;
        auto it = _configMap.find(name);
        if (_configMap.end() != it) {
            NAVI_KERNEL_LOG(ERROR, "duplicate config, name [%s]", name.c_str());
            return false;
        }
        _configMap.emplace(name, config);
    }
    return true;
}

void NaviRegistryConfigMap::mergeNotExist(const NaviRegistryConfigMap &other) {
    const auto &otherMap = other._configMap;
    for (const auto &pair : otherMap) {
        _configMap.emplace(pair);
    }
}

const NaviUnorderedMap<std::string, NaviRegistryConfig> &NaviRegistryConfigMap::getConfigMap() const {
    return _configMap;
}

void NaviRegistryConfigMap::setConfig(NaviRegistryConfig config) {
    _configMap[config.name] = std::move(config);
}

bool NaviRegistryConfigMap::hasConfig(const std::string &name) const {
    return nullptr != getConfig(name);
}

const NaviRegistryConfig *NaviRegistryConfigMap::getRegistryConfig(
        const std::string &name) const
{
    auto it = _configMap.find(name);
    if (_configMap.end() != it) {
        assert(it->second.config);
        return &it->second;
    } else {
        return nullptr;
    }
}

const autil::legacy::RapidValue *NaviRegistryConfigMap::getConfig(const std::string &name) const {
    auto it = _configMap.find(name);
    if (_configMap.end() != it) {
        assert(it->second.config);
        return it->second.config;
    } else {
        return nullptr;
    }
}
}
