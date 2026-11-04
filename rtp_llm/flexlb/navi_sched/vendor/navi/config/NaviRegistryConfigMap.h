#pragma once

#include "navi/common.h"
#include "autil/legacy/jsonizable.h"

namespace navi {

struct NaviRegistryConfig : public autil::legacy::Jsonizable
{
public:
    NaviRegistryConfig()
        : config(nullptr)
    {
    }
public:
    bool loadConfig(const std::string &name, const std::string &configStr);
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    std::shared_ptr<autil::legacy::RapidDocument> _document;
    std::string name;
    const autil::legacy::RapidValue *config = nullptr;
    std::string buildMode;
};

class NaviRegistryConfigMap {
public:
    bool init(const std::vector<NaviRegistryConfig> &configVec);
    void mergeNotExist(const NaviRegistryConfigMap &other);
    void setConfig(NaviRegistryConfig config);
    bool hasConfig(const std::string &name) const;
    const NaviRegistryConfig *getRegistryConfig(const std::string &name) const;
    const autil::legacy::RapidValue *getConfig(const std::string &name) const;
    const NaviUnorderedMap<std::string, NaviRegistryConfig> &getConfigMap() const;
private:
    NaviUnorderedMap<std::string, NaviRegistryConfig> _configMap;
};

}
