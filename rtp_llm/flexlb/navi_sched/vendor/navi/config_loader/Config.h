#pragma once

#include "navi/config_loader/Config.h"
#include "autil/legacy/jsonizable.h"

namespace navi {

class Config
{
public:
    static bool load(const std::string &naviPythonPath,
                     const std::string &configLoader,
                     const std::string &configPath,
                     const std::string &loadParam,
                     std::string &jsonConfig);
};

}
