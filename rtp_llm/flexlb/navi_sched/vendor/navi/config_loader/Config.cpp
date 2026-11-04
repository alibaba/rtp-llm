#include "navi/log/NaviLogger.h"
#include "navi/config_loader/Config.h"
#include "navi/config_loader/PythonEngine.h"

namespace navi {

bool Config::load(const std::string &naviPythonPath,
                  const std::string &configLoader,
                  const std::string &configPath,
                  const std::string &loadParam,
                  std::string &jsonConfig)
{
    return PythonEngine(naviPythonPath)
        .exec(configLoader, configPath, loadParam, jsonConfig);
}

}
