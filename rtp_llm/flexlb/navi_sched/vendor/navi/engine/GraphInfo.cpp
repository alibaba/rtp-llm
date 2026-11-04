#include "navi/engine/GraphInfo.h"
#include "navi/log/NaviLogger.h"
#include "navi/util/CommonUtil.h"
#include "autil/FileUtil.h"

namespace navi {

GraphInfo::GraphInfo(const std::string &configPath, const GraphConfig &config)
    : _configPath(configPath)
    , _config(config)
{
}

GraphInfo::~GraphInfo() {
}

bool GraphInfo::init() {
    if (!loadGraphDef()) {
        return false;
    }
    return true;
}

bool GraphInfo::loadGraphDef() {
    auto absPath = CommonUtil::getConfigAbsPath(_configPath, _config.graphFile);
    std::string fileContent;
    if (!autil::FileUtil::readFile(absPath, fileContent))
    {
        NAVI_KERNEL_LOG(ERROR, "read file [%s] failed",
                        absPath.c_str());
        return false;
    }
    if (!_graphDef.ParseFromString(fileContent)) {
        NAVI_KERNEL_LOG(ERROR, "parse graph def failed, file [%s], content: %s",
                        absPath.c_str(), fileContent.c_str());
        return false;
    }
    return true;
}

const GraphDef &GraphInfo::getGraphDef() const {
    return _graphDef;
}

}

