#include "navi/resource/ResourceErrorInfoCollectorR.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/ops/ResourceData.h"

namespace navi {

const std::string ResourceErrorInfoCollectorR::RESOURCE_ID = "navi.buildin.resource_error_info_collector_r";

ResourceErrorInfoCollectorR::ResourceErrorInfoCollectorR()
    : ExternalResource(RESOURCE_ID, RS_RUN_GRAPH_EXTERNAL)
{
}

ResourceErrorInfoCollectorR::~ResourceErrorInfoCollectorR() {
}

void ResourceErrorInfoCollectorR::updateErrorInfo(void *kernel, const std::shared_ptr<ResourceErrorInfo> &errorInfo) const {
    autil::ScopedLock lock(_mutex);
    if (!errorInfo) {
        _errorInfoMap.erase(kernel);
    } else {
        _errorInfoMap[kernel] = errorInfo;
    }
}

std::shared_ptr<ResourceErrorInfo> ResourceErrorInfoCollectorR::getErrorInfo() const {
    NaviUnorderedMap<void *, std::shared_ptr<ResourceErrorInfo>> errorInfoMap;
    {
        autil::ScopedLock lock(_mutex);
        errorInfoMap = _errorInfoMap;
    }
    std::shared_ptr<ResourceErrorInfo> errorInfo;
    for (const auto &pair : errorInfoMap) {
        const auto &kernelErrorInfo = pair.second;
        if (!kernelErrorInfo) {
            continue;
        }
        if (!errorInfo) {
            errorInfo = kernelErrorInfo;
        } else if (errorInfo->latterThan(*kernelErrorInfo)) {
            errorInfo = kernelErrorInfo;
        }
    }
    return errorInfo;
}

REGISTER_BUILDIN_RESOURCE(ResourceErrorInfoCollectorR);

}

