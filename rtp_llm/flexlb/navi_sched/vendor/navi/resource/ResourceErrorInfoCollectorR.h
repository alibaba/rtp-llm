#pragma once

#include "navi/engine/Resource.h"

namespace navi {

struct ResourceErrorInfo;

class ResourceErrorInfoCollectorR : public ExternalResource
{
public:
    ResourceErrorInfoCollectorR();
    ~ResourceErrorInfoCollectorR();
    ResourceErrorInfoCollectorR(const ResourceErrorInfoCollectorR &) = delete;
    ResourceErrorInfoCollectorR &operator=(const ResourceErrorInfoCollectorR &) = delete;
public:
    void updateErrorInfo(void *kernel, const std::shared_ptr<ResourceErrorInfo> &errorInfo) const;
    std::shared_ptr<ResourceErrorInfo> getErrorInfo() const;
public:
    static const std::string RESOURCE_ID;
private:
    RESOURCE_DEPEND_DECLARE();
private:
    mutable autil::ThreadMutex _mutex;
    mutable NaviUnorderedMap<void *, std::shared_ptr<ResourceErrorInfo>> _errorInfoMap;
};

NAVI_TYPEDEF_PTR(ResourceErrorInfoCollectorR);

}

