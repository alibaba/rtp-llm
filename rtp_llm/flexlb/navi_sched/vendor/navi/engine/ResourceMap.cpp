#include "navi/engine/ResourceMap.h"
#include "navi/engine/Resource.h"
#include "navi/util/CommonUtil.h"

namespace navi {

bool ResourceMap::addResource(const std::shared_ptr<Resource> &resource) {
    if (!resource) {
        return false;
    }
    return addResource(resource->getName(), resource);
}

bool ResourceMap::addResource(const ResourceMap &resourceMap) {
    auto map = resourceMap.getMap();
    bool ret = true;
    for (const auto &pair : map) {
        if (!addResource(pair.first, pair.second)) {
            NAVI_KERNEL_LOG(DEBUG, "add resource failed [%s]", pair.first.c_str());
            ret = false;
        }
    }
    return ret;
}

bool ResourceMap::addResource(const std::string &name, const std::shared_ptr<Resource> &resource) {
    if (name.empty()) {
        if (resource) {
            const auto &res = *resource;
            NAVI_KERNEL_LOG(ERROR, "resource name is empty, class [%s]", CommonUtil::parseSymbol(typeid(res).name()).c_str());
        } else {
            NAVI_KERNEL_LOG(ERROR, "resource name is empty, resource is null");
        }
        return false;
    }
    auto ret = _map.emplace(std::make_pair(name, resource));
    if (unlikely(!ret.second)) {
        if (resource != ret.first->second) {
            NAVI_KERNEL_LOG(DEBUG,
                            "resource [%s] already exists, old [%p] new [%p]",
                            name.c_str(),
                            ret.first->second.get(),
                            resource.get());
            return false;
        }
    }
    return true;
}

bool ResourceMap::overrideResource(const std::shared_ptr<Resource> &resource) {
    if (!resource) {
        return false;
    }
    return overrideResource(resource->getName(), resource);
}

std::ostream &operator<<(std::ostream &os, const ResourceMap &resourceMap) {
    auto map = resourceMap.getMap();
    os << "count: " << map.size() << ": ";
    os << "{";
    for (const auto &pair : map) {
        os << "(" << pair.first << ", " << pair.second.get() << ")";
    }
    return os << "}";
}

}
