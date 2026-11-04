#include "navi/engine/MultiLayerResourceMap.h"

namespace navi {

void MultiLayerResourceMap::addResourceMap(const ResourceMap *resourceMap) {
    assert(_index < N);
    _resourceMapVec[_index++] = resourceMap;
}

Resource *MultiLayerResourceMap::getResource(const std::string &name) const {
    for (size_t i = 0; i < _index; i++) {
        auto resourceMap = _resourceMapVec[i];
        if (resourceMap) {
            auto resource = resourceMap->getResource(name);
            if (resource) {
                return resource;
            }
        }
    }
    return nullptr;
}

Resource *MultiLayerResourceMap::getResource(const std::string &name, bool &exist) const {
    for (size_t i = 0; i < _index; i++) {
        auto resourceMap = _resourceMapVec[i];
        if (resourceMap) {
            auto resource = resourceMap->getResource(name, exist);
            if (exist) {
                return resource;
            }
        }
    }
    exist = false;
    return nullptr;
}

bool MultiLayerResourceMap::hasResource(const std::string &name) const {
    for (size_t i = 0; i < _index; i++) {
        auto resourceMap = _resourceMapVec[i];
        if (resourceMap) {
            if (resourceMap->hasResource(name)) {
                return true;
            }
        }
    }
    return false;
}

bool MultiLayerResourceMap::hasResource(const ResourcePtr &resource) const {
    if (!resource) {
        return false;
    }
    const auto &name = resource->getName();
    for (size_t i = 0; i < _index; i++) {
        auto resourceMap = _resourceMapVec[i];
        if (resourceMap) {
            auto depend = resourceMap->getResource(name);
            if (!depend) {
                continue;
            }
            return (resource.get() == depend);
        }
    }
    return false;
}

void MultiLayerResourceMap::collectResource(ResourceMap &resourceMap) const {
    for (size_t i = 0; i < _index; i++) {
        auto ptr = _resourceMapVec[i];
        if (ptr) {
            resourceMap.addResource(*ptr);
        }
    }
}

std::ostream &operator<<(std::ostream &os, const MultiLayerResourceMap &multiLayerResourceMap) {
    for (size_t i = 0; i < multiLayerResourceMap._index; i++) {
        auto resourceMap = multiLayerResourceMap._resourceMapVec[i];
        if (resourceMap) {
            os << autil::StringUtil::toString(*resourceMap) << ",";
        }
    }
    return os;
}

}

