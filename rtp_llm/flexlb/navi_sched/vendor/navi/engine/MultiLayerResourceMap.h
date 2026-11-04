#pragma once

#include "navi/engine/Resource.h"
#include "navi/engine/ResourceMap.h"

namespace navi {

class MultiLayerResourceMap {
private:
    static constexpr size_t N = 4;
public:
    MultiLayerResourceMap() {
    }
public:
    void addResourceMap(const ResourceMap *resourceMap);
    Resource *getResource(const std::string &name) const;
    Resource *getResource(const std::string &name, bool &exist) const;
    bool hasResource(const std::string &name) const;
    bool hasResource(const ResourcePtr &resource) const;
    void collectResource(ResourceMap &resourceMap) const;
public:
    friend std::ostream& operator<<(std::ostream& os,
                                    const MultiLayerResourceMap &multiLayerResourceMap);
private:
    size_t _index = 0;
    const ResourceMap *_resourceMapVec[N];
};

}

