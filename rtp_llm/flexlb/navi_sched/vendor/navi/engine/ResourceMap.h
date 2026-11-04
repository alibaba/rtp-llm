#pragma once

#include "navi/common.h"
#include "navi/log/NaviLogger.h"
#include "autil/Lock.h"

namespace navi {

class Resource;

class ResourceMap
{
public:
    typedef NaviUnorderedMap<std::string, std::shared_ptr<Resource>> MapType;
public:
    ResourceMap() {
    }
    ResourceMap(const MapType &map)
        : _map(map)
    {
    }
    ~ResourceMap() {
    }
public:
    template <typename T>
    T *getResource(const std::string &name) const;
    bool addResource(const std::shared_ptr<Resource> &resource);
    bool addResource(const ResourceMap &resourceMap);
    bool addResource(const std::string &name, const std::shared_ptr<Resource> &resource);
    Resource *getResource(const std::string &name) const {
        bool exist = false;
        return getResource(name, exist);
    }
    Resource *getResource(const std::string &name, bool &exist) const {
        auto it = _map.find(name);
        if (it != _map.end()) {
            exist = true;
            return it->second.get();
        } else {
            exist = false;
            return nullptr;
        }
    }
    bool overrideResource(const std::shared_ptr<Resource> &resource);
    bool overrideResource(const std::string &name, const std::shared_ptr<Resource> &resource) {
        _map[name] = resource;
        return true;
    }
    void setResource(const ResourceMap &resourceMap) {
        auto map = resourceMap.getMap();
        _map = std::move(map);
    }
    bool hasResource(const std::string &name) const {
        return _map.end() != _map.find(name);
    }
    void clear() {
        _map.clear();
    }
    void reserve(size_t count) {
        _map.reserve(count);
    }
    size_t size() const {
        return _map.size();
    }
    bool empty() const {
        return _map.empty();
    }
    const MapType &getMap() const {
        return _map;
    }
public:
    friend std::ostream& operator<<(std::ostream& os, const ResourceMap &resourceMap);
private:
    MapType _map;
};

template <typename T>
T *ResourceMap::getResource(const std::string &name) const {
    return dynamic_cast<T *>(getResource(name));
}

NAVI_TYPEDEF_PTR(ResourceMap);

}
