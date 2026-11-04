#pragma once

#include "navi/common.h"

namespace navi {

class StringMapWithSize
{
public:
    StringMapWithSize();
    ~StringMapWithSize();
    StringMapWithSize(const StringMapWithSize &) = delete;
    StringMapWithSize &operator=(const StringMapWithSize &) = delete;
public:
    std::shared_ptr<std::string> getStr(const std::string &str, size_t &hash);
    void removeStr(const std::shared_ptr<std::string> &str);
    void removeStr(size_t hash, const std::shared_ptr<std::string> &str);
    size_t getTotalSize() const {
        return _totalSize;
    }
    size_t size() const {
        return _strMap.size();
    }
private:
    std::hash<std::string> _hashFunc;
    NaviUnorderedMap<size_t, std::shared_ptr<std::string>> _strMap;
    size_t _totalSize = 0;
};

}

