#include "navi/engine/StringMapWithSize.h"

namespace navi {

StringMapWithSize::StringMapWithSize() {
}

StringMapWithSize::~StringMapWithSize() {
}

std::shared_ptr<std::string> StringMapWithSize::getStr(const std::string &str, size_t &hash) {
    hash = _hashFunc(str);
    auto it = _strMap.find(hash);
    if (_strMap.end() == it) {
        auto value = std::make_shared<std::string>(str);
        _strMap[hash] = value;
        _totalSize += value->capacity();
        return value;
    } else {
        return it->second;
    }
}

void StringMapWithSize::removeStr(const std::shared_ptr<std::string> &str) {
    if (str.use_count() > 2) {
        return;
    }
    if (!str) {
        return;
    }
    auto hash = _hashFunc(*str);
    removeStr(hash, str);
}

void StringMapWithSize::removeStr(size_t hash, const std::shared_ptr<std::string> &str) {
    if (str.use_count() > 2) {
        return;
    }
    if (!str) {
        return;
    }
    if (_strMap.erase(hash) > 0) {
        _totalSize -= str->capacity();
    }
}

}

