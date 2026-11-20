#pragma once

#include <memory>
#include <unordered_map>

namespace deep_jit {

// In-memory cache keyed by value type.
template <typename Key, typename Value>
class MemCache {
public:
    std::unordered_map<Key, std::shared_ptr<Value>> cache;

    template <typename Factory>
    std::shared_ptr<Value> get_or_create(const Key& key, Factory&& factory) {
        if (const auto iterator = cache.find(key); iterator != cache.end())
            return iterator->second;
        auto value = factory();
        cache.emplace(key, value);
        return value;
    }
};

}  // namespace deep_jit
