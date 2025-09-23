#pragma once
#include <iostream>
#include <list>
#include <vector>
#include <unordered_map>
#include <utility>
#include <functional>

template<typename KeyType, typename ValueType>
class LRUCache {
public:
    struct CacheSnapshot {
        int64_t                version;
        std::vector<ValueType> values;
    };

public:
    typedef typename std::list<std::pair<KeyType, ValueType>>::const_iterator CacheIterator;
    explicit LRUCache(size_t capacity): capacity_(capacity) {}

    void put(const KeyType& key, const ValueType& value);

    std::tuple<bool, ValueType> get(const KeyType& key);

    std::tuple<bool, ValueType> pop();

    std::tuple<bool, ValueType> popWithCond(const std::function<bool(const KeyType&, const ValueType&)>& cond);

    void clear();

    bool contains(const KeyType& key) const;

    void printCache() const;

    CacheIterator begin() {
        return items_list_.begin();
    }

    CacheIterator end() {
        return items_list_.end();
    }

    const std::list<std::pair<KeyType, ValueType>>& items() const {
        return items_list_;
    }

    size_t size() const {
        return items_list_.size();
    }

    bool empty() const {
        return items_list_.empty();
    }

    CacheSnapshot cacheSnapshot(int64_t latest_version) const;

private:
    size_t                                                                                   capacity_;
    std::list<std::pair<KeyType, ValueType>>                                                 items_list_;
    std::unordered_map<KeyType, typename std::list<std::pair<KeyType, ValueType>>::iterator> cache_items_map_;
    int64_t                                                                                  version = -1;
};

template<typename KeyType, typename ValueType>
void LRUCache<KeyType, ValueType>::put(const KeyType& key, const ValueType& value) {
    auto it = cache_items_map_.find(key);
    if (it != cache_items_map_.end()) {
        // 如果键已经存在，则更新值并移动到列表前端
        it->second->second = value;
        items_list_.splice(items_list_.begin(), items_list_, it->second);
    } else {
        // 如果达到容量，则删除最不常用的项
        if (items_list_.size() == capacity_) {
            auto last = items_list_.back().first;
            items_list_.pop_back();
            cache_items_map_.erase(last);
        }
        // 插入新项到列表前端
        items_list_.emplace_front(key, value);
        cache_items_map_[key] = items_list_.begin();
        version++;
    }
}

template<typename KeyType, typename ValueType>
std::tuple<bool, ValueType> LRUCache<KeyType, ValueType>::get(const KeyType& key) {
    auto it = cache_items_map_.find(key);
    if (it == cache_items_map_.end()) {
        return {false, ValueType()};
    }
    items_list_.splice(items_list_.begin(), items_list_, it->second);
    return {true, it->second->second};
}

template<typename KeyType, typename ValueType>
std::tuple<bool, ValueType> LRUCache<KeyType, ValueType>::pop() {
    return popWithCond([](const KeyType&, const ValueType&) { return true; });
}

template<typename KeyType, typename ValueType>
std::tuple<bool, ValueType>
LRUCache<KeyType, ValueType>::popWithCond(const std::function<bool(const KeyType&, const ValueType&)>& cond) {
    if (items_list_.empty()) {
        return {false, ValueType()};
    }

    // 从最不常用的项开始查找符合条件的项
    auto it = items_list_.rbegin();
    while (it != items_list_.rend()) {
        if (cond(it->first, it->second)) {
            auto key   = it->first;
            auto value = it->second;
            // 将 reverse_iterator 转换为 iterator 进行删除
            auto forward_it = std::next(it).base();
            items_list_.erase(forward_it);
            cache_items_map_.erase(key);
            version++;
            return {true, value};
        }
        ++it;
    }
    return {false, ValueType()};
}

template<typename KeyType, typename ValueType>
void LRUCache<KeyType, ValueType>::clear() {
    items_list_.clear();
    cache_items_map_.clear();
    version++;
}

template<typename KeyType, typename ValueType>
bool LRUCache<KeyType, ValueType>::contains(const KeyType& key) const {
    auto it = cache_items_map_.find(key);
    return it != cache_items_map_.end();
}

template<typename KeyType, typename ValueType>
void LRUCache<KeyType, ValueType>::printCache() const {
    for (auto it = items_list_.begin(); it != items_list_.end(); ++it) {
        std::cout << it->first << " : " << it->second << std::endl;
    }
}

template<typename KeyType, typename ValueType>
typename LRUCache<KeyType, ValueType>::CacheSnapshot
LRUCache<KeyType, ValueType>::cacheSnapshot(int64_t latest_version) const {
    std::vector<ValueType> values;
    values.reserve(items_list_.size());
    if (latest_version < version) {
        for (const auto& item : items_list_) {
            values.push_back(item.second);
        }
    }
    return CacheSnapshot{version, std::move(values)};
}
