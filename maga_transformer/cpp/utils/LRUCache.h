#pragma once
#include <iostream>
#include <list>
#include <unordered_map>
#include <utility>

template<typename KeyType, typename ValueType>
class LRUCache {
public:
    typedef typename std::list<std::pair<KeyType, ValueType>>::const_iterator CacheIterator;
    explicit LRUCache(size_t capacity) : capacity_(capacity) {}

    void put(const KeyType& key, const ValueType& value);

    std::tuple<bool, ValueType> get(const KeyType& key);

    std::tuple<bool, ValueType> pop();

    bool contains(const KeyType& key) const;

    void printCache();

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

private:
    size_t capacity_;
    std::list<std::pair<KeyType, ValueType>> items_list_;
    std::unordered_map<KeyType, typename std::list<std::pair<KeyType, ValueType>>::iterator> cache_items_map_;
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
    if (items_list_.empty()) {
        return {false, ValueType()};
    }
    // 删除最不常用的项
    auto last = items_list_.back().first;
    auto value = items_list_.back().second;
    items_list_.pop_back();
    cache_items_map_.erase(last);
    return {true, value};
}

template<typename KeyType, typename ValueType>
bool LRUCache<KeyType, ValueType>::contains(const KeyType& key) const {
    auto it = cache_items_map_.find(key);
    return it != cache_items_map_.end();
}

template<typename KeyType, typename ValueType>
void LRUCache<KeyType, ValueType>::printCache() {
    for (auto it = items_list_.begin(); it != items_list_.end(); ++it) {
        std::cout << it->first << " : " << it->second << std::endl;
    }
}