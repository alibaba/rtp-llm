#pragma once

#include <assert.h>
#include <optional>
#include <stdint.h>
#include <vector>

#include "alog/Logger.h"
#include "autil/Cache.h"
#include "autil/DoubleLinkList.h"
#include "autil/DoubleLinkListNode.h"
#include "autil/Lock.h"
#include "autil/Log.h"
#include "autil/StringUtil.h"
#include "autil/SynchronizedHashTable.h"
#include "autil/SynchronizedList.h"

namespace autil {

template <class K, class V>
struct CacheValue {
    V value;
    ListNode<K> *iter;

    CacheValue() : iter(0) {}

    CacheValue(const V &v, ListNode<K> *iterator = 0) : value(v), iter(iterator) {}
};

template <class K, class V, class GetSizeCallBack>
class LruCache : public Cache<K, V> {
public:
    LruCache(int64_t memSize, GetSizeCallBack getSizeCallBack = GetSizeCallBack(), int64_t elemAvgEstMemSize = -1);
    ~LruCache();

public:
    void warmUp(const std::vector<K> &keyList, const std::vector<V> &valueList) override;
    void invalidate(const std::vector<K> &keyList) override;
    bool invalidate(const K &key) override;

    bool put(const K &key, const V &val) override;
    bool get(const K &key, V &val) override;
    bool update(const K &key, const V &newVal) override;
    bool isInCache(const K &key) override;

    int64_t getCacheSizeUsed() const { return _memSizeUsed; }

    int64_t getCacheSize() const { return _memSize; }

    uint64_t getKeyCount() const { return _accessQueue.getSize(); }

    void setCacheSize(uint64_t memSize) override { _memSize = memSize; }

    GetSizeCallBack &getGetSizeCallBack() { return _getSizeCallBack; }

    int64_t getTotalQueryTimes() { return _totalQueryTimes; }

    int64_t getHitQueryTimes() { return _hitQueryTimes; }

    float getHitRatio() const override {
        if (_totalQueryTimes != 0) {
            return (float)_hitQueryTimes / _totalQueryTimes;
        } else {
            return 0.0f;
        }
    }

    void resetHitStatistics() {
        _totalQueryTimes = 0;
        _hitQueryTimes = 0;
    }

private:
    bool replacement(int64_t valSize);

private:
    GetSizeCallBack _getSizeCallBack;

    int64_t _memSize;
    int64_t _memSizeUsed;
    RecursiveThreadMutex _mutexMemSizeUsed;

    SynchronizedHashTable<K, CacheValue<K, V>> _hashTable;

    int64_t _totalQueryTimes;
    int64_t _hitQueryTimes;

    SynchronizedList<K> _accessQueue;

private:
    static alog::Logger *_logger;
};

template <class K, class V, class GetSizeCallBack>
LruCache<K, V, GetSizeCallBack>::LruCache(int64_t memSize, GetSizeCallBack getSizeCallBack, int64_t elemAvgEstMemSize)
    : _getSizeCallBack(getSizeCallBack)
    , _memSize(memSize)
    , _memSizeUsed(0)
    , _hashTable(elemAvgEstMemSize > 0 ? memSize / elemAvgEstMemSize
                                       : SynchronizedHashTable<K, CacheValue<K, V>>::BUCKET_NUM_DEF)
    , _totalQueryTimes(0)
    , _hitQueryTimes(0) {}

template <class K, class V, class GetSizeCallBack>
LruCache<K, V, GetSizeCallBack>::~LruCache() {}

template <class K, class V, class GetSizeCallBack>
bool LruCache<K, V, GetSizeCallBack>::replacement(int64_t valSize) {
    AUTIL_LOG(TRACE1, "enter replacement, _memSizeUsed=%ld, valSize=%ld", _memSizeUsed, valSize);
    if (_memSize < valSize) {
        AUTIL_LOG(WARN, "valSize exceeds the memory limit, _memSize=%ld", _memSize);
        return false;
    }

    while (valSize + _memSizeUsed > _memSize) {
        K nodeValue;
        bool exist = _accessQueue.getFront(nodeValue);
        if (!exist) {
            AUTIL_LOG(WARN, "Get Front From Access Queue Fail!");
            return false;
        }

        CacheValue<K, V> retVal;
        ReadWriteLock *pLock;
        HashEntry<K, CacheValue<K, V>> *entry = _hashTable.getHashEntry(nodeValue, pLock);
        pLock->wrlock();
        exist = entry->deleteGet(nodeValue, retVal);
        if (exist) {
            {
                ScopedLock l(_mutexMemSizeUsed);
                _memSizeUsed -= _getSizeCallBack(retVal.value);
            }
            ListNode<K> *iter = retVal.iter;
            _accessQueue.erase(iter);
            AUTIL_LOG(TRACE1, "delete succeeded, key=%s", StringUtil::toString(nodeValue).c_str());
        } else {
            AUTIL_LOG(TRACE1, "delete failed, key=%s", StringUtil::toString(nodeValue).c_str());
        }
        pLock->unlock();
    }
    AUTIL_LOG(TRACE1, "leave replacement, _memSizeUsed=%ld", _memSizeUsed);
    return true;
}

template <class K, class V, class GetSizeCallBack>
bool LruCache<K, V, GetSizeCallBack>::put(const K &key, const V &val) {

    int64_t valSize = _getSizeCallBack(val);
    if (valSize + _memSizeUsed > _memSize) {
        bool succ = replacement(valSize);
        if (!succ) {
            AUTIL_LOG(WARN, "put failed, because of replacement _memSizeUsed=%ld", _memSizeUsed);
            return false;
        }
    }

    ReadWriteLock *pLock;
    HashEntry<K, CacheValue<K, V>> *entry = _hashTable.getHashEntry(key, pLock);
    pLock->wrlock();

    // if key already exists in hash table and access queue,
    // it shouldn't be inserted again, but remove old node first

    CacheValue<K, V> cv(val);
    CacheValue<K, V> oldVal;
    CacheValue<K, V> *pretVal;
    bool firstTime = entry->putGet(key, cv, oldVal, pretVal); // FIXME: if overwrite called,
                                                              // memory size used need re-calculate
    if (!firstTime) {
        _accessQueue.moveToBack(oldVal.iter);
        pretVal->iter = oldVal.iter;
        assert(pretVal->iter);
        AUTIL_LOG(TRACE1, "overwrite existing key, key=%s", StringUtil::toString(key).c_str());

        int64_t oldSize = _getSizeCallBack(oldVal.value);
        ScopedLock l(_mutexMemSizeUsed);
        _memSizeUsed += valSize - oldSize;
    } else {
        pretVal->iter = _accessQueue.pushBack(key);
        ScopedLock l(_mutexMemSizeUsed);
        _memSizeUsed += valSize;
    }

    pLock->unlock();

    // std::cout << "Put to cache: key: " << key << std::endl;

    AUTIL_LOG(TRACE1, "put succeeded, key=%s, _memSizeUsed=%ld", StringUtil::toString(key).c_str(), _memSizeUsed);
    return true;
}

template <class K, class V, class GetSizeCallBack>
bool LruCache<K, V, GetSizeCallBack>::get(const K &key, V &val) {
    AUTIL_LOG(TRACE1, "enter get, key=%s", StringUtil::toString(key).c_str());
    ReadWriteLock *pLock;
    HashEntry<K, CacheValue<K, V>> *entry = _hashTable.getHashEntry(key, pLock);
    pLock->rdlock();

    CacheValue<K, V> cacheVal;
    bool exist = entry->get(key, cacheVal);
    if (exist) {
        _accessQueue.moveToBack(cacheVal.iter);
        val = cacheVal.value;
        AUTIL_LOG(TRACE1,
                  "value got, key=%s, valueSize=%ld",
                  StringUtil::toString(key).c_str(),
                  (int64_t)_getSizeCallBack(val));
        _hitQueryTimes++;
    } else {
        AUTIL_LOG(TRACE1, "key not exist, key=%s", StringUtil::toString(key).c_str());
    }
    pLock->unlock();
    _totalQueryTimes++;
    AUTIL_LOG(TRACE1, "leave get");
    return exist;
}

template <class K, class V, class GetSizeCallBack>
bool LruCache<K, V, GetSizeCallBack>::isInCache(const K &key) {
    ReadWriteLock *pLock;
    HashEntry<K, CacheValue<K, V>> *entry = _hashTable.getHashEntry(key, pLock);
    pLock->rdlock();

    CacheValue<K, V> cacheVal;
    bool ret = entry->get(key, cacheVal);
    pLock->unlock();
    return ret;
}

template <class K, class V, class GetSizeCallBack>
bool LruCache<K, V, GetSizeCallBack>::update(const K &key, const V &newVal) {
    return put(key, newVal);
}

template <class K, class V, class GetSizeCallBack>
bool LruCache<K, V, GetSizeCallBack>::invalidate(const K &key) {
    ReadWriteLock *pLock;
    HashEntry<K, CacheValue<K, V>> *entry = _hashTable.getHashEntry(key, pLock);
    pLock->wrlock();

    CacheValue<K, V> cacheVal;
    bool exist = entry->deleteGet(key, cacheVal);
    if (exist) {
        {
            ScopedLock l(_mutexMemSizeUsed);
            _memSizeUsed -= _getSizeCallBack(cacheVal.value);
        }
        ListNode<K> *iter = cacheVal.iter;
        _accessQueue.erase(iter);
        AUTIL_LOG(TRACE1, "invalidated key, key=%s", StringUtil::toString(key).c_str());
    }

    pLock->unlock();
    AUTIL_LOG(TRACE1, "not invalidated key, key=%s", StringUtil::toString(key).c_str());
    return exist;
}

template <class K, class V, class GetSizeCallBack>
void LruCache<K, V, GetSizeCallBack>::invalidate(const std::vector<K> &keyList) {
    typename std::vector<K>::const_iterator iter;
    for (iter = keyList.begin(); iter != keyList.end(); ++iter) {
        invalidate(*iter);
    }
}

template <class K, class V, class GetSizeCallBack>
void LruCache<K, V, GetSizeCallBack>::warmUp(const std::vector<K> &keyList, const std::vector<V> &valueList) {
    typename std::vector<K>::const_iterator iterK;
    typename std::vector<V>::const_iterator iterV;
    for (iterK = keyList.begin(), iterV = valueList.begin(); iterK != keyList.end() && iterV != valueList.end();
         ++iterK, ++iterV) {
        const K &key = *iterK;
        const V &val = *iterV;
        V retVal;
        bool exist = get(key, retVal);
        if (!exist) {
            put(key, val);
        }
    }
}

template <class K, class V, class GetSizeCallBack>
alog::Logger *LruCache<K, V, GetSizeCallBack>::_logger = alog::Logger::getLogger("autil.LruCache");

// for better performance when batch get and batch put

template <class K, class V>
struct CacheValueV2 {
    V value;
    DoubleLinkListNode<K> *iter;

    CacheValueV2() : iter(0) {}

    CacheValueV2(const V &v, DoubleLinkListNode<K> *iterator = 0) : value(v), iter(iterator) {}
};

template <class K, class V, class GetSizeCallBack, class HashTableType>
class UnsafeLruCache : public Cache<K, V> {
public:
    UnsafeLruCache(int64_t memSize,
                   GetSizeCallBack getSizeCallBack = GetSizeCallBack(),
                   int64_t elemAvgEstMemSize = -1);
    ~UnsafeLruCache();

public:
    void warmUp(const std::vector<K> &keyList, const std::vector<V> &valueList) override;
    void invalidate(const std::vector<K> &keyList) override;
    bool invalidate(const K &key) override;

    bool put(const K &key, const V &val) override;
    bool get(const K &key, V &val) override;
    bool update(const K &key, const V &newVal) override;
    bool isInCache(const K &key) override;

    int64_t getCacheSizeUsed() const { return _memSizeUsed.load(std::memory_order_relaxed); }

    int64_t getCacheSize() const { return _memSize.load(std::memory_order_relaxed); }

    int64_t getKeyCount() const { return _keyCount.load(std::memory_order_relaxed); }

    void setCacheSize(uint64_t memSize) override { _memSize.store(memSize, std::memory_order_relaxed); }

    GetSizeCallBack &getGetSizeCallBack() { return _getSizeCallBack; }

    int64_t getTotalQueryTimes() const { return _totalQueryTimes.load(std::memory_order_relaxed); }

    int64_t getHitQueryTimes() const { return _hitQueryTimes.load(std::memory_order_relaxed); }

    float getHitRatio() const override {
        auto totalQueryTimes = getTotalQueryTimes();
        if (totalQueryTimes > 0) {
            return (float)getHitQueryTimes() / totalQueryTimes;
        } else {
            return 0.0f;
        }
    }

    void resetHitStatistics() {
        _totalQueryTimes.store(0, std::memory_order_relaxed);
        _hitQueryTimes.store(0, std::memory_order_relaxed);
    }

private:
    bool replacement(int64_t valSize);

private:
    GetSizeCallBack _getSizeCallBack;

    std::atomic<int64_t> _memSize;
    std::atomic<int64_t> _memSizeUsed;
    std::atomic<int64_t> _keyCount;

    HashTableType _hashTable;
    DoubleLinkList<K> _accessQueue;

    std::atomic<int64_t> _totalQueryTimes;
    std::atomic<int64_t> _hitQueryTimes;

private:
    static alog::Logger *_logger;
};

template <class K, class V, class GetSizeCallBack, class HashTableType>
alog::Logger *UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::_logger =
    alog::Logger::getLogger("autil.UnsafeLruCache");

template <class K, class V, class GetSizeCallBack, class HashTableType>
UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::UnsafeLruCache(int64_t memSize,
                                                                     GetSizeCallBack getSizeCallBack,
                                                                     int64_t elemAvgEstMemSize)
    : _getSizeCallBack(getSizeCallBack)
    , _memSize(memSize)
    , _memSizeUsed(0)
    , _keyCount(0)
    , _totalQueryTimes(0)
    , _hitQueryTimes(0) {}

template <class K, class V, class GetSizeCallBack, class HashTableType>
UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::~UnsafeLruCache() {
    while (!_accessQueue.isEmpty()) {
        auto head = _accessQueue.getHead();
        _accessQueue.deleteAndFreeNode(head);
    }
}

// lock in caller
template <class K, class V, class GetSizeCallBack, class HashTableType>
bool UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::replacement(int64_t valSize) {
    AUTIL_LOG(TRACE1, "enter replacement, _memSizeUsed=%ld, valSize=%ld", getCacheSizeUsed(), valSize);

    if (getCacheSize() < valSize) {
        AUTIL_LOG(WARN, "valSize exceeds the memory limit, _memSize=%ld, valSize=%ld", getCacheSize(), valSize);
        return false;
    }

    while (valSize + getCacheSizeUsed() > getCacheSize()) {
        if (_accessQueue.isEmpty()) {
            AUTIL_LOG(WARN, "Get Front From Access Queue Fail!");
            return false;
        }
        const K &key = _accessQueue.getHead()->getData();
        auto hashIt = _hashTable.find(key);
        if (hashIt == _hashTable.end()) {
            AUTIL_LOG(WARN, "Key not found in hash table, key=%s", StringUtil::toString(key).c_str());
            _accessQueue.popFront();
            _keyCount.fetch_sub(1, std::memory_order_relaxed);
            continue;
        }

        auto &cachedValue = hashIt->second.value;
        _memSizeUsed.fetch_sub(_getSizeCallBack(cachedValue), std::memory_order_relaxed);
        _hashTable.erase(hashIt);
        _accessQueue.popFront();
        _keyCount.fetch_sub(1, std::memory_order_relaxed);
        AUTIL_LOG(TRACE1, "delete succeeded, key=%s", StringUtil::toString(key).c_str());
    }

    AUTIL_LOG(TRACE1, "leave replacement, _memSizeUsed=%ld", getCacheSizeUsed());
    return true;
}

template <class K, class V, class GetSizeCallBack, class HashTableType>
bool UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::put(const K &key, const V &val) {
    int64_t valSize = _getSizeCallBack(val);
    int64_t sizeDiff = 0;
    {
        if (valSize + getCacheSizeUsed() > getCacheSize()) {
            bool succ = replacement(valSize);
            if (!succ) {
                AUTIL_LOG(WARN, "put failed, because of replacement _memSizeUsed=%ld", getCacheSizeUsed());
                return false;
            }
        }

        auto [hashIt, tryFlag] = _hashTable.try_emplace(key, val, nullptr);
        if (!tryFlag) {
            auto &existVal = hashIt->second;
            int64_t oldValSize = _getSizeCallBack(existVal.value);
            _accessQueue.moveToBack(existVal.iter);
            existVal.value = val;
            sizeDiff = _getSizeCallBack(val) - oldValSize;
        } else {
            _accessQueue.pushBack(key);
            hashIt->second.iter = _accessQueue.getTail();
            _keyCount.fetch_add(1, std::memory_order_relaxed);
            sizeDiff = _getSizeCallBack(val);
        }
    }
    _memSizeUsed.fetch_add(sizeDiff, std::memory_order_relaxed);

    AUTIL_LOG(TRACE1, "put succeeded, key=%s, _memSizeUsed=%ld", StringUtil::toString(key).c_str(), getCacheSizeUsed());
    return true;
}

template <class K, class V, class GetSizeCallBack, class HashTableType>
bool UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::get(const K &key, V &val) {
    AUTIL_LOG(TRACE1, "enter get, key=%s", StringUtil::toString(key).c_str());
    bool exist = false;
    {
        auto hashIt = _hashTable.find(key);
        exist = (hashIt != _hashTable.end());
        if (exist) {
            auto &existVal = hashIt->second;
            val = existVal.value;
            _accessQueue.moveToBack(existVal.iter);

            AUTIL_LOG(TRACE1,
                      "value got, key=%s, valueSize=%ld",
                      StringUtil::toString(key).c_str(),
                      (int64_t)_getSizeCallBack(val));
        } else {
            AUTIL_LOG(TRACE1, "key not exist, key=%s", StringUtil::toString(key).c_str());
        }
    }
    if (exist) {
        _hitQueryTimes.fetch_add(1, std::memory_order_relaxed);
    }
    _totalQueryTimes.fetch_add(1, std::memory_order_relaxed);
    AUTIL_LOG(TRACE1, "leave get");
    return exist;
}

template <class K, class V, class GetSizeCallBack, class HashTableType>
bool UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::isInCache(const K &key) {

    return _hashTable.count(key);
}

template <class K, class V, class GetSizeCallBack, class HashTableType>
bool UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::update(const K &key, const V &newVal) {
    return put(key, newVal);
}

template <class K, class V, class GetSizeCallBack, class HashTableType>
bool UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::invalidate(const K &key) {
    bool exist = false;
    {
        auto hashIt = _hashTable.find(key);
        exist = (hashIt != _hashTable.end());
        if (exist) {
            auto &existVal = hashIt->second;
            _accessQueue.deleteAndFreeNode(existVal.iter);
            _keyCount.fetch_sub(1, std::memory_order_relaxed);
            _memSizeUsed.fetch_sub(_getSizeCallBack(existVal.value), std::memory_order_relaxed);
            _hashTable.erase(hashIt);
        }
    }

    AUTIL_LOG(TRACE1, "%s key, key=%s", exist ? "invalidated" : "not invalidated", StringUtil::toString(key).c_str());
    return exist;
}

template <class K, class V, class GetSizeCallBack, class HashTableType>
void UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::invalidate(const std::vector<K> &keyList) {
    typename std::vector<K>::const_iterator iter;
    for (iter = keyList.begin(); iter != keyList.end(); ++iter) {
        invalidate(*iter);
    }
}

template <class K, class V, class GetSizeCallBack, class HashTableType>
void UnsafeLruCache<K, V, GetSizeCallBack, HashTableType>::warmUp(const std::vector<K> &keyList,
                                                                  const std::vector<V> &valueList) {
    typename std::vector<K>::const_iterator iterK;
    typename std::vector<V>::const_iterator iterV;
    for (iterK = keyList.begin(), iterV = valueList.begin(); iterK != keyList.end() && iterV != valueList.end();
         ++iterK, ++iterV) {
        const K &key = *iterK;
        const V &val = *iterV;
        V retVal;
        bool exist = get(key, retVal);
        if (!exist) {
            put(key, val);
        }
    }
}

} // namespace autil
