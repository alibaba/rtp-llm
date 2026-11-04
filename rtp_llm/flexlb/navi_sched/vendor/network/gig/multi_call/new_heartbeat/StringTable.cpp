#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/StringTable.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/NewHeartbeat.pb.h"

namespace multi_call {

void StringTableShard::addHash(uint64_t hash, const std::string &str,
                               std::shared_ptr<std::string> *strPtr) {
    {
        autil::ScopedReadWriteLock lock(_lock, 'r');
        auto it = _table.find(hash);
        if (_table.end() != it) {
            if (strPtr) {
                *strPtr = it->second;
            }
            return;
        }
    }
    {
        auto value = std::make_shared<std::string>(str);
        autil::ScopedReadWriteLock lock(_lock, 'w');
        auto it = _table.find(hash);
        if (_table.end() == it) {
            _table[hash] = value;
            if (strPtr) {
                *strPtr = value;
            }
        } else {
            if (strPtr) {
                *strPtr = it->second;
            }
        }
        return;
    }
}

std::shared_ptr<std::string> StringTableShard::getStr(size_t hash) const {
    autil::ScopedReadWriteLock lock(_lock, 'r');
    auto it = _table.find(hash);
    if (_table.end() != it) {
        return it->second;
    } else {
        return nullptr;
    }
}

void StringTableShard::recycle() {
    std::vector<std::shared_ptr<std::string>> recycleStrs;
    {
        autil::ScopedReadWriteLock lock(_lock, 'w');
        for (auto it = _table.begin(); it != _table.end();) {
            if (1u == it->second.use_count()) {
                recycleStrs.push_back(it->second);
                it = _table.erase(it);
            } else {
                it++;
            }
        }
    }
    recycleStrs.clear();
}

size_t StringTable::addStr(const std::string &str, std::shared_ptr<std::string> *strPtr) {
    auto hash = _hashFunc(str);
    getShard(hash)->addHash(hash, str, strPtr);
    return hash;
}

void StringTable::addHash(uint64_t hash, const std::string &str,
                          std::shared_ptr<std::string> *strPtr) {
    getShard(hash)->addHash(hash, str, strPtr);
}

std::shared_ptr<std::string> StringTable::getStr(size_t hash) const {
    return getShard(hash)->getStr(hash);
}

void StringTable::recycle(bool force) {
    auto currentTime = autil::TimeUtility::currentTime();
    if (!force && _lastRecycleTimeUs + 10 * 1000 * 1000 > currentTime) {
        return;
    }
    _lastRecycleTimeUs = currentTime;
    for (auto shard : _shards) {
        shard->recycle();
    }
}

} // namespace multi_call
