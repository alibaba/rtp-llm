#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/NewHeartbeat.pb.h"

namespace multi_call {

class StringTableDef;

class StringTableShard
{
public:
    StringTableShard() {
    }
    ~StringTableShard() {
    }
    StringTableShard(const StringTableShard &) = delete;
    StringTableShard &operator=(const StringTableShard &) = delete;

public:
    void addHash(uint64_t hash, const std::string &str,
                 std::shared_ptr<std::string> *strPtr = nullptr);
    std::shared_ptr<std::string> getStr(size_t hash) const;
    void recycle();

private:
    mutable autil::ReadWriteLock _lock;
    std::unordered_map<size_t, std::shared_ptr<std::string>> _table;
};

class StringTable
{
public:
    StringTable() {
        for (size_t i = 0; i < SHARD_COUNT; i++) {
            _shards.push_back(new StringTableShard());
        }
    }
    ~StringTable() {
        for (auto shard : _shards) {
            delete shard;
        }
    }
    StringTable(const StringTable &) = delete;
    StringTable &operator=(const StringTable &) = delete;

public:
    size_t addStr(const std::string &str, std::shared_ptr<std::string> *strPtr = nullptr);
    void addHash(uint64_t hash, const std::string &str,
                 std::shared_ptr<std::string> *strPtr = nullptr);
    std::shared_ptr<std::string> getStr(size_t hash) const;
    void recycle(bool force);

private:
    static constexpr size_t SHARD_COUNT = 16;

private:
    StringTableShard *getShard(uint64_t hash) const {
        return _shards[hash % SHARD_COUNT];
    }

private:
    int64_t _lastRecycleTimeUs = 0;
    std::hash<std::string> _hashFunc;
    std::vector<StringTableShard *> _shards;
};

MULTI_CALL_TYPEDEF_PTR(StringTable);

} // namespace multi_call
