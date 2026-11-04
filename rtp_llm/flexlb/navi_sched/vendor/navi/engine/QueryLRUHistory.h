#pragma once

#include "navi/config/NaviConfig.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/engine/NaviError.h"
#include "navi/engine/StringMapWithSize.h"

namespace navi {

class QuerySample {
public:
    virtual ~QuerySample() {
    }
public:
    std::string arpcMethod;
    std::string httpPath;
    std::string clientIp;
    bool isHttp = false;
    bool naviWebQuery = false;
    SessionId session;
    std::string queryKey;
    std::string query;
    std::shared_ptr<std::string> queryStr;
    size_t queryHash = 0;
    int64_t responseSize = -1;
    NaviErrorPtr error;
    int64_t beginTime = 0;
    int64_t endTime = 0;
    int64_t dupCount = 0;
    int64_t queueUs = 0;
    int64_t computeUs = 0;
    int64_t lastBeginTimeUs = 0;
    int64_t lastEndTimeUs = 0;
};

NAVI_TYPEDEF_PTR(QuerySample);

class QueryLRUEntry {
public:
    QueryLRUEntry(StringMapWithSize &queryStrMap)
        : _queryStrMap(queryStrMap)
    {
    }
public:
    bool collect(QuerySample *sample);
    bool drop(int64_t remainCount);
    void toProto(int64_t &maxCount, QueryLRUEntryDef &historyDef, SymbolTableDef &symbolTableDef) const;
    void fromProto(const QueryLRUEntryDef &historyDef, const SymbolTableDef &symbolTableDef);
    void setId(int64_t id) {
        _id = id;
    }
    int64_t getId() const {
        return _id;
    }
    size_t size() const {
        return _querySampleList.size();
    }
    QuerySamplePtr getFirstSample() const {
        if (!_querySampleList.empty()) {
            return _querySampleList.front();
        } else {
            return nullptr;
        }
    }
private:
    bool dedupSample(const QuerySamplePtr &sample);
private:
    int64_t _id = 0;
    StringMapWithSize &_queryStrMap;
    NaviUnorderedMap<size_t, QuerySamplePtr> _querySampleMap;
    std::list<QuerySamplePtr> _querySampleList;
};

NAVI_TYPEDEF_PTR(QueryLRUEntry);

class QueryLRUHistory
{
public:
    QueryLRUHistory();
    ~QueryLRUHistory();
    QueryLRUHistory(const QueryLRUHistory &) = delete;
    QueryLRUHistory &operator=(const QueryLRUHistory &) = delete;
public:
    void updateConfig(const QueryLRUHistoryConfig &config);
    void parseConfig(const QueryLRUHistoryConfigDef &configDef);
    void fillConfig(QueryLRUHistoryConfigDef &configDef) const;
    void toProto(bool shortQueryHistory, QueryLRUHistoryDef &historyDef, SymbolTableDef &symbolTableDef) const;
    void fromProto(const QueryLRUHistoryDef &historyDef, const SymbolTableDef &symbolTableDef);
    void collect(QuerySample *sample);
    size_t getSampleCount() const {
        return _sampleCount;
    }
private:
    void dropToFit();
    void doDropToFit(int64_t threshold);
    void updatLRU(int64_t oldId, const QueryLRUEntryPtr &queryEntry);
    bool isQuotaOverflow() const;
private:
    static size_t getErrorKey(const NaviErrorPtr &error);
private:
    mutable autil::ReadWriteLock _lock;
    int64_t _idCounter = 0;
    QueryLRUHistoryConfig _config;
    StringMapWithSize _queryStrMap;
    NaviUnorderedMap<size_t, QueryLRUEntryPtr> _queryMap;
    std::map<int64_t, QueryLRUEntryPtr> _queryLRU;
    size_t _sampleCount = 0;
};

}

