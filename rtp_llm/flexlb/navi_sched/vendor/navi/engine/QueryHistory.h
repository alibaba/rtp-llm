#pragma once

#include "navi/engine/QueryLRUHistory.h"
#include <arpc/common/LockFreeQueue.h>

namespace http_arpc {
class HTTPRPCServerClosure;
}

namespace navi {

class NaviArpcRequestData;
class NaviResult;
struct RegistryArpcParam;

class QueryHistory
{
public:
    QueryHistory();
    ~QueryHistory();
    QueryHistory(const QueryHistory &) = delete;
    QueryHistory &operator=(const QueryHistory &) = delete;
public:
    void updateConfig(const QueryHistoryConfig &config);
    void toProto(bool shortQueryHistory, QueryHistoryDef &historyDef, SymbolTableDef &symbolTableDef) const;
    void fromProto(const QueryHistoryDef &historyDef, const SymbolTableDef &symbolTableDef);
    void addQuery(const std::shared_ptr<NaviArpcRequestData> &requestData,
                  const RegistryArpcParam &arpcParam,
                  http_arpc::HTTPRPCServerClosure *httpClosure,
                  const std::shared_ptr<NaviResult> &naviResult,
                  const std::string &queryKey,
                  google::protobuf::Message *response,
                  bool needDebug);
    std::pair<size_t, size_t> getQueryCount() const {
        return std::make_pair(_normalHistory.getSampleCount(), _errorHistory.getSampleCount());
    }
public:
    void workLoop();
private:
    void parseConfig(const QueryHistoryConfigDef &configDef);
    void fillConfig(QueryHistoryConfigDef &configDef) const;
    bool checkTimeInterval(int64_t beginTimeUs, bool hasError, int64_t &prevLogTimeUs) const;
    bool updateLogTime(int64_t beginTimeUs, bool hasError, int64_t prevLogTimeUs);
private:
    int64_t _sampleIntervalUs = DEFAULT_HISTORY_QUERY_SAMPLE_INTERVAL_US;
    arpc::common::LockFreeQueue<QuerySample *> _queryQueue;
    volatile int64_t _prevLogTimeUs = 0;
    volatile int64_t _prevErrorLogTimeUs = 0;
    QueryLRUHistory _normalHistory;
    QueryLRUHistory _errorHistory;
};

}

