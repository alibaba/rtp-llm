#pragma once

#include "navi/engine/UpdateHistory.h"
#include "navi/engine/QueryHistory.h"
#include "autil/LoopThread.h"

namespace navi {

class NaviHistory
{
public:
    NaviHistory();
    ~NaviHistory();
    NaviHistory(const NaviHistory &) = delete;
    NaviHistory &operator=(const NaviHistory &) = delete;
public:
    bool init(const std::string &naviName);
public:
    void updateConfig(const HistoryConfig &config);
    void
    addUpdateEvent(const std::string &configStr, const NaviErrorPtr &error, int64_t beginTimeUs, int64_t endTimeUs);
    void addControlEvent(UpdateHistoryEventType type, int64_t beginTimeUs, int64_t endTimeUs);
    void addQuery(const std::shared_ptr<NaviArpcRequestData> &requestData,
                  const RegistryArpcParam &arpcParam,
                  http_arpc::HTTPRPCServerClosure *httpClosure,
                  const std::shared_ptr<NaviResult> &naviResult,
                  const std::string &queryKey,
                  google::protobuf::Message *response,
                  bool needDebug);
public:
    void toProto(bool shortQueryHistory, NaviHistoryDef &historyDef) const;
private:
    bool doInit(const std::string &naviName);
    void fromProto(const NaviHistoryDef &historyDef);
    void saveToFile();
    void workLoop();
private:
    std::string _fileName;
    UpdateHistory _updateHistory;
    QueryHistory _queryHistory;
    autil::LoopThreadPtr _loopThread;
};

class UpdateHistoryScope {
public:
    UpdateHistoryScope(const std::string &configStr)
        : _beginTimeUs(autil::TimeUtility::currentTime())
        , _configStr(configStr)
    {
    }
    ~UpdateHistoryScope() {
        if (_history) {
            _history->addUpdateEvent(_configStr, _error, _beginTimeUs, autil::TimeUtility::currentTime());
        }
    }
public:
    void setConfigStr(const std::string &str) {
        _configStr = str;
    }
    void setHistory(NaviHistory *history) {
        _history = history;
    }
    void setError(const NaviErrorPtr &error) {
        _error = error;
    }
    const NaviErrorPtr &getError() const {
        return _error;
    }
private:
    NaviHistory *_history = nullptr;
    int64_t _beginTimeUs = 0;
    std::string _configStr;
    NaviErrorPtr _error;
};

}

