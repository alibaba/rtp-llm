#pragma once

#include "navi/common.h"
#include "navi/config/NaviConfig.h"
#include "navi/engine/NaviError.h"
#include "navi/engine/StringMapWithSize.h"

namespace navi {

struct UpdateEvent {
    UpdateHistoryEventType type = NHET_UNKNOWN;
    size_t configHash = 0;
    std::shared_ptr<std::string> configStr;
    NaviErrorPtr error;
    int64_t beginTimeUs = 0;
    int64_t endTimeUs = 0;
    int64_t dupCount = 0;
    int64_t lastBeginTimeUs = 0;
    int64_t lastEndTimeUs = 0;
    uint64_t loadCount = 0;
};

NAVI_TYPEDEF_PTR(UpdateEvent);

class UpdateHistory;
NAVI_TYPEDEF_PTR(UpdateHistory);

class UpdateHistory
{
public:
    UpdateHistory();
    ~UpdateHistory();
    UpdateHistory(const UpdateHistory &) = delete;
    UpdateHistory &operator=(const UpdateHistory &) = delete;
public:
    void updateConfig(const UpdateHistoryConfig &config);
    bool
    addUpdateEvent(const std::string &configStr, const NaviErrorPtr &error, int64_t beginTimeUs, int64_t endTimeUs);
    void addControlEvent(UpdateHistoryEventType type, int64_t beginTimeUs, int64_t endTimeUs);
    void toProto(UpdateHistoryDef &historyDef, SymbolTableDef &symbolTableDef) const;
    void fromProto(const UpdateHistoryDef &historyDef, const SymbolTableDef &symbolTableDef);
    size_t getEventCount() const;
private:
    bool dedupControlEvent(UpdateHistoryEventType type, int64_t beginTimeUs, int64_t endTimeUs);
    bool dedupEvent(UpdateHistoryEventType type,
                    const std::string &configStr,
                    const NaviErrorPtr &error,
                    int64_t beginTimeUs,
                    int64_t endTimeUs);
    bool doDedupEvent(UpdateHistoryEventType type,
                      const UpdateEventPtr &event,
                      const std::string &configStr,
                      const NaviErrorPtr &error);
    void dropToFit();
    bool isQuotaOverflow() const;
    void parseConfig(const UpdateHistoryConfigDef &configDef);
    void fillConfig(UpdateHistoryConfigDef &configDef) const;
private:
    mutable autil::ReadWriteLock _lock;
    UpdateHistoryConfig _config;
    std::list<UpdateEventPtr> _updateEvents;
    StringMapWithSize _configStrMap;
};

}

