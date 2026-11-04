#include "navi/engine/UpdateHistory.h"
#include "navi/log/NaviLogger.h"

namespace navi {

UpdateHistory::UpdateHistory() {
}

UpdateHistory::~UpdateHistory() {
}

void UpdateHistory::addControlEvent(UpdateHistoryEventType type, int64_t beginTimeUs, int64_t endTimeUs) {
    autil::ScopedWriteLock scope(_lock);
    if (dedupControlEvent(type, beginTimeUs, endTimeUs)) {
        return;
    }
    auto event = std::make_shared<UpdateEvent>();
    event->type = type;
    event->beginTimeUs = beginTimeUs;
    event->endTimeUs = endTimeUs;
    if (NHET_START == type) {
        event->loadCount = _updateEvents.size();
    }
    _updateEvents.push_back(event);
}

void UpdateHistory::updateConfig(const UpdateHistoryConfig &config) {
    autil::ScopedWriteLock scope(_lock);
    _config = config;
    dropToFit();
}

bool UpdateHistory::addUpdateEvent(const std::string &configStr,
                                 const NaviErrorPtr &error,
                                 int64_t beginTimeUs,
                                 int64_t endTimeUs)
{
    auto type = NHET_UPDATE;
    autil::ScopedWriteLock scope(_lock);
    if (dedupEvent(type, configStr, error, beginTimeUs, endTimeUs)) {
        return false;
    }
    auto event = std::make_shared<UpdateEvent>();
    event->type = type;
    event->configStr = _configStrMap.getStr(configStr, event->configHash);
    event->error = error;
    event->beginTimeUs = beginTimeUs;
    event->endTimeUs = endTimeUs;
    _updateEvents.push_back(event);
    dropToFit();
    return true;
}

bool UpdateHistory::dedupControlEvent(UpdateHistoryEventType type, int64_t beginTimeUs, int64_t endTimeUs) {
    if (_updateEvents.empty()) {
        return false;
    }
    const auto &event = _updateEvents.back();
    if (event->type != type) {
        return false;
    }
    event->dupCount++;
    event->lastBeginTimeUs = beginTimeUs;
    event->lastEndTimeUs = endTimeUs;
    return true;
}

bool UpdateHistory::dedupEvent(UpdateHistoryEventType type,
                             const std::string &configStr,
                             const NaviErrorPtr &error,
                             int64_t beginTimeUs,
                             int64_t endTimeUs)
{
    if (!error) {
        return false;
    }
    if (_updateEvents.empty()) {
        return false;
    }
    const auto &event = _updateEvents.back();
    auto ret = doDedupEvent(type, event, configStr, error);
    if (ret) {
        event->dupCount++;
        event->lastBeginTimeUs = beginTimeUs;
        event->lastEndTimeUs = endTimeUs;
    }
    return ret;
}

bool UpdateHistory::doDedupEvent(UpdateHistoryEventType type,
                               const UpdateEventPtr &event,
                               const std::string &configStr,
                               const NaviErrorPtr &error)
{
    if (type != event->type) {
        return false;
    }
    if (*event->configStr != configStr) {
        return false;
    }
    const auto &eventError = event->error;
    if (!eventError) {
        return false;
    }
    if (eventError->ec != error->ec) {
        return false;
    }
    if (eventError->location.SerializeAsString() != error->location.SerializeAsString()) {
        return false;
    }
    const auto &errorEventA = eventError->errorEvent;
    const auto &errorEventB = error->errorEvent;
    if (!errorEventA) {
        if (errorEventB) {
            return false;
        } else {
            return true;
        }
    }
    if (!errorEventB) {
        return false;
    }
    if (errorEventA->message == errorEventB->message && errorEventA->file == errorEventB->file &&
        errorEventA->line == errorEventB->line && errorEventA->func == errorEventB->func &&
        errorEventA->level == errorEventB->level)
    {
        return true;
    }
    return false;
}

size_t UpdateHistory::getEventCount() const {
    autil::ScopedReadLock scope(_lock);
    return _updateEvents.size();
}

void UpdateHistory::toProto(UpdateHistoryDef &historyDef, SymbolTableDef &symbolTableDef) const {
    autil::ScopedReadLock scope(_lock);
    if (_updateEvents.empty()) {
        return;
    }
    auto &symbolTable = *symbolTableDef.mutable_table();
    for (const auto &event : _updateEvents) {
        auto eventDef = historyDef.add_events();
        eventDef->set_type(event->type);
        if (event->configStr) {
            eventDef->set_config_hash(event->configHash);
            auto it = symbolTable.find(event->configHash);
            if (symbolTable.end() == it) {
                symbolTable[event->configHash] = *event->configStr;
            }
        }
        if (event->error) {
            event->error->toProto(*eventDef->mutable_error());
        }
        eventDef->set_begin_time_us(event->beginTimeUs);
        eventDef->set_end_time_us(event->endTimeUs);
        eventDef->set_dup_count(event->dupCount);
        eventDef->set_last_begin_time_us(event->lastBeginTimeUs);
        eventDef->set_last_end_time_us(event->lastEndTimeUs);
        eventDef->set_load_count(event->loadCount);
    }
    fillConfig(*historyDef.mutable_config());
}

void UpdateHistory::fromProto(const UpdateHistoryDef &historyDef, const SymbolTableDef &symbolTableDef) {
    autil::ScopedWriteLock scope(_lock);
    auto count = historyDef.events_size();
    const auto &symbolTable = symbolTableDef.table();
    for (int64_t i = 0; i < count; i++) {
        const auto &eventDef = historyDef.events(i);
        auto event = std::make_shared<UpdateEvent>();
        event->type = eventDef.type();
        if (NHET_UPDATE == event->type) {
            auto configHash = eventDef.config_hash();
            auto it = symbolTable.find(configHash);
            if (symbolTable.end() != it) {
                const auto &configStr = it->second;
                event->configStr = _configStrMap.getStr(configStr, event->configHash);
            } else {
                event->configHash = configHash;
                size_t tmp;
                event->configStr = _configStrMap.getStr("error, config not exist", tmp);
            }
        }
        if (eventDef.has_error()) {
            auto error = std::make_shared<NaviError>();
            error->fromProto(eventDef.error());
            event->error = error;
        }
        event->beginTimeUs = eventDef.begin_time_us();
        event->endTimeUs = eventDef.end_time_us();
        event->dupCount = eventDef.dup_count();
        event->lastBeginTimeUs = eventDef.last_begin_time_us();
        event->lastEndTimeUs = eventDef.last_end_time_us();
        event->loadCount = eventDef.load_count();
        _updateEvents.push_back(event);
    }
    parseConfig(historyDef.config());
}

void UpdateHistory::parseConfig(const UpdateHistoryConfigDef &configDef) {
    _config.maxSize = configDef.max_size();
    _config.maxCount = configDef.max_count();
}

void UpdateHistory::fillConfig(UpdateHistoryConfigDef &configDef) const {
    configDef.set_max_size(_config.maxSize);
    configDef.set_max_count(_config.maxCount);
}

void UpdateHistory::dropToFit() {
    while (isQuotaOverflow() && !_updateEvents.empty()) {
        auto event = _updateEvents.front();
        _updateEvents.pop_front();
        _configStrMap.removeStr(event->configStr);
    }
}

bool UpdateHistory::isQuotaOverflow() const {
    constexpr size_t EVENT_SIZE = (sizeof(UpdateEvent) + sizeof(UpdateEventPtr));
    auto eventCount = _updateEvents.size();
    return (eventCount > _config.maxCount) ||
           ((_configStrMap.getTotalSize() + eventCount * EVENT_SIZE) > _config.maxSize);
}

}

