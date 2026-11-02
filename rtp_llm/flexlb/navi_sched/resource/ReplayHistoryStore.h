#pragma once

#include "aios/navi_sched/data/ReplayControlData.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace navi_sched {

class ReplayHistoryStore {
public:
    ~ReplayHistoryStore();
    bool configure(const std::string &directory, size_t historyLimit,
                   const std::string &legacyDirectory = {});
    bool append(const NaviSchedScheduleRequest &request, int64_t eventTimeUs);
    ReplayHistorySummary getSummary(const std::string &clientId) const;
    bool loadTimeRange(int64_t startTimeUs, int64_t endTimeUs,
                       std::vector<ReplayHistorySample> &samples,
                       std::string &error) const;
    bool loadRecent(size_t count, std::vector<ReplayHistorySample> &samples,
                    std::string &error) const;
    bool persistent() const;
    size_t residentRequestCount() const;

private:
    struct IndexEntry {
        int64_t eventTimeUs = 0;
        uint64_t recordOffset = 0;
        uint64_t payloadOffset = 0;
        uint64_t payloadSize = 0;
        uint32_t checksum = 0;
        uint32_t recordVersion = 0;
    };

    bool scanLocked() const;
    bool refreshLocked() const;
    bool loadLocked(const std::vector<IndexEntry> &entries,
                    std::vector<ReplayHistorySample> &samples,
                    std::string &error) const;
    bool compactLocked();
    void retainNewestLocked() const;

private:
    mutable std::mutex _mutex;
    std::string _dataFile;
    size_t _historyLimit = 20000;
    mutable uint64_t _validDataSize = 0;
    mutable uint64_t _persistedRecordCount = 0;
    mutable uint64_t _incompatibleRecordCount = 0;
    mutable std::vector<IndexEntry> _index;
    mutable int _indexedFd = -1;
    std::vector<ReplayHistorySample> _volatileSamples;
};

}
