#pragma once

#include "navi/common.h"
#include "navi/config/NaviConfig.h"

namespace navi {

struct LocalSubGraphInternal;
class CacheSummaryDef;

class LocalSubGraphCache
{
public:
    LocalSubGraphCache(const std::string &bizName);
    ~LocalSubGraphCache();
    LocalSubGraphCache(const LocalSubGraphCache &) = delete;
    LocalSubGraphCache &operator=(const LocalSubGraphCache &) = delete;
public:
    bool init(const CacheConfig &config);
    void reclaim();
    bool putGraph(size_t key, const std::shared_ptr<LocalSubGraphInternal> &graph);
    std::shared_ptr<LocalSubGraphInternal> getGraph(size_t key) const;
    void fillProto(CacheSummaryDef &summaryDef) const;
private:
    inline bool needReclaim() const {
        return (_maxCacheSize > 0 && _totalSize > _maxCacheSize);
    }
private:
    std::string _bizName;
    int64_t _maxCacheSize = -1;
    mutable autil::ReadWriteLock _lock;
    volatile size_t _totalSize = 0;
    size_t _reclaimCount = 0;
    NaviUnorderedMap<size_t, std::shared_ptr<LocalSubGraphInternal>> _graphMap;
};

}
