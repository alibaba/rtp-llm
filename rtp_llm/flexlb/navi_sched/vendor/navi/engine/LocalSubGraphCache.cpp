#include "navi/engine/LocalSubGraphCache.h"
#include "navi/engine/LocalSubGraphInternal.h"
#include "navi/proto/NaviSummary.pb.h"

namespace navi {

LocalSubGraphCache::LocalSubGraphCache(const std::string &bizName)
    : _bizName(bizName)
{
}

LocalSubGraphCache::~LocalSubGraphCache() {
}

bool LocalSubGraphCache::init(const CacheConfig &config) {
    _maxCacheSize = config.maxCacheSizeMB * 1024 * 1024;
    return true;
}

bool LocalSubGraphCache::putGraph(size_t key, const LocalSubGraphInternalPtr &graph) {
    if (!graph) {
        return true;
    }
    graph->hashKey = key;
    graph->updateAccessTime();
    auto graphSize = graph->getSize();
    if (_maxCacheSize > 0 && graphSize > _maxCacheSize) {
        return false;
    }
    {
        autil::ScopedWriteLock scope(_lock);
        if (unlikely(needReclaim())) {
            NAVI_KERNEL_LOG(DEBUG,
                            "put failed, need reclaim, graph size [%ld], cache totalSize [%lu] graph count [%lu]",
                            graphSize,
                            _totalSize,
                            _graphMap.size());
            return false;
        }
        auto it = _graphMap.find(key);
        if (_graphMap.end() != it) {
            return false;
        }
        _totalSize += graphSize;
        _graphMap[key] = graph;
        return true;
    }
}

LocalSubGraphInternalPtr LocalSubGraphCache::getGraph(size_t key) const {
    autil::ScopedReadLock scope(_lock);
    auto it = _graphMap.find(key);
    if (_graphMap.end() == it) {
        return LocalSubGraphInternalPtr();
    } else {
        it->second->updateAccessTime();
        return it->second;
    }
}

void LocalSubGraphCache::reclaim() {
    if (!needReclaim()) {
        NAVI_KERNEL_LOG(DEBUG, "biz [%s], graph cache total size [%lu]", _bizName.c_str(), _totalSize);
        return;
    }
    std::map<int64_t, size_t> reclaimMap;
    size_t beforeReclaimSize = 0;
    {
        autil::ScopedReadLock scope(_lock);
        if (!needReclaim()) {
            return;
        }
        for (const auto &pair : _graphMap) {
            if (1 == pair.second.use_count()) {
                reclaimMap.emplace(pair.second->lastAccessTime, pair.first);
            }
        }
        beforeReclaimSize = _totalSize;
    }
    std::vector<std::shared_ptr<LocalSubGraphInternal>> reclaimVec;
    reclaimVec.reserve(32);
    size_t afterReclaimSize = 0;
    size_t graphCountAfterReclaim = 0;
    {
        autil::ScopedWriteLock scope(_lock);
        if (!needReclaim()) {
            return;
        }
        for (const auto &pair : reclaimMap) {
            auto it = _graphMap.find(pair.second);
            if (_graphMap.end() == it) {
                continue;
            }
            if (1 != it->second.use_count()) {
                continue;
            }
            if (!needReclaim()) {
                break;
            }
            reclaimVec.push_back(it->second);
            _totalSize -= it->second->getSize();
            _graphMap.erase(it);
        }
        afterReclaimSize = _totalSize;
        graphCountAfterReclaim = _graphMap.size();
    }
    _reclaimCount += reclaimVec.size();
    if (!reclaimVec.empty()) {
        NAVI_KERNEL_LOG(
            INFO,
            "biz [%s], reclaim graph cache, reclaim count [%lu] size [%lu], cache totalSize [%lu] graph count [%lu]",
            _bizName.c_str(),
            reclaimVec.size(),
            beforeReclaimSize - afterReclaimSize,
            _totalSize,
            graphCountAfterReclaim);
    }
}

void LocalSubGraphCache::fillProto(CacheSummaryDef &summaryDef) const {
    summaryDef.Clear();
    NaviUnorderedMap<size_t, std::shared_ptr<LocalSubGraphInternal>> graphMap;
    {
        autil::ScopedReadLock scope(_lock);
        graphMap = _graphMap;
        summaryDef.set_max_cache_size(_maxCacheSize);
        summaryDef.set_reclaim_count(_reclaimCount);
    }
    summaryDef.set_total_count(graphMap.size());
    constexpr size_t MAX_SUMMARY_COUNT = 128;
    std::vector<LocalSubGraphInternal *> graphVec;
    {
        graphVec.reserve(graphMap.size());
        for (const auto &pair : graphMap) {
            graphVec.push_back(pair.second.get());
        }
        std::sort(graphVec.begin(), graphVec.end(), [](LocalSubGraphInternal *lhs, LocalSubGraphInternal *rhs) {
            return rhs->getSize() < lhs->getSize();
        });
    }
    auto maxCount = std::min(MAX_SUMMARY_COUNT, graphVec.size());
    for (int i = 0; i < maxCount; i++) {
        auto graphSummaryDef = summaryDef.add_graph_summarys();
        graphVec[i]->fillProto(*graphSummaryDef);
    }
}

}

