#include "navi/engine/QueryLRUHistory.h"
#include "navi/log/NaviLogger.h"
#include "autil/HashUtil.h"

namespace navi {

constexpr size_t SAMPLE_SIZE = (sizeof(QuerySample) + 2 * sizeof(QuerySamplePtr));

bool QueryLRUEntry::collect(QuerySample *sample) {
    QuerySamplePtr samplePtr(sample);
    sample->queryStr = _queryStrMap.getStr(sample->query, sample->queryHash);
    if (dedupSample(samplePtr)) {
        return false;
    }
    sample->query.clear();
    _querySampleList.push_back(samplePtr);
    _querySampleMap[sample->queryHash] = samplePtr;
    return true;
}

bool QueryLRUEntry::dedupSample(const QuerySamplePtr &sample) {
    auto it = _querySampleMap.find(sample->queryHash);
    if (_querySampleMap.end() != it) {
        const auto &prevSample = it->second;
        prevSample->session = sample->session;
        prevSample->clientIp = sample->clientIp;
        prevSample->isHttp = sample->isHttp;
        prevSample->dupCount++;
        prevSample->lastBeginTimeUs = sample->beginTime;
        prevSample->lastEndTimeUs = sample->endTime;
        _queryStrMap.removeStr(sample->queryStr);
        return true;
    }
    return false;
}

bool QueryLRUEntry::drop(int64_t remainCount) {
    if (_querySampleList.size() > remainCount) {
        auto event = _querySampleList.front();
        _querySampleList.pop_front();
        _queryStrMap.removeStr(event->queryStr);
        _querySampleMap.erase(event->queryHash);
        return true;
    } else {
        return false;
    }
}

void QueryLRUEntry::toProto(int64_t &maxCount, QueryLRUEntryDef &historyDef, SymbolTableDef &symbolTableDef) const {
    if (_querySampleList.empty()) {
        return;
    }
    auto &symbolTable = *symbolTableDef.mutable_table();
    for (auto it = _querySampleList.rbegin(); it != _querySampleList.rend(); ++it) {
        const auto &sample = *it;
        if (sample->beginTime > sample->endTime) {
            continue;
        }
        auto sampleDef = historyDef.add_querys();
        if (sample->error) {
            sample->error->toProto(*sampleDef->mutable_error());
        } else {
            NaviError::fillSessionId(sampleDef->mutable_session(), sample->session);
        }
        auto queryHash = sample->queryHash;
        if (sample->queryStr) {
            auto it = symbolTable.find(queryHash);
            if (symbolTable.end() == it) {
                symbolTable[queryHash] = *sample->queryStr;
            }
        }
        auto queryRpcInfoDef = sampleDef->mutable_rpc_info();
        queryRpcInfoDef->set_arpc_method(sample->arpcMethod);
        queryRpcInfoDef->set_http_path(sample->httpPath);
        queryRpcInfoDef->set_client_ip(sample->clientIp);
        queryRpcInfoDef->set_is_http(sample->isHttp);
        queryRpcInfoDef->set_navi_web_query(sample->naviWebQuery);
        sampleDef->set_query_key(sample->queryKey);
        sampleDef->set_query_hash(queryHash);
        sampleDef->set_response_size(sample->responseSize);
        sampleDef->set_begin_time_us(sample->beginTime);
        sampleDef->set_end_time_us(sample->endTime);
        sampleDef->set_dup_count(sample->dupCount);
        sampleDef->set_queue_us(sample->queueUs);
        sampleDef->set_compute_us(sample->computeUs);
        sampleDef->set_last_begin_time_us(sample->lastBeginTimeUs);
        sampleDef->set_last_end_time_us(sample->lastEndTimeUs);
        if (maxCount > 0) {
            maxCount--;
        }
        if (0 == maxCount) {
            break;
        }
    }
}

void QueryLRUEntry::fromProto(const QueryLRUEntryDef &historyDef, const SymbolTableDef &symbolTableDef) {
    auto count = historyDef.querys_size();
    const auto &symbolTable = symbolTableDef.table();
    for (int64_t i = count - 1; i >= 0; i--) {
        const auto &queryDef = historyDef.querys(i);
        if (queryDef.begin_time_us() > queryDef.end_time_us()) {
            continue;
        }
        auto querySample = std::make_shared<QuerySample>();
        if (queryDef.has_error()) {
            auto error = std::make_shared<NaviError>();
            error->fromProto(queryDef.error());
            querySample->error = error;
            querySample->session = querySample->error->id;
        } else {
            NaviError::parseSessionId(queryDef.session(), querySample->session);
        }
        const auto &queryRpcInfoDef = queryDef.rpc_info();
        querySample->arpcMethod = queryRpcInfoDef.arpc_method();
        querySample->httpPath = queryRpcInfoDef.http_path();
        querySample->clientIp = queryRpcInfoDef.client_ip();
        querySample->isHttp = queryRpcInfoDef.is_http();
        querySample->naviWebQuery = queryRpcInfoDef.navi_web_query();
        querySample->queryKey = queryDef.query_key();
        {
            auto queryHash = queryDef.query_hash();
            auto it = symbolTable.find(queryHash);
            if (symbolTable.end() != it) {
                const auto &queryStr = it->second;
                querySample->queryStr = _queryStrMap.getStr(queryStr, querySample->queryHash);
            } else {
                NAVI_KERNEL_LOG(ERROR, "query hash not found, ignored");
                continue;
            }
        }
        querySample->responseSize = queryDef.response_size();
        querySample->beginTime = queryDef.begin_time_us();
        querySample->endTime = queryDef.end_time_us();
        querySample->dupCount = queryDef.dup_count();
        querySample->queueUs = queryDef.queue_us();
        querySample->computeUs = queryDef.compute_us();
        querySample->lastBeginTimeUs = queryDef.last_begin_time_us();
        querySample->lastEndTimeUs = queryDef.last_end_time_us();
        _querySampleList.push_back(querySample);
        _querySampleMap[querySample->queryHash] = querySample;
    }
}

QueryLRUHistory::QueryLRUHistory() {
}

QueryLRUHistory::~QueryLRUHistory() {
}

void QueryLRUHistory::updateConfig(const QueryLRUHistoryConfig &config) {
    autil::ScopedWriteLock scope(_lock);
    _config = config;
    dropToFit();
}

void QueryLRUHistory::collect(QuerySample *sample) {
    auto entryKey = getErrorKey(sample->error);
    autil::ScopedWriteLock scope(_lock);
    QueryLRUEntryPtr queryEntry;
    int64_t oldId = -1;
    auto it = _queryMap.find(entryKey);
    if (_queryMap.end() == it) {
        queryEntry = std::make_shared<QueryLRUEntry>(_queryStrMap);
        _queryMap[entryKey] = queryEntry;
    } else {
        queryEntry = it->second;
        oldId = queryEntry->getId();
    }
    queryEntry->setId(_idCounter++);
    if (queryEntry->collect(sample)) {
        _sampleCount++;
    }
    updatLRU(oldId, queryEntry);
    dropToFit();
}

void QueryLRUHistory::updatLRU(int64_t oldId, const QueryLRUEntryPtr &queryEntry) {
    if (oldId >= 0) {
        _queryLRU.erase(oldId);
    }
    _queryLRU[queryEntry->getId()] = queryEntry;
}

void QueryLRUHistory::dropToFit() {
    doDropToFit(10);
    if (isQuotaOverflow()) {
        doDropToFit(0);
    }
}

void QueryLRUHistory::doDropToFit(int64_t threshold) {
    for (auto it = _queryLRU.begin(); _queryLRU.end() != it;) {
        auto &entry = *it->second;
        while (isQuotaOverflow()) {
            if (entry.drop(threshold)) {
                _sampleCount--;
            } else {
                break;
            }
        }
        if (0 == entry.size()) {
            it = _queryLRU.erase(it);
        } else {
            it++;
        }
    }
}

bool QueryLRUHistory::isQuotaOverflow() const {
    if (_sampleCount > _config.maxCount) {
        return true;
    }
    auto totalSize = SAMPLE_SIZE * _sampleCount + _queryStrMap.getTotalSize() +
                     2 * _queryMap.size() * sizeof(std::pair<size_t, QueryLRUEntryPtr>);
    return totalSize > _config.maxSize;
}

void QueryLRUHistory::toProto(bool shortQueryHistory, QueryLRUHistoryDef &historyDef, SymbolTableDef &symbolTableDef) const {
    autil::ScopedReadLock scope(_lock);
    int64_t maxCount = shortQueryHistory ? 5 : -1;
    for (auto it = _queryLRU.rbegin(); it != _queryLRU.rend(); ++it) {
        if (0 == maxCount) {
            break;
        }
        it->second->toProto(maxCount, *historyDef.add_entrys(), symbolTableDef);
    }
}

void QueryLRUHistory::fromProto(const QueryLRUHistoryDef &historyDef, const SymbolTableDef &symbolTableDef) {
    autil::ScopedWriteLock scope(_lock);
    int64_t count = historyDef.entrys_size();
    for (int64_t i = count - 1; i >= 0; i--) {
        const auto &entryDef = historyDef.entrys(i);
        if (entryDef.querys_size() == 0) {
            continue;
        }
        auto lruEntry = std::make_shared<QueryLRUEntry>(_queryStrMap);
        lruEntry->fromProto(entryDef, symbolTableDef);
        lruEntry->setId(_idCounter++);
        auto querySample = lruEntry->getFirstSample();
        if (!querySample) {
            continue;
        }
        auto entryKey = getErrorKey(querySample->error);
        _queryMap[entryKey] = lruEntry;
        updatLRU(-1, lruEntry);
        _sampleCount += lruEntry->size();
    }
    dropToFit();
}

void QueryLRUHistory::parseConfig(const QueryLRUHistoryConfigDef &configDef) {
    _config.maxSize = configDef.max_size();
    _config.maxCount = configDef.max_count();
}

void QueryLRUHistory::fillConfig(QueryLRUHistoryConfigDef &configDef) const {
    configDef.set_max_size(_config.maxSize);
    configDef.set_max_count(_config.maxCount);
    configDef.set_sample_size(SAMPLE_SIZE);
    configDef.set_str_map_size(_queryStrMap.size());
    configDef.set_str_map_mem_size(_queryStrMap.getTotalSize());
}

size_t QueryLRUHistory::getErrorKey(const NaviErrorPtr &error) {
    size_t hash = 0;
    if (error) {
        autil::HashUtil::combineHash(hash, 1);
        autil::HashUtil::combineHash(hash, (size_t)error->ec);
        if (error->errorEvent) {
            autil::HashUtil::combineHash(hash, 2);
            autil::HashUtil::combineHash(hash, error->errorEvent->file);
            autil::HashUtil::combineHash(hash, (size_t)error->errorEvent->line);
            autil::HashUtil::combineHash(hash, error->errorEvent->func);
            autil::HashUtil::combineHash(hash, (size_t)error->errorEvent->level);
        } else {
            autil::HashUtil::combineHash(hash, 3);
        }
    } else {
        autil::HashUtil::combineHash(hash, 0);
    }
    return hash;
}

}

