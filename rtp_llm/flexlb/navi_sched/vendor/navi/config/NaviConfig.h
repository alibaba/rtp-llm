#pragma once

#include "navi/common.h"
#include "navi/config/GraphConfig.h"
#include "navi/config/LogConfig.h"
#include "navi/config/NaviRegistryConfigMap.h"
#include <multi_call/config/FlowControlConfig.h>
#include "autil/legacy/jsonizable.h"
#include <map>

namespace navi {

class ConcurrencyConfigDef;
class EngineConfigDef;
class SnapshotConfigDef;

class UpdateHistoryConfig : public autil::legacy::Jsonizable {
public:
    UpdateHistoryConfig();
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    int64_t maxSize;
    int64_t maxCount;
};

class QueryLRUHistoryConfig : public autil::legacy::Jsonizable {
public:
    QueryLRUHistoryConfig();
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    int64_t maxSize;
    int64_t maxCount;
};

class QueryHistoryConfig : public autil::legacy::Jsonizable {
public:
    QueryHistoryConfig();
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    int64_t sampleIntervalUs;
    QueryLRUHistoryConfig normal;
    QueryLRUHistoryConfig error;
};

class HistoryConfig : public autil::legacy::Jsonizable {
public:
    HistoryConfig();
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    UpdateHistoryConfig update;
    QueryHistoryConfig query;
};

class BizPublishInfos : public autil::legacy::Jsonizable {
public:
    BizPublishInfos() {
    }
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    std::set<std::string> bizAliasNames;
    int64_t version = 0;
    bool disablePublish = false;
};

class ResourceReplaceConfig  : public autil::legacy::Jsonizable {
public:
    ResourceReplaceConfig() {
    }
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    std::string from;
    std::string to;
};

class CacheConfig : public autil::legacy::Jsonizable {
public:
    CacheConfig();
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    int64_t maxCacheSizeMB;
};

class NaviBizConfig : public autil::legacy::Jsonizable {
public:
    NaviBizConfig();
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    NaviPartId partCount;
    std::set<NaviPartId> partIds;
    std::string configPath;
    std::string metaInfo;
    std::vector<std::string> modules;
    std::set<std::string> kernels;
    std::set<std::string> kernelBlacklist;
    std::set<std::string> resources;
    std::vector<NaviRegistryConfig> kernelConfigVec;
    std::vector<NaviRegistryConfig> resourceConfigVec;
    std::map<std::string, GraphConfig> graphConfigMap;
    BizPublishInfos publishInfos;
    std::shared_ptr<multi_call::FlowControlConfig> flowControlConfig;
    std::vector<ResourceReplaceConfig> resourceReplaceInfos;
    CacheConfig cacheConfig;
};

NAVI_TYPEDEF_PTR(NaviBizConfig);

class ConcurrencyConfig : public autil::legacy::Jsonizable {
public:
    ConcurrencyConfig();
    ConcurrencyConfig(int threadNum_, size_t queueSize_, size_t processingSize_);
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
    void fillProto(ConcurrencyConfigDef &def) const;
public:
    int threadNum;
    size_t minThreadNum;
    size_t maxThreadNum;
    size_t maxThreadPoolQueueSize;
    size_t queueSize;
    size_t processingSize;
    int threadNiceValue;
};

class EngineConfig : public autil::legacy::Jsonizable {
public:
    EngineConfig();
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
    void fillProto(EngineConfigDef &def) const;
public:
    bool disablePerf;
    bool disableSymbolTable;
    ConcurrencyConfig builtinTaskQueue;
    std::map<std::string, ConcurrencyConfig> extraTaskQueueMap;
};

typedef std::map<std::string, NaviBizConfig> NaviBizConfigMap;

class NaviConfig : public autil::legacy::Jsonizable
{
public:
    NaviConfig();
    ~NaviConfig();
private:
    NaviConfig(const NaviConfig &);
    NaviConfig &operator=(const NaviConfig &);
public:
    bool loadConfig(const std::string &configStr);
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
    void fillSnapshotConfig(SnapshotConfigDef &configDef) const;
public:
    static bool parseToDocument(std::string jsonStr,
                                autil::legacy::RapidDocument &document);
    static bool prettyDumpToStr(const autil::legacy::Jsonizable &t, std::string &configStr);
public:
    autil::legacy::RapidDocument document;
    std::string configStr;
    std::string configPath;
    std::vector<std::string> modules;
    LogConfig logConfig;
    EngineConfig engineConfig;
    std::set<std::string> resources;
    std::set<std::string> kernels;
    std::vector<NaviRegistryConfig> snapshotResourceConfigVec;
    std::string defaultBizName;
    NaviBizConfigMap bizMap;
    size_t sleepBeforeUpdateUs = 0;
    size_t sleepAfterUpdateUs = 0;
    bool ignoreProduceError = true;
    bool collectInitMetric = true;
    bool disableGigAllReplicaStopped = false;
    int64_t initTimeoutMs = DEFAULT_TIMEOUT_MS;
    int64_t gigSmoothIntervalUs = 10 * 1000 * 1000;
    HistoryConfig historyConfig;
};

NAVI_TYPEDEF_PTR(NaviConfig);

}
