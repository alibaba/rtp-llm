#include "navi/config/NaviConfig.h"
#include "autil/legacy/jsonizable_exception.h"
#include "rapidjson/prettywriter.h"
#include "navi/log/NaviLogger.h"
#include "navi/proto/NaviSummary.pb.h"
#include "rapidjson/error/en.h"

namespace navi {

UpdateHistoryConfig::UpdateHistoryConfig()
    : maxSize(DEFAULT_HISTORY_MAX_UPDATE_MEM_SIZE)
    , maxCount(DEFAULT_HISTORY_MAX_UPDATE_EVENT_COUNT)
{
}

void UpdateHistoryConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("max_size", maxSize, maxSize);
    json.Jsonize("max_count", maxCount, maxCount);
}

QueryLRUHistoryConfig::QueryLRUHistoryConfig()
    : maxSize(DEFAULT_HISTORY_MAX_QUERY_MEM_SIZE / 2)
    , maxCount(DEFAULT_HISTORY_MAX_QUERY_EVENT_COUNT / 2)
{
}

void QueryLRUHistoryConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("max_size", maxSize, maxSize);
    json.Jsonize("max_count", maxCount, maxCount);
}

QueryHistoryConfig::QueryHistoryConfig()
    : sampleIntervalUs(DEFAULT_HISTORY_QUERY_SAMPLE_INTERVAL_US)
{
}

void QueryHistoryConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("sample_interval_us", sampleIntervalUs, sampleIntervalUs);
    json.Jsonize("normal_query", normal, normal);
    json.Jsonize("error_query", error, error);
}

HistoryConfig::HistoryConfig() {
}

void HistoryConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("update_history", update, update);
    json.Jsonize("query_history", query, query);
}

void BizPublishInfos::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("biz_alias_names", bizAliasNames, bizAliasNames);
    json.Jsonize("version", version, version);
    json.Jsonize("disable_publish", disablePublish, disablePublish);
}

void ResourceReplaceConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("from", from, from);
    json.Jsonize("to", to, to);
}

CacheConfig::CacheConfig()
    : maxCacheSizeMB(256)
{
}

void CacheConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("max_cache_size_mb", maxCacheSizeMB, maxCacheSizeMB);
}

NaviBizConfig::NaviBizConfig()
    : partCount(INVALID_NAVI_PART_COUNT)
{
}

void NaviBizConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("part_count", partCount);
    json.Jsonize("part_ids", partIds);
    json.Jsonize("config_path", configPath, configPath);
    json.Jsonize("meta_info", metaInfo, metaInfo);
    json.Jsonize("modules", modules, modules);
    json.Jsonize("kernel_list", kernels, kernels);
    json.Jsonize("kernel_blacklist", kernelBlacklist, kernelBlacklist);
    json.Jsonize("resource_list", resources, resources);
    json.Jsonize("graph_config", graphConfigMap, graphConfigMap);
    json.Jsonize("kernel_config", kernelConfigVec, kernelConfigVec);
    json.Jsonize("resource_config", resourceConfigVec, resourceConfigVec);
    json.Jsonize("publish_infos", publishInfos, publishInfos);
    json.Jsonize("resource_replace_infos", resourceReplaceInfos, resourceReplaceInfos);
    json.Jsonize("cache_config", cacheConfig, cacheConfig);
    if (json.GetMode() == FROM_JSON) {
        std::string flowControlStr;
        json.JsonizeAsString("flow_control_config", flowControlStr, std::string());
        if (!flowControlStr.empty()) {
            auto config = std::make_shared<multi_call::FlowControlConfig>();
            json.Jsonize("flow_control_config", *config);
            flowControlConfig = config;
        } else {
            flowControlConfig.reset();
        }
    } else {
        if (flowControlConfig) {
            json.Jsonize("flow_control_config", *flowControlConfig);
        }
    }
}

ConcurrencyConfig::ConcurrencyConfig()
    : threadNum(DEFAULT_THREAD_NUMBER)
    , minThreadNum(DEFAULT_THREAD_NUMBER)
    , maxThreadNum(DEFAULT_THREAD_NUMBER)
    , maxThreadPoolQueueSize(0u)
    , queueSize(DEFAULT_QUEUE_SIZE)
    , processingSize(DEFAULT_PROCESSING_SIZE)
    , threadNiceValue(DEFAULT_THREAD_NICE_VALUE)
{}

ConcurrencyConfig::ConcurrencyConfig(int threadNum_, size_t queueSize_, size_t processingSize_)
    : threadNum(threadNum_)
    , minThreadNum(DEFAULT_THREAD_NUMBER)
    , maxThreadNum(DEFAULT_THREAD_NUMBER)
    , queueSize(queueSize_)
    , processingSize(processingSize_)
    , threadNiceValue(DEFAULT_THREAD_NICE_VALUE)
{}

void ConcurrencyConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("thread_num", threadNum, threadNum);
    json.Jsonize("min_thread_num", minThreadNum, minThreadNum);
    json.Jsonize("max_thread_num", maxThreadNum, maxThreadNum);
    json.Jsonize("max_thread_pool_queue_size", maxThreadPoolQueueSize, maxThreadPoolQueueSize);
    json.Jsonize("queue_size", queueSize, queueSize);
    json.Jsonize("processing_size", processingSize, processingSize);
    json.Jsonize("thread_nice_value", threadNiceValue, threadNiceValue);
}

void ConcurrencyConfig::fillProto(ConcurrencyConfigDef &def) const {
    def.set_thread_num(threadNum);
    def.set_min_thread_num(minThreadNum);
    def.set_max_thread_num(maxThreadNum);
    def.set_max_thread_pool_queue_size(maxThreadPoolQueueSize);
    def.set_queue_size(queueSize);
    def.set_processing_size(processingSize);
    def.set_thread_nice_value(threadNiceValue);
}

EngineConfig::EngineConfig()
    : disablePerf(false)
    , disableSymbolTable(false)
{
}

void EngineConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("disable_perf", disablePerf, disablePerf);
    json.Jsonize("disable_symbol_table", disableSymbolTable, disableSymbolTable);
    json.Jsonize("builtin_task_queue", builtinTaskQueue, builtinTaskQueue);
    json.Jsonize("extra_task_queue", extraTaskQueueMap, extraTaskQueueMap);
}

void EngineConfig::fillProto(EngineConfigDef &def) const {
    def.set_disable_perf(disablePerf);
    def.set_disable_symbol_table(disableSymbolTable);
    builtinTaskQueue.fillProto(*def.mutable_builtin_task_queue());
    for (const auto &pair : extraTaskQueueMap) {
        auto &queueDef = (*def.mutable_extra_task_queues())[pair.first];
        pair.second.fillProto(queueDef);
    }
}

NaviConfig::NaviConfig() {
}

NaviConfig::~NaviConfig() {
}

bool NaviConfig::loadConfig(const std::string &jsonStr) {
    auto paddedConfigStr = jsonStr + SIMD_PADING_STR;
    document.Parse(paddedConfigStr.c_str());
    if (document.HasParseError()) {
        NAVI_KERNEL_LOG(ERROR, "invalid navi config [%s], error [%s (%lu)]",
                        jsonStr.c_str(),
                        rapidjson::GetParseError_En(document.GetParseError()),
                        document.GetErrorOffset());
        return false;
    }
    try {
        autil::legacy::FromRapidValue(*this, document);
    } catch (const autil::legacy::ExceptionBase &e) {
        NAVI_KERNEL_LOG(ERROR, "invalid config [%s], msg [%s]", jsonStr.c_str(),
                        e.GetMessage().c_str());
        return false;
    }
    configStr = jsonStr;
    return true;
}

bool NaviConfig::prettyDumpToStr(const autil::legacy::Jsonizable &t,
                                 std::string &configStr)
{
    try {
        auto compactJson = FastToJsonString(t);
        autil::legacy::RapidDocument prettyDoc;
        auto ret = parseToDocument(std::move(compactJson), prettyDoc);
        assert(ret);
        (void)ret;
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(
            buffer);
        prettyDoc.Accept(writer);
        configStr.assign(buffer.GetString(), buffer.GetSize());
    } catch (const autil::legacy::ExceptionBase &e) {
        NAVI_KERNEL_LOG(ERROR, "convert config to json failed");
        return false;
    }
    return true;
}

void NaviConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("config_path", configPath, configPath);
    json.Jsonize("modules", modules, modules);
    json.Jsonize("log_config", logConfig, logConfig);
    json.Jsonize("engine_config", engineConfig, engineConfig);
    json.Jsonize("kernel_list", kernels, kernels);
    json.Jsonize("resource_list", resources, resources);
    json.Jsonize("default_biz", defaultBizName, defaultBizName);
    json.Jsonize("biz_config", bizMap, bizMap);
    json.Jsonize("resource_config", snapshotResourceConfigVec,
                 snapshotResourceConfigVec);
    json.Jsonize("sleep_before_update_us", sleepBeforeUpdateUs,
                 sleepBeforeUpdateUs);
    json.Jsonize("sleep_after_update_us", sleepAfterUpdateUs,
                 sleepAfterUpdateUs);
    json.Jsonize("ignore_produce_error", ignoreProduceError, ignoreProduceError);
    json.Jsonize("collect_init_metric", collectInitMetric, collectInitMetric);
    json.Jsonize("disable_gig_all_replica_stopped", disableGigAllReplicaStopped, disableGigAllReplicaStopped);
    json.Jsonize("init_timeout_ms", initTimeoutMs, initTimeoutMs);
    json.Jsonize("gig_smooth_interval_us", gigSmoothIntervalUs, gigSmoothIntervalUs);
    json.Jsonize("history_config", historyConfig, historyConfig);
}

void NaviConfig::fillSnapshotConfig(SnapshotConfigDef &configDef) const {
    configDef.set_config_path(configPath);
    for (const auto &path : modules) {
        configDef.add_modules(path);
    }
    logConfig.fillProto(*configDef.mutable_log_config());
    engineConfig.fillProto(*configDef.mutable_engine_config());
    configDef.set_default_biz_name(defaultBizName);
    configDef.set_sleep_before_update_us(sleepBeforeUpdateUs);
    configDef.set_sleep_after_update_us(sleepAfterUpdateUs);
    configDef.set_ignore_produce_error(ignoreProduceError);
    configDef.set_collect_init_metric(collectInitMetric);
    configDef.set_init_timeout_ms(initTimeoutMs);
    configDef.set_gig_smooth_interval_us(gigSmoothIntervalUs);
    configDef.set_disable_gig_all_replica_stopped(disableGigAllReplicaStopped);
}

bool NaviConfig::parseToDocument(std::string jsonStr,
                                 autil::legacy::RapidDocument &document)
{
    if (jsonStr.empty()) {
        jsonStr = "{}";
    }
    jsonStr += SIMD_PADING_STR;
    document.Parse(jsonStr.c_str());
    if (document.HasParseError()) {
        NAVI_KERNEL_LOG(ERROR, "invalid json, error [%s (%lu)], json: %s",
                        rapidjson::GetParseError_En(document.GetParseError()),
                        document.GetErrorOffset(), jsonStr.c_str());
        return false;
    }
    return true;
}
}
