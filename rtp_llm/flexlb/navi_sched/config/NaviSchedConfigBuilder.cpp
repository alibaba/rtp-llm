#include "rtp_llm/flexlb/navi_sched/config/NaviSchedConfigBuilder.h"
#include "autil/legacy/jsonizable.h"

namespace navi_sched {

autil::legacy::json::JsonArray NaviSchedConfigBuilder::makeStringArray(
    const std::vector<std::string> &values)
{
    autil::legacy::json::JsonArray result;
    for (const auto &value : values) {
        result.emplace_back(value);
    }
    return result;
}

autil::legacy::json::JsonArray NaviSchedConfigBuilder::makeNamedConfigArray(
    const NaviSchedConfigBuilder::NamedConfigs &configs)
{
    autil::legacy::json::JsonArray result;
    for (const auto &[name, configJson] : configs) {
        autil::legacy::json::JsonMap value;
        value["name"] = name;
        value["config"] = autil::legacy::json::ParseJson(configJson);
        result.emplace_back(std::move(value));
    }
    return result;
}

autil::legacy::json::JsonMap NaviSchedConfigBuilder::makeSubscribeConfig(
    const NaviSchedRuntimeConfig &config)
{
    autil::legacy::json::JsonMap subscribeConfig;
    const auto &vip = config.vipSubscription;
    if (!vip.services.empty()) {
        autil::legacy::json::JsonMap vipConfig;
        vipConfig["jmenv_dom"] = vip.jmenvDomain;
        vipConfig["services"] = makeStringArray(vip.services);
        if (!vip.serviceKey.empty()) {
            vipConfig["service_key"] = vip.serviceKey;
        }
        if (0 != vip.grpcPortDiff) {
            vipConfig["grpc_port_diff"] = vip.grpcPortDiff;
        }
        vipConfig["support_heartbeat"] = true;
        vipConfig["query_full_api"] = true;
        subscribeConfig["vip"] = std::move(vipConfig);
    }
    return subscribeConfig;
}

std::string NaviSchedConfigBuilder::build(const NaviSchedRuntimeConfig &config,
                                          const std::vector<std::string> &kernelList,
                                          bool createGigClient,
                                          const NamedConfigs &kernelConfigs,
                                          const NamedConfigs &bizResourceConfigs,
                                          const std::vector<std::string> &bizResourceList,
                                          size_t grpcClientIoThreadNum,
                                          const std::map<std::string, size_t> &extraTaskQueueThreads)
{
    if (config.bizName.empty()) {
        return "";
    }
    autil::legacy::json::JsonMap taskQueue;
    taskQueue["thread_num"] = config.threadNum;
    taskQueue["queue_size"] = config.queueSize;
    taskQueue["processing_size"] = config.processingSize;
    autil::legacy::json::JsonMap engineConfig;
    engineConfig["builtin_task_queue"] = std::move(taskQueue);
    if (!extraTaskQueueThreads.empty()) {
        autil::legacy::json::JsonMap queues;
        for (const auto &[name, threadNum] : extraTaskQueueThreads) {
            if (name.empty() || threadNum == 0) {
                return "";
            }
            autil::legacy::json::JsonMap queue;
            queue["thread_num"] = threadNum;
            queue["queue_size"] = config.queueSize;
            queue["processing_size"] = config.processingSize;
            queues[name] = std::move(queue);
        }
        engineConfig["extra_task_queue"] = std::move(queues);
    }
    engineConfig["disable_perf"] = config.disablePerf;
    engineConfig["disable_symbol_table"] = config.disableSymbolTable;

    autil::legacy::json::JsonMap biz;
    biz["part_count"] = 1;
    biz["part_ids"] = autil::legacy::json::JsonArray{int32_t(0)};
    autil::legacy::json::JsonArray kernels;
    for (const auto &kernel : kernelList) {
        kernels.emplace_back(kernel);
    }
    biz["kernel_list"] = std::move(kernels);
    biz["kernel_config"] = makeNamedConfigArray(kernelConfigs);
    biz["resource_config"] = makeNamedConfigArray(bizResourceConfigs);
    biz["resource_list"] = makeStringArray(bizResourceList);
    autil::legacy::json::JsonMap bizConfig;
    bizConfig[config.bizName] = std::move(biz);

    autil::legacy::json::JsonArray resourceConfig;
    if (createGigClient) {
        autil::legacy::json::JsonMap initConfig;
        initConfig["subscribe_config"] = makeSubscribeConfig(config);
        if (grpcClientIoThreadNum > 0) {
            autil::legacy::json::JsonMap streamConfig;
            streamConfig["thread_num"] = grpcClientIoThreadNum;
            autil::legacy::json::JsonMap connectionConfig;
            connectionConfig["grpc_stream"] = std::move(streamConfig);
            initConfig["connection_config"] = std::move(connectionConfig);
        }
        autil::legacy::json::JsonMap gigConfig;
        gigConfig["init_config"] = std::move(initConfig);
        autil::legacy::json::JsonMap gigResource;
        gigResource["name"] = std::string("navi.buildin.gig_client");
        gigResource["config"] = std::move(gigConfig);
        resourceConfig.emplace_back(std::move(gigResource));
    }

    autil::legacy::json::JsonMap naviConfig;
    naviConfig["version"] = 0;
    naviConfig["engine_config"] = std::move(engineConfig);
    naviConfig["config_path"] = std::string(".");
    naviConfig["biz_config"] = std::move(bizConfig);
    naviConfig["resource_config"] = std::move(resourceConfig);
    return autil::legacy::ToJsonString(naviConfig, true);
}

} // namespace navi_sched
