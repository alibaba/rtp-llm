#include "aios/navi_sched/config/NaviSchedConfigBuilder.h"
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

autil::legacy::json::JsonMap NaviSchedConfigBuilder::makeLocalNode(
    const NaviSchedLocalNode &node)
{
    autil::legacy::json::JsonMap value;
    value["biz_name"] = node.bizName;
    value["part_count"] = node.partCount;
    value["part_id"] = node.partId;
    value["version"] = node.version;
    value["weight"] = node.weight;
    value["ip"] = node.ip;
    value["http_port"] = node.httpPort;
    value["tcp_port"] = node.tcpPort;
    value["grpc_port"] = node.grpcPort;
    value["support_heartbeat"] = node.supportHeartbeat;
    return value;
}

autil::legacy::json::JsonMap NaviSchedConfigBuilder::makeCm2Subscription(
    const NaviSchedCm2Subscription &config)
{
    autil::legacy::json::JsonMap value;
    value["zk_host"] = config.zkHost;
    value["zk_path"] = config.zkPath;
    if (!config.cm2Part.empty()) {
        value["cm2_part"] = makeStringArray(config.cm2Part);
    }
    if (!config.cm2Group.empty()) {
        value["cm2_group"] = makeStringArray(config.cm2Group);
    }
    if (!config.cm2PartNoTopoInfo.empty()) {
        value["cm2_part_no_topo_info"] = makeStringArray(config.cm2PartNoTopoInfo);
    }
    if (config.cm2AllGroupTopo) {
        value["cm2_all_group_topo"] = true;
    }
    if (!config.serviceKey.empty()) {
        value["service_key"] = config.serviceKey;
    }
    return value;
}

autil::legacy::json::JsonMap NaviSchedConfigBuilder::makeSubscribeConfig(
    const NaviSchedRuntimeConfig &config)
{
    autil::legacy::json::JsonMap subscribeConfig;
    autil::legacy::json::JsonArray localNodes;
    for (const auto &node : config.localNodes) {
        localNodes.emplace_back(makeLocalNode(node));
    }
    subscribeConfig["local"] = std::move(localNodes);
    if (1 == config.cm2Subscriptions.size()) {
        subscribeConfig["cm2"] = makeCm2Subscription(config.cm2Subscriptions.front());
    } else if (!config.cm2Subscriptions.empty()) {
        autil::legacy::json::JsonArray cm2Configs;
        for (const auto &cm2 : config.cm2Subscriptions) {
            cm2Configs.emplace_back(makeCm2Subscription(cm2));
        }
        subscribeConfig["cm2_configs"] = std::move(cm2Configs);
    }
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
    const auto &spectrum = config.spectrumSubscription;
    if (!spectrum.serviceId.empty()) {
        autil::legacy::json::JsonMap spectrumConfig;
        spectrumConfig["service_id"] = spectrum.serviceId;
        if (!spectrum.physicalServiceId.empty()) {
            spectrumConfig["physical_service_id"] = spectrum.physicalServiceId;
        }
        spectrumConfig["grpc_port_diff"] = spectrum.grpcPortDiff;
        subscribeConfig["spectrum"] = std::move(spectrumConfig);
    }
    return subscribeConfig;
}

std::string NaviSchedConfigBuilder::build(
    const NaviSchedRuntimeConfig &config, const std::vector<std::string> &kernelList,
    bool createGigClient, const NamedConfigs &kernelConfigs,
    const NamedConfigs &bizResourceConfigs,
    const std::vector<std::string> &bizResourceList, size_t grpcClientIoThreadNum,
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

}
