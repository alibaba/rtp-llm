#include <signal.h>
#include <unistd.h>
#include <iostream>

#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceManager.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"

using namespace rtp_llm;

volatile bool gStopFlag = false;

void signalHandler(int signum) {
    gStopFlag = true;
}

void registerSignal() {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
}

SubscribeServiceConfig makeConfig() {
    NacosSubscribeServiceConfig nacos_config;
    nacos_config.server_host = "127.0.0.1:8848";
    nacos_config.clusters.push_back("TestNamingService1");
    nacos_config.clusters.push_back("TestNamingService2");

    SubscribeServiceConfig config;
    config.nacos_configs.push_back(nacos_config);
    return config;
}

int main(int argc, char** argv) {
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(DEBUG);

    registerSignal();

    auto config                    = makeConfig();
    auto subscribe_service_manager = std::make_shared<SubscribeServiceManager>();
    if (!subscribe_service_manager->init(config)) {
        std::cout << "manager init failed" << std::endl;
        return -1;
    }

    while (!gStopFlag) {
        std::vector<std::shared_ptr<const TopoNode>> topo_nodes;
        if (!subscribe_service_manager->getTopoNodes(topo_nodes)) {
            std::cout << "get topo nodes failed" << std::endl;
            return -1;
        }

        std::cout << "toponode size is " << topo_nodes.size() << std::endl;
        for (auto& toponode : topo_nodes) {
            std::cout << *toponode << std::endl;
        }
        sleep(1);
    }

    return 0;
}