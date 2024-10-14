#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceManager.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"

#include <signal.h>
#include <unistd.h>
#include <iostream>

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
    CM2SubscribeServiceConfig cm2_config;
    cm2_config.zk_host = "search-zk-write-cm2-ea119cloud-pre.vip.tbsite.net:12181";  // pre
    cm2_config.zk_path = "/cm_server_common";
    cm2_config.clusters.push_back("mainse_excellent_rank");
    cm2_config.clusters.push_back("mainse_summary_pre");

    SubscribeServiceConfig config;
    config.cm2_configs.push_back(cm2_config);

    return config;
}

int main(int argc, char** argv) {
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);

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