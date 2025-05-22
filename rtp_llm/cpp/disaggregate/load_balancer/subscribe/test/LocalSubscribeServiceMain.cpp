#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceManager.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"

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
    LocalNodeJsonize node1("biz1", "127.0.0.1", 12345);
    LocalNodeJsonize node2("biz2", "127.0.0.1", 12345);
    LocalNodeJsonize node3("biz2", "127.0.0.1", 12346);

    LocalSubscribeServiceConfig local_config;
    local_config.nodes.push_back(node1);
    local_config.nodes.push_back(node2);
    local_config.nodes.push_back(node3);

    SubscribeServiceConfig config;
    config.local_configs.push_back(local_config);

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