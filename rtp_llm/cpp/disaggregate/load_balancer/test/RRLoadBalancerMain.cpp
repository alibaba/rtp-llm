#include "rtp_llm/cpp/disaggregate/load_balancer/RRLoadBalancer.h"

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

LoadBalancerInitParams makeConfig() {
    LocalNodeJsonize node1("biz1", "127.0.0.1", 12345);
    LocalNodeJsonize node2("biz2", "127.0.0.1", 12345);
    LocalNodeJsonize node3("biz2", "127.0.0.1", 12346);

    LocalSubscribeServiceConfig local_config;
    local_config.nodes.push_back(node1);
    local_config.nodes.push_back(node2);
    local_config.nodes.push_back(node3);

    SubscribeServiceConfig config;
    config.local_configs.push_back(local_config);

    LoadBalancerInitParams params;
    params.subscribe_config = config;

    return params;
}

int main(int argc, char** argv) {
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);

    registerSignal();

    auto config        = makeConfig();
    auto load_balancer = std::make_shared<RRLoadBalancer>();
    if (!load_balancer->init(config)) {
        std::cout << "load_balancer init failed" << std::endl;
        return -1;
    }

    while (!gStopFlag) {
        auto host = load_balancer->chooseHost("biz1");
        if (!host) {
            std::cout << "choose biz1 host failed" << std::endl;
        } else {
            std::cout << "choose biz1 host " << host->ip << ":" << host->rpc_port << std::endl;
        }

        host = load_balancer->chooseHost("biz2");
        if (!host) {
            std::cout << "choose biz2 host failed" << std::endl;
        } else {
            std::cout << "choose biz2 host " << host->ip << ":" << host->rpc_port << std::endl;
        }
        sleep(1);
    }

    return 0;
}