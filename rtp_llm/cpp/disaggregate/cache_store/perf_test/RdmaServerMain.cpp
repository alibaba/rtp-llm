#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServer.h"
#include <unistd.h>
#include "kmonitor/client/KMonitorFactory.h"

using namespace rtp_llm;
using namespace std;

volatile bool gStopFlag = false;
void          handleSignal(int sig) {
    std::cout << "receive server stopped sig: " << sig << std::endl;
    gStopFlag = true;
}

void registerSignalHandler() {
    signal(SIGINT, handleSignal);
    signal(SIGQUIT, handleSignal);
    signal(SIGTERM, handleSignal);
    signal(SIGUSR1, handleSignal);
    signal(SIGUSR2, handleSignal);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " <port>" << std::endl;
        return -1;
    }
    int port = atoi(argv[1]);

    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);

    // Init kmonitor
    std::string kmonConf = R"(
        {
            "tenant_name": "default",
            "service_name": "rtp_llm",
            "system_metrics": true,
            "sink_period": 1,
            "sink_address": "localhost:4141",
            "global_tags": {
            },
            "assigned": false
        }
    )";
    kmonitor::KMonitorFactory::Init(kmonConf);
    kmonitor::KMonitorFactory::Start();

    auto rdma_server = std::make_shared<RdmaServer>(createMemoryUtilImpl(true));
    if (!rdma_server || !rdma_server->init(port, 4, 2)) {
        std::cout << "make rdma server failed" << std::endl;
        return -1;
    }

    while (!gStopFlag) {
        usleep(1000 * 1000);
    }

    std::cout << "ready to exit" << std::endl;
    return 0;
}
